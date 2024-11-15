import logging

from bs4 import BeautifulSoup, NavigableString, PageElement, Tag
from selenium.webdriver.remote.webelement import WebElement
from selenium import webdriver
from selenium.webdriver.common.by import By

from browser_use.dom.views import DomContentItem, ProcessedDomContent
from browser_use.utils import time_execution_sync


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class DomService:
	def __init__(self, driver: webdriver.Chrome):
		self.driver = driver
		self.xpath_cache = {}  # Add cache at instance level

	def get_clickable_elements(self) -> ProcessedDomContent:
		# Clear xpath cache on each new DOM processing
		self.xpath_cache = {}
		html_content = self.driver.page_source
		return self._process_content(html_content)

	@time_execution_sync('--_process_content')
	def _process_content(self, html_content: str) -> ProcessedDomContent:
		"""
		Process HTML content to extract and clean relevant elements.
		Args:
		    html_content: Raw HTML string to process
		Returns:
		    ProcessedDomContent: Processed DOM content

		@dev TODO: instead of of using enumerated index, use random 4 digit numbers -> a bit more tokens BUT updates on the screen wont click on incorrect items -> tricky because you have to consider that same elements need to have the same index ...
		"""

		# Parse HTML content using BeautifulSoup with html.parser
		soup = BeautifulSoup(html_content, 'html.parser')

		output_items: list[DomContentItem] = []
		selector_map: dict[int, str] = {}
		current_index = 0

		dom_queue = (
			[(element, [], None) for element in reversed(list(soup.body.children))]
			if soup.body
			else []
		)

		while dom_queue:
			element, path_indices, current_xpath = dom_queue.pop()

			if isinstance(element, Tag):
				if not self._is_element_accepted(element):
					element.decompose()
					continue

				# Generate simple xpath using tag name and sibling index
				siblings = (
					list(element.parent.find_all(element.name, recursive=False))
					if element.parent
					else []
				)
				sibling_index = siblings.index(element) + 1 if siblings else 1
				current_path = path_indices + [(element.name, sibling_index)]

				# Generate xpath string for current element
				element_xpath = '//' + '/'.join(f'{tag}[{idx}]' for tag, idx in current_path)

				# Add children to queue with their path information
				for child in reversed(list(element.children)):
					dom_queue.append((child, current_path, None))

				# Process interactive or leaf elements
				if self._is_interactive_element(element) or self._is_leaf_element(element):
					if (
						self._is_active(element)
						and self._is_top_element(element, xpath=element_xpath)
						and self._is_visible(element, xpath=element_xpath)
					):
						text_content = self._extract_text_from_all_children(element)
						tag_name = element.name

						attributes = self._get_essential_attributes(element)

						output_string = f"<{tag_name}{' ' + attributes if attributes else ''}>{text_content}</{tag_name}>"

						depth = len(current_path)

						output_items.append(
							DomContentItem(
								index=current_index, text=output_string, clickable=True, depth=depth
							)
						)

						selector_map[current_index] = element_xpath

						current_index += 1

			elif isinstance(element, NavigableString) and element.strip():
				if self._is_visible(element, xpath=current_xpath):
					# Skip text nodes that are direct children of already processed elements
					if element.parent in [e[0] for e in dom_queue]:
						continue

					text_content = self._cap_text_length(element.strip())
					if text_content:
						depth = len(path_indices)
						output_items.append(
							DomContentItem(
								index=current_index, text=text_content, clickable=False, depth=depth
							)
						)
						current_index += 1

		return ProcessedDomContent(items=output_items, selector_map=selector_map)

	def _cap_text_length(self, text: str, max_length: int = 150) -> str:
		if len(text) > max_length:
			half_length = max_length // 2
			return text[:half_length] + '...' + text[-half_length:]
		return text

	def _extract_text_from_all_children(self, element: Tag) -> str:
		# Tell BeautifulSoup that button tags can contain content
		# if not hasattr(element.parser, 'BUTTON_TAGS'):
		# 	element.parser.BUTTON_TAGS = set()

		text_content = ''
		for child in element.descendants:
			if isinstance(child, NavigableString):
				current_child_text = child.strip()
			else:
				current_child_text = child.get_text(strip=True)

			text_content += '\n' + current_child_text

		return self._cap_text_length(text_content.strip()) or ''

	def _is_interactive_element(self, element: Tag) -> bool:
		"""Check if element is interactive based on tag name and attributes."""
		interactive_elements = {
			'a',
			'button',
			'details',
			'embed',
			'input',
			'label',
			'menu',
			'menuitem',
			'object',
			'select',
			'textarea',
			'summary',
			# 'dialog',
			# 'div',
		}

		interactive_roles = {
			'button',
			'menu',
			'menuitem',
			'link',
			'checkbox',
			'radio',
			'slider',
			'tab',
			'tabpanel',
			'textbox',
			'combobox',
			'grid',
			'listbox',
			'option',
			'progressbar',
			'scrollbar',
			'searchbox',
			'switch',
			'tree',
			'treeitem',
			'spinbutton',
			'tooltip',
			# 'dialog',  # added
			# 'alertdialog',  # added
			'menuitemcheckbox',
			'menuitemradio',
		}

		return (
			element.name in interactive_elements
			or element.get('role') in interactive_roles
			or element.get('aria-role') in interactive_roles
			or element.get('tabindex') == '0'
		)

	def _is_leaf_element(self, element: Tag) -> bool:
		"""Check if element is a leaf element."""
		if not element.get_text(strip=True):
			return False

		if not list(element.children):
			return True

		# Check for simple text-only elements
		children = list(element.children)
		if len(children) == 1 and isinstance(children[0], str):
			return True

		return False

	def _is_element_accepted(self, element: Tag) -> bool:
		"""Check if element is accepted based on tag name and special cases."""
		leaf_element_deny_list = {'svg', 'iframe', 'script', 'style', 'link', 'meta'}

		# First check if it's in deny list
		if element.name in leaf_element_deny_list:
			return False

		return element.name not in leaf_element_deny_list

	def _get_essential_attributes(self, element: Tag) -> str:
		"""
		Collects essential attributes from an element.
		Args:
		    element: The BeautifulSoup PageElement
		Returns:
		    A string of formatted essential attributes
		"""
		essential_attributes = [
			'id',
			'class',
			'href',
			'src',
			'readonly',
			'disabled',
			'checked',
			'selected',
			'role',
			'type',  # Important for inputs, buttons
			'name',  # Important for form elements
			'value',  # Current value of form elements
			'placeholder',  # Helpful for understanding input purpose
			'title',  # Additional descriptive text
			'alt',  # Alternative text for images
			'for',  # Important for label associations
			'autocomplete',  # Form field behavior
		]

		# Collect essential attributes that have values
		attrs = []
		for attr in essential_attributes:
			if attr in element.attrs:
				element_attr = element[attr]
				if isinstance(element_attr, str):
					element_attr = element_attr[:50]
				elif isinstance(element_attr, (list, tuple)):
					element_attr = ' '.join(str(v)[:50] for v in element_attr)

				attrs.append(f'{attr}="{element_attr}"')

		state_attributes_prefixes = (
			'aria-',
			'data-',
		)

		# Collect data- attributes
		for attr in element.attrs:
			if attr.startswith(state_attributes_prefixes):
				attrs.append(f'{attr}="{element[attr]}"')

		return ' '.join(attrs)

	def _is_visible(self, element: Tag | NavigableString, xpath: str | None = None) -> bool:
		if not isinstance(element, Tag):
			return self._is_text_visible(element, xpath)

		element_id = element.get('id', '')
		if element_id:
			js_selector = f'document.getElementById("{element_id}")'
		else:
			js_selector = f'document.evaluate("{xpath}", document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue'

		visibility_check = f"""
			return (function() {{
				const element = {js_selector};
				if (!element) {{ return false; }}
				return Boolean(element.checkVisibility({{
					checkOpacity: true,
					checkVisibilityCSS: true
				}}));
			}}());
		"""

		try:
			is_visible = self.driver.execute_script(visibility_check)
			return bool(is_visible)
		except Exception:
			return False

	def _is_text_visible(self, element: NavigableString, parent_xpath: str | None = None) -> bool:
		"""Check if text node is visible using JavaScript."""
		parent = element.parent
		if not parent:
			return False

		visibility_check = f"""
			return (function() {{
				const parent = document.evaluate("{parent_xpath}", document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
				if (!parent) {{ return false; }}
				
				const range = document.createRange();
				const textNode = parent.childNodes[{list(parent.children).index(element)}];
				range.selectNodeContents(textNode);
				const rect = range.getBoundingClientRect();
				
				if (rect.width === 0 || rect.height === 0 || 
					rect.top < 0 || rect.top > window.innerHeight) {{
					return false;
				}}
				
				return Boolean(parent.checkVisibility({{
					checkOpacity: true,
					checkVisibilityCSS: true
				}}));
			}}());
		"""
		try:
			is_visible = self.driver.execute_script(visibility_check)
			return bool(is_visible)
		except Exception:
			return False

	def _is_top_element(self, element: Tag | NavigableString, xpath: str) -> bool:
		"""Check if element is the topmost at its position."""
		check_top = f"""
			return (function() {{
				const elem = document.evaluate("{xpath}", document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
				if (!elem) {{ return false; }}
				
				const rect = elem.getBoundingClientRect();
				const points = [
					{{x: rect.left + rect.width * 0.25, y: rect.top + rect.height * 0.25}},
					{{x: rect.left + rect.width * 0.75, y: rect.top + rect.height * 0.25}}, 
					{{x: rect.left + rect.width * 0.25, y: rect.top + rect.height * 0.75}},
					{{x: rect.left + rect.width * 0.75, y: rect.top + rect.height * 0.75}},
					{{x: rect.left + rect.width / 2, y: rect.top + rect.height / 2}}
				];
				
				return Boolean(points.some(point => {{
					const topEl = document.elementFromPoint(point.x, point.y);
					let current = topEl;
					while (current && current !== document.body) {{
						if (current === elem) return true;
						current = current.parentElement;
					}}
					return false;
				}}));
			}}());
		"""
		try:
			is_top = self.driver.execute_script(check_top)
			return bool(is_top)
		except Exception:
			logger.error(f'Error checking top element: {element}')
			return False

	def _is_active(self, element: Tag) -> bool:
		"""Check if element is active (not disabled)."""
		return not (
			element.get('disabled') is not None
			or element.get('hidden') is not None
			or element.get('aria-disabled') == 'true'
		)

	# def _quick_element_filter(self, element: PageElement) -> bool:
	# 	"""
	# 	Quick pre-filter to eliminate elements before expensive checks.
	# 	Returns True if element passes initial filtering.
	# 	"""
	# 	if isinstance(element, NavigableString):
	# 		# Quick check for empty or whitespace-only strings
	# 		return bool(element.strip())

	# 	if not isinstance(element, Tag):
	# 		return False

	# 	style = element.get('style')

	# 	# Quick attribute checks that would make element invisible/non-interactive
	# 	if any(
	# 		[
	# 			element.get('aria-hidden') == 'true',
	# 			element.get('hidden') is not None,
	# 			element.get('disabled') is not None,
	# 			style and ('display: none' in style or 'visibility: hidden' in style),
	# 			element.has_attr('class')
	# 			and any(cls in element['class'] for cls in ['hidden', 'invisible']),
	# 			# Common hidden class patterns
	# 			element.get('type') == 'hidden',
	# 		]
	# 	):
	# 		return False

	# 	# Skip elements that definitely won't be interactive or visible
	# 	non_interactive_display = ['none', 'hidden']
	# 	computed_style = element.get('style', '') or ''
	# 	if any(display in computed_style for display in non_interactive_display):
	# 		return False

	# 	return True
