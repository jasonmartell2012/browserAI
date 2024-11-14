import logging

from browser_use.agent.views import ActionResult, AgentAction
from browser_use.browser.service import Browser, MainContentExtractor
from browser_use.controller.registry.service import Registry
from browser_use.controller.views import (
	ClickElementAction,
	DoneAction,
	ExtractPageContentAction,
	GoToUrlAction,
	InputTextAction,
	OpenTabAction,
	SearchGoogleAction,
	SwitchTabAction,
)
from browser_use.utils import time_execution_sync

logger = logging.getLogger(__name__)


class Controller:
	def __init__(self, keep_open: bool = False):
		self.browser = Browser(keep_open=keep_open)
		self.registry = Registry()
		self._register_default_actions()

	def _register_default_actions(self):
		"""Register all default browser actions"""

		# Basic Navigation Actions
		@self.registry.action(
			'Search Google', param_model=SearchGoogleAction, requires_browser=True
		)
		def search_google(params: SearchGoogleAction, browser: Browser):
			driver = browser._get_driver()
			driver.get(f'https://www.google.com/search?q={params.query}')
			browser.wait_for_page_load()

		@self.registry.action('Navigate to URL', param_model=GoToUrlAction, requires_browser=True)
		def go_to_url(params: GoToUrlAction, browser: Browser):
			driver = browser._get_driver()
			driver.get(params.url)
			browser.wait_for_page_load()

		@self.registry.action('Go back', requires_browser=True)
		def go_back(browser: Browser):
			driver = browser._get_driver()
			driver.back()
			browser.wait_for_page_load()

		# Element Interaction Actions
		@self.registry.action(
			'Click element', param_model=ClickElementAction, requires_browser=True
		)
		def click_element(params: ClickElementAction, browser: Browser):
			state = browser._cached_state
			if params.index not in state.selector_map:
				raise Exception(f'Element index {params.index} not found in selector map')

			xpath = state.selector_map[params.index]
			driver = browser._get_driver()
			initial_handles = len(driver.window_handles)

			for _ in range(params.num_clicks):
				try:
					browser._click_element_by_xpath(xpath)
					msg = f'🖱️  Clicked element {params.index}: {xpath}'
					if params.num_clicks > 1:
						msg += f' ({_ + 1}/{params.num_clicks} clicks)'
					logger.info(msg)
				except Exception as e:
					logger.warning(f'Element no longer available after {_ + 1} clicks: {str(e)}')
					break

			if len(driver.window_handles) > initial_handles:
				browser.handle_new_tab()

		@self.registry.action('Input text', param_model=InputTextAction, requires_browser=True)
		def input_text(params: InputTextAction, browser: Browser):
			state = browser._cached_state
			if params.index not in state.selector_map:
				raise Exception(f'Element index {params.index} not found in selector map')

			xpath = state.selector_map[params.index]
			browser._input_text_by_xpath(xpath, params.text)
			logger.info(f'⌨️  Input text "{params.text}" into element {params.index}: {xpath}')

		# Tab Management Actions
		@self.registry.action('Switch tab', param_model=SwitchTabAction, requires_browser=True)
		def switch_tab(params: SwitchTabAction, browser: Browser):
			browser.switch_tab(params.handle)

		@self.registry.action('Open new tab', param_model=OpenTabAction, requires_browser=True)
		def open_tab(params: OpenTabAction, browser: Browser):
			browser.open_tab(params.url)

		# Content Actions
		@self.registry.action(
			'Extract page content', param_model=ExtractPageContentAction, requires_browser=True
		)
		def extract_content(params: ExtractPageContentAction, browser: Browser):
			driver = browser._get_driver()
			content = MainContentExtractor.extract(driver.page_source, output_format=params.value)
			return content

		@self.registry.action('Complete task', param_model=DoneAction, requires_browser=True)
		def done(params: DoneAction, browser: Browser):
			logger.info(f'✅ Done on page {browser._cached_state.url}\n\n: {params.text}')
			return params.text

	def action(self, description: str, **kwargs):
		"""Decorator for registering custom actions"""
		return self.registry.action(description, **kwargs)

	@time_execution_sync('--act')
	def act(self, action: AgentAction) -> ActionResult:
		"""Execute an action"""
		try:
			for action_name, params in action.model_dump(exclude_unset=True).items():
				if params is not None:
					result = self.registry.execute_action(action_name, params, browser=self.browser)
					return ActionResult(extracted_content=str(result) if result else None)
			return ActionResult()
		except Exception as e:
			return ActionResult(error=str(e))
