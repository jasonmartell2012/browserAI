from typing import Literal, Optional

from pydantic import BaseModel

from browser_use.browser.views import BrowserState


class SearchGoogleControllerAction(BaseModel):
	query: str


class GoToUrlControllerAction(BaseModel):
	url: str


class ClickElementControllerAction(BaseModel):
	id: int
	num_clicks: int = 1


class InputTextControllerAction(BaseModel):
	id: int
	text: str


class DoneControllerAction(BaseModel):
	text: str


class SwitchTabControllerAction(BaseModel):
	handle: str  # The window handle to switch to


class OpenTabControllerAction(BaseModel):
	url: str


class ControllerActions(BaseModel):
	"""
	Controller actions you can use to interact.
	"""

	search_google: Optional[SearchGoogleControllerAction] = None
	go_to_url: Optional[GoToUrlControllerAction] = None
	nothing: Optional[Literal[True]] = None
	go_back: Optional[Literal[True]] = None
	done: Optional[DoneControllerAction] = None
	click_element: Optional[ClickElementControllerAction] = None
	input_text: Optional[InputTextControllerAction] = None
	extract_page_content: Optional[Literal[True]] = None
	switch_tab: Optional[SwitchTabControllerAction] = None
	open_tab: Optional[OpenTabControllerAction] = None

	@staticmethod
	def description() -> str:
		return """
- Search Google:
   {"search_google": {"query": "Your search query"}}
- Navigate to URL:
   {"go_to_url": {"url": "https://example.com"}}
- Wait/Do nothing:
   {"nothing": true}
- Go back:
   {"go_back": true}
- Click an interactive element by its given ID and number how many times you want to click it (default is 1):
   {"click_element": {"id": 1, "num_clicks": 2}}
- Input text into an interactive element by its ID:
   {"input_text": {"id": 1, "text": "Your text"}}
- Get page content:
   {"extract_page_content": true}
- Open new tab:
   {"open_tab": {"url": "https://example.com"}}
- Switch tab:
   {"switch_tab": {"handle": "tab-id"}}
- Complete task:
   {"done": {"text": "Final result message"}}
"""


class ControllerPageState(BrowserState):
	screenshot: Optional[str] = None
	tabs: list[dict] = []  # Add tabs info to state

	def model_dump(self) -> dict:
		dump = super().model_dump()
		# Add a summary of available tabs
		if self.tabs:
			dump['available_tabs'] = [
				f"Tab {i+1}: {tab['title']} ({tab['url']})" for i, tab in enumerate(self.tabs)
			]
		return dump
