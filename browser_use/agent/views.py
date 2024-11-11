from inspect import signature
from typing import Any, Callable, Dict, Optional, Type

from pydantic import BaseModel, create_model

from browser_use.controller.views import (
	ClickElementControllerAction,
	ControllerActions,
	InputTextControllerAction,
)


class AgentState(BaseModel):
	valuation_previous_goal: str
	memory: str
	next_goal: str


class AgentOutput(ControllerActions):
	pass


class Output(BaseModel):
	current_state: AgentState
	action: AgentOutput


class ClickElementControllerHistoryItem(ClickElementControllerAction):
	xpath: str | None


class InputTextControllerHistoryItem(InputTextControllerAction):
	xpath: str | None


class AgentHistory(AgentOutput):
	click_element: Optional[ClickElementControllerHistoryItem] = None
	input_text: Optional[InputTextControllerHistoryItem] = None
	url: str


class ActionResult(BaseModel):
	done: bool
	extracted_content: Optional[str] = None
	error: Optional[str] = None


class CustomAction(BaseModel):
	"""Base model for custom actions"""

	name: str
	description: str
	function: Callable

	def __init__(self, description: str, function: Callable, **kwargs):
		super().__init__(
			name=function.__name__, description=description, function=function, **kwargs
		)

	def execute(self, params: dict) -> ActionResult:
		try:
			result = self.function(**params)
			return ActionResult(done=False, extracted_content=str(result) if result else None)
		except Exception as e:
			return ActionResult(done=False, error=str(e))

	def get_prompt_description(self) -> str:
		"""Generate description for system prompt"""
		sig = signature(self.function)
		params = ', '.join(
			f'{name}: {param.annotation.__name__}' for name, param in sig.parameters.items()
		)
		return f'- {self.name} ({params}):\n   {self.description}'


def create_agent_output_class(custom_actions: list[CustomAction]) -> Type[BaseModel]:
	"""Dynamically create AgentOutput class with custom actions"""
	# Create fields for custom actions
	custom_fields = {action.name: (Optional[dict[str, Any]], None) for action in custom_actions}

	# Create the combined model
	return create_model('DynamicAgentOutput', __base__=ControllerActions, **custom_fields)
