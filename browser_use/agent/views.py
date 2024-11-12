from __future__ import annotations

from inspect import signature
from typing import Any, Callable, ClassVar, Dict, Optional, Type

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
		sig = signature(self.function)
		params = {name: param.annotation.__name__ for name, param in sig.parameters.items()}
		param_str = ', '.join(f'"{k}": "{v}"' for k, v in params.items())
		total = f'- {self.description}:\n{{"{self.name}": {{{param_str}}}}}'

		return total


class CustomActionsHelper(BaseModel):
	"""Container for custom actions"""

	custom_actions: Dict[str, CustomAction] = {}
	_cached_model: ClassVar[Optional[Type[CustomActionsHelper]]] = None

	@classmethod
	def get_or_create_models(
		cls, custom_actions: Optional[list[CustomAction]] = None
	) -> Type[CustomActionsHelper]:
		"""Creates or returns cached model with custom actions"""
		if cls._cached_model is None:
			# Create fields for custom actions
			custom_fields: Dict[str, tuple[Type, Any]] = {
				action.name: (Optional[Dict[str, Any]], None) for action in (custom_actions or [])
			}

			# Create model with custom action fields and store custom_actions
			cls._cached_model = create_model(
				'CustomActions',
				__base__=cls,
				custom_actions=(
					Dict[str, CustomAction],
					{action.name: action for action in (custom_actions or [])},
				),
				**custom_fields,
			)

		return cls._cached_model


class AgentAction(ControllerActions, CustomActionsHelper):
	"""Combined class that inherits both controller and custom actions"""

	pass


class Output(BaseModel):
	current_state: AgentState
	action: AgentAction


class ClickElementControllerHistoryItem(ClickElementControllerAction):
	xpath: str | None


class InputTextControllerHistoryItem(InputTextControllerAction):
	xpath: str | None


class AgentHistory(AgentAction):
	click_element: Optional[ClickElementControllerHistoryItem] = None
	input_text: Optional[InputTextControllerHistoryItem] = None
	url: str
