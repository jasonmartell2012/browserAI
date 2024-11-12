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
		"""Generate description for system prompt"""
		sig = signature(self.function)
		params = {name: param.annotation.__name__ for name, param in sig.parameters.items()}
		param_str = ', '.join(f'"{k}": "{v}"' for k, v in params.items())
		total = f'- {self.description}:\n{{"{self.name}": {{{param_str}}}}}'

		return total


class AgentAction(ControllerActions):
	"""Base class for dynamic agent output that extends ControllerActions"""

	custom_actions: Dict[str, Optional[Dict[str, Any]]] = {}

	_cached_model: ClassVar[Optional[Type[AgentAction]]] = None
	_cached_output_model: ClassVar[Optional[Type[BaseModel]]] = None

	@classmethod
	def get_or_create_models(
		cls, custom_actions: Optional[list[CustomAction]] = None
	) -> tuple[Type[AgentAction], Type[BaseModel]]:
		"""Gets or creates both dynamic models, caching them for reuse"""
		if cls._cached_model is None or cls._cached_output_model is None:
			# Create dynamic agent output model
			custom_fields: Dict[str, tuple[Type, Any]] = {
				action.name: (Optional[Dict[str, Any]], None) for action in (custom_actions or [])
			}

			cls._cached_model = create_model(
				'AgentActionWithActions', __base__=cls, **custom_fields
			)

			# Create output model after agent output model
			cls._cached_output_model = create_model(
				'AgentOutput',
				current_state=(AgentState, ...),
				action=(cls._cached_model, ...),
			)

		return cls._cached_model, cls._cached_output_model


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
