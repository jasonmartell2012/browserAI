from __future__ import annotations

from inspect import signature
from typing import Any, Callable, ClassVar, Dict, Optional, Type

from pydantic import BaseModel, ConfigDict, create_model

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
	_function: Callable
	prompt_description: str = ''

	model_config = ConfigDict(
		arbitrary_types_allowed=True,
		# Use list instead of set for excluded fields
		json_schema_extra={'exclude': ['_function', 'prompt_description']},
	)

	def __init__(self, description: str, function: Callable, **kwargs):
		# Generate prompt description before super().__init__
		sig = signature(function)
		params = {name: param.annotation.__name__ for name, param in sig.parameters.items()}
		param_str = ', '.join(f'"{k}": "{v}"' for k, v in params.items())
		prompt_desc = f'- {description}:\n{{"{function.__name__}": {{{param_str}}}}}'

		super().__init__(
			name=function.__name__,
			description=description,
			_function=function,
			prompt_description=prompt_desc,
			**kwargs,
		)

	def execute(self, params: dict) -> ActionResult:
		try:
			result = self._function(**params)
			return ActionResult(done=False, extracted_content=str(result) if result else None)
		except Exception as e:
			return ActionResult(done=False, error=str(e))


class CustomActionsHelper(BaseModel):
	"""Container for custom actions"""

	_custom_actions: Dict[str, CustomAction] = {}
	_cached_model: ClassVar[Optional[Type[CustomActionsHelper]]] = None

	model_config = ConfigDict(
		arbitrary_types_allowed=True,
		# Use list instead of set for excluded fields
		json_schema_extra={'exclude': ['_custom_actions']},
	)

	@classmethod
	def get_or_create_models(
		cls, custom_actions: Optional[list[CustomAction]] = None
	) -> Type[CustomActionsHelper]:
		if cls._cached_model is None:
			# Create fields for custom actions
			custom_fields: Dict[str, tuple[Type, Any]] = {
				action.name: (Optional[Dict[str, Any]], None) for action in (custom_actions or [])
			}

			# Create model with custom action fields
			cls._cached_model = create_model('CustomActions', __base__=cls, **custom_fields)

			# Store custom actions in private field
			if custom_actions:
				cls._cached_model._custom_actions = {
					action.name: action for action in custom_actions
				}

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
