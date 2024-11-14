from __future__ import annotations

from inspect import signature
from typing import Any, Callable, ClassVar, Dict, Optional, Type

from openai import RateLimitError
from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from browser_use.browser.views import BrowserState
from browser_use.controller.views import (
	ControllerActions,
)


class ActionRegistry:
	"""Registry for custom actions that can be used by the agent"""

	_actions: Dict[str, tuple[str, Callable]] = {}

	@classmethod
	def register(cls, description: str):
		"""Decorator to register custom actions with their descriptions"""

		def decorator(func: Callable):
			cls._actions[func.__name__] = (description, func)
			return func

		return decorator

	@classmethod
	def get_registered_actions(cls) -> list[RegisteredAction]:
		"""Get all registered actions as RegisteredAction instances"""
		return [
			RegisteredAction(description=desc, function=func)
			for func_name, (desc, func) in cls._actions.items()
		]


class RegisteredAction(BaseModel):
	"""Model for custom actions with their metadata"""

	name: str
	description: str
	prompt_description: str = ''
	_function: ClassVar[Dict[str, Callable]] = {}

	model_config = ConfigDict(arbitrary_types_allowed=True)

	def __init__(self, description: str, function: Callable, **kwargs):
		sig = signature(function)
		params = {name: param.annotation.__name__ for name, param in sig.parameters.items()}
		param_str = ', '.join(f'"{k}": "{v}"' for k, v in params.items())
		prompt_desc = f'- {description}:\n{{"{function.__name__}": {{{param_str}}}}}'

		self._function[function.__name__] = function

		super().__init__(
			name=function.__name__,
			description=description,
			prompt_description=prompt_desc,
			**kwargs,
		)

	def execute(self, params: dict) -> ActionResult:
		try:
			func = self._function[self.name]
			result = func(**params)
			return ActionResult(extracted_content=str(result) if result else None)
		except Exception as e:
			return ActionResult(error=str(e))


class ActionResult(BaseModel):
	"""Result of executing an action"""

	is_done: Optional[bool] = False
	extracted_content: Optional[str] = None
	error: Optional[str] = None


class AgentBrain(BaseModel):
	"""Current state of the agent"""

	valuation_previous_goal: str
	memory: str
	next_goal: str


class AgentAction(ControllerActions):
	"""Base class combining controller and custom actions"""

	_cached_model: ClassVar[Optional[Type[AgentAction]]] = None
	_registered_actions: ClassVar[Dict[str, RegisteredAction]] = {}

	@classmethod
	def combine_actions(
		cls, custom_actions: Optional[list[RegisteredAction]] = None
	) -> Type[AgentAction]:
		"""Create or return cached model with combined actions"""
		if cls._cached_model is None:
			custom_actions = custom_actions or ActionRegistry.get_registered_actions()
			cls._registered_actions = {action.name: action for action in custom_actions}

			controller_fields = {
				name: (field.annotation, field.default)
				for name, field in ControllerActions.model_fields.items()
			}

			custom_fields = {
				action.name: (Optional[Dict[str, Any]], None) for action in custom_actions
			}

			cls._cached_model = create_model(
				'AgentAction', __base__=cls, **{**controller_fields, **custom_fields}
			)

		return cls._cached_model

	def is_controller_action(self) -> bool:
		return any(
			getattr(self, field_name) is not None for field_name in ControllerActions.model_fields
		)

	def is_custom_action(self) -> bool:
		return any(
			getattr(self, action_name) is not None for action_name in self._registered_actions
		)

	def get_custom_action_and_params(self) -> tuple[RegisteredAction, dict]:
		for action_name, action in self._registered_actions.items():
			params = getattr(self, action_name)
			if params is not None:
				return action, params
		raise ValueError('No custom action found')


class AgentOutput(BaseModel):
	"""Output model for agent

	@dev note: this model is extended with custom actions in AgentService. You can also use some fields that are not in this model as provided by the linter, as long as they are registered in the DynamicActions model.
	"""

	model_config = ConfigDict(arbitrary_types_allowed=True)

	current_state: AgentBrain
	action: AgentAction

	@staticmethod
	def type_with_custom_actions(custom_actions: Type[AgentAction]) -> Type['AgentOutput']:
		"""Extend actions with custom actions"""
		return create_model(
			'AgentOutput',
			__base__=AgentOutput,
			action=(custom_actions, Field(...)),  # Properly annotated field with no default
			__module__=AgentOutput.__module__,
		)


class AgentHistory(BaseModel):
	"""History item for agent actions"""

	model_output: AgentOutput | None
	result: ActionResult
	state: BrowserState

	model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())


class AgentError:
	"""Container for agent error handling"""

	VALIDATION_ERROR = 'Invalid model output format. Please follow the correct schema.'
	RATE_LIMIT_ERROR = 'Rate limit reached. Waiting before retry.'
	NO_VALID_ACTION = 'No valid action found'

	@staticmethod
	def format_error(error: Exception) -> str:
		"""Format error message based on error type"""
		if isinstance(error, ValidationError):
			return f'{AgentError.VALIDATION_ERROR}\nDetails: {str(error)}'
		if isinstance(error, RateLimitError):
			return AgentError.RATE_LIMIT_ERROR
		return f'Unexpected error: {str(error)}'
