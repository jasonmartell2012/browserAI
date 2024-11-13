from __future__ import annotations

from inspect import signature
from typing import Any, Callable, ClassVar, Dict, Optional, Type, TypeVar

from pydantic import BaseModel, ConfigDict, create_model

from browser_use.controller.views import (
	ClickElementControllerAction,
	ControllerActions,
	ControllerPageState,
	InputTextControllerAction,
)

T = TypeVar('T', bound='DynamicActions')


class CustomActionRegistry:
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
	def get_registered_actions(cls) -> list[CustomAction]:
		"""Get all registered actions as CustomAction instances"""
		return [
			CustomAction(description=desc, function=func)
			for func_name, (desc, func) in cls._actions.items()
		]


class CustomAction(BaseModel):
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


class AgentState(BaseModel):
	"""Current state of the agent"""

	valuation_previous_goal: str
	memory: str
	next_goal: str


class DynamicActions(BaseModel):
	"""Base class combining controller and custom actions"""

	_cached_model: ClassVar[Optional[Type[DynamicActions]]] = None
	_custom_actions: ClassVar[Dict[str, CustomAction]] = {}

	@classmethod
	def get_or_create_model(
		cls: Type[T], custom_actions: Optional[list[CustomAction]] = None
	) -> Type[T]:
		"""Create or return cached model with combined actions"""
		if cls._cached_model is None:
			custom_actions = custom_actions or CustomActionRegistry.get_registered_actions()
			cls._custom_actions = {action.name: action for action in custom_actions}

			controller_fields = {
				name: (field.annotation, field.default)
				for name, field in ControllerActions.model_fields.items()
			}

			custom_fields = {
				action.name: (Optional[Dict[str, Any]], None) for action in custom_actions
			}

			cls._cached_model = create_model(
				'DynamicActions', __base__=cls, **{**controller_fields, **custom_fields}
			)

		return cls._cached_model

	def is_controller_action(self) -> bool:
		return any(
			getattr(self, field_name) is not None for field_name in ControllerActions.model_fields
		)

	def is_custom_action(self) -> bool:
		return any(getattr(self, action_name) is not None for action_name in self._custom_actions)

	def get_custom_action_and_params(self) -> tuple[CustomAction, dict]:
		for action_name, action in self._custom_actions.items():
			params = getattr(self, action_name)
			if params is not None:
				return action, params
		raise ValueError('No custom action found')


class AgentAction(DynamicActions, ControllerActions):
	"""Concrete implementation of combined actions"""

	pass


# History models
class ClickElementControllerHistoryItem(ClickElementControllerAction):
	xpath: str | None


class InputTextControllerHistoryItem(InputTextControllerAction):
	xpath: str | None


class DynamicOutput(BaseModel):
	"""Factory for creating Output models with dynamic actions"""

	_cached_model: ClassVar[Optional[Type[BaseModel]]] = None

	model_config = ConfigDict(arbitrary_types_allowed=True)

	@classmethod
	def get_or_create_model(cls, action_model: Type[DynamicActions]) -> Type[BaseModel]:
		"""Create or get cached output model"""
		if cls._cached_model is None:
			# Create combined action model
			combined_action = create_model(
				'CombinedAction',
				__base__=(action_model, ControllerActions),
			)

			# Create output model
			cls._cached_model = create_model(
				'Output',
				__base__=BaseModel,  # Make it inherit from BaseModel
				current_state=(AgentState, ...),
				action=(combined_action, ...),
			)

		return cls._cached_model


class AgentHistory(BaseModel):
	"""History item for agent actions"""

	model_output: BaseModel  # Change from Type[DynamicOutput] to BaseModel
	result: ActionResult
	state: ControllerPageState

	model_config = ConfigDict(arbitrary_types_allowed=True)
