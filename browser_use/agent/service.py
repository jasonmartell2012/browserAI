from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional, Type, TypeVar, cast

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from openai import RateLimitError
from pydantic import BaseModel, ValidationError

from browser_use.agent.prompts import AgentMessagePrompt, AgentSystemPrompt
from browser_use.agent.views import (
	Action,
	ActionResult,
	AgentError,
	AgentHistory,
	DynamicActions,
	DynamicOutput,
)
from browser_use.controller.service import Controller
from browser_use.controller.views import ControllerActions, ControllerPageState
from browser_use.utils import time_execution_async

load_dotenv()
logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


class Agent:
	def __init__(
		self,
		task: str,
		llm: BaseChatModel,
		controller: Optional[Controller] = None,
		use_vision: bool = True,
		save_conversation_path: Optional[str] = None,
		max_failures: int = 5,
		retry_delay: int = 10,
	):
		self.task = task
		self.use_vision = use_vision
		self.llm = llm
		self.save_conversation_path = save_conversation_path

		# Controller setup
		self.controller_injected = controller is not None
		self.controller = controller or Controller()

		# Action and output models setup
		self._setup_action_models()

		# Message history setup
		self.messages = self._initialize_messages()

		# Tracking variables
		self.history: list[AgentHistory] = []
		self.n_steps = 1
		self.consecutive_failures = 0
		self.max_failures = max_failures
		self.retry_delay = retry_delay

		if save_conversation_path:
			logger.info(f'Saving conversation to {save_conversation_path}')

	def _setup_action_models(self) -> None:
		"""Setup dynamic action and output models"""
		custom_actions = Action.get_registered_actions()
		self.DynamicActions = DynamicActions.combine_actions(custom_actions)
		self.OutputModel = DynamicOutput.type_with_custom_actions(
			custom_actions=self.DynamicActions
		)

	def _initialize_messages(self) -> list[BaseMessage]:
		"""Initialize message history with system and first message"""
		system_prompt = AgentSystemPrompt(
			self.task, default_action_description=self._get_action_description()
		).get_system_message()
		first_message = HumanMessage(content=f'Your task is: {self.task}')
		return [system_prompt, first_message]

	def _get_action_description(self) -> str:
		"""Get combined description of all available actions"""
		base_description = ControllerActions.description()
		custom_descriptions = '\n'.join(
			action.prompt_description for action in Action.get_registered_actions()
		)
		return base_description + custom_descriptions if custom_descriptions else base_description

	@time_execution_async('--step')
	async def step(self) -> None:
		"""Execute one step of the task"""

		logger.info(f'\n📍 Step {self.n_steps}')
		state = self.controller.get_current_state(screenshot=self.use_vision)

		try:
			model_output = await self.get_next_action(state)
			action = model_output.action
			result = self._execute_action(action)

			# Success
			self.consecutive_failures = 0

		except Exception as e:
			result = self._handle_step_error(e, state)
			model_output = None

		self._update_messages_with_result(result)
		self._make_history_item(model_output, state, result)

	def _handle_step_error(self, error: Exception, state: ControllerPageState) -> ActionResult:
		"""Handle all types of errors that can occur during a step"""
		error_msg = str(error)
		prefix = f'Failed {self.consecutive_failures + 1}/{self.max_failures} times:\n '

		if isinstance(error, ValidationError):
			error_msg = f'{AgentError.VALIDATION_ERROR}\nDetails: {str(error)}'
			logger.error(f'{prefix}{error_msg}')
			self.consecutive_failures += 1

		elif isinstance(error, RateLimitError):
			error_msg = f'{AgentError.RATE_LIMIT_ERROR}\nWaiting {self.retry_delay} seconds...'
			logger.warning(f'{prefix}{error_msg}')
			time.sleep(self.retry_delay)
			self.consecutive_failures += 1

		elif isinstance(error, ValueError):
			error_msg = f'{AgentError.NO_VALID_ACTION}\n{error}'
			logger.error(f'{prefix}{error_msg}')
			self.consecutive_failures += 1

		else:
			error_msg = f'Unknown error: {str(error)}'
			logger.error(f'{prefix}{error_msg}')
			self.consecutive_failures += 1

		return ActionResult(error=error_msg)

	def _execute_action(self, action: Any) -> ActionResult:
		"""Execute the appropriate action type"""
		if action.is_controller_action():
			return self.controller.act(action)

		elif action.is_custom_action():
			custom_action, params = action.get_custom_action_and_params()
			return custom_action.execute(params)

		else:
			raise ValueError(f'Unknown action: {action}')

	def _update_messages_with_result(self, result: ActionResult) -> None:
		"""Update message history with action results"""
		if result.extracted_content:
			self.messages.append(HumanMessage(content=result.extracted_content))
		if result.error:
			self.messages.append(HumanMessage(content=result.error))

	def _make_history_item(
		self,
		model_output: DynamicOutput | None,
		state: ControllerPageState,
		result: ActionResult,
	) -> None:
		"""Create and store history item"""
		history_item = AgentHistory(model_output=model_output, result=result, state=state)
		self.history.append(history_item)

	@time_execution_async('--get_next_action')
	async def get_next_action(self, state: ControllerPageState) -> DynamicOutput:
		"""Get next action from LLM based on current state"""
		new_message = AgentMessagePrompt(state).get_user_message()
		input_messages = self.messages + [new_message]

		structured_llm = self.llm.with_structured_output(self.OutputModel)
		response: DynamicOutput = await structured_llm.ainvoke(input_messages)  # type: ignore

		self._update_message_history(state, response)
		self._log_response(response)
		self._save_conversation(input_messages, response)

		return response

	def _update_message_history(self, state: ControllerPageState, response: Any) -> None:
		"""Update message history with new interactions"""
		history_message = AgentMessagePrompt(state).get_message_for_history()
		self.messages.append(history_message)
		self.messages.append(AIMessage(content=response.model_dump_json(exclude_unset=True)))
		self.n_steps += 1

	def _log_response(self, response: Any) -> None:
		"""Log the model's response"""
		if 'Success' in response.current_state.valuation_previous_goal:
			emoji = '👍'
		elif 'Failed' in response.current_state.valuation_previous_goal:
			emoji = '⚠️'
		else:
			emoji = '🤷'

		logger.info(f'{emoji} Evaluation: {response.current_state.valuation_previous_goal}')
		logger.info(f'🧠 Memory: {response.current_state.memory}')
		logger.info(f'🎯 Next Goal: {response.current_state.next_goal}')
		logger.info(f'🛠️ Action: {response.action.model_dump_json(exclude_unset=True)}')

	def _save_conversation(self, input_messages: list[BaseMessage], response: Any) -> None:
		"""Save conversation history to file if path is specified"""
		if not self.save_conversation_path:
			return

		with open(self.save_conversation_path + f'_{self.n_steps}.txt', 'w') as f:
			self._write_messages_to_file(f, input_messages)
			self._write_response_to_file(f, response)

	def _write_messages_to_file(self, f: Any, messages: list[BaseMessage]) -> None:
		"""Write messages to conversation file"""
		for message in messages:
			f.write('=' * 33 + f' {message.__class__.__name__} ' + '=' * 33 + '\n\n')

			if isinstance(message.content, list):
				for item in message.content:
					if isinstance(item, dict) and item.get('type') == 'text':
						f.write(item['text'].strip() + '\n')
			elif isinstance(message.content, str):
				try:
					content = json.loads(message.content)
					f.write(json.dumps(content, indent=2) + '\n')
				except json.JSONDecodeError:
					f.write(message.content.strip() + '\n')

			f.write('\n')

	def _write_response_to_file(self, f: Any, response: Any) -> None:
		"""Write model response to conversation file"""
		f.write('=' * 33 + ' Response ' + '=' * 33 + '\n\n')
		f.write(json.dumps(json.loads(response.model_dump_json()), indent=2))

	async def run(self, max_steps: int = 100) -> list[AgentHistory]:
		"""Execute the task with maximum number of steps"""
		try:
			logger.info(f'🚀 Starting task: {self.task}')

			for step in range(max_steps):
				if self._too_many_failures():
					break

				await self.step()

				if self._is_task_complete():
					logger.info('✅ Task completed successfully')
					break
			else:
				logger.info('❌ Failed to complete task in maximum steps')

			return self.history

		finally:
			if not self.controller_injected:
				self.controller.browser.close()

	def _too_many_failures(self) -> bool:
		"""Check if we should stop due to too many failures"""
		if self.consecutive_failures >= self.max_failures:
			logger.error(f'❌ Stopping due to {self.max_failures} consecutive failures')
			return True
		return False

	def _is_task_complete(self) -> bool:
		"""Check if the task has been completed successfully"""
		return bool(self.history and self.history[-1].result.is_done)
