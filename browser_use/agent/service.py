from __future__ import annotations

import json
import logging
from typing import Any, Callable, Optional, Type, TypeVar, Union, cast

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, create_model

from browser_use.agent.prompts import AgentMessagePrompt, AgentSystemPrompt
from browser_use.agent.views import (
	ActionResult,
	AgentAction,
	AgentHistory,
	AgentState,
	ClickElementControllerHistoryItem,
	CustomAction,
	CustomActionRegistry,
	DynamicActions,
	DynamicOutput,
	InputTextControllerHistoryItem,
)
from browser_use.controller.service import ControllerService
from browser_use.controller.views import ControllerActions, ControllerPageState
from browser_use.utils import time_execution_async

load_dotenv()
logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)
OutputT = TypeVar('OutputT', bound=BaseModel)


class AgentService:
	def __init__(
		self,
		task: str,
		llm: BaseChatModel,
		controller: Optional[ControllerService] = None,
		use_vision: bool = True,
		save_conversation_path: Optional[str] = None,
	):
		"""
		Agent service that handles task execution using LLM and custom actions.

		Args:
			task: Task to be performed
			llm: Language model to use
			controller: Optional existing controller service
			use_vision: Whether to use vision capabilities
			save_conversation_path: Optional path to save conversation history
		"""
		self.task = task
		self.use_vision = use_vision

		# Get registered custom actions
		custom_actions = CustomActionRegistry.get_registered_actions()

		# Get dynamic action model with registered actions
		self.DynamicActions = DynamicActions.get_or_create_model(custom_actions)

		# Create dynamic output model
		self.Output = DynamicOutput.get_or_create_model(self.DynamicActions)

		# Type hint the output model
		self.output_model: Type[OutputT] = cast(Type[OutputT], self.Output)

		self.controller_injected = controller is not None
		self.controller = controller or ControllerService()
		self.llm = llm

		# Initialize prompts with action descriptions
		system_prompt = AgentSystemPrompt(
			task, default_action_description=self._get_action_description()
		).get_system_message()

		# Initialize message history
		first_message = HumanMessage(content=f'Your task is: {task}')
		self.messages: list[BaseMessage] = [system_prompt, first_message]
		self.n = 0

		self.save_conversation_path = save_conversation_path
		if save_conversation_path is not None:
			logger.info(f'Saving conversation to {save_conversation_path}')

		self.action_history: list[AgentHistory] = []

	def _get_action_description(self) -> str:
		"""Get combined description of all available actions"""
		base_description = ControllerActions.description()

		# Get descriptions from registry
		custom_descriptions = '\n'.join(
			action.prompt_description for action in CustomActionRegistry.get_registered_actions()
		)

		return base_description + custom_descriptions if custom_descriptions else base_description

	async def run(self, max_steps: int = 100) -> tuple[Optional[bool], list[AgentHistory]]:
		"""Execute the task with maximum number of steps"""
		try:
			logger.info(f'🚀 Starting task: {self.task}')

			for i in range(max_steps):
				action, result = await self.step()

				if result.done:
					logger.info('✅ Task completed successfully')
					return action.done, self.action_history

			logger.info('❌ Failed to complete task in maximum steps')
			return None, self.action_history
		finally:
			if not self.controller_injected:
				self.controller.browser.close()

	@time_execution_async('--step')
	async def step(self) -> tuple[AgentHistory, ActionResult]:
		"""Execute one step of the task"""
		state = self.controller.get_current_state(screenshot=self.use_vision)
		action = await self.get_next_action(state)

		# Handle controller actions
		if action.is_controller_action():
			result = self.controller.act(action)

		# Handle custom actions
		elif action.is_custom_action():
			custom_action, params = action.get_custom_action_and_params()
			result = custom_action.execute(params)
		else:
			result = ActionResult(done=False, error=f'No valid action found: {action}')
			logger.error(f'No valid action found: {action}')

		# include result in model
		if result.extracted_content is not None:
			self.messages.append(HumanMessage(content=result.extracted_content))
		if result.error is not None:
			self.messages.append(HumanMessage(content=result.error))

		# Update history
		history_item = self._make_history_item(action, state)
		self.action_history.append(history_item)

		return history_item, result

	def _make_history_item(self, action: AgentAction, state: ControllerPageState) -> AgentHistory:
		"""Create history item from action and state"""
		return AgentHistory(
			search_google=action.search_google,
			go_to_url=action.go_to_url,
			nothing=action.nothing,
			go_back=action.go_back,
			done=action.done,
			click_element=ClickElementControllerHistoryItem(
				id=action.click_element.id, xpath=state.selector_map.get(action.click_element.id)
			)
			if action.click_element and state.selector_map.get(action.click_element.id)
			else None,
			input_text=InputTextControllerHistoryItem(
				id=action.input_text.id,
				xpath=state.selector_map.get(action.input_text.id),
				text=action.input_text.text,
			)
			if action.input_text and state.selector_map.get(action.input_text.id)
			else None,
			extract_page_content=action.extract_page_content,
			switch_tab=action.switch_tab,
			open_tab=action.open_tab,
			url=state.url,
		)

	@time_execution_async('--get_next_action')
	async def get_next_action(self, state: ControllerPageState) -> AgentAction:
		"""Get next action from LLM based on current state"""
		new_message = AgentMessagePrompt(state).get_user_message()
		logger.debug(f'current tabs: {state.tabs}')
		input_messages = self.messages + [new_message]

		# Use dynamic output model with proper typing
		structured_llm = self.llm.with_structured_output(self.output_model)
		response = await structured_llm.ainvoke(input_messages)

		# Update message history
		history_new_message = AgentMessagePrompt(state).get_message_for_history()
		self.messages.append(history_new_message)
		self.messages.append(AIMessage(content=response.model_dump_json(exclude_unset=True)))

		logger.info(
			f'💭 Thought: {response.current_state.model_dump_json(exclude_unset=True, indent=4)}'
		)
		logger.info(f'➡️  Action: {response.action.model_dump_json(exclude_unset=True)}')

		self._save_conversation(input_messages, response)

		return response.action

	def _save_conversation(self, input_messages: list[BaseMessage], response: Any) -> None:
		"""Save conversation history to file if path is specified"""
		if self.save_conversation_path is not None:
			with open(self.save_conversation_path + f'_{self.n}.txt', 'w') as f:
				# Write messages with proper formatting
				for message in input_messages:
					f.write('=' * 33 + f' {message.__class__.__name__} ' + '=' * 33 + '\n\n')

					if isinstance(message.content, list):
						# Handle vision model messages
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

				# Write final response
				f.write('=' * 33 + ' Response ' + '=' * 33 + '\n\n')
				f.write(json.dumps(json.loads(response.model_dump_json()), indent=2))
