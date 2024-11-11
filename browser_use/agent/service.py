import json
import logging

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from browser_use.agent.prompts import AgentMessagePrompt, AgentSystemPrompt
from browser_use.agent.views import (
	ActionResult,
	AgentHistory,
	AgentOutput,
	ClickElementControllerHistoryItem,
	CustomAction,
	InputTextControllerHistoryItem,
	Output,
	create_agent_output_class,
)
from browser_use.controller.service import ControllerService
from browser_use.controller.views import (
	ControllerActions,
	ControllerPageState,
)
from browser_use.utils import time_execution_async

load_dotenv()
logger = logging.getLogger(__name__)


class AgentService:
	def __init__(
		self,
		task: str,
		llm: BaseChatModel,
		custom_actions: list[CustomAction] = None,
		controller: ControllerService | None = None,
		use_vision: bool = True,
		save_conversation_path: str | None = None,
	):
		"""
		Agent service.

		Args:
			task (str): Task to be performed.
			llm (AvailableModel): Model to be used.
			controller (ControllerService | None): You can reuse an existing or (automatically) create a new one.
		"""
		self.task = task
		self.use_vision = use_vision
		self.custom_actions = {action.name: action for action in (custom_actions or [])}

		self.controller_injected = controller is not None
		self.controller = controller or ControllerService()

		self.llm = llm
		system_prompt = AgentSystemPrompt(
			task, default_action_description=self._get_action_description()
		).get_system_message()

		# Init messages
		first_message = HumanMessage(content=f'Your task is: {task}')
		self.messages: list[BaseMessage] = [system_prompt, first_message]
		self.n = 0

		self.save_conversation_path = save_conversation_path
		if save_conversation_path is not None:
			logger.info(f'Saving conversation to {save_conversation_path}')

		self.action_history: list[AgentHistory] = []

		# Create dynamic AgentOutput with registered custom actions
		self.AgentOutputClass = create_agent_output_class(custom_actions or [])

	async def run(self, max_steps: int = 100):
		"""
		Execute the task.

		@dev ctrl+c to interrupt
		"""

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
		state = self.controller.get_current_state(screenshot=self.use_vision)
		action = await self.get_next_action(state)

		# Handle controller actions
		if any(getattr(action, field) for field in ControllerActions.model_fields):
			result = self.controller.act(action)
		# Handle custom actions
		elif any(getattr(action, name) for name in self.custom_actions):
			action_name = next(name for name in self.custom_actions if getattr(action, name))
			params = getattr(action, action_name) or {}
			result = self.custom_actions[action_name].execute(params)
		else:
			result = ControllerActionResult(done=False, error=f'Invalid action type: {action}')

		# Update history
		history_item = self._make_history_item(action, state)
		self.action_history.append(history_item)

		return history_item, result

	def _make_history_item(self, action: AgentOutput, state: ControllerPageState) -> AgentHistory:
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
	async def get_next_action(self, state: ControllerPageState) -> AgentOutput:
		# TODO: include state, actions, etc.

		new_message = AgentMessagePrompt(state).get_user_message()
		logger.debug(f'current tabs: {state.tabs}')
		input_messages = self.messages + [new_message]

		structured_llm = self.llm.with_structured_output(Output, include_raw=False)

		response: Output = await structured_llm.ainvoke(input_messages)  # type: ignore

		# Only append the output message
		history_new_message = AgentMessagePrompt(state).get_message_for_history()
		self.messages.append(history_new_message)
		self.messages.append(AIMessage(content=response.model_dump_json(exclude_unset=True)))
		logger.info(
			f'💭 Thought: {response.current_state.model_dump_json(exclude_unset=True, indent=4)}'
		)
		logger.info(f'➡️  Action: {response.action.model_dump_json(exclude_unset=True)}')
		self._save_conversation(input_messages, response)

		return response.action

	def _get_action_description(self) -> str:
		base_description = AgentOutput.description()
		if self.custom_actions:
			custom_descriptions = '\nCustom Actions:\n' + '\n'.join(
				action.get_prompt_description() for action in self.custom_actions.values()
			)
			return base_description + custom_descriptions
		return base_description

	def _save_conversation(self, input_messages: list[BaseMessage], response: Output):
		if self.save_conversation_path is not None:
			with open(self.save_conversation_path + f'_{self.n}.txt', 'w') as f:
				# Write messages with proper formatting
				for message in input_messages:
					f.write('=' * 33 + f' {message.__class__.__name__} ' + '=' * 33 + '\n\n')

					# Handle different content types
					if isinstance(message.content, list):
						# Handle vision model messages
						for item in message.content:
							if isinstance(item, dict) and item.get('type') == 'text':
								f.write(item['text'].strip() + '\n')
					elif isinstance(message.content, str):
						try:
							# Try to parse and format JSON content
							content = json.loads(message.content)
							f.write(json.dumps(content, indent=2) + '\n')
						except json.JSONDecodeError:
							# If not JSON, write as regular text
							f.write(message.content.strip() + '\n')

					f.write('\n')

				# Write final response as formatted JSON
				f.write('=' * 33 + ' Response ' + '=' * 33 + '\n\n')
				f.write(json.dumps(json.loads(response.model_dump_json()), indent=2))
