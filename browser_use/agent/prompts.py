from langchain_core.messages import HumanMessage, SystemMessage

from browser_use.controller.views import ControllerPageState


class AgentSystemPrompt:
	def __init__(self, task: str, default_action_description: str):
		self.task = task
		self.default_action_description = default_action_description

	def response_format(self) -> str:
		"""
		Returns the response format for the agent.

		Returns:
		    str: Response format
		"""
		return """{{
			"current_state": {{
				"valuation_previous_goal": "String starting with "Success" or "Failed:" evaluate if the previous goal was successful and if failed describe why",
				"memory": "String to store progress information for the overall task to rememeber until the end of the task",
				"next_goal": "String describing the next immediate goal which can be achieved with one action"
			}},
			"action": {{
				// EXACTLY ONE of the following available actions must be specified
			}}
		}}"""

	def example_response(self) -> str:
		"""
		Returns an example response for the agent.

		Returns:
		    str: Example response
		"""
		return """{"current_state": {"valuation_previous_goal": "Success", "memory": "We applied already for 3/7 jobs, 1. ..., 2. ..., 3. ...", "next_goal": "Click on the button x to apply for the next job"}, "action": {"click_element": {"index": 44,"num_clicks": 1}}}"""

	def important_rules(self) -> str:
		"""
		Returns the important rules for the agent.

		Returns:
		    str: Important rules
		"""
		return """
	1. Only use indexes that exist in the input list for click or input text actions
	2. Use extract_page_content to get more page information
	3. If stuck, try alternative approaches, go back, search google
	4. Use extract_page_content followed by done action to complete task
	5. If an image is provided, use it to understand the context
	6. ALWAYS respond in the RESPONSE FORMAT with valid JSON:
	7. If the page is empty use actions like "go_to_url", "search_google" or "open_tab"
	8. If you need to click multiple times the same element use the num_clicks parameter with the number of clicks to be more efficient
	"""

	def get_system_message(self) -> SystemMessage:
		"""
		Get the system prompt for the agent.

		Returns:
		    str: Formatted system prompt
		"""

		AGENT_PROMPT = f"""
    
	You are an AI agent that helps users interact with websites. You receive a list of interactive elements from the current webpage and must respond with specific actions.
    
	INPUT FORMAT:
    33:\t<button>Interactive element</button> (33 is the index to interact with)
    _: Not clickable, only for your context
	\t: Tab indent (1 tab for depth 1 etc.). This is to help you understand which elements belong to each other.

	You have to respond in the following RESPONSE FORMAT: 
	{self.response_format()}

	Your AVAILABLE ACTIONS:
    {self.default_action_description}

	Example:
	{self.example_response()}
	
	IMPORTANT RULES:
	{self.important_rules()}
	Remember: Choose EXACTLY ONE action per response. Invalid combinations or multiple actions will be rejected.
    """
		return SystemMessage(content=AGENT_PROMPT)


class AgentMessagePrompt:
	def __init__(self, state: ControllerPageState):
		self.state = state

	def get_user_message(self) -> HumanMessage:
		state_description = f"""
Current url: {self.state.url}
Available tabs:
{self.state.tabs}
Interactive elements:
{self.state.dom_items_to_string()}
        """

		if self.state.screenshot:
			# Format message for vision model
			return HumanMessage(
				content=[
					{'type': 'text', 'text': state_description},
					{
						'type': 'image_url',
						'image_url': {'url': f'data:image/png;base64,{self.state.screenshot}'},
					},
				]
			)

		return HumanMessage(content=state_description)

	def get_message_for_history(self) -> HumanMessage:
		return HumanMessage(content=f'Step url: {self.state.url}')
