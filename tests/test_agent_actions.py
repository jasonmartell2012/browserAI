import pytest
from langchain_openai import ChatOpenAI

from browser_use.agent.service import AgentService
from browser_use.agent.views import ActionResult
from browser_use.controller.service import ControllerService


@pytest.fixture
def llm():
	"""Initialize language model for testing"""
	return ChatOpenAI(model='gpt-4o')


@pytest.fixture
async def agent_with_controller():
	"""Create agent with controller for testing"""
	controller = ControllerService(keep_open=False)
	yield controller
	controller.browser.close()


@pytest.mark.asyncio
async def test_ecommerce_interaction(llm, agent_with_controller):
	"""Test complex ecommerce interaction sequence"""
	agent = AgentService(
		task="Go to amazon.com, search for 'laptop', filter by 4+ stars, and find the price of the first result",
		llm=llm,
		controller=agent_with_controller,
	)

	history = await agent.run(max_steps=20)

	# Verify sequence of actions
	action_sequence = []
	for h in history:
		if getattr(h.model_output.action, 'go_to_url', None):
			action_sequence.append('navigate')
		elif getattr(h.model_output.action, 'input_text', None):
			action_sequence.append('input')
			# Check that the input is 'laptop'
			inp = h.model_output.action.input_text.text.lower()
			if inp == 'laptop':
				action_sequence.append('input_exact_correct')
			elif 'laptop' in inp:
				action_sequence.append('correct_in_input')
			else:
				action_sequence.append('incorrect_input')

		elif getattr(h.model_output.action, 'click_element', None):
			action_sequence.append('click')

	# Verify essential steps were performed
	assert 'navigate' in action_sequence  # Navigated to Amazon
	assert 'input' in action_sequence  # Entered search term
	assert 'click' in action_sequence  # Clicked search/filter
	assert 'input_exact_correct' in action_sequence or 'correct_in_input' in action_sequence


@pytest.mark.asyncio
async def test_error_recovery(llm, agent_with_controller):
	"""Test agent's ability to recover from errors"""
	agent = AgentService(
		task='Navigate to nonexistent-site.com and then recover by going to google.com',
		llm=llm,
		controller=agent_with_controller,
	)

	history = await agent.run(max_steps=10)

	# Verify error handling
	error_action = next((h for h in history if h.result.error is not None), None)
	assert error_action is not None

	# Verify recovery
	recovery_action = next(
		(
			h
			for h in history
			if getattr(h.model_output.action, 'go_to_url', None)
			and 'google.com' in h.model_output.action.go_to_url.url
		),
		None,
	)
	assert recovery_action is not None


@pytest.mark.asyncio
async def test_find_contact_email(llm, agent_with_controller):
	"""Test agent's ability to find contact email on a website"""
	agent = AgentService(
		task='Go to https://browser-use.com/ and find out the contact email',
		llm=llm,
		controller=agent_with_controller,
	)

	history = await agent.run(max_steps=10)

	# Verify the agent navigated to the website
	navigate_action = next(
		(
			h
			for h in history
			if getattr(h.model_output.action, 'go_to_url', None)
			and 'browser-use.com' in h.model_output.action.go_to_url.url
		),
		None,
	)
	assert navigate_action is not None

	# Verify the agent found the contact email
	email_action = next(
		(h for h in history if 'info@browser-use.com' in h.result.extracted_content), None
	)
	assert email_action is not None


@pytest.mark.asyncio
async def test_agent_finds_installation_command(llm, agent_with_controller):
	"""Test agent's ability to find the pip installation command for browser-use on the web"""
	agent = AgentService(
		task='Find the pip installation command for the browser use repo',
		llm=llm,
		controller=agent_with_controller,
	)

	history = await agent.run(max_steps=10)

	# Verify the agent found the correct installation command
	install_command_action = next(
		(h for h in history if 'pip install browser-use' in h.result.extracted_content), None
	)
	assert install_command_action is not None
