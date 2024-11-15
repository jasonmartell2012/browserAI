import asyncio

import pytest
from langchain_openai import ChatOpenAI

from browser_use.agent.service import Agent
from browser_use.controller.service import Controller


@pytest.fixture
def llm():
	"""Initialize the language model"""
	return ChatOpenAI(model='gpt-4o')  # Use appropriate model


@pytest.fixture
async def controller():
	"""Initialize the controller with self-registered actions"""
	controller = Controller()

	# Define custom actions without Pydantic models
	@controller.action('Print a message')
	def print_message(message: str):
		print(f'Message: {message}')
		return f'Printed message: {message}'

	@controller.action('Add two numbers')
	def add_numbers(a: int, b: int):
		result = a + b
		return f'The sum is {result}'

	@controller.action('Concatenate strings')
	def concatenate_strings(str1: str, str2: str):
		result = str1 + str2
		return f'Concatenated string: {result}'

	try:
		yield controller
	finally:
		if controller.browser:
			controller.browser.close(force=True)


@pytest.mark.asyncio
async def test_self_registered_actions_no_pydantic(llm, controller):
	"""Test self-registered actions with individual arguments"""
	agent = Agent(
		task="First, print the message 'Hello, World!'. Then, add 10 and 20. Next, concatenate 'foo' and 'bar'.",
		llm=llm,
		controller=controller,
	)
	history = await agent.run(max_steps=10)
	# Check that custom actions were executed
	actions = [h.model_output.action for h in history if h.model_output and h.model_output.action]
	action_names = [list(action.model_dump(exclude_unset=True).keys())[0] for action in actions]

	assert 'print_message' in action_names
	assert 'add_numbers' in action_names
	assert 'concatenate_strings' in action_names


# Additional test with mixed arguments


@pytest.mark.asyncio
async def test_mixed_arguments_actions(llm, controller):
	"""Test actions with mixed argument types"""

	# Define another action during the test
	@controller.action('Calculate the area of a rectangle')
	def calculate_area(length: float, width: float):
		area = length * width
		return f'The area is {area}'

	agent = Agent(
		task='Calculate the area of a rectangle with length 5.5 and width 3.2.',
		llm=llm,
		controller=controller,
	)
	history = await agent.run(max_steps=5)

	# Check that the action was executed
	actions = [h.model_output.action for h in history if h.model_output and h.model_output.action]
	action_names = [list(action.model_dump(exclude_unset=True).keys())[0] for action in actions]

	assert 'calculate_area' in action_names
	# check result
	correct = 'The area is 17.6'
	assert correct in [h.result.extracted_content for h in history if h.model_output]
