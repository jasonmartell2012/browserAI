import asyncio
import time

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
	"""Initialize the controller"""
	controller = Controller()
	try:
		yield controller
	finally:
		if controller.browser:
			controller.browser.close(force=True)


async def collect_statistics(history):
	"""Collect statistics from the agent's history"""
	total_steps = len(history)
	total_tokens = sum([h.token_usage.total_tokens for h in history if hasattr(h, 'token_usage')])
	total_cost = sum([h.token_usage.total_cost for h in history if hasattr(h, 'token_usage')])
	return total_steps, total_tokens, total_cost


@pytest.mark.asyncio
async def test_refresh_page_100_times(llm, controller):
	"""Stress test: Refresh the page 100 times"""
	agent = Agent(
		task='Refresh the current page 100 times.',
		llm=llm,
		controller=controller,
	)
	start_time = time.time()
	history = await agent.run(max_steps=100)
	end_time = time.time()

	total_time = end_time - start_time
	total_steps, total_tokens, total_cost = await collect_statistics(history)

	print(f'Total time: {total_time:.2f} seconds')
	print(f'Total steps: {total_steps}')
	print(f'Total tokens used: {total_tokens}')
	print(f'Estimated cost: ${total_cost:.4f}')
	# Check for rate limit errors in history
	errors = [h.result.error for h in history if h.result and h.result.error]
	rate_limit_errors = [e for e in errors if 'rate limit' in e.lower()]
	assert len(rate_limit_errors) == 0, 'Rate limit errors occurred during the test'


@pytest.mark.asyncio
async def test_open_10_tabs_and_extract_content(llm, controller):
	"""Stress test: Open 10 tabs and extract content"""
	agent = Agent(
		task='Open new tabs with example.com, example.net, example.org, and seven more example sites. Then, extract the content from each.',
		llm=llm,
		controller=controller,
	)
	start_time = time.time()
	history = await agent.run(max_steps=50)
	end_time = time.time()

	total_time = end_time - start_time
	total_steps, total_tokens, total_cost = await collect_statistics(history)

	print(f'Total time: {total_time:.2f} seconds')
	print(f'Total steps: {total_steps}')
	print(f'Total tokens used: {total_tokens}')
	print(f'Estimated cost: ${total_cost:.4f}')
	# Check for errors
	errors = [h.result.error for h in history if h.result and h.result.error]
	assert len(errors) == 0, 'Errors occurred during the test'
