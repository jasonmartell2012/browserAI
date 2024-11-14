import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio

from langchain_anthropic import ChatAnthropic

from browser_use.agent.service import Agent
from browser_use.controller.service import Controller

task = 'Open 3 wikipedia pages in different tabs and summarize the content of all pages.'
controller = Controller()
model = ChatAnthropic(
	model_name='claude-3-5-sonnet-20240620', timeout=25, stop=None, temperature=0.3
)
agent = Agent(task, model, controller, use_vision=True)


async def main():
	await agent.run()


asyncio.run(main())
