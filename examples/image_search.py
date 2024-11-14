import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio

from langchain_openai import ChatOpenAI

from browser_use.agent.service import Agent
from browser_use.controller.service import Controller

people = ['Albert Einstein', 'Oprah Winfrey', 'Steve Jobs']
task = f'Opening new tabs and searching for images for these people: {", ".join(people)}. Then ask me for further instructions.'
controller = Controller(keep_open=True)
model = ChatOpenAI(model='gpt-4o')
agent = Agent(task, model, controller, use_vision=True)


async def main():
	await agent.run()


asyncio.run(main())
