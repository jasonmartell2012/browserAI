"""
@dev You need to add ANTHROPIC_API_KEY to your environment variables.
"""

import os
import sys

from browser_use.agent.views import CustomAction

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio

from langchain_openai import ChatOpenAI

from browser_use.agent.service import AgentService


def save_job_to_file(title: str, link: str, company: str):
	with open('jobs.txt', 'a') as f:
		f.write(f'{title} - {link} - {company}\n')


def read_jobs_from_file() -> str:
	with open('jobs.txt', 'r') as f:
		return f.read()


custom_actions = [
	CustomAction(description='Read jobs from file you saved before', function=read_jobs_from_file),
	CustomAction(description='Save job details to file', function=save_job_to_file),
]


task = 'Find 5 developer jobs in Zurich and save each to a file. In the end read the file and return the content.'
model = ChatOpenAI(model='gpt-4o')
agent = AgentService(task, model, custom_actions=custom_actions)


async def main():
	last_action, result = await agent.run()

	print(last_action)
	print(result)


asyncio.run(main())


# run with: python -m examples.extend_actions
