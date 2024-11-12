"""
@dev You need to add ANTHROPIC_API_KEY to your environment variables.
"""

import os
import sys
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio

from langchain_openai import ChatOpenAI

from browser_use.agent.service import AgentService
from browser_use.agent.views import CustomActionRegistry


@CustomActionRegistry.register(description='Save job details to file')
def save_job_to_file(title: str, link: str, company: str) -> None:
	with open('jobs.txt', 'a') as f:
		f.write(f'{title} - {link} - {company}\n')


@CustomActionRegistry.register(description='Read jobs from file you saved before')
def read_jobs_from_file() -> str:
	with open('jobs.txt', 'r') as f:
		return f.read()


async def main():
	task = 'Save job to file : Backend Developer, https://www.google.com/backend-developer, Google. Then read jobs and return the content.'
	# task = 'Find 5 developer jobs in Zurich and save each to a file. In the end read the file and return the content.'
	model = ChatOpenAI(model='gpt-4o')

	# Custom actions are automatically loaded from registry
	agent = AgentService(task, model)

	last_action, result = await agent.run()
	print(last_action)
	print(result)


if __name__ == '__main__':
	asyncio.run(main())


# run with: python -m examples.extend_actions
