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
def save_job_to_file(title: str, link: str, company: str, salary: str) -> None:
	with open('jobs.txt', 'a') as f:
		f.write(f'{title} - {link} - {company} - {salary}\n')


@CustomActionRegistry.register(description='Read jobs from file you saved before')
def read_jobs_from_file() -> str:
	with open('jobs.txt', 'r') as f:
		return f.read()


async def main():
	task = 'Find 10 software developer jobs in San Francisco at YC startups in google and save them to the file.'
	model = ChatOpenAI(model='gpt-4o')
	agent = AgentService(task, model)

	result = await agent.run()

	for item in result:
		# print model outputs
		print(item.model_output)


if __name__ == '__main__':
	asyncio.run(main())


# run with: python -m examples.extend_actions
