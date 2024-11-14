"""
@dev You need to add ANTHROPIC_API_KEY to your environment variables.
"""

import os
import sys
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio

from langchain_openai import ChatOpenAI

from browser_use.agent.service import Agent
from browser_use.agent.views import ActionRegistry


@ActionRegistry.register(description='Save job details to file')
def save_job_to_file(title: str, link: str, company: str, salary: str) -> None:
	with open('jobs.txt', 'a') as f:
		f.write(f'{title} - {link} - {company} - {salary}\n')


@ActionRegistry.register(description='Save people who starred the repo to file')
def save_starred_people(usernames: list[str]) -> None:
	with open('starred_people.txt', 'a') as f:
		for username in usernames:
			f.write(f'{username}\n')


@ActionRegistry.register(description='Read jobs from file you saved before')
def read_jobs_from_file() -> str:
	with open('jobs.txt', 'r') as f:
		return f.read()


@ActionRegistry.register(
	description='Call this action if you need more information from the user e.g. login credentials'
)
def ask_human_for_more_information(question: str) -> str:
	answer = input('\n' + question + '\nInput: \n')
	return answer


@ActionRegistry.register(description='Save socks to file')
def save_socks_to_file(socks: list[str]) -> None:
	with open('socks.txt', 'a') as f:
		for sock in socks:
			f.write(f'{sock}\n')


async def main():
	task = 'Find 10 software developer jobs in San Francisco at YC startups in google and save them to the file.'
	# task = 'Find 10 flights from San Francisco to New York in one way in skyscanner and save them to the file.'
	# task = 'Go to github to https://github.com/gregpr07/browser-use repo and get all people who starred the repo, save them to the file and click to the next page until you get all pages.'
	# task = 'Read jobs from file and start applying for them one by one - if you need more information from the user call ask_human_for_more_information action.'
	task = 'I want socks with a theme of friends tv show and find the cheapest ones on amazon.'
	model = ChatOpenAI(model='gpt-4o')
	agent = Agent(task, model)

	result = await agent.run()

	for item in result:
		# print model outputs
		print(item.model_output)


if __name__ == '__main__':
	asyncio.run(main())


# run with: python -m examples.extend_actions
