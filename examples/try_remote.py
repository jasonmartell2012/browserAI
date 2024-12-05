"""
Simple example of using a remote browser, using Anchor Browser.

@dev You need to add ANCHOR_BROWSER_API_KEY to your environment variables.
@dev You need to add ANTHROPIC_API_KEY to your environment variables.
"""

import os
import sys

from langchain_openai import ChatOpenAI

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import asyncio

from browser_use import Agent
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.controller.service import Controller


def get_llm(provider: str):
	if provider == 'openai':
		return ChatOpenAI(model='gpt-4o', temperature=0.0)
	else:
		raise ValueError(f'Unsupported provider: {provider}')


parser = argparse.ArgumentParser()
parser.add_argument('query', type=str, help='The query to process')
parser.add_argument(
	'--provider',
	type=str,
	choices=['openai', 'anthropic'],
	default='openai',
	help='The model provider to use (default: openai)',
)

args = parser.parse_args()

llm = get_llm(args.provider)

wss_url = f'wss://connect.anchorbrowser.io?apiKey={os.getenv("ANCHOR_BROWSER_API_KEY")}'

browser = Browser(config=BrowserConfig(wss_url=wss_url))

agent = Agent(
	task=args.query,
	llm=llm,
	controller=Controller(),
	browser=browser,
)


async def main():
	await agent.run()


asyncio.run(main())
