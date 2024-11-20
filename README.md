# 🌐 Browser Use

Make websites accessible for AI agents 🤖.

[![GitHub stars](https://img.shields.io/github/stars/gregpr07/browser-use?style=social)](https://github.com/gregpr07/browser-use/stargazers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Discord](https://img.shields.io/discord/1303749220842340412?color=7289DA&label=Discord&logo=discord&logoColor=white)](https://link.browser-use.com/discord)

Browser use is the easiest way to connect your AI agents with the browser. If you have used Browser Use for your project feel free to show it off in our [Discord](https://link.browser-use.com/discord).

# Quick start

With pip:

```bash
pip install browser-use
```

Spin up your agent:

```python
from langchain_openai import ChatOpenAI
from browser_use import Agent

agent = Agent(
    task="Find a one-way flight from Bali to Oman on 12 January 2025 on Google Flights. Return me the cheapest option.",
    llm=ChatOpenAI(model="gpt-4o"),
)

# ... inside an async function
await agent.run()
```

And don't forget to add your API keys to your `.env` file.

```bash
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
```

# Demos


<div>
    <video src="https://github.com/MagMueller/examples-browser-use/raw/main/hugging_face/hugging_face_high_quality.mp4" controls style="max-width:300px;">
    </video>
    <p><i>Prompt: Look up models with a license of cc-by-sa-4.0 and sort by most likes on Hugging face, save top 5 to file. (1x speed)</i></p>
</div>

<div>
    <img src="https://github.com/MagMueller/examples-browser-use/blob/main/captcha/captcha%20big.gif" alt="Solving Captcha" style="max-width:300px;">
    <p><i>Prompt: Solve the captcha on captcha.com/demos/features/captcha-demo.aspx . (2x speed) 
    NOTE: captchas are hard. For this example it works. But e.g. for iframes it does not.</i></p>
</div>

# Features ⭐

- Vision + html extraction
- Automatic multi-tab management
- Extract clicked elements XPaths and repeat exact LLM actions
- Add custom actions (e.g. save to file, push to database, notify me, get human input)
- Self-correcting
- Use any LLM supported by LangChain (e.g. gpt4o, gpt4o mini, claude 3.5 sonnet, llama 3.1 405b, etc.)

## Register custom actions

If you want to add custom actions your agent can take, you can register them like this:

```python
from browser_use.agent.service import Agent
from browser_use.browser.service import Browser
from browser_use.controller.service import Controller

# Initialize controller first
controller = Controller()

@controller.action('Ask user for information')
def ask_human(question: str, display_question: bool) -> str:
	return input(f'\n{question}\nInput: ')
```

Or define your parameters using Pydantic

```python
class JobDetails(BaseModel):
  title: str
  company: str
  job_link: str
  salary: Optional[str] = None

@controller.action('Save job details which you found on page', param_model=JobDetails, requires_browser=True)
def save_job(params: JobDetails, browser: Browser):
	print(params)

  # use the browser normally
  browser.driver.get(params.job_link)
```

and then run your agent:

```python
model = ChatAnthropic(model_name='claude-3-5-sonnet-20240620', timeout=25, stop=None, temperature=0.3)
agent = Agent(task=task, llm=model, controller=controller)

await agent.run()
```

## Get XPath history

To get the entire history of everything the agent has done, you can use the output of the `run` method:

```python
history: list[AgentHistory] = await agent.run()

print(history)
```

## More examples

For more examples see the [examples](examples) folder or join the [Discord](https://link.browser-use.com/discord) and show off your project.

## Telemetry

We collect anonymous usage data to help us understand how the library is being used and to identify potential issues. There is no privacy risk, as no personal information is collected. We collect data with PostHog.

You can opt out of telemetry by setting the `ANONYMIZED_TELEMETRY=false` environment variable.

# Contributing

Contributions are welcome! Feel free to open issues for bugs or feature requests.

## Local Setup

1. Create a virtual environment and install dependencies:

```bash
# To install all dependencies including dev
pip install . ."[dev]"
```

2. Add your API keys to the `.env` file:

```bash
cp .env.example .env
```

or copy the following to your `.env` file:

```bash
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
```

You can use any LLM model supported by LangChain by adding the appropriate environment variables. See [langchain models](https://python.langchain.com/docs/integrations/chat/) for available options.

### Building the package

```bash
hatch build
```

Feel free to join the [Discord](https://link.browser-use.com/discord) for discussions and support.

---

<div align="center">
  <b>Star ⭐ this repo if you find it useful!</b><br>
  Made with ❤️ by the Browser-Use team
</div>
