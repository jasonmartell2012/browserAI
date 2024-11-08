from typing import Optional

from pydantic import BaseModel, Field

from browser_use.controller.views import (
	ClickElementControllerAction,
	ControllerActions,
	InputTextControllerAction,
)


class AskHumanAgentAction(BaseModel):
	question: str = Field(description='Question to the human')


class AgentState(BaseModel):
	valuation_previous_goal: str = Field(
		default='', description='Valuation if the previous goal was successful or what went wrong'
	)
	memory: str = Field(default='', description='Memory of the current state')
	next_goal: str = Field(default='', description='Description of the next immediate goal')


class AgentOnlyAction(BaseModel):
	ask_human: Optional[AskHumanAgentAction] = Field(
		default=None, description='Ask for human help / information'
	)


class AgentOutput(ControllerActions, AgentOnlyAction):
	pass


class Output(BaseModel):
	current_state: AgentState
	action: AgentOutput


class ClickElementControllerHistoryItem(ClickElementControllerAction):
	xpath: str | None


class InputTextControllerHistoryItem(InputTextControllerAction):
	xpath: str | None


class AgentHistory(AgentOutput):
	click_element: Optional[ClickElementControllerHistoryItem] = None
	input_text: Optional[InputTextControllerHistoryItem] = None
	url: str
