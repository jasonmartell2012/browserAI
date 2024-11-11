from typing import Optional

from pydantic import BaseModel, Field

from browser_use.controller.views import (
	ClickElementControllerAction,
	ControllerActions,
	InputTextControllerAction,
)


class AskHumanAgentAction(BaseModel):
	question: str


class AgentState(BaseModel):
	valuation_previous_goal: str
	memory: str
	next_goal: str


class AgentOnlyAction(BaseModel):
	ask_human: Optional[AskHumanAgentAction]


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
