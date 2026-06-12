"""Exception hierarchy for the agent runner."""

from __future__ import annotations


class AgentRunnerError(Exception):
    """Base class for all agent-runner errors."""


class ActionParseError(AgentRunnerError):
    """The model's Action payload was missing, malformed, or invalid."""


class ActionExecutionError(AgentRunnerError):
    """An action could not be executed (e.g. selector never appeared)."""
