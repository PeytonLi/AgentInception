"""Pydantic models for the HTTP API — CONTRACTS §6 and §8."""

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

Mode = Literal["baseline", "mi"]

ALLOWED_ACTIONS = {"goto", "click", "dismiss_modal", "done"}


class StepRequest(BaseModel):
    session_id: str
    mode: Mode
    task: str
    url: str
    page_key: str  # computed by agent-runner via shared page_key()
    dom_text: Optional[str] = None  # REQUIRED in baseline mode; null in mi mode
    dom_token_count: int = Field(
        ge=0
    )  # full-DOM count the runner computes in both modes
    history: list[str] = Field(default_factory=list)
    step: int

    @model_validator(mode="after")
    def _baseline_needs_dom(self) -> "StepRequest":
        if self.mode == "baseline" and not self.dom_text:
            raise ValueError("dom_text is required in baseline mode (CONTRACTS §6)")
        return self


class StepResponse(BaseModel):
    action: dict
    bank_found: bool
    injected_layers: list[int]
    visible_tokens: int
    baseline_tokens: int


class FramePayload(BaseModel):
    jpeg_base64: str
