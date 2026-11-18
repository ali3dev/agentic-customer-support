
from typing import Literal

from pydantic import BaseModel, Field, StrictBool


class SourceMetadata(BaseModel):
    citation_id: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    source_path: str = Field(min_length=1)
    section_title: str = Field(min_length=1)
    score: float = Field(ge=-1.0, le=1.0)


class TokenUsage(BaseModel):
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    cost_usd: float | None = Field(default=None, ge=0)


class AgentResponse(BaseModel):
    message: str
    sources: list[SourceMetadata] = Field(default_factory=list)
    usage: TokenUsage | None = None


class SupportResponse(BaseModel):
    status: Literal[
        "answered",
        "needs_clarification",
        "needs_human_review",
    ]

    message: str

    needs_human_review: StrictBool
