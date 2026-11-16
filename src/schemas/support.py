
from typing import Literal

from pydantic import BaseModel, StrictBool


class SupportResponse(BaseModel):
    status: Literal[
        "answered",
        "needs_clarification",
        "needs_human_review",
    ]

    message: str

    needs_human_review: StrictBool
