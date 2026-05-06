from __future__ import annotations

from pydantic import BaseModel, Field


class QueryNamesOutput(BaseModel):
    names: list[str] = Field(default_factory=list)
