from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage] = Field(min_length=1)
    temperature: Optional[float] = 0.2
    stream: Optional[bool] = False


class ModelCard(BaseModel):
    id: str
    object: str = "model"
