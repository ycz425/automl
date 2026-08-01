from pydantic import BaseModel, Field
from typing import TypeVar, Generic
from app.graph.schemas.clarification_request import GeneratedClarificationRequest

TData = TypeVar("TData")
TClarifiableField = TypeVar("TClarifiableField")

class LLMOutput(BaseModel, Generic[TData, TClarifiableField]):
    data: TData
    clarification_request: GeneratedClarificationRequest[TClarifiableField] | None = Field(default=None, description="Clarification that is absolutely necessary to continue. Return null when it can be determined confidently enough to proceed.")
    