from pydantic import BaseModel, Field
from typing import Generic, TypeVar, Literal

TClarifiableField = TypeVar("TClarifiableField")

class GeneratedClarificationRequest(BaseModel, Generic[TClarifiableField]):
    fields: list[TClarifiableField] = Field(description="Ambiguous or conflicting field(s) that require clarification. Must only contain values from ClarifiableField.")
    reason: str = Field(description="Concise explanation on why clarification is needed.")
    question: str = Field(description="Concise question(s) to ask the user.")

class ClarificationRequest(BaseModel):
    source: Literal['user_request', 'dataset_analysis']
    request: GeneratedClarificationRequest[TClarifiableField]

