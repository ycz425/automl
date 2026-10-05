from pydantic import BaseModel, Field
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from app.graph.schemas.state import AutoMLState  # state.py imports these schemas, so import it for typing only

class ClarifiableModel(BaseModel):
    # Literal of the field names a user's clarification is allowed to change; set by each subclass.
    clarifiable_fields: ClassVar[Any]

    def problems(self, state: "AutoMLState") -> list[str]:
        raise NotImplementedError


class Clarification[T](BaseModel):
    field: T = Field(description="Name of the single field that the user's clarification changes.")
    new_value: Any = Field(description="The complete new value for that field, in the same JSON shape as the field's existing value (a string, a list of strings, an object such as a metric entry, or null to clear the field). It replaces the old value entirely, so include every item for list fields.")


class Clarifications[T](BaseModel):
    clarifications: list[Clarification[T]] = Field(default_factory=list, description="One entry for each field that the user's clarification changes, and nothing else. Return an empty list if the clarification does not resolve any field.")