from pydantic import BaseModel

class ClarifiableModel(BaseModel):
    def problems(self) -> str:
        raise NotImplementedError