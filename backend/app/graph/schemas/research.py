from pydantic import BaseModel


class Point(BaseModel):
    source: str
    page_number: int
    text: str
    score: float


class SearchResult(BaseModel):
    points: list[Point]
    