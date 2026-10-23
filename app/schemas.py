from pydantic import BaseModel
from typing import List

class EntityResponse(BaseModel):
    text: str
    label: str
    start_char: int
    end_char: int

class ArticleRequest(BaseModel):
    text: str

class NERResponse(BaseModel):
    entities: List[EntityResponse]