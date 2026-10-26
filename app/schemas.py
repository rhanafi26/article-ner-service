from pydantic import BaseModel, Field
from typing import List, Optional

class EntityItem(BaseModel):
    text: str
    label: str
    start: int
    end: int

class MetaInfo(BaseModel):
    entity_count: int
    model: str

class ArticleRequest(BaseModel):
    title: str = Field(..., description="Judul artikel berita")
    body: str = Field(..., description="Isi lengkap artikel berita")
    model_type: Optional[str] = Field("adapted", description="'adapted' atau 'baseline'")

class NERResponse(BaseModel):
    entities: List[EntityItem]
    meta: MetaInfo