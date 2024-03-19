"""FastAPI models"""

from typing import List
from pydantic import BaseModel


class ResultModel(BaseModel):
    rec_texts: List[str]
    rec_scores: List[float]
    det_polygons: List[List[int]]
    det_scores: List[float]
