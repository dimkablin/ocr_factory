"""Main FastAPI entry point."""
import io
import tempfile
import cv2
from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from matplotlib import pyplot as plt
import numpy as np
from ai_models.ocr import MODELS_FACTORY
from api.app.models import ResultModel


router = APIRouter()


@router.get("/get-model-names/", response_model=list)
async def get_model_names() -> list:
    """Return a list of model names."""
    return MODELS_FACTORY.get_model_names()


@router.get("/get-current-model/", response_model=str)
async def get_current_model() -> str:
    """Return the name of the current model."""
    return MODELS_FACTORY.get_model().get_model_name()


@router.post("/ocr/", response_model=ResultModel)
async def ocr(image: UploadFile) -> ResultModel:
    """Predict function."""

    # Read the content of the uploaded image file
    image = await image.read()

    # Convert the bytes to an image using OpenCV
    image = np.frombuffer(image, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    result = MODELS_FACTORY([image])[0]
    return result


@router.post("/change-model/")
async def change_model(model_name: str):
    """Change the model"""

    # change the model by its name
    MODELS_FACTORY.change_model(model_name)

    return JSONResponse(
        status_code=200,
        content={"message": f"Model changed to {model_name}."}
    )
