"""Main FastAPI entry point."""
import logging
import gradio as gr

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.app.middleware import BackendMiddleware
from api.app.endpoint import router

from ai_models.ocr import MODELS_FACTORY

# LOGGING CONFIG SETTING
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logging.info("Running ocr model factory.")


# APP CONFIG SET
app = FastAPI(
    title="Backend API",
    docs_url="/docs",
    openapi_url="/openapi.json",
    openapi_tags=[{
        "name": "Backend API",
        "description": "API for ocr models factory method."
    }]
)

# MIDDLEWARE CONFIG SET
app.add_middleware(BackendMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    allow_credentials=True
)
app.include_router(router, tags=["ai_models"])
app.mount("/docs", StaticFiles(directory="../docs"), name="docs")

# @app.get("/")
# def read_main():
#     return {"message": "This is your main app"}

#####GRADIO
def ocr_inf(image):
    result = MODELS_FACTORY([image])[0]
    return result

def model_change(x):
    MODELS_FACTORY.change_model(x)

def get_model_names():
    return MODELS_FACTORY.get_model_names()

def get_cur_model():
    return MODELS_FACTORY.get_model().get_model_name()

with gr.Blocks() as demo:
    with gr.Tab("Модель"):
        with gr.Row():
            image_input = gr.Image()
            image_output = gr.Textbox()
        image_button = gr.Button("Распознать")

    with gr.Tab("Настройки"):
            model_drop = gr.Dropdown(get_model_names(), value=get_cur_model(), label="Текущая модель")
            text_button = gr.Button("Применить")

    text_button.click(model_change, inputs=model_drop)
    image_button.click(ocr_inf, inputs=image_input, outputs=image_output)

gradio_app = gr.mount_gradio_app(app, demo, path="/")
