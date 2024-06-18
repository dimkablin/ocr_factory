"""Main FastAPI entry point."""
import logging
import gradio as gr

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.app.middleware import BackendMiddleware
from api.app.endpoint import router
from api.app.counter import invoke_model_use

from ai_models.ocr import MODELS_FACTORY
from utils.visualize import draw_bounding_boxes

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
    invoke_model_use(get_cur_model())#count how many model calls
    result = MODELS_FACTORY([image])[0]
    image_out = draw_bounding_boxes(image, result)
    return result, image_out

def model_change(x):
    MODELS_FACTORY.change_model(x)
    gr.Info(f"Модель была изменена на: {x}")

def get_model_names():
    return MODELS_FACTORY.get_model_names()

def get_cur_model():
    return MODELS_FACTORY.get_model().get_model_name()

with gr.Blocks() as demo:
    gr.Markdown(
    """
    # OCR factory
    Контейнер для распознавания текста при помощи моделей easyOCR, tesseract.\n
    Выберите модель во вкладке настройки (по стандарту выбрана модель tesseract),
    загрузите ваш документ и нажимите кнопку распознать.
    """
    )
    with gr.Tab("Модель"):
        with gr.Row():
            image_input = gr.Image()
            image_output = gr.Image()
        text_output = gr.Textbox(label="Результат")
        image_button = gr.Button("Распознать")

    with gr.Tab("Настройки"):
            model_drop = gr.Dropdown(get_model_names(), value=get_cur_model(), label="Текущая модель")
            text_button = gr.Button("Применить")

    text_button.click(model_change, inputs=model_drop)
    image_button.click(ocr_inf, inputs=image_input, outputs=[text_output, image_output])

gradio_app = gr.mount_gradio_app(app, demo, path="/")
