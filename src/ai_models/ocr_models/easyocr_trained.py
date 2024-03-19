"""Trained EasyOCR Initialization."""
from typing import Any
import easyocr

from ai_models.ocr_models.ocr_interface import OCRInterface
from env import USE_CUDA
from api.app.models import ResultModel


class EasyOCRInitedTrained(OCRInterface):
    """ Initialized EasyOCR model """
    def __init__(self):
        self.languages = ['ru']
        self.use_cuda = USE_CUDA
        self.model = easyocr.Reader(
            self.languages,
            gpu=self.use_cuda,
            model_storage_directory='ai_models/weights/ocr/easyocr/model',
            user_network_directory='ai_models/weights/ocr/easyocr/user_network',
            download_enabled=False,
            recog_network='ru_custom'
        )

    def __call__(self, inputs, *args, **kwargs) -> ResultModel:
        results = []
        for image in inputs:

            horizontal_boxes, free_boxes = self.model.detect(image)
            outputs = self.model.recognize(image, horizontal_boxes[0], free_boxes[0])

            result = ResultModel(rec_texts=[], rec_scores=[], det_polygons=[], det_scores=[])

            for bbox, text, conf in outputs:
                result.det_scores.append(1)
                result.det_polygons.append([int(coord) for xy in bbox for coord in xy])
                result.rec_scores.append(conf)
                result.rec_texts.append(text)
            results.append(result)

        return results

    def __str__(self):
        return f"EasyOCR lang {self.languages}"

    @staticmethod
    def get_model_name() -> str:
        """Return model type."""
        return "EasyOCR Trained"
