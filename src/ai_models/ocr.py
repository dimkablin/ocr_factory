""" Factory Method - Design Pattern """
from ai_models.ocr_models.easyocr import EasyOCRInited
from ai_models.ocr_models.easyocr_trained import EasyOCRInitedTrained
from ai_models.ocr_models.pytesseract import PyTesseractInited
from ai_models.ocr_models.pytesseract_trained import PyTesseractTrained

from ai_models.ocr_models.ocr_interface import OCRInterface


class OCRFactory:
    """ Factory Method - Design Pattern implementation """

    MODEL_MAP = {
        PyTesseractInited.get_model_type(): PyTesseractInited,
        PyTesseractTrained.get_model_type(): PyTesseractTrained,
        EasyOCRInited.get_model_type(): EasyOCRInited,
        EasyOCRInitedTrained.get_model_type(): EasyOCRInitedTrained
    }

    # get first model
    MODEL = MODEL_MAP[PyTesseractInited.get_model_type()]()

    @classmethod
    def __call__(cls, *args, **kwargs) -> dict:
        """Call the current model"""
        result = cls.MODEL(*args, **kwargs)

        # grabage collector will delete this objects from CPU memory
        args, kwargs = None, None

        return result

    @classmethod
    def get_model(cls) -> OCRInterface:
        """ Create OCR model by its name """
        return cls.MODEL

    @classmethod
    def get_model_names(cls):
        """Return a list of model names"""
        return list(cls.MODEL_MAP.keys())

    @classmethod
    def change_model(cls, model_name: str) -> None:
        """Change the model"""

        if model_name not in cls.get_model_names():
            return False

        # delete model from DEVICE
        if hasattr(cls.MODEL, 'model'):
            del cls.MODEL.model

        cls.MODEL = cls.MODEL_MAP[model_name]()

        # success
        return True


MODELS_FACTORY = OCRFactory()
