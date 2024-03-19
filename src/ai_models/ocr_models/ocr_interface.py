""" The ocr interface """

from abc import ABC, abstractmethod
from api.app.models import ResultModel


class OCRInterface(ABC):
    """ Interface of ocr ai_models """
    @abstractmethod
    def __call__(self, *args, **kwargs) -> ResultModel:
        """
        This function returns a dictionary with specific 
        keys and annotated value types.
        """

    @staticmethod
    def get_model_name() -> str:
        """Return model type."""

    @abstractmethod
    def __str__(self) -> str:
        """Return model description."""
