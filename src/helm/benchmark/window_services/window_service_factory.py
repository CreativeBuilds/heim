from helm.proxy.models import get_model, get_model_names_with_tag, Model, CLIP_TOKENIZER_TAG
from .image_generation.clip_window_service import CLIPWindowService
from .image_generation.dalle2_window_service import DALLE2WindowService
from .image_generation.lexica_search_window_service import LexicaSearchWindowService
from .window_service import WindowService
from .tokenizer_service import TokenizerService


class WindowServiceFactory:
    @staticmethod
    def get_window_service(model_name: str, service: TokenizerService) -> WindowService:
        """
        Returns a `WindowService` given the name of the model.
        Make sure this function returns instantaneously on repeated calls.
        """
        model: Model = get_model(model_name)
        organization: str = model.organization
        engine: str = model.engine

        window_service: WindowService

        if organization == "openai":
            if engine == "dalle-2":
                window_service = DALLE2WindowService(service)
            else:
                raise ValueError(f"Unhandled OpenAI model: {engine}")
        elif model_name in get_model_names_with_tag(CLIP_TOKENIZER_TAG):
            window_service = CLIPWindowService(service)
        elif model_name == "lexica/search-stable-diffusion-1.5":
            window_service = LexicaSearchWindowService(service)
        else:
            raise ValueError(f"Unhandled model name: {model_name}")

        return window_service
