from dataclasses import dataclass, field
from typing import List, Dict

# Different modalities
TEXT_MODEL_TAG: str = "text"
IMAGE_MODEL_TAG: str = "image"
CODE_MODEL_TAG: str = "code"
EMBEDDING_MODEL_TAG: str = "embedding"
TEXT_TO_IMAGE_MODEL_TAG: str = "text_to_image"

# Some model APIs have limited functionalities
FULL_FUNCTIONALITY_TEXT_MODEL_TAG: str = "full_functionality_text"
LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG: str = "limited_functionality_text"

# ChatML format
CHATML_MODEL_TAG: str = "chatml"

# OpenAI Chat format
OPENAI_CHATGPT_MODEL_TAG: str = "openai_chatgpt"

# For Anthropic models
ANTHROPIC_MODEL_TAG: str = "anthropic"

# For OpenAI models with wider context windows
WIDER_CONTEXT_WINDOW_TAG: str = "openai_wider_context_window"  # huggingface/gpt2 tokenizer, 4000 tokens
GPT_TURBO_CONTEXT_WINDOW_TAG: str = "gpt_turbo_context_window"  # cl100k_base tokenizer, 4000 tokens
GPT_TURBO_16K_CONTEXT_WINDOW_TAG: str = "gpt_turbo_16k_context_window"  # cl100k_base tokenizer, 8000 tokens
GPT4_CONTEXT_WINDOW_TAG: str = "gpt4_context_window"  # cl100k_base tokenizer, 8192 tokens
GPT4_32K_CONTEXT_WINDOW_TAG: str = "gpt4_32k_context_window"  # cl100k_base tokenizer, 32768 tokens

# For AI21 Jurassic-2 models with wider context windows
AI21_WIDER_CONTEXT_WINDOW_TAG: str = "ai21_wider_context_window"

# For AI21 Jurassic-2 Jumbo
# AI21 has recommended using a sequence length of 6000 tokens to avoid OOMs.
AI21_JURASSIC_2_JUMBO_CONTEXT_WINDOW_TAG: str = "ai21_jurassic_2_jumbo_context_window"  # 6000

# To fetch models that use these tokenizers
GPT2_TOKENIZER_TAG: str = "gpt2_tokenizer"
AI21_TOKENIZER_TAG: str = "ai21_tokenizer"
COHERE_TOKENIZER_TAG: str = "cohere_tokenizer"
OPT_TOKENIZER_TAG: str = "opt_tokenizer"
GPTJ_TOKENIZER_TAG: str = "gptj_tokenizer"
GPT4_TOKENIZER_TAG: str = "gpt4_tokenizer"
GPTNEO_TOKENIZER_TAG: str = "gptneo_tokenizer"
CLIP_TOKENIZER_TAG: str = "clip_tokenizer"

# Models which emit garbage tokens when temperature=0.
BUGGY_TEMP_0_TAG: str = "buggy_temp_0"

# Models that are used for ablations and fine-grained analyses.
# These models are selected specifically because of their low marginal cost to evaluate.
ABLATION_MODEL_TAG: str = "ablation"

# Some models (e.g., T5) have stripped newlines.
# So we cannot use \n as a stop sequence for these models.
NO_NEWLINES_TAG: str = "no_newlines"

# Some models (e.g., UL2) require a prefix (e.g., [NLG]) in the
# prompts to indicate the mode before doing inference.
NLG_PREFIX_TAG: str = "nlg_prefix_tag"

# Whether the HuggingFace model needs to be loaded locally
LOCAL_HUGGINGFACE_MODEL_TAG: str = "local_huggingface_model"

# Some models can follow instructions.
INSTRUCTION_FOLLOWING_MODEL_TAG: str = "instruction_following"


@dataclass
class Model:
    """
    Represents a model that we can make requests to.  Conceptually, an instance
    of `Model` is tied more to the hosting implementation (where can we send
    requests) rather than the conceptual model.  These are the same for closed
    models, but different for open-source models.  Note: for all the metadata
    and documentation about the model itself, see `ModelField` in `schema.py`.
    """

    # Model group, used to determine quotas (e.g. "huggingface").
    # This group is only for user accounts, not benchmarking, and should probably
    # called something else.
    group: str

    # Name of the specific model (e.g. "huggingface/gpt-j-6b")
    # The name is <hosting_organization>/<model_name> or
    # <creator_organization>/<model_name>
    # There is also `<creator_organization>` (see `ModelField`).
    name: str

    # Tags corresponding to the properties of the model.
    tags: List[str] = field(default_factory=list)

    @property
    def organization(self) -> str:
        """
        Extracts the organization from the model name.
        Example: 'ai21/j1-jumbo' => 'ai21'
        """
        return self.name.split("/")[0]

    @property
    def engine(self) -> str:
        """
        Extracts the model engine from the model name.
        Example: 'ai21/j1-jumbo' => 'j1-jumbo'
        """
        return self.name.split("/")[1]


# For the list of available models, see the following docs:
# Note that schema.yaml has much of this information now.
# Over time, we should add more information there.

ALL_MODELS = [
    # Text-to-image models
    Model(
        group="magma",
        name="AlephAlpha/m-vader",
        tags=[TEXT_TO_IMAGE_MODEL_TAG, CLIP_TOKENIZER_TAG],
    ),
    Model(
        group="giga_gan",
        name="adobe/giga-gan",
        tags=[TEXT_TO_IMAGE_MODEL_TAG, CLIP_TOKENIZER_TAG],
    ),
    Model(
        group="firefly",
        name="adobe/firefly",
        # TODO: add TEXT_TO_IMAGE_MODEL_TAG later after the first batch of results
        tags=[CLIP_TOKENIZER_TAG],
    ),
    Model(
        group="dall_e2",
        name="openai/dalle-2",
        tags=[TEXT_TO_IMAGE_MODEL_TAG],
    ),
    Model(
        group="lexica",
        name="lexica/search-stable-diffusion-1.5",
        tags=[TEXT_TO_IMAGE_MODEL_TAG],
    ),
    Model(
        group="deepfloyd_if",
        name="DeepFloyd/IF-I-M-v1.0",
        tags=[TEXT_TO_IMAGE_MODEL_TAG, CLIP_TOKENIZER_TAG],
    ),
    Model(
        group="deepfloyd_if",
        name="DeepFloyd/IF-I-L-v1.0",
        tags=[TEXT_TO_IMAGE_MODEL_TAG, CLIP_TOKENIZER_TAG],
    ),
    Model(
        group="deepfloyd_if",
        name="DeepFloyd/IF-I-XL-v1.0",
        tags=[TEXT_TO_IMAGE_MODEL_TAG, CLIP_TOKENIZER_TAG],
    ),
    Model(
        group="mindall-e",
        name="kakaobrain/mindall-e",
        tags=[TEXT_TO_IMAGE_MODEL_TAG, CLIP_TOKENIZER_TAG],
    ),
    Model(
        group="dall_e",
        name="craiyon/dalle-mini",
        tags=[TEXT_TO_IMAGE_MODEL_TAG, CLIP_TOKENIZER_TAG],
    ),
    Model(
        group="dall_e",
        name="craiyon/dalle-mega",
        tags=[TEXT_TO_IMAGE_MODEL_TAG, CLIP_TOKENIZER_TAG],
    ),
    Model(
        group="cogview",
        name="thudm/cogview2",
        tags=[TEXT_TO_IMAGE_MODEL_TAG, CLIP_TOKENIZER_TAG],
    ),
    Model(
        group="huggingface_diffusers",
        name="huggingface/dreamlike-photoreal-v2-0",
        tags=[TEXT_TO_IMAGE_MODEL_TAG, CLIP_TOKENIZER_TAG],
    ),
    Model(
        group="huggingface_diffusers",
        name="huggingface/dreamlike-diffusion-v1-0",
        tags=[TEXT_TO_IMAGE_MODEL_TAG, CLIP_TOKENIZER_TAG],
    ),
    Model(
        group="huggingface_diffusers",
        name="huggingface/openjourney-v1-0",
        tags=[TEXT_TO_IMAGE_MODEL_TAG, CLIP_TOKENIZER_TAG],
    ),
    Model(
        group="huggingface_diffusers",
        name="huggingface/openjourney-v2-0",
        tags=[TEXT_TO_IMAGE_MODEL_TAG, CLIP_TOKENIZER_TAG],
    ),
    Model(
        group="huggingface_diffusers",
        name="huggingface/redshift-diffusion",
        tags=[TEXT_TO_IMAGE_MODEL_TAG, CLIP_TOKENIZER_TAG],
    ),
    Model(
        group="huggingface_diffusers",
        name="huggingface/promptist-stable-diffusion-v1-4",
        tags=[TEXT_TO_IMAGE_MODEL_TAG, CLIP_TOKENIZER_TAG],
    ),
    Model(
        group="huggingface_diffusers",
        name="huggingface/stable-diffusion-v1-4",
        tags=[TEXT_TO_IMAGE_MODEL_TAG, CLIP_TOKENIZER_TAG],
    ),
    Model(
        group="huggingface_diffusers",
        name="huggingface/stable-diffusion-v1-5",
        tags=[TEXT_TO_IMAGE_MODEL_TAG, CLIP_TOKENIZER_TAG],
    ),
    Model(
        group="huggingface_diffusers",
        name="huggingface/stable-diffusion-v2-base",
        tags=[TEXT_TO_IMAGE_MODEL_TAG, CLIP_TOKENIZER_TAG],
    ),
    Model(
        group="huggingface_diffusers",
        name="huggingface/stable-diffusion-v2-1-base",
        tags=[TEXT_TO_IMAGE_MODEL_TAG, CLIP_TOKENIZER_TAG],
    ),
    Model(
        group="huggingface_diffusers",
        name="huggingface/stable-diffusion-safe-weak",
        tags=[TEXT_TO_IMAGE_MODEL_TAG, CLIP_TOKENIZER_TAG],
    ),
    Model(
        group="huggingface_diffusers",
        name="huggingface/stable-diffusion-safe-medium",
        tags=[TEXT_TO_IMAGE_MODEL_TAG, CLIP_TOKENIZER_TAG],
    ),
    Model(
        group="huggingface_diffusers",
        name="huggingface/stable-diffusion-safe-strong",
        tags=[TEXT_TO_IMAGE_MODEL_TAG, CLIP_TOKENIZER_TAG],
    ),
    Model(
        group="huggingface_diffusers",
        name="huggingface/stable-diffusion-safe-max",
        tags=[TEXT_TO_IMAGE_MODEL_TAG, CLIP_TOKENIZER_TAG],
    ),
    Model(
        group="huggingface_diffusers",
        name="huggingface/vintedois-diffusion-v0-1",
        tags=[TEXT_TO_IMAGE_MODEL_TAG, CLIP_TOKENIZER_TAG],
    ),
]

MODEL_NAME_TO_MODEL: Dict[str, Model] = {model.name: model for model in ALL_MODELS}


def get_model(model_name: str) -> Model:
    """Get the `Model` given the name."""
    if model_name not in MODEL_NAME_TO_MODEL:
        raise ValueError(f"No model with name: {model_name}")

    return MODEL_NAME_TO_MODEL[model_name]


def get_model_group(model_name: str) -> str:
    """Get the model's group given the name."""
    model: Model = get_model(model_name)
    return model.group


def get_all_models() -> List[str]:
    """Get all model names."""
    return list(MODEL_NAME_TO_MODEL.keys())


def get_models_by_organization(organization: str) -> List[str]:
    """
    Gets models by organization e.g., ai21 => ai21/j1-jumbo, ai21/j1-grande, ai21-large.
    """
    return [model.name for model in ALL_MODELS if model.organization == organization]


def get_model_names_with_tag(tag: str) -> List[str]:
    """Get all the name of the models with tag `tag`."""
    return [model.name for model in ALL_MODELS if tag in model.tags]


def get_models_with_tag(tag: str) -> List[Model]:
    """Get all models with tag `tag`."""
    return [model for model in ALL_MODELS if tag in model.tags]


def get_all_text_models() -> List[str]:
    """Get all text model names."""
    return get_model_names_with_tag(TEXT_MODEL_TAG)


def get_all_code_models() -> List[str]:
    """Get all code model names."""
    return get_model_names_with_tag(CODE_MODEL_TAG)


def is_text_to_image_model(model_name: str) -> bool:
    model: Model = get_model(model_name)
    return TEXT_TO_IMAGE_MODEL_TAG in model.tags


def get_all_instruction_following_models() -> List[str]:
    """Get all instruction-following model names."""
    return get_model_names_with_tag(INSTRUCTION_FOLLOWING_MODEL_TAG)
