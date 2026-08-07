"""DiBO's self-contained Transformers wrapper for the pinned LLaDA model.

The paper-faithful checkpoints have a tokenizer-sized input embedding and a
larger, untied output projection. The pinned LLaDA implementation uses one
configuration field for both dimensions, so this wrapper reconstructs the
stored shape exactly when a DiBO safetensors export is loaded.
"""

import torch.nn as nn

from .configuration_llada import LLaDAConfig
from .modeling_llada import (
    LLaDAModel,
    LLaDAModelLM,
    create_model_config_from_pretrained_config,
)


class DiBOLLaDAModelLM(LLaDAModelLM):
    """LLaDA model with separately recorded input and output vocabulary sizes."""

    config_class = LLaDAConfig

    def __init__(self, config: LLaDAConfig):
        input_size = int(
            getattr(config, "dibo_input_embedding_size", config.vocab_size)
        )
        output_size = int(
            getattr(
                config,
                "dibo_output_embedding_size",
                getattr(config, "embedding_size", config.vocab_size),
            )
        )
        if bool(config.weight_tying) and input_size != output_size:
            raise ValueError(
                "A tied DiBO LLaDA model must use identical input and output vocabulary sizes."
            )

        model_config = create_model_config_from_pretrained_config(config)
        # Match LLaDAModelLM's native CPU construction before from_pretrained
        # streams the safetensors weights into this model.
        model_config.init_device = "cpu"
        model_config.vocab_size = input_size
        model_config.embedding_size = input_size
        model = LLaDAModel(model_config)
        if not model_config.weight_tying:
            model.transformer["ff_out"] = nn.Linear(
                model_config.d_model,
                output_size,
                bias=model_config.include_bias,
                device=model_config.init_device,
            )
        super().__init__(config, model=model)
