"""AdaLLaMA model configuration"""

from transformers import LlamaConfig

class AdaLlamaConfig(LlamaConfig):
    r"""
    AdaLLaMA model configuration
    """

    model_type = "adallama"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        num_prefix_layers=16,
        **kwargs
    ):  
        super().__init__(**kwargs)
        self.num_prefix_layers = num_prefix_layers