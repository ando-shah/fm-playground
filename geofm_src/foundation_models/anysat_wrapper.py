import torch.nn as nn
import torch
from torch import Tensor
from einops import rearrange
from geofm_src.foundation_models.galileo.util import construct_galileo_input, MaskedOutput

from typing import Any


from geofm_src.engine.model import EvalModelWrapper


EXPECTED_CHANNELS = {
    "s2": 10,
    "s1": 3,
    "s1-asc": 2,
    "spot": 3,
}


class AnySatWrapper(EvalModelWrapper):
    def load_encoder(self, blk_indices):
        self.encoder = torch.hub.load("gastruc/anysat", self.model_config.anysat_torchhub_id, pretrained=True, flash_attn=False)

        # Dont think there is a norm layer to be used here
        self.norm = nn.Identity()
    
        
        # prepare hooks
        for idx in blk_indices:
            self.encoder.model.blocks[idx].register_forward_hook(
                lambda m, i, o: self._cache_block(o))

    def _cache_block(self,x):
        self.cache.append(x)


    def format_input(self, x: Tensor, input_key: str) -> dict[str, Tensor]:
        """Process the input tensor and stack per-sample outputs to form a single MaskedOutput.

        Args:
            x (Tensor): Input tensor of shape [B, C, H, W] (for s1, s1-asc, or s2/spot).
            input_key (str): Specifies how to interpret the channels.

        Returns:
            dict[str, Tensor]: A dictionary mapping names as expected by the encoder.
        """
        # Validate expected channel count.
        expected = EXPECTED_CHANNELS.get(input_key)
        if expected is None:
            raise ValueError(f"Unsupported input key: {input_key}")
        assert x.shape[1] == expected, (
            f"Input tensor for {input_key} should have {expected} channels"
        )

        # For keys that require a time dimension unsqueeze and rearrange.
        if input_key in {"s1", "s1-asc", "s2"}:
            x = x.unsqueeze(1)  # Now shape: [B, 1, C, H, W]
            x = rearrange(x, "B T C H W -> B H W T C")

        # Process each element in the batch through the Galileo input function, but perhaps also need
        # batched_collate_fn?
        outputs = [construct_galileo_input(**{input_key: x[i]}) for i in range(x.shape[0])]

        # Stack each field of the MaskedOutput over the batch dimension for batched input.
        batched = MaskedOutput(
            space_time_x=torch.stack([o.space_time_x for o in outputs], dim=0),
            space_x=torch.stack([o.space_x for o in outputs], dim=0),
            time_x=torch.stack([o.time_x for o in outputs], dim=0),
            static_x=torch.stack([o.static_x for o in outputs], dim=0),
            space_time_mask=torch.stack([o.space_time_mask for o in outputs], dim=0),
            space_mask=torch.stack([o.space_mask for o in outputs], dim=0),
            time_mask=torch.stack([o.time_mask for o in outputs], dim=0),
            static_mask=torch.stack([o.static_mask for o in outputs], dim=0),
            months=torch.stack([o.months for o in outputs], dim=0),
        )

        return {
            "s_t_x": batched.space_time_x,
            "sp_x": batched.space_x,
            "t_x": batched.time_x,
            "st_x": batched.static_x,
            "s_t_m": batched.space_time_mask,
            "sp_m": batched.space_mask,
            "t_m": batched.time_mask,
            "st_m": batched.static_mask,
            "months": batched.months,
        }

    def get_blocks(self, x):
        self.cache = []
        # TODO these arguments will be different for segmentation 
        self.encoder(self.format_input(x, self.model_config.input_key), patch_size=10, output='tile')
        blocks = self.cache
        self.cache = [] 
        return blocks

    def default_blocks_to_featurevec(self, block_list):
        x = block_list[-1][:, 1:,:].mean(dim=1)
        # TODO not sure how to configure the norm layer above
        # x = self.norm(x)
        return x