import torch.nn as nn
import torch
from torch import Tensor
from einops import repeat, rearrange


from geofm_src.engine.model import EvalModelWrapper



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
        """Format input tensor to be passed to the model.
        
        According to https://github.com/gastruc/AnySat?tab=readme-ov-file#format-your-data
        Args:
            x (Tensor): input tensor
        Returns:
            dict[str, Tensor]: formatted input tensor
        """
        match input_key:
            case "s2":
                assert x.shape[1] == 10, "Input tensor for s2 should have 10 channels"
                # unsqueeze time dimension
                x = x.unsqueeze(1)
                dates = torch.arange(x.shape[1]).float()

            case "s1":
                assert x.shape[1] == 3, "Input tensor for s1 should have 3 channels"
                # unsqueeze time dimension
                x = x.unsqueeze(1)
                dates = torch.arange(x.shape[1]).float()
            case "s1-asc":
                assert x.shape[1] == 2, "Input tensor for s1-asc should have 2 channels"
                # unsqueeze time dimension
                x = x.unsqueeze(1)
                dates = torch.arange(x.shape[1]).float()

            case "spot":
                assert x.shape[1] == 3, "Input tensor for spot should have 3 channels"
                dates = None

        anysat_input = {input_key: x}
        if dates is not None:
            dates = repeat(dates, 't -> b t', b=x.shape[0])
            anysat_input[f'{input_key}_dates'] = dates.to(x.device)


        return anysat_input

    def get_blocks(self, x):
        self.cache = []
        # TODO these arguments will be different for segmentation 
        self.encoder(self.format_input(x, self.model_config.input_key), patch_size=10, output='tile')
        blocks = self.cache
        self.cache = [] 
        return blocks

    def default_input_to_feature_list(self, x: Tensor) -> list[torch.Tensor]:
        self.cache = []
        self.encoder(self.format_input(x, self.model_config.input_key), patch_size=10, output='dense', output_modality=self.model_config.input_key)
        blocks = self.cache
        self.cache = []
        patch_size = int(blocks[0].size(1) ** 0.5)
        out = [rearrange(f[:, 1:, :], "b (h w) c -> b c h w", h=patch_size, w=patch_size) for f in blocks]
        return out

    def default_blocks_to_featurevec(self, block_list):
        x = block_list[-1][:, 1:,:].mean(dim=1)
        # TODO not sure how to configure the norm layer above
        # x = self.norm(x)
        return x

