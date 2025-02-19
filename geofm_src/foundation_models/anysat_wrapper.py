import torch.nn as nn
import torch
from torch import Tensor
from einops import repeat
from .lightning_task import LightningClsRegTask, LightningSegmentationTask



def load_encoder():
    encoder = torch.hub.load(
        "gastruc/anysat", "anysat", pretrained=True, flash_attn=False)
    return encoder

def format_input(x: Tensor, input_key: str) -> dict[str, Tensor]:
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

    print("FORMATTED INPUTS")
    for key, val in anysat_input.items():
        print(key, val.shape)
        print("stats", val.mean(), val.std(), val.min(), val.max())
    
    return anysat_input


class AnySatClsReg(LightningClsRegTask):
    def __init__(self, args, model_config, data_config):
        super().__init__(args, model_config, data_config)

        self.encoder = load_encoder()

        self.linear_classifier = nn.Linear(
            model_config.embed_dim, data_config.num_classes)
        self.dot_str_of_linear_classifier = None

        self.freeze_and_return_params()

    def forward(self, x):
        x = format_input(x, self.data_config.input_key)
        # https://github.com/gastruc/AnySat?tab=readme-ov-file#extract-features
        # patch size must be multiple of 10
        x = self.encoder(x, patch_size=10, output="tile")
        x = self.linear_classifier(x)
        return x



class AnySatSegmentation(LightningSegmentationTask):
    def __init__(self, args, model_config, data_config):
        super().__init__(args, model_config, data_config)

        self.encoder = load_encoder()
        self._build_default_segm_modules()
        self.freeze_and_return_params()

    def _forward_feats(self, x: Tensor) -> Tensor:
        """Forward pass to get features from the encoder.
        
        Args:
            x (Tensor): input tensor

        Returns:
            Tensor: features from the model
        """
        x = format_input(x, self.data_config.input_key)
        feats = self.encoder(samples, patch_size=10, output="dense", output_modality=self.data_config.input_key)
        return feats



# Model factory for different dinov2 tasks
def AnySatModel(args, model_config, data_config):
    task = data_config.task
    if task in ["classification",'regression']:
        return AnySatClsReg(args, model_config, data_config)
    elif task == "segmentation":
        return AnySatSegmentation(args, model_config, data_config)
    else:
        raise NotImplementedError("Task not supported")
