import os
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange
from torchvision.datasets.utils import download_url

from .lightning_task import LightningClsRegTask, LightningSegmentationTask
from .galileo.model import Encoder
from .galileo.util import (
    construct_galileo_input,
    MaskedOutput,
    SPACE_TIME_BANDS_GROUPS_IDX,
)


EXPECTED_CHANNELS = {
    "s2": 10,
    "s1": 3,
    "s1-asc": 2,
    "spot": 3,
}


def format_input(x: Tensor, input_key: str) -> dict[str, Tensor]:
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


def load_encoder(model_config):
    URL = "https://hf.co/nasaharvest/galileo/resolve/main/models/base/{}"
    pretrained_path = model_config.get("pretrained_path", None)
    print(f"PRETRAINED PATH: {pretrained_path}")

    if pretrained_path and not os.path.exists(pretrained_path):
        # Download weights and config if they don't exist.
        download_url(
            URL.format("encoder.pt"),
            os.path.dirname(pretrained_path),
            filename=os.path.basename(pretrained_path),
        )
        download_url(
            URL.format("config.json"),
            os.path.dirname(pretrained_path),
            filename="config.json",
        )
    path = pretrained_path if pretrained_path else None

    encoder = Encoder.load_from_folder(Path(os.path.dirname(path)), device="cpu")
    return encoder


class GalileoClsReg(LightningClsRegTask):
    def __init__(self, args, model_config, data_config):
        """Initalizes the GalileoClsReg model.

        Args:
            args: Command-line arguments or similar.
            model_config (Dict[str, Any]): Configuration for the model.
            data_config (Dict[str, Any]): Configuration for the data.
        """
        super().__init__(args, model_config, data_config)
        self.encoder = load_encoder(model_config)
        self.linear_classifier = nn.Linear(
            model_config.embed_dim, data_config.num_classes
        )
        self.dot_str_of_linear_classifier = None
        self.freeze_and_return_params()
        self.patch_size = model_config.patch_size

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the encoder and linear classifier.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Class predictions or regression outputs.
        """
        x = format_input(x, self.data_config.input_key)
        if self.training_mode == "linear_probe":
            with torch.no_grad():
                outputs = self.encoder(patch_size=self.patch_size, **x)
        else:
            outputs = self.encoder(patch_size=self.patch_size, **x)

        # Unpack outputs from encoder
        s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, months = outputs

        # Average tokens to derive features based on
        # https://github.com/nasaharvest/galileo/blob/e0581f735763fa5752b1d07b378384a3f28ca697/src/galileo.py#L1409
        feat = self.encoder.average_tokens(
            s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m
        )
        x = self.linear_classifier(feat)
        return x


class GalileoSegmentation(LightningSegmentationTask):
    def __init__(self, args, model_config, data_config):
        super().__init__(args, model_config, data_config)

        self.encoder = load_encoder(model_config)

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
        feats = self.encoder(
            samples,
            patch_size=10,
            output="dense",
            output_modality=self.data_config.input_key,
        )
        return feats


# Model factory for different dinov2 tasks
def GalileoModel(args, model_config, data_config):
    task = data_config.task
    if task in ["classification", "regression"]:
        return GalileoClsReg(args, model_config, data_config)
    elif task == "segmentation":
        return GalileoSegmentation(args, model_config, data_config)
    else:
        raise NotImplementedError("Task not supported")
