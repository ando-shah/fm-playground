import torch.nn as nn
import torch
import os

# use mmsegmentation for upernet+mae
from .lightning_task import LightningClassificationTask, LightningSegmentationTask
from einops import rearrange
from torchvision.datasets.utils import download_url

from .base import LinearHead


def load_encoder(model_config): 
    URL = "https://huggingface.co/wangyi111/softcon/resolve/main/{}"

    # load dino model
    encoder = torch.hub.load("facebookresearch/dinov2", model_config.dinov2_torchhub_id)
    # add softcon input layer
    encoder.patch_embed.proj = torch.nn.Conv2d(
        model_config.num_channels, 384, kernel_size=(14, 14), stride=(14, 14)
    )

    # look for Softcon pretrained weights
    if model_config.get("pretrained_path", None):
        path = model_config.pretrained_path
        if not os.path.exists(path):
            download_url(
                URL.format(os.path.basename(path)),
                os.path.dirname(path),
                filename=os.path.basename(path),
            )

        ckpt_vit14 = torch.load(path, map_location="cpu")
        msg = encoder.load_state_dict(ckpt_vit14)
        print(msg)

    return encoder


class SoftConClassification(LightningClassificationTask):
    """SoftCon model for classification."""


    def __init__(self, args, model_config, data_config):
        super().__init__(args, model_config, data_config)

        self.encoder = load_encoder(model_config)

        # TODO this embed dim could be pulled from the encoder, to remove the need for the arg
        self.linear_classifier = LinearHead(
            in_features=model_config.embed_dim, num_classes=data_config.num_classes
        )
        self.dot_str_of_linear_classifier = None

        self.freeze_and_return_params()

    def forward(self, samples):
        out = self.encoder.forward_features(samples)
        global_pooled = out["x_norm_patchtokens"].mean(dim=1)
        out_logits = self.linear_classifier(global_pooled)
        return out_logits, global_pooled


class SoftConSegmentation(LightningSegmentationTask):
    """SoftCon Model for Segmentation."""

    def __init__(self, args, model_config, data_config):
        super().__init__(args, model_config, data_config)

        self.encoder = load_encoder(model_config)
        self._build_default_segm_modules()
        self.freeze_and_return_params()

    def _forward_feats(self, samples):
        outputs = self.encoder.get_intermediate_layers(samples, [4, 6, 10, 11])
        feats = [
            rearrange(out, "n (h w) c -> n c h w", h=int(out.size(1) ** 0.5))
            for out in outputs
        ]
        return feats




# Model factory for different dinov2 tasks
def SoftConModel(args, model_config, data_config):
    task = data_config.task
    if task == "classification":
        return SoftConClassification(args, model_config, data_config)
    elif args.task == "regression":
        return SoftConRegression(args, model_config, data_config)
    elif args.task == "segmentation":
        return SoftConSegmentation(args, model_config, data_config)
    else:
        raise NotImplementedError("Task not supported")
