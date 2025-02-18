from .lightning_task import LightningClassificationTask, LightningSegmentationTask
from .CROMA.use_croma import PretrainedCROMA
import torch
import os
from torchvision.datasets.utils import download_url



def load_encoder(model_config):
    URL = "https://huggingface.co/antofuller/CROMA/resolve/main/{}"

    if model_config.get("pretrained_path", None):
        path = model_config.pretrained_path
        if not os.path.exists(path):
            # download the weights from HF
            download_url(
                URL.format(os.path.basename(path)),
                os.path.dirname(path),
                filename=os.path.basename(path),
            )
    else:
        path = None

    encoder = PretrainedCROMA(
        pretrained_path=path,
        size=model_config.size,
        modality=model_config.modality,
        image_resolution=model_config.image_resolution,
    )
    return encoder



class CromaClassification(LightningClassificationTask):

    def __init__(self, args, model_config, data_config):
        super().__init__(args, model_config, data_config)

        self.encoder = load_encoder(model_config)

        # assign classification head
        #   we take pre-trained layer norm and only replace the linear layer,
        #   so in linear probing, layer norm is frozen, only linear layer is unfrozen!
        del self.encoder.s2_GAP_FFN[2:]
        self.linear_classifier = torch.nn.Linear(
            self.encoder.s2_GAP_FFN[1].in_features, data_config.num_classes)
        self.encoder.s2_GAP_FFN[1] = self.linear_classifier
        self.dot_str_of_linear_classifier = 's2_GAP_FFN.1' 

        self.freeze_and_return_params()

    def forward(self, samples):
        all_output = self.encoder(optical_images=samples)
        out_logits = all_output["optical_GAP"]
        feats = all_output["optical_encodings"]
        return (out_logits, feats)


class CromaSegmentation(LightningSegmentationTask):
    def __init__(self, args, model_config, data_config):
        super().__init__(args, model_config, data_config)

        self.encoder = load_encoder(model_config)
        self._build_default_segm_modules()
        self.freeze_and_return_params()

    def _forward_feats(self, samples):
        feats = self.encoder(optical_images=samples)["out_feats"]
        return feats



# Model factory for different dinov2 tasks
def CromaModel(args, model_config, data_config):
    task = data_config.task
    if task == "classification":
        return CromaClassification(args, model_config, data_config)
    elif args.task == "regression":
        return CromaRegression(args, model_config, data_config)
    elif args.task == "segmentation":
        return CromaSegmentation(args, model_config, data_config)
    else:
        raise NotImplementedError("Task not supported")
