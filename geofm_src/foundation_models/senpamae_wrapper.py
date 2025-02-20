import torch
import os
from torchvision.datasets.utils import download_url
from peft import LoraConfig, get_peft_model
from .SenPaMAE.model import vit_base_patch16
import numpy as np
from geofm_src.engine.model import EvalModelWrapper




class SenPaMAEWrapper(EvalModelWrapper):
    URL = "https://drive.google.com/file/d/16IoG47yzdyUnPqUgaV8ofeja5RgQjlAz"
    # url = 'https://drive.usercontent.google.com/download?id=16IoG47yzdyUnPqUgaV8ofeja5RgQjlAz&export=download&authuser=0&confirm=t&uuid=9e279667-af3a-4f3a-a648-bec3452a1450&at=AIrpjvMEDRsz82ufHQy8sUmSk5j5%3A1739180929862'

    def load_encoder(self, blk_indices):
        model_config = self.model_config

        encoder = vit_base_patch16(
            image_size=model_config.image_resolution,
            num_channels=model_config.num_channels,
            emb_dim=model_config.embed_dim,
        )

        # download weights
        if model_config.get("pretrained_path", None):
            path = model_config.pretrained_path
            if not os.path.exists(path):
                raise NotImplementedError('Need to manually download weights from above link')
                download_url(
                    URL.format(os.path.basename(path)),
                    os.path.dirname(path),
                    filename=os.path.basename(path),
                )

        # load weights
        check_point = torch.load(path, map_location="cpu")
        encoder.load_state_dict(check_point, strict=False)

        # register fwd hooks
        for idx in blk_indices:
            encoder.transformer[idx].register_forward_hook(
                lambda m, i, o: self._cache_block(o))

        # set variables
        self.encoder = encoder
        self.norm = self.encoder.layer_norm

        self.process_srfs()

    def _cache_block(self,x):
        self.cache.append(x)


    def process_srfs(self):
        # SRF loading
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)  # Go up one level

        srf_path = os.path.join(
            parent_dir,
            "foundation_models/SenPaMAE/responsefunctions",
            self.data_config.senpamae_srf_name,
        )
        if not os.path.exists(srf_path):
            raise ValueError(f"SRF not found at {srf_path}")
        self.srf = np.load(srf_path).T
        subset_channels = self.data_config.senpamae_channels
        self.srf = self.srf[subset_channels, :]
        self.srf = torch.from_numpy(self.srf).float()
        self.srf = self.srf.unsqueeze(0)

        # Convert band_gsds to numpy array and index it
        self.band_gsds = np.array(self.data_config.band_gsds)[
            self.data_config.senpamae_channels
        ]
        self.band_gsds = torch.tensor(self.band_gsds).float().unsqueeze(0)

        print("SRF shape: ", self.srf.shape)
        print("Selected GSDs: ", self.band_gsds, self.band_gsds.shape)


    def apply_peft(self, encoder, lora_cfg: dict):
        """
        Apply LoRA to the last few layers of the encoder using PEFT.
        """

        print("LORA: Applying PEFT: ", lora_cfg)

        # Configure LoRA
        peft_config = LoraConfig(
            r=lora_cfg.get("lora_rank", 16),  # Rank of LoRA
            lora_alpha=lora_cfg.get("lora_alpha", 16),  # Scaling factor for LoRA
            target_modules=lora_cfg.get(
                "lora_target_modules", "blocks.*.attn.qkv"
            ),  # ["qkv", "proj"]
            lora_dropout=lora_cfg.get("lora_dropout", 0.0),  # Dropout rate for LoRA
            bias=lora_cfg.get("bias", "none"),
            task_type=lora_cfg.get(
                "lora_task_type", None
            ),  # Task type (use appropriate type for your model), "SEQ_CLS"
        )

        # Wrap the encoder with PEFT
        self.encoder = get_peft_model(encoder, peft_config)


    def get_blocks(self, x):
        self.cache = []

        x = x[:, self.data_config.senpamae_channels, :, :]

        device = x.device
        srf = self.srf.to(device)
        gsd = self.band_gsds.to(device)
        self.encoder(x, gsd=gsd, rf=srf)

        blocks_list = self.cache
        self.cache = []

        return blocks_list

    def default_blocks_to_featurevec(self, block_list):
        # no official recommendation by the authors, using this for now
        return self.norm(block_list[-1]).mean(dim=1)
