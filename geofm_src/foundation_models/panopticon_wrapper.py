import torch
from geofm_src.engine.model import EvalModelWrapper


class PanopticonWrapper(EvalModelWrapper):

    def _load_encoder(self, blk_indices):
        self.encoder = torch.hub.load(
            '/home/hk-project-pai00028/tum_mhj8661/code/PanOpticOn',
            'panopticon_from_eval_config',
            self.model_config,
            source='local')
        self.norm = self.encoder.norm
        self.blk_indices = blk_indices

        wavelengths = self.data_config.wavelengths_mean_nm
        self.register_buffer('chn_ids',
            torch.tensor([wl for wl in wavelengths]).unsqueeze(0))
    
    def get_blocks(self, x):
        x = dict(imgs=x, chn_ids=self.chn_ids.expand(x.size(0), -1))

        if self.encoder.chunked_blocks:
            x_blocks = self.encoder._get_intermediate_layers_chunked(x, self.blk_indices)
        else:
            x_blocks = self.encoder._get_intermediate_layers_not_chunked(x, self.blk_indices)

        return x_blocks

    def default_blocks_to_featurevec(self, block_list):
        return self.norm(block_list[-1])[:,0]

    def replace_pe(self, num_channels):
        raise NotImplementedError('No need :)')