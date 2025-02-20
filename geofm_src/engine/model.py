import torch.nn as nn


class EvalModelWrapper(nn.Module):
    def __init__(self, model_config, data_config): 
        super().__init__()
        self.model_config = model_config
        self.data_config = data_config


    def load_encoder(self, blk_indices):
        """ 
        Loads the encoder. Also needs to save the encoder and the norm function for  
        normalizing features in self.encoder and self.norm respectively.
        Can also initialize objects needed for self.forward_feats and 
        self.default_blocks_to_cls (e.g. register forward hooks).
        """
        self.encoder = None
        self.norm = None
        raise NotImplementedError()
    
    def get_blocks(self, x):
        """ 
        Main function to extract features. Extracting blocks allows segmentation
        and multiple different mappings from blocks to a single representation. 
        For classification, you can also put define custom aggregation method 
        (e.g. the default method suggested by the authors) in self.default_blocks_to_cls.

        Input: 
            x: input tensor of size [b,c,h,w]
            indices: list of indices of intermediate features to return

        Output: 
            list of intermediate features indexed by indices, 
            each element is of size [b,p,d] where p is the number of patches
            and d the embedding dimension
        """
        raise NotImplementedError()

        # segm needs b,d,p,p
        # accel_cls needs b,pp,c

    def default_blocks_to_featurevec(self, block_list):
        """
        Input: 
            block_list: output of forward_feats

        Output:
            feature vector of size [b,d]
        """
        raise NotImplementedError()
