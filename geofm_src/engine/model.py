import torch.nn as nn


class EvalModelWrapper(nn.Module):
    def __init__(self, model_config, data_config): 
        super().__init__()
        self.model_config = model_config
        self.data_config = data_config


    def load_encoder(self, blk_indices):
        """ 
        Loads the encoder and prepares any functionality needed for extracting the 
        blocks index by blk_indices (e.g. register forward hooks) in get_blocks.
        Also needs to save the encoder and the norm function for  
        normalizing features in self.encoder and self.norm respectively.

        Input:
            blk_indices: list of indices of intermediate features to return
        """
        self.encoder = None
        self.norm = None
        raise NotImplementedError()
    
    def get_blocks(self, x):
        """ 
        Main function to extract features. Extracting blocks allows segmentation
        and multiple different mappings from blocks to a single representation. 

        Input: 
            x: input tensor of size [b,c,h,w]
            indices: list of indices of intermediate features to return

        Output: 
            list of intermediate features indexed by indices, 
            each element is of size [b,p,d] where p is the number of patches
            and d the embedding dimension
        """
        raise NotImplementedError()


    def default_blocks_to_featurevec(self, block_list):
        """
        Takes the output of get_blocks and returns a single feature vector computed 
        from this list. During classification / regression, essentially this happens:

            block_list = wrapper.get_blocks(x)
            feature_vec = wrapper.default_blocks_to_featurevec(block_list)
            logits = linear_classifier_head(feature_vec)
        
        This function should be the default method suggested by the authors of 
        the model to extract features from the blocks (or only the last block).
        The advantage of separating this function from get_blocks is that we 
        can test different aggregation functions during linear probing.

        This function is only needed for classification / regression!

        Input: 
            block_list: output of forward_feats

        Output:
            feature vector of size [b,d]
        """
        raise NotImplementedError()
