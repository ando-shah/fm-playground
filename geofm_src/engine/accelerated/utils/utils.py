# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
# 
# adjusted from https://github.com/facebookresearch/dinov2/blob/main/dinov2/

import logging
from typing import Dict, Optional

import torch
from torch import nn
from torchmetrics import MetricCollection

from .data import SamplerType, make_data_loader
from . import distributed
from .logger import MetricLogger
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import os
import pandas as pd
import time
import sys

logger = logging.getLogger("eval")

def has_ddp_wrapper(m: nn.Module) -> bool:
    return isinstance(m, DistributedDataParallel)

def remove_ddp_wrapper(m: nn.Module) -> nn.Module:
    return m.module if has_ddp_wrapper(m) else m

def blocks_to_cls(blocks_list, use_n_blocks, pooling, norm=None, default_blocks_to_featurevec=None):
    """ different ways to process features from backbone blocks to single feature vectors

    Inputs:
        blocks_list: list of output tensors of all blocks, each item is a 
            tensor of shape (b, p, c) where p is number of patches 
    
    Output:
        output: tensor of shape (b, c) where c is the feature dimension
    """
    blocks_list = blocks_list[-use_n_blocks:] # list of output tensors of last n blocks

    if pooling == 'avgpool': # corresponds to DINOv2 avgpool=True
        blocks_list = [norm(x) for x in blocks_list]
        class_tokens = [x[:, 0] for x in blocks_list]

        output = torch.cat(
            (
                torch.cat(class_tokens, dim=-1),
                torch.mean(blocks_list[-1][:, 1:], dim=1),
            ),
            dim=-1,
        )
        output = output.reshape(output.shape[0], -1)

    elif pooling == 'cls': # corresponds to DINOv2 avgpool=False
        blocks_list = [norm(x) for x in blocks_list]
        class_tokens = [x[:, 0] for x in blocks_list]
        output = torch.cat(class_tokens, dim=-1)

    elif pooling == 'normalized_default': # consistent with vanilla DINOv2 eval knn
        output = default_blocks_to_featurevec(blocks_list)
        output = nn.functional.normalize(output, dim=1, p=2) # need to normalize knn! Big performance drop if not done ..

    elif pooling == 'default':
        output = default_blocks_to_featurevec(blocks_list)

    else:
        raise ValueError(f"Pooling {pooling} not supported")

    return output.float()


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    data_loader,
    postprocessors: Dict[str, nn.Module],
    metrics: Dict[str, MetricCollection],
    device: torch.device,
    criterion: Optional[nn.Module] = None,
):
    model.eval()
    if criterion is not None:
        criterion.eval()

    for metric in metrics.values():
        metric = metric.to(device)

    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    for samples, targets, *_ in metric_logger.log_every(data_loader, 10, header):
        outputs = model(samples.to(device))
        targets = targets.to(device)

        if criterion is not None:
            loss = criterion(outputs, targets)
            metric_logger.update(loss=loss.item())

        for k, metric in metrics.items():
            metric_inputs = postprocessors[k](outputs, targets)
            metric.update(**metric_inputs)

    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger}")

    stats = {k: metric.compute() for k, metric in metrics.items()}
    metric_logger_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return metric_logger_stats, stats


def all_gather_and_flatten(tensor_rank):
    tensor_all_ranks = torch.empty(
        distributed.get_global_size(),
        *tensor_rank.shape,
        dtype=tensor_rank.dtype,
        device=tensor_rank.device,
    )
    tensor_list = list(tensor_all_ranks.unbind(0))
    if distributed.is_enabled():
        dist.all_gather(tensor_list, tensor_rank.contiguous())
    else:
        tensor_all_ranks = tensor_rank.unsqueeze(0)
    return tensor_all_ranks.flatten(end_dim=1)


def extract_features(model, dataset, dl_cfg, gather_on_cpu=False):
    dataset_with_enumerated_targets = DatasetWithEnumeratedTargets(dataset)
    sample_count = len(dataset_with_enumerated_targets)
    sampler_type = SamplerType.DISTRIBUTED if distributed.is_enabled() else SamplerType.EPOCH
    data_loader = make_data_loader(
        dataset=dataset_with_enumerated_targets,
        sampler_type=sampler_type,
        drop_last=False,
        shuffle=False,
        **dl_cfg
    )
    return extract_features_with_dataloader(model, data_loader, sample_count, gather_on_cpu)


@torch.inference_mode()
def extract_features_with_dataloader(model, data_loader, sample_count, gather_on_cpu=False):
    gather_device = torch.device("cpu") if gather_on_cpu else torch.device("cuda")
    metric_logger = MetricLogger(delimiter="  ")
    features, all_labels = None, None
    for samples, (index, labels_rank) in metric_logger.log_every(data_loader, 10):
        if isinstance(samples, dict):
            for k, v in samples.items():
                if isinstance(v, torch.Tensor):
                    samples[k] = v.cuda(non_blocking=True)
        else:
            samples = samples.cuda(non_blocking=True)
        labels_rank = labels_rank.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        features_rank = model(samples).float()

        # init storage feature matrix
        if features is None:
            features = torch.zeros(sample_count, features_rank.shape[-1], device=gather_device)
            labels_shape = list(labels_rank.shape)
            labels_shape[0] = sample_count
            all_labels = torch.full(labels_shape, fill_value=-1, device=gather_device)
            logger.info(f"Storing features into tensor of shape {features.shape}")

        # share indexes, features and labels between processes
        index_all = all_gather_and_flatten(index).to(gather_device)
        features_all_ranks = all_gather_and_flatten(features_rank).to(gather_device)
        labels_all_ranks = all_gather_and_flatten(labels_rank).to(gather_device)

        # update storage feature matrix
        if len(index_all) > 0:
            features.index_copy_(0, index_all, features_all_ranks)
            all_labels.index_copy_(0, index_all, labels_all_ranks)

    logger.info(f"Features shape: {tuple(features.shape)}")
    logger.info(f"Labels shape: {tuple(all_labels.shape)}")

    assert torch.all(all_labels > -1)

    return features, all_labels


def collect_csv(output_dir, return_df='relpath', sleep=0.05):
    """ recursively create all aggregation csv files in nested folder structure """
    
    def add_lvl(df: pd.DataFrame):
        """ for pretty to_string() convert relpath to index with lvl0, lvl1, ... """
        if df.shape[0] == 0:
            return df
        df['lvl'] = df['relpath'].apply(lambda x: x.split('/'))
        max_level = max([len(l) for l in df['lvl']])
        for i in range(max_level):
            df[f'lvl{i}'] = df['lvl'].apply(lambda x: x[i] if len(x) > i else '')
        df.set_index([f'lvl{i}' for i in range(max_level)] + ['metric'], inplace=True)
        df.drop(columns=['lvl'], inplace=True)
        df.sort_index(inplace=True)
        return df
    
    def rec(abspath):
        sub_dirs = [p for p in os.listdir(abspath) if os.path.isdir(os.path.join(abspath,p))]
        sub_results = [rec(os.path.join(abspath,p)) for p in sub_dirs]
        sub_results = [r for r in sub_results if len(r) > 0]

        if len(sub_results) == 0:
            if 'results.csv' in os.listdir(abspath):
                if sleep > 0.0:
                    time.sleep(sleep)
                df = pd.read_csv(os.path.join(abspath, 'results.csv'))
                df['relpath'] = ''
                return (os.path.basename(abspath), df)
            return ()
        
        else:
            dfs = []
            for d, df in sub_results:
                df['relpath'] = df['relpath'].apply(lambda x: os.path.normpath(os.path.join(d,x)))
                dfs.append(df)
            df = pd.concat(dfs).reset_index(drop=True)

            df.to_csv(os.path.join(abspath, 'results.csv'))
            with open(os.path.join(abspath, 'results.txt'),'w+') as f:
                df_print = add_lvl(df.copy()).drop(columns=['relpath'])
                f.write(df_print.to_string())

            return (os.path.basename(abspath), df)
                
    t = rec(output_dir)
    if len(t) == 0:
        df = pd.DataFrame(columns=['metric','value','best_classifier','relpath'])
    else:
        df = t[1]
    
    if return_df == 'lvl':
        return add_lvl(df).drop(columns=['relpath'])
    elif return_df == 'relpath':
        return df
    elif return_df == 'all':
        return add_lvl(df)
    else:
        raise ValueError(f'Unknown return_df {return_df}')