from geofm_src.factory import create_dataset
from omegaconf import OmegaConf
import hydra
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from geofm_src.engine.accelerated.utils.data import compute_dataset_stats
import os

root = '/home/hk-project-pai00028/tum_mhj8661/code/fm-playground/geofm_src/configs/dataset'

ds_configs = [
    # 'benv2_s1.yaml',
    # 'benv2_s2_12b.yaml',
    # 'corine_202b.yaml',
    # 'corine_sd.yaml',
    # 'digital_typhoon_3b.yaml', # TODO
    # 'fmow_8b.yaml', # TODO
    # 'geobench_brick_kiln_13b.yaml', # TODO
    'geobench_eurosat_13b.yaml',
    'geobench_cashew_12b.yaml',
    'geobench_chesapeake_4b.yaml',
    'geobench_forestnet.yaml',
    'geobench_neontree.yaml',
    'geobench_nzcattle.yaml',
    'geobench_pv4ger_cls.yaml',
    'geobench_sacrop_12b.yaml',
    'geobench_so2sat_10b.yaml',
    'hyperview_150b.yaml',
    'l8_6b.yaml',
    'tropical_cyclone.yaml'
]


def get_config(relpath):
    # imitate hydra.compose, weird bugs when hyra.instantiate is executed in notebook
    p = os.path.join(root, relpath)
    cfg = OmegaConf.load(p)

    defaults = [OmegaConf.load(os.path.join(root, f'{r}.yaml')) for r in cfg.get('defaults', [])]
    cfg = OmegaConf.merge(*defaults, cfg)
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

    return cfg



for relpath in ds_configs:
    
    cfg = get_config(relpath)
    print('--------------------------', cfg.dataset_name )


    print(OmegaConf.to_yaml(cfg))
    train, val, test = create_dataset(cfg)

    ds = dict(
        train=train,
        val=val,
        test=test
    )

    for n,d in ds.items():
        print(f'computing stats for {n}')
        compute_dataset_stats(d, batch_size=256, subset=-1, num_workers=32, num_channels=cfg.num_channels)