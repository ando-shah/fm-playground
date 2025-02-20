import datetime
import os
from pathlib import Path
import warnings
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import MLFlowLogger, WandbLogger
from lightning import Trainer
from lightning.pytorch.strategies import DDPStrategy
from datasets.data_module import BenchmarkDataModule
from lightning.pytorch import seed_everything
from factory import create_model
import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from geofm_src.engine.accelerated.utils.logging import setup_logger

from geofm_src.engine.accelerated.linear import run_eval_linear
from geofm_src.engine.lightning import LightningClsRegTask, LightningSegmentationTask


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore")


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )


@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    is_fastdevrun = cfg.trainer.get('fast_dev_run', False)
    task = cfg.dataset.task
    training_mode = cfg.model.training_mode

    # setup output dir
    if cfg.output_dir is None:
        experiment_name = f"{cfg.model.model_type}/{cfg.model.training_mode}/{cfg.dataset.dataset_name}"
        args_defining_run = {
            "lr": "lr",
            "batch_size": "bsz",
            "epochs": "e",
        }
        assert all([not "." in k for k in args_defining_run.keys()]), (
            "cannot contain nested '.' yet"
        )
        run_name = "_".join([f"{v}={cfg[k]}" for k, v in args_defining_run.items()])

        suff = cfg.output_dir_suffix if cfg.output_dir_suffix is not None else ""
        cfg.output_dir = os.path.join(
            os.environ["ODIR"], experiment_name, suff, run_name
        )

    else:
        assert cfg.output_dir_suffix is None, (
            "output_dir_suffix is only used when output_dir is not set"
        )
        experiment_name = os.path.basename(os.path.normpath(cfg.output_dir))
        run_name = (
            f"{experiment_name}_run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

    # check if task already executed
    if os.path.exists(os.path.join(cfg.output_dir, "results.csv")):
        if cfg.overwrite:
            print(f"Overwriting existing output dir: {cfg.output_dir}")
        else:
            print(f"Output dir already exists: {cfg.output_dir}. Skipping job.")
            return

    seed_everything(cfg.seed)

    # Scale learning rate
    assert (cfg.lr == -1) != (cfg.base_lr == -1), "either lr or base_lr should be set"
    if cfg.lr == -1:
        cfg.lr = cfg.base_lr * cfg.num_gpus

    # print & save config
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(cfg.output_dir, "config.yaml"))
    print(OmegaConf.to_yaml(cfg))

    # create model
    model_wrapper = create_model(cfg.model, cfg.dataset)

    # create datamodule
    cfg.dataset.image_resolution = cfg.model.image_resolution
    data_module = BenchmarkDataModule(
        dataset_config=cfg.dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_mem,
        seed=cfg.seed,
    )

    # assign engine
    engine = 'lightning'
    if training_mode in ['linear_probe','knn']:
        assert training_mode != 'knn', 'not impolemented yet'
        engine = 'accelerated'

    # execute training with correct engine
    if engine == 'lightning':

        if task in ['classification','regression']:
            model_wrapper.load_encoder(cfg.model.default_cls_blk_indices)
            pl_task = LightningClsRegTask(cfg, cfg.model, cfg.dataset, model_wrapper)
        elif task == 'segmentation':
            model_wrapper.load_encoder(cfg.model.segm_blk_indices)
            pl_task = LightningSegmentationTask(cfg, cfg.model, cfg.dataset, model_wrapper)
        else:
            raise NotImplementedError()

        # Setup logger
        if cfg.logger == "mlflow":
            logger = MLFlowLogger(
                experiment_name=experiment_name,
                run_name=run_name,
                tracking_uri=f"file:{os.path.join(os.environ['ODIR'], '_mlruns')}",)
        elif cfg.logger == 'wandb':
            raise NotImplementedError()


        # Callbacks
        model_monitor = "val_miou" if task == "segmentation" else "val_acc1"
        callbacks = [
            ModelCheckpoint(
                dirpath=os.path.join(cfg.output_dir, "checkpoints"),
                filename="best_model-{epoch}",
                monitor=model_monitor,
                mode="max",
                save_last=True,
            ),
            LearningRateMonitor(logging_interval="epoch"),
        ]

        # Initialize trainer

        if cfg.num_gpus == 0: # cpu
            trainer = Trainer(
                logger=logger,
                callbacks=callbacks,
                accelerator='cpu',
                max_epochs=cfg.epochs,
                num_sanity_val_steps=0,
                **cfg.trainer)

        elif cfg.num_gpus == 1: # single gpu
            trainer = Trainer(
                logger=logger,
                callbacks=callbacks,
                accelerator='gpu',
                devices=cfg.num_gpus,
                max_epochs=cfg.epochs,
                num_sanity_val_steps=0,
                **cfg.trainer)

        else: # ddp on multiple gpus
            trainer = Trainer(
                logger=logger,
                callbacks=callbacks,
                accelerator='gpu',
                strategy=DDPStrategy(find_unused_parameters=False),
                devices=cfg.num_gpus,
                max_epochs=cfg.epochs,
                num_sanity_val_steps=0,
                **cfg.trainer)


        # Train
        trainer.fit(pl_task, data_module, ckpt_path=cfg.resume if cfg.resume else None)

        if is_fastdevrun:
            print('No eval for fastdevrun.')
            return

        # Test
        best_checkpoint_path = callbacks[0].best_model_path
        results_list = trainer.test(pl_task, data_module, ckpt_path=best_checkpoint_path)
        results_dict = {k:v for l in results_list for k,v in l.items()}
        results = pd.DataFrame([results_dict])


    elif engine == 'accelerated':
        
        data_module.setup()
        setup_logger('eval', to_sysout=True, filename=os.path.join(cfg.output_dir, 'log.txt'))
        model_wrapper.load_encoder(cfg.model.accel_cls_blk_indices)

        dl_cfg = OmegaConf.create(dict(
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
        ))

        heads_cfg = OmegaConf.create(dict(
            n_last_blocks_list = [1,4],
            pooling = ['avgpool', 'cls', 'default'],
            learning_rates = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1, 0.2, 0.3, 0.5, 1,3,5,10],
        ))

        if task == 'classification':
            criterion_cfg = {'id': 'CrossEntropyLoss'}
            val_metrics = [{'id': 'MulticlassAccuracy'}]
        elif task == 'regression':
            criterion_cfg = {'id': 'MSELoss'}
            val_metrics = [{'id': 'RMSE'}]
        else:
            raise NotImplementedError()

        results_list = run_eval_linear(
            model_wrapper,
            cfg.output_dir,
            data_module.dataset_train,
            data_module.dataset_val,
            [data_module.dataset_test],
            cfg.dataset.num_classes,
            dl_cfg,
            heads_cfg,
            cfg.epochs,
            eval_period_epoch = 1,
            criterion_cfg = criterion_cfg,
            val_metrics = val_metrics,
        )

        results = pd.DataFrame(results_list)
        results = results[['metric_str','val','best_classifier']]
        results.rename(columns={'metric_str':'metric'}, inplace=True)
        results.reset_index(drop=True, inplace=True)
        print(f'Results: \n\n{results.to_string()}\n')

    else:
        raise ValueError(f'Unknown engine: {engine}')

    # save results
    results.to_csv(os.path.join(cfg.output_dir, "results.csv"), index=False)


if __name__ == "__main__":
    os.environ["MODEL_WEIGHTS_DIR"] = os.getenv("MODEL_WEIGHTS_DIR", "./fm_weights")
    main()
