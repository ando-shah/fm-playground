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

    print("DATASET CONFIG")
    print(cfg.dataset)

    print("MODEL CONFIG")
    print(cfg.model)

    print("OTHER")
    print(cfg)

    # setup output dir
    if cfg.output_dir is None:
        experiment_name = f"{cfg.model.model_type}/{cfg.model.training_mode}/{cfg.dataset.dataset_name}"
        args_defining_run = {
            'lr': 'lr',
            'batch_size': 'bsz',
            'epochs': 'e',}
        assert all([not '.' in k for k in args_defining_run.keys()]), "cannot contain nested '.' yet"
        run_name = '_'.join([f"{v}={cfg[k]}" for k,v in args_defining_run.items()])

        suff = cfg.output_dir_suffix if cfg.output_dir_suffix is not None else ''
        cfg.output_dir = os.path.join(os.environ['ODIR'], experiment_name, suff, run_name)

    else:
        assert cfg.output_dir_suffix is None, "output_dir_suffix is only used when output_dir is not set"
        experiment_name = os.path.basename(os.path.normpath(cfg.output_dir))
        run_name = f"{experiment_name}_run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # check if task already executed
    if os.path.exists(os.path.join(cfg.output_dir, 'results.csv')):
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
    task = cfg.dataset.task

    # print & save config
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(cfg.output_dir, 'config.yaml'))
    print(OmegaConf.to_yaml(cfg))


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
        

    # Initialize data module
    cfg.dataset.image_resolution = cfg.model.image_resolution
    data_module = BenchmarkDataModule(
        dataset_config=cfg.dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_mem,
        seed = cfg.seed,
    )

    # Create model (assumed to be a LightningModule)
    model = create_model(cfg, cfg.model, cfg.dataset)
    print_trainable_parameters(model)

    # Train
    trainer.fit(model, data_module, ckpt_path=cfg.resume if cfg.resume else None)

    if is_fastdevrun:
        print('No eval for fastdevrun.')
        return

    # Test
    best_checkpoint_path = callbacks[0].best_model_path
    results_list = trainer.test(model, data_module, ckpt_path=best_checkpoint_path)
    results_dict = {k:v for l in results_list for k,v in l.items()}
    results = pd.DataFrame([results_dict])

    # save results
    results.to_csv(os.path.join(cfg.output_dir, 'results.csv'), index=False)

if __name__ == "__main__":
    main()
