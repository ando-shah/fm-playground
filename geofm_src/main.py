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
from geofm_src.engine.accelerated.utils.logger import setup_logger, plot_curves

from geofm_src.engine.accelerated.linear import run_eval_linear
from geofm_src.engine.lightning_task import LightningClsRegTask, LightningSegmentationTask
import logging
import json

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
    os.environ['CDIR'] = os.path.join(os.environ['REPO_PATH'], 'geofm_src/configs/')

    # assign engine
    if training_mode in ['linear_probe','knn']:
        engine = 'accelerated'
    else:
        engine = 'lightning'

    # engine specific input handling
    default_config_dir = os.path.join(os.environ['REPO_PATH'], 'geofm_src/configs/task_defaults/')
    if engine == 'accelerated':
        defaults = OmegaConf.load(os.path.join(default_config_dir, 'linear_probe_accel.yaml'))
        print(OmegaConf.to_yaml(defaults))
        cfg = OmegaConf.merge(defaults, cfg)
        print(OmegaConf.to_yaml(cfg))


        assert OmegaConf.is_list(cfg.lr), 'lr should be a list for accelerated engine'
        assert training_mode != 'knn', 'not impolemented yet'
        assert cfg.num_gpus == 1, 'accelerated only supports single gpu for now'

        args_defining_run = {
            "batch_size": "bsz",
            "epochs": "e",
        }

    elif engine == 'lightning':
        defaults = OmegaConf.load(os.path.join(default_config_dir, 'lightning.yaml'))
        cfg = OmegaConf.merge(defaults, cfg)

        assert all([k not in cfg for k in ['pooling','n_last_blocks_list']]), 'only for accelerated engine'

        args_defining_run = {
            "lr": "lr",
            "batch_size": "bsz",
            "epochs": "e",
        }

        # Scale learning rate
        assert (cfg.lr == -1) != (cfg.base_lr == -1), "either lr or base_lr should be set"
        if cfg.lr == -1:
            cfg.lr = cfg.base_lr * cfg.num_gpus

    # setup output dir
    experiment_name = os.path.relpath(cfg.output_dir, os.environ['ODIR'])
    if cfg.add_defining_args:
            
        assert all([not "." in k for k in args_defining_run.keys()]), (
            "cannot contain nested '.' yet"
        )
        run_name = "_".join([f"{v}={cfg[k]}" for k, v in args_defining_run.items()])

        cfg.output_dir = os.path.join(
            cfg.output_dir, run_name
        )

    else:
        run_name = (
            f"{experiment_name}_run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

    print(experiment_name)
    print(cfg.output_dir)

    # check if task already executed
    if os.path.exists(os.path.join(cfg.output_dir, "results.csv")):
        if cfg.overwrite:
            print(f"Overwriting existing output dir: {cfg.output_dir}")
        else:
            print(f"Output dir already exists: {cfg.output_dir}. Skipping job.")
            return

    seed_everything(cfg.seed)



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
        logger = logging.getLogger("eval")
        model_wrapper.load_encoder(cfg.model.accel_cls_blk_indices)

        dl_cfg = OmegaConf.create(dict(
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
        ))

        heads_cfg = OmegaConf.create(dict(
            n_last_blocks_list = cfg.n_last_blocks_list,
            pooling = cfg.pooling,
            learning_rates = cfg.lr,
        ))

        if task == 'classification':
            if cfg.dataset.multilabel:
                logger.info('Multilabel classification')
                criterion_cfg = {'id': 'MultiLabelSoftMarginLoss'}
                val_metrics = [{'id': 'MultilabelAccuracy'}]
            else:
                logger.info('Multiclass classification')
                criterion_cfg = {'id': 'CrossEntropyLoss'}
                val_metrics = [{'id': 'MulticlassAccuracy'}]
        elif task == 'regression':
            logger.info('Regression')
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
            eval_period_epoch = 10,
            criterion_cfg = criterion_cfg,
            val_metrics = val_metrics,
        )

        results = pd.DataFrame(results_list)
        results = results[['metric_str','val','best_classifier']]
        results.rename(columns={'metric_str':'metric'}, inplace=True)
        results.reset_index(drop=True, inplace=True)
        print(f'Results: \n\n{results.to_string()}\n')


        # logging

        # process loss file
        loss_file = os.path.join(cfg.output_dir, 'linear_probe_all_losses.csv')
        losses = pd.read_csv(loss_file).reset_index(drop=True)
        classifiers = losses.columns[1:]

        # process metrics file
        metrics_file = os.path.join(cfg.output_dir, 'linear_probe_all_metrics.json')
        metrics_by_cls = {}
        with open(metrics_file, 'r') as f:
            lines = f.readlines()
        for l in lines:
            d = json.loads(l)
            cls = d['classifier']
            if cls not in metrics_by_cls:
                metrics_by_cls[cls] = {}
            if 'test' in d['prefix']:
                key = d['prefix']
            else:
                key = d['iteration']
            metrics_by_cls[cls][key] = {k:v for k,v in d.items() if k not in ['prefix','iteration','classifier']}

        if cfg.logger == 'mlflow':
            logger.info('Logging to mlflow')
            import mlflow
            mlflow.set_tracking_uri(f"file:{os.path.join(os.environ['ODIR'], '_mlruns')}")
            mlflow.set_experiment(os.path.join(experiment_name, run_name))
            for cls in classifiers:
                with mlflow.start_run(run_name=cls):
                    # example: classifier_blocks_4_pooling_default_lr_2_50000
                    params = dict(
                        blocks = cls.split('_')[2],
                        pooling = cls.split('_')[4],
                        lr = float('.'.join(cls.split('_')[-2:])) ,)
                    mlflow.log_params(params)

                    for i in range(losses.shape[0]):
                        mlflow.log_metric(f'loss', losses.at[i,cls], step=losses.at[i, 'iteration'])

                    for i, metrics in metrics_by_cls[cls].items():
                        if isinstance(i, int):
                            for name, val in metrics.items():
                                mlflow.log_metric(f'val/{name}', val, step=int(i))
                        else:
                            for name, val in metrics.items():
                                mlflow.log_metric(f'{i}/{name}', val)
                
        else:
            raise NotImplementedError()
        
        plot_curves(cfg.output_dir) # plot average curve into .png file
    else:
        raise ValueError(f'Unknown engine: {engine}')

    # save results
    results.to_csv(os.path.join(cfg.output_dir, "results.csv"), index=False)


if __name__ == "__main__":
    os.environ["MODEL_WEIGHTS_DIR"] = os.getenv("MODEL_WEIGHTS_DIR", "./fm_weights")
    main()
