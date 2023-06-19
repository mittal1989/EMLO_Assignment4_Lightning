from typing import Tuple, Dict

import lightning as L
import torch
import hydra
from omegaconf import DictConfig
import timm
from copper import utils
import os

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def eval(cfg: DictConfig) -> Tuple[dict, dict]:
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "trainer": trainer,
    }

    # if cfg.get("train"):
    #     log.info("Starting training!")
    #     trainer.fit(model=model, datamodule=datamodule,
    #                 ckpt_path=cfg.get("ckpt_path"))

    # train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting evalution!")
        ckpt_path = ""
        if cfg.get("ckpt_path"):
        # if ckpt_path == "":
        #     log.warning(
        #         "Best ckpt not found! Using current weights for testing...")
            latest_checkpoint_path = None
            for root, dirs, files in os.walk("./outputs/"):
                for file in files:
                    if file.endswith(".ckpt"):
                        if latest_checkpoint_path is None or file > latest_checkpoint_path:
                            latest_checkpoint_path = os.path.join(root, file)
            
            print("latest_checkpoint_path",latest_checkpoint_path)

            trainer.test(model=model, datamodule=datamodule, ckpt_path=latest_checkpoint_path)
            log.info(f"Best ckpt path: {ckpt_path}")

    eval_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**eval_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig):
    # train the model
    metric_dict, _ = eval(cfg)

    # this will be used by hydra later for optimization
    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
