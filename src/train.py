# import os
from http.client import ImproperConnectionState
from typing import List, Optional

import hydra
from omegaconf import DictConfig

# import torch
# from pytorch_lightning import (
#     Callback,
#     LightningDataModule,
#     LightningModule,
#     Trainer,
#     seed_everything,
# )
# from pytorch_lightning.loggers import LightningLoggerBase

import os
from typing import List

import torch
from pytorch_lightning import LightningModule,Trainer, seed_everything
from pytorch_lightning.loggers import CSVLogger

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")

from src.dqn_lightning_module import DQNLitModule


# def train(config: DictConfig) -> Optional[float]:
def train(config: DictConfig) -> Optional[float]:
    #avail_gpus = torch.cuda.device_count() # 使用するGPUを実行時に動的に得る場合
    
    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        seed_everything(config.seed, workers=True)
        
    
    # # Convert relative ckpt path to absolute path if necessary
    # ckpt_path = config.trainer.get("resume_from_checkpoint")
    # if ckpt_path and not os.path.isabs(ckpt_path):
    #     config.trainer.resume_from_checkpoint = os.path.join(
    #         hydra.utils.get_original_cwd(), ckpt_path
    #     )
    
    model: LightningModule = hydra.utils.instantiate(config.model) # デフォルトは, model = DQNLitModule()
    # model = DQNLitModule()


    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, logger=CSVLogger(save_dir = config.original_work_dir + "/logs/"),
    )
    

    # trainer = Trainer(
    #     accelerator = "auto",
    #     # devices=avail_gpus if torch.cuda.is_available() else None, # 使用するGPUを実行時に動的に得る場合
    #     devices = [0,1,2], #DP処理
    #     strategy = "ddp", #DDP処理したい場合は更に追加
    #     min_epochs = 1,
    #     max_epochs = 200,
    #     val_check_interval = 100,
    #     logger=CSVLogger(save_dir = "logs/"),
    # )

    trainer.fit(model)


