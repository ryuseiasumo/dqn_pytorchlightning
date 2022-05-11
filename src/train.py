# import os
from typing import List, Optional

# import hydra
# from omegaconf import DictConfig

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
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")

from src.dqn_lightning_module import DQNLitModule


# def train(config: DictConfig) -> Optional[float]:
def train() -> Optional[float]:
    avail_gpus = torch.cuda.device_count()
    
    model = DQNLitModule()

    trainer = Trainer(
        accelerator = "auto",
        # devices=avail_gpus if torch.cuda.is_available() else None,  # limiting got iPython runs
        devices = [0,1,2], #DP処理
        # strategy = "ddp", #DDP処理したい場合は更に追加
        max_epochs = 150,
        val_check_interval = 50,
        logger=CSVLogger(save_dir = "logs/"),
    )

    trainer.fit(model)


