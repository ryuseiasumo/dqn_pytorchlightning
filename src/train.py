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
    # avail_gpus = min(1, torch.cuda.device_count())
    avail_gpus = torch.cuda.device_count()
    
    model = DQNLitModule()

    trainer = Trainer(
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        max_epochs=150,
        val_check_interval=50,
        logger=CSVLogger(save_dir="logs/"),
    )

    trainer.fit(model)


