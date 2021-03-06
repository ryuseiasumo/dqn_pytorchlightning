# import os
from http.client import ImproperConnectionState
from typing import List, Optional

import hydra
from omegaconf import DictConfig

import os
from typing import List

from pytorch_lightning import LightningModule,Trainer, seed_everything
from pytorch_lightning.loggers import CSVLogger
# from pytorch_lightning import (
#     Callback,
#     LightningDataModule,
#     LightningModule,
#     Trainer,
#     seed_everything,
# )
from pytorch_lightning.loggers import LightningLoggerBase

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")

from src.dqn_lightning_module import DQNLitModule
from src.models.q_net import QNet


def train(config: DictConfig) -> Optional[float]:
    #avail_gpus = torch.cuda.device_count() # 使用するGPUを実行時に動的に得る場合
    
    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        seed_everything(config.seed, workers=True)
        
    
    # Convert relative ckpt path to absolute path if necessary
    ckpt_path = config.trainer.get("resume_from_checkpoint")
    if ckpt_path and not os.path.isabs(ckpt_path):
        config.trainer.resume_from_checkpoint = os.path.join(
            hydra.utils.get_original_cwd(), ckpt_path
        )

    q_net: QNet = hydra.utils.instantiate(config.model)
    target_net: QNet = hydra.utils.instantiate(config.model)
    
    model: LightningModule = hydra.utils.instantiate(config.lightning_module, q_net = q_net, target_net = target_net)

    # Init lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                logger.append(hydra.utils.instantiate(lg_conf))


    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, logger=logger
    )
    
    trainer.fit(model)


