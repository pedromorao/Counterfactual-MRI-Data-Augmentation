import os
from utils.constants import (
    WANDB_API_KEY,
    CUDA_VISIBLE_DEVICES,
)
os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
os.environ['WANDB_API_KEY'] = WANDB_API_KEY

import wandb
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from pl_modules.DukePreMRI import DukePreMRI
from pl_models.FeaturesModModel import FeaturesModModel

from utils.constants import (
    SPATIAL_SIZE,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    FEATURE_MOD_CHECKPOINT_PATH
)

# consants
BATCH_SIZE = 32
WANDB_PROJECT_NAME = 'feature_mod'

datamodule = DukePreMRI(batch_size=BATCH_SIZE,
                        test_batch_size=BATCH_SIZE*4,
                        train_ratio=TRAIN_RATIO,
                        val_ratio=VAL_RATIO,
                        test_ratio=TEST_RATIO
)

datamodule.prepare_data()


test_params = [(steps, gs) for steps in [50,75] for gs in [3,5,7]]

for steps, gs in test_params:

    model = FeaturesModModel.load_from_checkpoint(FEATURE_MOD_CHECKPOINT_PATH)

    model.eval()
    
    model.hparams.test_params = {'guidance_scale': gs, 'steps': steps}

    # initialise the wandb logger and name your wandb project
    wandb_logger = WandbLogger(project=WANDB_PROJECT_NAME,name=f'steps-{steps},gs-{gs};-3')

    # train the model
    trainer = Trainer(logger=wandb_logger,
                        precision='16-mixed',
    )

    trainer.test(model=model, datamodule=datamodule, ckpt_path=FEATURE_MOD_CHECKPOINT_PATH)
    
    wandb_logger.experiment.config.update({'test_params': {'guidance_scale': gs, 'steps': steps}}, allow_val_change=True)

    wandb.finish()