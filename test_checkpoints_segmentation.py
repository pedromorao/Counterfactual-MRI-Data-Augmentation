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
from pl_models.UNet_Segmentation import UNet_Segmentation

from utils.constants import (
    SPATIAL_SIZE,
    SEG_TRAIN_RATIO,
    SEG_VAL_RATIO,
    SEG_TEST_RATIO,
    SEGMENTATION_CHECKPOINT_PATH
)

# consants
BATCH_SIZE = 256
WANDB_PROJECT_NAME = 'segmentation'

# datamodule = DukePreMRI(batch_size=BATCH_SIZE,
#                         train_ratio = 1,
#                         val_ratio = 0,
#                         test_ratio = 0,
#                         shuffle_train=False,
#                         from_manufacturer='SIEMENS',
#                         with_segmentation=True,
#                         only_with_segmentation=True,
#                         only_without_segmentation=False,
# )

datamodule = DukePreMRI(batch_size=BATCH_SIZE,
                        train_ratio = SEG_TRAIN_RATIO,
                        val_ratio = SEG_VAL_RATIO,
                        test_ratio = SEG_TEST_RATIO,
                        with_segmentation=True,
                        only_with_segmentation=True,
                        only_without_segmentation=False,
                        from_manufacturer='GE MEDICAL SYSTEMS',
)

datamodule.prepare_data()

model = UNet_Segmentation.load_from_checkpoint(SEGMENTATION_CHECKPOINT_PATH)

model.eval()

# initialise the wandb logger and name your wandb project
wandb_logger = WandbLogger(project=WANDB_PROJECT_NAME)

# train the model
trainer = Trainer(logger=wandb_logger)

trainer.test(model=model, datamodule=datamodule, ckpt_path=SEGMENTATION_CHECKPOINT_PATH)
# datamodule.setup(stage='fit')
# trainer.test(model=model, dataloaders=datamodule.train_dataloader(), ckpt_path=SEGMENTATION_CHECKPOINT_PATH)

wandb.finish()