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
from pl_models.FeaturesPredModel import FeaturesPredModel

from utils.constants import (
    SPATIAL_SIZE,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
)

# consants
BATCH_SIZE = 512
WANDB_PROJECT_NAME = 'feature_pred'

os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
os.environ['WANDB_API_KEY'] = WANDB_API_KEY

datamodule = DukePreMRI(batch_size=BATCH_SIZE,
                        train_ratio = TRAIN_RATIO,
                        val_ratio = VAL_RATIO,
                        test_ratio = TEST_RATIO,
                        only_without_segmentation=True
)

datamodule.prepare_data()

model = FeaturesPredModel(model_hparams={'batch_size':datamodule.batch_size,
                                        'spatial_size':SPATIAL_SIZE},
                          optimizer_name='Adam',
                          optimizer_hparams={'lr':1e-5,
                                            'weight_decay':1e-4},
                          features_dims=datamodule.features_dims,
                          features_labels=datamodule.features_labels,
                          features_scale=datamodule.features_scale,
                          catg_features_weights=datamodule.catg_classes_weights
)

# initialise the wandb logger and name your wandb project
wandb_logger = WandbLogger(project=WANDB_PROJECT_NAME)

checkpoint_callback = ModelCheckpoint(monitor="val/loss", mode="min", save_last=True, save_top_k=1)

# train the model
trainer = Trainer(max_epochs=200,
                  callbacks=[checkpoint_callback],
                  logger=wandb_logger
)

trainer.fit(model=model, datamodule=datamodule)

trainer.test(model=model, datamodule=datamodule)

wandb.finish()