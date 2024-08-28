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
    TEST_RATIO
)

# consants
BATCH_SIZE = 32
WANDB_PROJECT_NAME = 'feature_mod'

datamodule = DukePreMRI(batch_size=BATCH_SIZE,
                        test_batch_size=BATCH_SIZE*4,
                        train_ratio=TRAIN_RATIO,
                        val_ratio=VAL_RATIO,
                        test_ratio=TEST_RATIO,
                        only_without_segmentation=True
)

datamodule.prepare_data()

model = FeaturesModModel(model_hparams={
                            'spatial_size':SPATIAL_SIZE,
                            'batch_size':datamodule.batch_size,
                            'features_dim':sum(datamodule.features_dims.values()),
                            'conditioning':'hybrid',
                            'features_dropout':0.15,
                            'with_segmentation': datamodule.with_segmentation,
                            'segmentation_dropout':0.0
                        },
                        unet_hprams={
                            'spatial_dims':2,
                            'out_channels': 1,
                            'num_res_blocks': 2,
                            'num_channels': (64, 64, 128, 128, 256, 256),
                            'attention_levels': (False, False, True, False, True, False),
                            'num_head_channels': 8,
                            'transformer_num_layers': 1,
                            'use_flash_attention': True
                        },
                        noise_scheduler_hprams={
                            'schedule':'cosine',
                            'num_train_timesteps':1000,
                            'prediction_type':'epsilon'
                        },
                        optimizer_name='Adam',
                        optimizer_hparams={
                            'lr':1e-4,
                            'weight_decay':1e-3
                        },
                        test_params={
                            'guidance_scale':5,
                            'steps':25
                        }
)


from utils.constants import (
    FEATURE_MOD_CHECKPOINT_PATH
)
model = FeaturesModModel.load_from_checkpoint(FEATURE_MOD_CHECKPOINT_PATH)

# initialise the wandb logger and name your wandb project
wandb_logger = WandbLogger(project=WANDB_PROJECT_NAME)

checkpoint_callback = ModelCheckpoint(monitor="val/loss", mode="min", save_last=True, save_top_k=1)

# train the model
trainer = Trainer(max_epochs=15,
                  callbacks=[checkpoint_callback],
                  logger=wandb_logger,
                  precision='16-mixed',
)

trainer.fit(model=model, datamodule=datamodule)

trainer.test(model=model, datamodule=datamodule)

wandb.finish()