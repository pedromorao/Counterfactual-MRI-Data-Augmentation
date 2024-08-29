import os
from utils.constants import (
    WANDB_API_KEY,
    CUDA_VISIBLE_DEVICES,
)
os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
os.environ['WANDB_API_KEY'] = WANDB_API_KEY

from utils.generate_aug_images import generate_aug_images

import torch
import wandb
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping

from pl_models.UNet_Segmentation import UNet_Segmentation
from pl_modules.DukePreMRI import DukePreMRI
from monai.networks.layers.factories import Norm

from utils.constants import (
    SPATIAL_SIZE,
    SEG_TRAIN_RATIO,
    SEG_VAL_RATIO,
    SEG_TEST_RATIO,
    AUG_RATIO
)

# consants
BATCH_SIZE = 256
WANDB_PROJECT_NAME = 'segmentation-v3'

os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
os.environ['WANDB_API_KEY'] = WANDB_API_KEY

MANUFACTURER_TO_AUG = ['GE MEDICAL SYSTEMS']
REFERENCE_MANUFACTURER = ['SIEMENS']
GS = [3,5,7]
STEPS = [25,50,75]
aug_ratio = 1

test_params = [(steps, gs) for steps in STEPS for gs in GS]

for (steps, gs) in test_params:

    for manufacturer_to_aug, reference_manufacturer in zip(MANUFACTURER_TO_AUG, REFERENCE_MANUFACTURER):

        generate_aug_images(manufacturer_to_aug=manufacturer_to_aug, reference_manufacturer=reference_manufacturer,
                            gs=gs,steps=steps,
                            aug_ratio=aug_ratio)

        datamodule = DukePreMRI(batch_size=BATCH_SIZE,
                                train_ratio = SEG_TRAIN_RATIO,
                                val_ratio = SEG_VAL_RATIO,
                                test_ratio = SEG_TEST_RATIO,
                                from_manufacturer=manufacturer_to_aug,
                                with_segmentation=True,
                                only_with_segmentation=True,
                                only_without_segmentation=False,
                                use_aug=True,
        )

        datamodule.prepare_data()

        # confirm no augmented images in val or test sets
        for file in datamodule.train_files:
            if 'aug_images' in file['image']:
                print('Augmented images in train set')
                break
        for file in datamodule.val_files:
            if 'aug_images' in file['image']:
                print('Augmented images in val set')
                break
                
        for file in datamodule.test_files:
            if 'aug_images' in file['image']:
                print('Augmented images in test set')
                break

        model = UNet_Segmentation(model_hparams={'batch_size':datamodule.batch_size,
                                                'spatial_size':SPATIAL_SIZE,
                                                'loss':'dice',
                                                'include_background':True,
                                                'weights':None, #datamodule.calc_weigths()
                                                'aug_ratio':AUG_RATIO if datamodule.use_aug==True else None},
                                unet_hprams={'spatial_dims':2,
                                            'in_channels':1,
                                            'out_channels':3,
                                                'channels':(32, 64, 128, 256, 512, 512),
                                                'strides':(2, 2, 2, 2, 2),
                                                'num_res_units':2,
                                                'dropout':0,
                                                'norm':Norm.BATCH},
                                optimizer_name='Adam',
                                optimizer_hparams={'lr':2e-3,
                                                    'weight_decay':1e-3}
        )

        # initialise the wandb logger and name your wandb project
        wandb_logger = WandbLogger(project=WANDB_PROJECT_NAME,
                                name=manufacturer_to_aug+' trained in '+manufacturer_to_aug+f'-steps{steps},gs-{gs},aug_ratio-{aug_ratio}')
                                
        checkpoint_callback = EarlyStopping(monitor="val/loss", patience=25, verbose=False, mode="min")

        # train the model
        trainer = Trainer(max_epochs=600,
                        callbacks=[checkpoint_callback],
                        logger=wandb_logger,
                        log_every_n_steps=25,
                        check_val_every_n_epoch=1
        )

        trainer.fit(model=model, datamodule=datamodule)

        trainer.test(model=model, datamodule=datamodule)

        wandb.finish()

        # Test in SIEMENS SCANNERS

        # consants
        BATCH_SIZE = 256

        datamodule = DukePreMRI(batch_size=BATCH_SIZE,
                                train_ratio = 1,
                                val_ratio = 0,
                                test_ratio = 0,
                                shuffle_train=False,
                                from_manufacturer=reference_manufacturer,
                                with_segmentation=True,
                                only_with_segmentation=True,
                                only_without_segmentation=False,
        )


        datamodule.prepare_data()

        model.eval()


        # initialise the wandb logger and name your wandb project
        wandb_logger = WandbLogger(project=WANDB_PROJECT_NAME,
                                name=reference_manufacturer+' trained in '+manufacturer_to_aug+f'-steps{steps}-gs{gs}-aug_ratio-{aug_ratio}')

        # train the model
        trainer = Trainer(logger=wandb_logger)

        datamodule.setup(stage='fit')
        trainer.test(model=model, dataloaders=datamodule.train_dataloader())

        wandb.finish()
