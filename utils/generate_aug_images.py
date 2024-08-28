import os
import glob
from utils.constants import (
    CUDA_VISIBLE_DEVICES,
)
os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES

import torch
import pandas as pd
import random
from tqdm import tqdm
import numpy as np
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from pl_modules.DukePreMRI import DukePreMRI
from pl_models.UNet_Segmentation import UNet_Segmentation
from pl_models.FeaturesModModel import FeaturesModModel

from utils.constants import (
    SEG_TRAIN_RATIO,
    SEG_VAL_RATIO,
    SEG_TEST_RATIO,
    FEATURE_MOD_CHECKPOINT_PATH,
    AUGMENTED_IMAGES_PATH,
    DATASETS_PATH,
    RANDOM_SEED,
    AUG_RATIO
)

def generate_aug_images(manufacturer_to_aug, reference_manufacturer, gs, steps, aug_ratio=AUG_RATIO):
    # consants
    BATCH_SIZE = 32*3

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    datamodule_to_augment = DukePreMRI(batch_size=BATCH_SIZE,
                            train_ratio = SEG_TRAIN_RATIO,
                            val_ratio = SEG_VAL_RATIO,
                            test_ratio = SEG_TEST_RATIO,
                            from_manufacturer=manufacturer_to_aug,
                            with_ID=True,
                            with_segmentation=True,
                            only_with_segmentation=True,
                            only_without_segmentation=False
    )
    
    datamodule_to_augment.prepare_data()
    
    datamodule_reference = DukePreMRI(batch_size=BATCH_SIZE,
                        train_ratio = 1,
                        val_ratio = 0,
                        test_ratio = 0,
                        shuffle_train=False,
                        from_manufacturer=reference_manufacturer,
                        with_segmentation=True,
                        only_with_segmentation=True,
                        only_without_segmentation=False,
    )   
    datamodule_reference.prepare_data()

    
    model = FeaturesModModel.load_from_checkpoint(FEATURE_MOD_CHECKPOINT_PATH)
    model.eval()
    
    datamodule_to_augment.setup(stage='fit')
    
    df = pd.read_csv(os.path.join(DATASETS_PATH,'dataset.csv')).fillna('')
    
    # Remove previous generated files
    df['aug_path'] = ''
    files = glob.glob(os.path.join(AUGMENTED_IMAGES_PATH,'*.pt'))
    for f in files:
        os.remove(f)
        
    while aug_ratio > 0:

        with torch.no_grad():
            for _, batch in enumerate(tqdm(datamodule_to_augment.train_dataloader())):
                
                if np.random.rand() < aug_ratio:
                
                    images,features = batch["image"].to(model.device),batch["features"].to(model.device)
                    patient_IDs = batch["Patient ID"]
                    slices = batch["slice"]
                    
                    new_features = torch.stack([torch.tensor(file['features'],device=model.device, dtype=torch.float32) for file in random.sample(datamodule_reference.train_files, images.shape[0])])
                    
                    _,edited_imgs,_ = model.edit(images,
                                                new_features,
                                                # original_features=features,
                                                guidance_scale=gs,
                                                start_t=steps
                                                )

                    for i in range(edited_imgs.shape[0]):
                        aug_path = os.path.join(AUGMENTED_IMAGES_PATH,patient_IDs[i]+'-'+f'{slices[i]}'+'.pt')
                        torch.save(edited_imgs[i].detach().cpu().clone(), aug_path)
                        
                        if df.loc[(df['Patient ID'] == patient_IDs[i]) & (df['slice']==slices[i].item()), ['aug_path']].values.item() == '':
                            df.loc[(df['Patient ID'] == patient_IDs[i]) & (df['slice']==slices[i].item()), ['aug_path']] = aug_path
                        else:
                            df.loc[(df['Patient ID'] == patient_IDs[i]) & (df['slice']==slices[i].item()), ['aug_path']] += ',' + aug_path
                        
        aug_ratio = aug_ratio - 1
    
    df.to_csv(os.path.join(DATASETS_PATH, "dataset.csv"), index=False)