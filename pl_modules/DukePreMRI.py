import numpy as np
import random
import lightning as L
import pandas as pd
import os
from monai.transforms import AsDiscrete
from monai.data import DataLoader, CacheDataset, Dataset
from monai.data.utils import pad_list_data_collate

from utils.dataset import *
from utils.constants import (
    DATASETS_PATH,
    FEATURES,
    CATEGORICAL_FEATURES,
    CONTINUOUS_FEATURES,
    RANDOM_SEED
    )
from utils.transforms import (
    DEFAULT_TRANSFORMS,
    DEFAULT_TRANSFORMS_WITH_SEGMENTATION,
)

class DukePreMRI(L.LightningDataModule):
    def __init__(self,
                 batch_size = 16,
                 test_batch_size = None,
                 train_ratio = 0.7,
                 val_ratio = 0.15,
                 test_ratio = 0.15,
                 with_segmentation = False,
                 only_with_segmentation = False,
                 only_without_segmentation = True,
                 features = FEATURES,
                 transforms = {'train': None},
                 from_manufacturer = None,
                 shuffle_train=True,
                 use_aug = False,
                 with_ID=False,
                 only_use_aug=False,
                 subset=1):
        super().__init__()
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.with_segmentation = with_segmentation
        self.only_with_segmentation = only_with_segmentation
        self.only_without_segmentation = only_without_segmentation
        self.features_names = features
        self.subset = subset
        self.test_batch_size = batch_size if test_batch_size is None else test_batch_size
        self.shuffle_train = shuffle_train
        self.with_ID = with_ID
        self.use_aug = True if only_use_aug==True else use_aug
        self.only_use_aug = only_use_aug
        
        assert not (self.only_with_segmentation and self.only_without_segmentation)
        
        if type(transforms) is dict:
            if transforms['train'] is None:
                self.transforms =  {'train':DEFAULT_TRANSFORMS if self.with_segmentation==False else DEFAULT_TRANSFORMS_WITH_SEGMENTATION}

        else:
            self.transforms = {'train':transforms}
            
        if 'val' not in self.transforms.keys():
            self.transforms['val'] = self.transforms['train']
            
        if 'test' not in self.transforms.keys():
            self.transforms['test'] = self.transforms['val']
        
        self.df = pd.read_csv(os.path.join(DATASETS_PATH,'dataset.csv')).fillna("")

        # Encode dataset
        self.features_labels = {}
        self.features_dims = {}
        self.features_scale = {}

        for feature in self.features_names:
            if feature in CATEGORICAL_FEATURES:
                
                encoding = {name: value for (value, name) in enumerate(self.df[feature].unique())}
                self.features_labels[feature] = {value: name for (value, name) in enumerate(self.df[feature].unique())}
                self.df[feature] = self.df[feature].apply(lambda x: encoding[x])
                self.features_dims[feature] = len(self.df[feature].unique())

            elif feature in CONTINUOUS_FEATURES:

                max_ = self.df[feature].max()
                self.df[feature] = self.df[feature].apply(lambda x: float(x)/max_)
                self.features_scale[feature] = max_
                self.features_dims[feature] = 1
        
        if self.only_with_segmentation==True:
            self.df = self.df[self.df['seg_path']!=""]
            
        elif self.only_without_segmentation == True:
            self.df = self.df[self.df['seg_path']==""]
            
        if from_manufacturer == 'SIEMENS':
            self.df = self.df[self.df['Manufacturer']=='SIEMENS']
        elif from_manufacturer == 'GE MEDICAL SYSTEMS':
            self.df = self.df[self.df['Manufacturer']=='GE MEDICAL SYSTEMS']
        
    def prepare_data(self):
        
        catg_features = [feature for feature in self.features_names if feature in CATEGORICAL_FEATURES]
        
        # Split the dataset into train, val and test subsets
        ids_with_seg = self.df[self.df['seg_path']!=""]['Patient ID'].unique()[:,None]
        ids_without_seg = self.df[self.df['seg_path']==""]['Patient ID'].unique()[:,None]
       
        if self.only_with_segmentation==True or len(ids_without_seg)==0:
            ids_train_without_seg, ids_val_without_seg, ids_test_without_seg = set(),set(),set()
        else:
            # Set Reproducibility
            np.random.seed(RANDOM_SEED)
            random.seed(RANDOM_SEED)
            ids_train_without_seg, ids_val_without_seg, ids_test_without_seg = train_val_test_split(self.df,
                                                                                                    ids_without_seg,
                                                                                                    catg_features,
                                                                                                    train_ratio=self.train_ratio,
                                                                                                    val_ratio=self.val_ratio,
                                                                                                    test_ratio=self.test_ratio)

        if self.only_without_segmentation==True or len(ids_with_seg)==0:
            ids_train_with_seg, ids_val_with_seg, ids_test_with_seg = set(),set(),set()
        else:
            # Set Reproducibility
            np.random.seed(RANDOM_SEED)
            random.seed(RANDOM_SEED)
            ids_train_with_seg, ids_val_with_seg, ids_test_with_seg = train_val_test_split(self.df,
                                                                                            ids_with_seg,
                                                                                            catg_features,
                                                                                            train_ratio=self.train_ratio,
                                                                                            val_ratio=self.val_ratio,
                                                                                            test_ratio=self.test_ratio)


        ids_train, ids_val, ids_test = (ids_train_with_seg.union(ids_train_without_seg),
                                        ids_val_with_seg.union(ids_val_without_seg),
                                        ids_test_with_seg.union(ids_test_without_seg)
        )

        self.catg_classes_weights = {label: get_weights(self.df[self.df['Patient ID'].isin(ids_train)][label].value_counts().sort_index()) for label in catg_features}

        for feature,weights in self.catg_classes_weights.items():
            if self.features_dims[feature] > len(weights):
                print(f"Feature: \"{feature}\" has cases missing in the train set")
        
        self.train_files = []
        for  _, row in self.df[self.df['Patient ID'].isin(ids_train)].iterrows():
            
            file = {'image': row['file_path'],
                    'features': features_to_vec(row, self.features_dims)}

            if self.with_ID==True:
                file.update({'Patient ID':  row['Patient ID'],
                            'slice': row['slice']})
                
            if self.with_segmentation==True:
                file.update({'segmentation':  row['seg_path']})
            
            if self.only_use_aug==False:
                self.train_files.append(file.copy())
            
            if  self.use_aug==True and row['aug_path']!='':
                for path in row['aug_path'].split(','):
                    file['image'] = path
                    self.train_files.append(file.copy())
                
        self.val_files = []
        for  _, row in self.df[self.df['Patient ID'].isin(ids_val)].iterrows():
            
            file = {'image': row['file_path'],
                    'features': features_to_vec(row, self.features_dims)}

            if self.with_ID==True:
                file.update({'Patient ID':  row['Patient ID'],
                            'slice': row['slice']})
                
            if self.with_segmentation==True:
                file.update({'segmentation':  row['seg_path']})
                
            self.val_files.append(file)
            
        self.test_files = []
        for  _, row in self.df[self.df['Patient ID'].isin(ids_test)].iterrows():
            
            file = {'image': row['file_path'],
                    'features': features_to_vec(row, self.features_dims)}

            if self.with_ID==True:
                file.update({'Patient ID':  row['Patient ID'],
                            'slice': row['slice']})
                
            if self.with_segmentation==True:
                file.update({'segmentation':  row['seg_path']})

            self.test_files.append(file)
            
        # Set Reproducibility
        np.random.seed(RANDOM_SEED)
        random.seed(RANDOM_SEED)
        # for testing with a smaller dataset
        if self.subset < 1:            
            self.train_files = random.sample(self.train_files, int(len(self.train_files) * self.subset) + 1)
            if len(self.val_files)> 0:
                self.val_files = random.sample(self.val_files, int(len(self.val_files) * self.subset) + 1)
            if len(self.test_files)> 0:
                self.test_files = random.sample(self.test_files, int(len(self.test_files) * self.subset) + 1)
        elif self.subset==1:
            random.shuffle(self.val_files)
            random.shuffle(self.test_files)
            random.shuffle(self.train_files)
        
    def setup(self, stage: str):
        if stage == "fit":
            self.train_ds = CacheDataset(data=self.train_files, transform=self.transforms['train'], cache_rate=1.0, num_workers=8)
            with torch.no_grad():
                self.val_ds = CacheDataset(data=self.val_files, transform=self.transforms['val'], cache_rate=1.0, num_workers=8)

        if stage == "test":
            with torch.no_grad():
                self.test_ds = CacheDataset(data=self.test_files, transform=self.transforms['test'], cache_rate=1.0, num_workers=8)
    
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=self.shuffle_train)

    @torch.no_grad
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False)

    @torch.no_grad
    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.test_batch_size, shuffle=False)
    
    @torch.no_grad
    def calc_weigths(self):
        train_data = Dataset(data=self.train_files, transform=self.transforms['train'])
        
        weights = torch.zeros(3)

        for _, data in enumerate(train_data):
            
            label = AsDiscrete(to_onehot=3)(data["segmentation"])
            
            pixel_number = label.shape[1]

            weights += ((pixel_number + 3) / (label.sum(dim=(1, 2)) + 1)**(1/2)) / len(train_data)        

        return weights