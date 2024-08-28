import torch
import numpy as np
from skmultilearn.model_selection import iterative_train_test_split

def get_weights(value_counts):
    value_counts = value_counts.sort_index().values
    
    weights = 1 / value_counts
    weights = len(weights) * weights / weights.sum()

    return weights

def one_hot(a, num_classes):
  return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

def features_to_vec(features_row, features_dims):
    features_vec = []
    
    if  torch.is_tensor(features_row):
        i=0
        for j,feature in enumerate(sorted(features_dims.keys())):
            if features_dims[feature] == 1:
                features_vec.append(features_row[:,j][:,None])
            else:
                features_vec.append(torch.nn.functional.one_hot(torch.tensor(features_row[:,j], dtype=torch.int64), features_dims[feature]))
            i+=features_dims[feature]
        return torch.concat(features_vec,dim=1)
    
    for feature in sorted(features_dims.keys()):
        if features_dims[feature] == 1:
            features_vec.append(np.array([features_row[feature]]))
        else:
            features_vec.append(one_hot(np.array(features_row[feature]), features_dims[feature]))
            
    return np.concatenate(features_vec)

def vec_to_features(features_vec, features_dims):
    features = []

    if  torch.is_tensor(features_vec):
        i=0
        for feature in sorted(features_dims.keys()):
            if features_dims[feature] == 1:
                features.append(features_vec[:,i])
            else:
                features.append(features_vec[:,i:i+features_dims[feature]].argmax(dim=-1))
            i+=features_dims[feature]
        
        return torch.stack(features,dim=1)
        
    i=0
    for feature in sorted(features_dims.keys()):
        if features_dims[feature] == 1:
            features.append(features_vec[i])
        else:
            features.append(features_vec[i:i+features_dims[feature]].argmax(axis=-1))
        i+=features_dims[feature]
    
    return features

def get_feature(features_vec,feature,features_dims,features_scale,as_indices=False):
    # if as_indices==True of 'both' the continuous features will be unnormalized to the original values
    i=0
    for feature_name in sorted(features_dims.keys()):
        if feature_name==feature:
            break
        i+=features_dims[feature_name]
    
    wanted_feature_vec = features_vec[:,i:i+features_dims[feature_name]].squeeze(-1)
      
    if as_indices==True:
        if features_dims[feature_name] == 1:
            return wanted_feature_vec*features_scale[feature_name]
        else:
            return wanted_feature_vec.argmax(dim=-1)
    elif as_indices == 'both':
        if features_dims[feature_name] == 1:
            return wanted_feature_vec,wanted_feature_vec*features_scale[feature_name]
        else:
            return wanted_feature_vec,wanted_feature_vec.argmax(dim=-1)
        
    return wanted_feature_vec

def  train_val_test_split(df, ids, catg_features, train_ratio, val_ratio, test_ratio):
    "Split train, val and test sets ids are an array of shape Nx1, if test ratio == 0 test set = val set"
    
    if val_ratio==0:
        val_ratio = test_ratio
        test_ratio = 0
    
    if train_ratio == 1:
        return  set(ids.squeeze(-1)), set(), set()

    df = df[df['Patient ID'].isin(set(ids.squeeze()))]
    
    stratify_cols = df.groupby('Patient ID')[catg_features].min().values
    
    ids_train, _, ids_test, stratify_cols = iterative_train_test_split(X=ids,
                                                                    y=stratify_cols,
                                                                    test_size = 1 - train_ratio)
    if test_ratio == 0:
        
        ids_val = ids_test
        
    else:
        
        ids_val, _, ids_test, _ = iterative_train_test_split(X=ids_test,
                                                            y=stratify_cols,
                                                            test_size = test_ratio/(test_ratio + val_ratio)) 
    
    ids_train, ids_val, ids_test = set(ids_train.squeeze(-1)), set(ids_val.squeeze(-1)), set(ids_test.squeeze(-1))
    
    # If the train set has equal or less than 2 cases force them to the train set
    value_counts = {label: df[['Patient ID']+[label]].groupby('Patient ID').min().value_counts() for label in catg_features}

    forced_ids_to_trainset = set()
    df_by_PatienID = df[['Patient ID']+catg_features].groupby('Patient ID').min()
    for _, row in df_by_PatienID.iterrows():
        for label in value_counts.keys():
            if value_counts[label].loc[row[label]].values[0]<=2:
                forced_ids_to_trainset.add(row.name)

    for PatientID in forced_ids_to_trainset:
        ids_train.add(PatientID)
        
        if PatientID in ids_val:
            ids_val.remove(PatientID)

        if PatientID in ids_test:
            ids_test.remove(PatientID)
    
    return ids_train, ids_val, ids_test
