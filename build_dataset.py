import pandas as pd
import os
import glob
import torch
import pydicom as dicom

from monai.transforms import(
    Compose,
    LoadImage,
    EnsureChannelFirst,
    Resize,
    Orientation
)
from utils.constants import SPATIAL_SIZE,SEGMENTATIONS_FILES,SEGMENTATIONS_PATH,SLICES_BOUND

from utils.constants import (
    MANIFEST_FILE,
    DATASETS_PATH,
    FEATURES_NAMES,
    SLICES_BOUND,
    SEGMENTATIONS_FILES,
    SEGMENTATIONS_PATH
)

# Read datasets
filename_mapping = pd.read_csv(os.path.join(DATASETS_PATH,'Breast-Cancer-MRI-filepath_filename-mapping.csv'),low_memory=False)
filename_mapping.set_index('descriptive_path', inplace=True)

boxes_df = pd.read_csv(os.path.join(DATASETS_PATH,'Annotation_Boxes.csv'))
boxes_df.set_index('Patient ID', inplace=True)

clinical_features = pd.read_csv(os.path.join(DATASETS_PATH, "Clinical_and_Other_Features.csv"), skiprows=1)
clinical_features = clinical_features.rename(columns={feature: feature.strip() for feature in clinical_features.columns})
clinical_features.set_index('Patient ID', inplace=True)

read_segmentation = Compose([
            LoadImage(dtype=torch.float32),
            EnsureChannelFirst(),
            Resize(spatial_size=(SPATIAL_SIZE[0],SPATIAL_SIZE[1],-1), mode='nearest-exact')
])

def fix_path(path):
    # fix discriptive path error
    path = os.path.normpath(path)
    path_components = path.split(os.sep)[-5:]
    
    path_components[-4] = path_components[-4].replace("_", "")
    path_components[-3] = (path_components[-3]
                           .replace("W  WO", "W + W/O")
                           .replace("WWO","W/WO")
                           .replace("W/WO CONTRAST W","W WO CONTRAST W" )
                           .replace("W AND WO","W AND W/O")
                           .replace("W-O","W-O")
                           .replace("-NA","")
                           .replace("BREASTROUTINE","BREAST^ROUTINE")
                           .replace("BREASTGENERAL","BREAST^GENERAL")
                           .replace("BREASTGENERAL","BREAST^GENERAL")                           
                           .replace("BREASTlesion","BREAST^lesion")
                            .replace("breastlesion","breast^lesion")
                           .replace("-e1","-e+1")
                        )
    path_components[-2] = (path_components[-2]
                           .replace("Ph1ax", "Ph1/ax")
                           .replace("Ph1Ax", "Ph1/Ax")
                           .replace("Ph2ax", "Ph2/ax")
                           .replace("Ph2Ax", "Ph2/Ax")
                           .replace("Ph3ax", "Ph3/ax")
                           .replace("Ph3Ax", "Ph3/Ax")
                           .replace("Ph4ax", "Ph4/ax")
                           .replace("Ph4Ax", "Ph4/Ax")
                           .replace(" c-"," +c-")
                        )
    slice_number = path_components[-1].split('.')[0]
    path_components[-1] = '1-'+slice_number.split('-')[-1].zfill(3)+'.dcm'

    return '/'.join(path_components)
    

def get_original_path_and_filename(path):
    path = fix_path(path)
    
    if path.__contains__('Segmentation'):
        return 'Segmentation'
    
    fixes=[("W WO CONTRAST W", "WWO CONTRAST W"),
            ("W + W/O", "W/ & WO"),
            ("W/ & WO", "W & WO"),
            ("W & WO", "W & W/O")]
    
    if path not in filename_mapping.index:
        path_components = path.split('/')
        path_components[-1] = '/'+path_components[-1]
        path = '/'.join(path_components)

    if path not in filename_mapping.index:
        
        for fix in fixes:
            path_components = path.split('//')
            path = '/'.join(path_components)
            
            path = path.replace(fix[0],fix[1])
            
            if path not in filename_mapping.index:
                path_components = path.split('/')
                path_components[-1] = '/'+path_components[-1]
                path = '/'.join(path_components)

            if path in filename_mapping.index:
                break
            
    return filename_mapping.loc[path]['original_path_and_filename']

def slice_from_original_path(path):
    return int(path.split('.')[-2][-3:])

def patientID_from_original_path(path):
    id = path.split('/')[1]
    return id

def pos_from_original_path(path):
    id = patientID_from_original_path(path)
    slice_ = slice_from_original_path(path)
    
    start_slice = int(boxes_df.loc[id]['Start Slice'])
    end_slice = int(boxes_df.loc[id]['End Slice'])
    
    if slice_ >= start_slice and slice_ <= end_slice:
        return 1
    else:
        return 0
    
def feature_from_original_path(path, feature_name):
    id = patientID_from_original_path(path)
    value = clinical_features.loc[id][feature_name]
    
    if feature_name in ['TE (Echo Time)', 'TR (Repetition Time)']:
        return float(value)
    
    feature_values = dict(reversed(name_value.strip().split('=')) for name_value in clinical_features[feature_name].iloc[0].split(','))

    if value in feature_values.keys():
        name = feature_values[value]
        
        if name == 'MMAGNEVIST':
            name = 'MAGNEVIST'
        
        elif name == '1.494':
            name = '1.5'
        
        elif name == '2.8936':
            name = '3'
        
    else:
        name = pd.NA
    
    return name

def main():
    
    # Get file paths
    files = glob.glob(os.path.join(MANIFEST_FILE,'**','*.dcm'),recursive=True)
    
    # Build dataset
    df = pd.DataFrame({'file_path':files,'original_path_and_filename':[get_original_path_and_filename(file) for file in files]})

    # Add features to the dataset
    df = df[df['original_path_and_filename'].str.contains('pre')]
    df['Patient ID'] = df['original_path_and_filename'].apply(patientID_from_original_path)
    df['slice'] = df['original_path_and_filename'].apply(slice_from_original_path)
    df['pos'] = df['original_path_and_filename'].apply(pos_from_original_path)
    for feature in FEATURES_NAMES:
        df[feature] = df['original_path_and_filename'].apply(lambda x: feature_from_original_path(x,feature))
    
    # Remove first b and last b slices
    max_slice_per_PatientID = df.groupby('Patient ID')['slice'].max().reset_index()
    
    def within_slice_bounds(path, bound):
        patientID = patientID_from_original_path(path)
        slice_ = slice_from_original_path(path)
        max_slice_idx = max_slice_per_PatientID.index[max_slice_per_PatientID["Patient ID"] == patientID]
        return (bound < slice_ and  slice_ <= (max_slice_per_PatientID['slice'][max_slice_idx].iloc[0] - bound))

    df = df[df['original_path_and_filename'].apply(lambda x: within_slice_bounds(x, SLICES_BOUND))]
        
    # Remove missing values
    ids_with_missing_values = set(df.loc[df.isna().any(axis=1)]['Patient ID'].unique())
    print(f"Removing {len(ids_with_missing_values)} Patient IDs with missing values")
    df = df[df['Patient ID'].apply(lambda x: x not in ids_with_missing_values)]
    
    # Preprocess and save segmentation paths
    seg_files = glob.glob(os.path.join(SEGMENTATIONS_FILES, '**','*.nii.gz'), recursive=True)
    df = df.reindex(columns = df.columns.tolist() + ["seg_path"] + ["aug_path"], fill_value='')
       
    for file in seg_files:
        id = os.path.normpath(file).split(os.sep)[-1].split('.')[0]
        
        segmentation = read_segmentation(file)
        
        # Correct orientation in last dimension 
        if id in set(df['Patient ID']):
            image_path = df[df['Patient ID'] == id].iloc[0]['file_path']
            orientation = dicom.dcmread(os.path.join(os.path.split(image_path)[0],'1-002.dcm')).SliceLocation - dicom.dcmread(os.path.join(os.path.split(image_path)[0],'1-001.dcm')).SliceLocation
            orientation = 'S' if orientation>0 else 'I'
            max_slice = df[df['Patient ID'] == id].groupby('Patient ID')['slice'].max().item() + SLICES_BOUND
            
        for i in df.index[df['Patient ID'] == id].tolist():
            slice_ = df.loc[i]['slice']
            
            segmentation_path = os.path.join(SEGMENTATIONS_PATH, id + f'-{slice_}'.zfill(3) + '.pt' )
            
            slice_ind = slice_-1 if orientation=='S' else max_slice - slice_
            torch.save(Orientation(axcodes='RA')(segmentation[:,:,:,slice_ind]).clone(), segmentation_path)
            
            df.loc[i, ["seg_path"]] = segmentation_path
    
    # Save dataset
    df.to_csv(os.path.join(DATASETS_PATH, "dataset.csv"), index=False)
    
if __name__=="__main__": 
    main() 