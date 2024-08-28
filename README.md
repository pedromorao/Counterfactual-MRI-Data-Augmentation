# Counterfactual MRI Data Augmentation


## Installation

You can install dependencies using pip:
``` pip install -r requirements.txt```

Download the Duke Breast Dataset with the option descriptive paths and add the path to the file `utils\constants.py` on **MANIFEST_FILE**.

Then run the script `build_dataset.py` to nuild the dataset.csv with all the image path information and IAP.

## File Paths
Paths are defined in `utils\constants.py` that may need to be changed.

-   **MANIFEST_FILE**: Path to the manifest file where the Duke Breast Dataset should be downloaded, default is `/workspace/data`.
-   **SEGMENTATIONS_FILES**: Path where the segmentated 3D images are downloaded, default is `/workspace/segmentations_files`.
-   **SEGMENTATIONS_PATH**: Path where the sliced 2D images will be stored after processing the SEGMENTATIONS_FILES , default is `/workspace/segmentations`.
-   **AUGMENTED_IMAGES_PATH**: Path for storing augmented images, default is `/workspace/aug_images`.

## Checkpoints
Paths are defined in `utils\constants.py` that may need to be changed.

-   **FEATURE_PRED_CHECKPOINT_PATH**: Path to the feature prediction model checkpoint, default is `feature_pred/od902wkw/checkpoints/last.ckpt`.
-   **FEATURE_MOD_CHECKPOINT_PATH**: Path to the feature modification model checkpoint, default is `feature_mod/pamr38jf/checkpoints/last.ckpt`.
-   **SEGMENTATION_CHECKPOINT_PATH**: Path to the segmentation model checkpoint, default is `segmentation-v3/bhwg2k6w/checkpoints/epoch=44-step=1710.ckpt`.


## Training and Evaluating

To train the models the first model needs to be trained by running the `train_feature_pred_model.py` script file and then changing the **FEATURE_PRED_CHECKPOINT_PATH** path in `utils\constants.py` to the path of the checkpoint of the model trained.
Then the script `train_feature_mod_model.py` should be used to train the cDDGM and the path **FEATURE_MOD_CHECKPOINT_PATH** changed to the model wheights checkpoint created.
Finally to train a segmentation model the `train_seg_model.py` should be used but it will train multiple models according to a grid search of *steps* and *gs* that can be changed inside the script.

The test evaluation is also run when running the training scripts, but to evaluate further cDDGM image modifications the notebook `Eval FeatureModModel.ipynb` can be used.