import os
import seaborn as sns
from matplotlib import colors
import matplotlib.patches as mpatches

RANDOM_SEED=123

# consants
WANDB_API_KEY = ''
CUDA_VISIBLE_DEVICES = "0"

MANIFEST_FILE = '/workspace/data'
DATASETS_PATH = 'datasets'
SEGMENTATIONS_FILES = 'segmentations_files'
SEGMENTATIONS_PATH = '/workspace/segmentations'
AUGMENTED_IMAGES_PATH = '/workspace/aug_images'


FEATURE_PRED_CHECKPOINT_PATH = 'feature_pred/od902wkw/checkpoints/last.ckpt' 
FEATURE_MOD_CHECKPOINT_PATH = 'feature_mod/pamr38jf/checkpoints/last.ckpt'
SEGMENTATION_CHECKPOINT_PATH = 'segmentation-v3/bhwg2k6w/checkpoints/epoch=44-step=1710.ckpt'

AUG_RATIO = 1

LOWER_PERCENTILE_NORM = 10
UPPER_PERCENTILE_NORM = 99
WITH_CLIP = False

TRAIN_RATIO = 0.75
VAL_RATIO = 0.10
TEST_RATIO = 0.15

SEG_TRAIN_RATIO = 0.75
SEG_VAL_RATIO = 0.25
SEG_TEST_RATIO = 0

FEATURES = ["Manufacturer Model Name",
            "Scan Options",
            "Field Strength (Tesla)",
            "Flip Angle",
            "Slice Thickness",
            "TE (Echo Time)",
            "TR (Repetition Time)"]

CATEGORICAL_FEATURES = ["Manufacturer",
                        "Manufacturer Model Name",
                        "Scan Options",
                        "Patient Position During MRI",
                        "Field Strength (Tesla)",
                        "Contrast Agent",
                        "Acquisition Matrix"]

CONTINUOUS_FEATURES = [ "Slice Thickness",
                        "Flip Angle",
                        "FOV Computed (Field of View) in cm",
                        "TE (Echo Time)",
                        "TR (Repetition Time)"]

FEATURES_NAMES = CATEGORICAL_FEATURES + CONTINUOUS_FEATURES

SLICES_BOUND = 20

SPATIAL_SIZE=(224,224)

SEGMENTATION_LABELS = ["Background", "Breast", "FGT"]
COLLOR_PALETTE = sns.color_palette(as_cmap=True)
COLLOR_PALETTE = ['black'] + COLLOR_PALETTE[0:2]
CMAP = colors.ListedColormap(COLLOR_PALETTE)
BOUNDS = [-0.5,0.5,1.5,2.5]
PATCHES = [ mpatches.Patch(color=COLLOR_PALETTE[i], label=SEGMENTATION_LABELS[i]) for i in range(len(COLLOR_PALETTE))]
NORM = colors.BoundaryNorm(BOUNDS, CMAP.N)