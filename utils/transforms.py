import torch
from monai.transforms import(
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    Resized,
    ScaleIntensityRangePercentilesd,
    Orientationd,
    Lambdad,
    LoadImage,
    EnsureChannelFirst,
    Orientation,
    Resize,
    Lambda,
)
from utils.constants import SPATIAL_SIZE,LOWER_PERCENTILE_NORM, UPPER_PERCENTILE_NORM,WITH_CLIP

scale_intensity =  ScaleIntensityRangePercentilesd(
                        keys=["image"],
                        lower=LOWER_PERCENTILE_NORM,
                        upper=UPPER_PERCENTILE_NORM,
                        b_min=0.0,
                        b_max=1.0,
                        clip=WITH_CLIP
                    )

read_img = Compose([LoadImage(reader='pydicomreader'),
                    EnsureChannelFirst(),
                    Resize(spatial_size=SPATIAL_SIZE),
                    Orientation(axcodes='RA'),
                    Lambda(func=lambda x: torch.tensor(x, dtype=torch.float32))])

DEFAULT_TRANSFORMS = Compose([
                    Lambdad(keys=["image"], func=read_img),
                    scale_intensity,
                    Lambdad(keys=["features"], func=lambda x: torch.tensor(x, dtype=torch.float32))
                ])

DEFAULT_TRANSFORMS_WITH_SEGMENTATION = Compose([
                    Lambdad(keys=["image"], func=lambda x: torch.load(x) if ('aug_images' in x) else read_img(x)),
                    Lambdad(keys=["segmentation"], func=torch.load),
                    scale_intensity,
                    Lambdad(keys=["features"], func=lambda x: torch.tensor(x, dtype=torch.float32)),
                ])