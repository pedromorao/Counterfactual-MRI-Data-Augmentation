from torch import nn
from monai.networks.layers.factories import Pool
from generative.networks.nets.diffusion_model_unet import DiffusionModelUNet

class UNet_CondDiff(DiffusionModelUNet):
    def __init__(self, num_channels, num_class_embeds, *args, **kwargs):
        super().__init__(num_channels=num_channels, num_class_embeds=num_class_embeds, *args, **kwargs)
        
        # time
        time_embed_dim = num_channels[0] * 4
        self.time_embed = nn.Sequential(
            nn.Linear(num_channels[0], time_embed_dim), nn.SiLU(), nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        # class embedding
        self.num_class_embeds = num_class_embeds
        if num_class_embeds is not None:
            self.class_embedding = nn.Sequential(
                nn.Linear(num_class_embeds, time_embed_dim, bias=False), nn.SiLU()
            )
