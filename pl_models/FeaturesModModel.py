import torch
import torch.optim as optim
from torch.nn.functional import mse_loss
import lightning as L
import wandb
from tqdm import tqdm
from generative.networks.schedulers import NoiseSchedules, DDPMScheduler
from generative.networks.schedulers.ddpm import DDPMPredictionType
from generative.metrics import FIDMetric, MMDMetric, SSIMMetric
from generative.metrics.ssim import KernelType,_gaussian_kernel
from monai.transforms import ScaleIntensityRangePercentiles
from monai.utils import convert_data_type
from monai.utils.type_conversion import convert_to_dst_type
import torch.nn.functional as F

from models.UNet_CondDiff import UNet_CondDiff
from pl_models.FeaturesPredModel import FeaturesPredModel
from utils.constants import FEATURE_PRED_CHECKPOINT_PATH,WITH_CLIP,LOWER_PERCENTILE_NORM, UPPER_PERCENTILE_NORM

# Modified structural similarity from monai generative metrics based on the original formula
# to include only the structular similarity (without contrast and luminosity)
# https://www.researchgate.net/publication/4071876_Multiscale_structural_similarity_for_image_quality_assessment
class OnlyStructure_SSIMMetric(SSIMMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def _compute_metric(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: Predicted image.
                It must be a 2D or 3D batch-first tensor [B,C,H,W] or [B,C,H,W,D].
            y: Reference image.
                It must be a 2D or 3D batch-first tensor [B,C,H,W] or [B,C,H,W,D].

        Raises:
            ValueError: when `y_pred` is not a 2D or 3D image.
        """
        dims = y_pred.ndimension()
        if self.spatial_dims == 2 and dims != 4:
            raise ValueError(
                f"y_pred should have 4 dimensions (batch, channel, height, width) when using {self.spatial_dims} "
                f"spatial dimensions, got {dims}."
            )

        if self.spatial_dims == 3 and dims != 5:
            raise ValueError(
                f"y_pred should have 4 dimensions (batch, channel, height, width, depth) when using {self.spatial_dims}"
                f" spatial dimensions, got {dims}."
            )

        # compute only sctructural part of SSIM
        if y.shape != y_pred.shape:
            raise ValueError(f"y_pred and y should have same shapes, got {y_pred.shape} and {y.shape}.")

        y_pred = convert_data_type(y_pred, output_type=torch.Tensor, dtype=torch.float)[0]
        y = convert_data_type(y, output_type=torch.Tensor, dtype=torch.float)[0]

        num_channels = y_pred.size(1)

        if self.kernel_type == KernelType.GAUSSIAN:
            kernel = _gaussian_kernel(self.spatial_dims, num_channels, self.kernel_size, self.kernel_sigma)
        elif self.kernel_type == KernelType.UNIFORM:
            kernel = torch.ones((num_channels, 1, *self.kernel_size)) / torch.prod(torch.tensor(self.kernel_size))
        
        kernel = convert_to_dst_type(src=kernel, dst=y_pred)[0]

        
        c1 = (self.k1 * self.data_range) ** 2  # stability constant for luminance
        c2 = (self.k2 * self.data_range) ** 2  # stability constant for contrast

        conv_fn = getattr(F, f"conv{self.spatial_dims}d")
        mu_x = conv_fn(y_pred, kernel, groups=num_channels)
        mu_y = conv_fn(y, kernel, groups=num_channels)
        mu_xx = conv_fn(y_pred * y_pred, kernel, groups=num_channels)
        mu_yy = conv_fn(y * y, kernel, groups=num_channels)
        mu_xy = conv_fn(y_pred * y, kernel, groups=num_channels)

        sigma_x = mu_xx - mu_x * mu_x
        sigma_y = mu_yy - mu_y * mu_y
        sigma_xy = mu_xy - mu_x * mu_y

        ssim_value_full_image = (sigma_xy + c2/2) / (sigma_x*sigma_y +  c2/2)

        ssim_per_batch: torch.Tensor = ssim_value_full_image.view(ssim_value_full_image.shape[0], -1).mean(
            1, keepdim=True
        )
        
        return ssim_per_batch

# Fixed a bug that appeared when using cosine sheduling
@NoiseSchedules.add_def("cosine", "Cosine schedule")
def _cosine_beta(num_train_timesteps: int, s: float = 8e-3):
    """
    Cosine noise schedule, see https://arxiv.org/abs/2102.09672

    Args:
        num_train_timesteps: number of timesteps
        s: smoothing factor, default 8e-3 (see referenced paper)

    Returns:
        (betas, alphas, alpha_cumprod) values
    """
    x = torch.linspace(0, num_train_timesteps, num_train_timesteps + 1)
    alphas_cumprod = torch.cos(((x / num_train_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod /= alphas_cumprod[0].item()
    alphas = torch.clip(alphas_cumprod[1:] / alphas_cumprod[:-1], 0.0001, 0.9999)
    betas = 1.0 - alphas
    return betas

# Changed the cliping from the DDPMScheduler from monai-generative
class Modified_DDPMScheduler(DDPMScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def step(
        self, model_output: torch.Tensor, timestep: int, sample: torch.Tensor, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output: direct output from learned diffusion model.
            timestep: current discrete timestep in the diffusion chain.
            sample: current instance of sample being created by diffusion process.
            generator: random number generator.

        Returns:
            pred_prev_sample: Predicted previous sample
        """
        if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type in ["learned", "learned_range"]:
            model_output, predicted_variance = torch.split(model_output, sample.shape[1], dim=1)
        else:
            predicted_variance = None

        # 1. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[timestep - 1] if timestep > 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        if self.prediction_type == DDPMPredictionType.EPSILON:
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        elif self.prediction_type == DDPMPredictionType.SAMPLE:
            pred_original_sample = model_output
        elif self.prediction_type == DDPMPredictionType.V_PREDICTION:
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output

        # 3. Clip "predicted x_0"
        if self.clip_sample:
            if WITH_CLIP==False:
                scaler = ScaleIntensityRangePercentiles(lower=LOWER_PERCENTILE_NORM,
                                                        upper=UPPER_PERCENTILE_NORM,
                                                        b_min=0.0,
                                                        b_max=1.0,
                                                        channel_wise=True,
                                                        clip=WITH_CLIP)
                pred_original_sample = scaler(pred_original_sample)
                
            else:
                pred_original_sample = torch.clamp(pred_original_sample, 0, 1)

        # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * self.betas[timestep]) / beta_prod_t
        current_sample_coeff = self.alphas[timestep] ** (0.5) * beta_prod_t_prev / beta_prod_t

        # 5. Compute predicted previous sample µ_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

        # 6. Add noise
        variance = 0
        if timestep > 0:
            noise = torch.randn(
                model_output.size(), dtype=model_output.dtype, layout=model_output.layout, generator=generator
            ).to(model_output.device)
            variance = (self._get_variance(timestep, predicted_variance=predicted_variance) ** 0.5) * noise

        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample, pred_original_sample
        

# Functions from Monai generative metrics tutorial to evaluate the FID score in the test set
def subtract_mean(x: torch.Tensor) -> torch.Tensor:
    mean = [0.406, 0.456, 0.485]
    x[:, 0, :, :] -= mean[0]
    x[:, 1, :, :] -= mean[1]
    x[:, 2, :, :] -= mean[2]
    return x


def spatial_average(x: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
    return x.mean([2, 3], keepdim=keepdim)


def get_features(image,radnet):
    # If input has just 1 channel, repeat channel to have 3 channels
    if image.shape[1]:
        image = image.repeat(1, 3, 1, 1)

    # Change order from 'RGB' to 'BGR'
    image = image[:, [2, 1, 0], ...]

    # Subtract mean used during training
    image = subtract_mean(image)

    # Get model outputs
    with torch.no_grad():
        feature_image = radnet.forward(image)
        # flattens the image spatially
        feature_image = spatial_average(feature_image, keepdim=False)

    return feature_image


class FeaturesModModel(L.LightningModule):
    def __init__(self,
                 model_hparams,
                 unet_hprams,
                 noise_scheduler_hprams,
                 optimizer_name,
                 optimizer_hparams,
                 scheduler_name=None,
                 scheduler_hparams=None,
                 test_params={'guidance_scale':3,
                              'steps':50},):
        super().__init__()
        self.save_hyperparameters()

        if model_hparams['conditioning']=='hybrid':
            with_conditioning = True,
            cross_attention_dim = model_hparams['features_dim']
            num_class_embeds = model_hparams['features_dim']
            
        elif model_hparams['conditioning']=='cross_attention':
            with_conditioning = True,
            cross_attention_dim = model_hparams['features_dim']
            num_class_embeds = None
            
        elif model_hparams['conditioning']=='time_emb':
            with_conditioning = False,
            cross_attention_dim = None
            num_class_embeds = model_hparams['features_dim']
            
        self.model = UNet_CondDiff(**unet_hprams,
                                    in_channels = 2 if model_hparams['with_segmentation']==True else 1,
                                    with_conditioning=with_conditioning,
                                    cross_attention_dim=cross_attention_dim,
                                    num_class_embeds=num_class_embeds,
                                   )
        self.scheduler = Modified_DDPMScheduler(**noise_scheduler_hprams)
    
    def forward(self, x, timesteps, features):
        if self.hparams.model_hparams['conditioning']=='hybrid':
            return self.model(x=x,
                                timesteps=timesteps,
                                class_labels=features,
                                context=features.unsqueeze(1))
            
        elif self.hparams.model_hparams['conditioning']=='cross_attention':
            return self.model(x=x,
                                timesteps=timesteps,
                                context=features.unsqueeze(1))
            
        elif self.hparams.model_hparams['conditioning']=='time_emb':
            return self.model(x=x,
                                timesteps=timesteps,
                                class_labels=features)

    def configure_optimizers(self):
        if self.hparams.optimizer_name == "Adam":
            # AdamW is Adam with a correct implementation of weight decay (see here
            # for details: https://arxiv.org/pdf/1711.05101.pdf)
            optimizer = optim.AdamW(self.parameters(), **self.hparams.optimizer_hparams)
        else:
            assert False, f'Unknown optimizer: "{self.hparams.optimizer_name}"'
        return optimizer
    
    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        imgs,features = batch["image"],batch["features"]
        
        # Apply feature dropout
        dropout_mask = torch.rand((features.shape[0],1), device=self.device) > self.hparams.model_hparams['features_dropout']
        features = features * dropout_mask
        
        timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (imgs.shape[0],), device=self.device).long()

        noise = torch.randn_like(imgs, device=self.device)
        noisy_imgs = self.scheduler.add_noise(original_samples=imgs, noise=noise, timesteps=timesteps)

        # Add segmentation to noisy images
        if self.hparams.model_hparams['with_segmentation']==True:
            segmentation = batch["segmentation"]

            # apply segmentation dropout
            dropout_mask = torch.rand((segmentation.shape[0],1,1,1), device=self.device) > self.hparams.model_hparams['segmentation_dropout']
            segmentation = segmentation * dropout_mask
    
            noisy_imgs = torch.cat([noisy_imgs,segmentation],dim=1)
        else:
            segmentation=None

        noise_pred = self(x=noisy_imgs,
                            timesteps=timesteps,
                            features=features)
        
        if self.scheduler.prediction_type == 'v_prediction':
            target = self.scheduler.get_velocity(imgs, noise, timesteps)
        elif self.scheduler.prediction_type == 'sample':
            target = imgs
        elif self.scheduler.prediction_type == 'epsilon':
            target = noise

        loss = mse_loss(noise_pred, target)
        
        self.log("train/loss", loss, prog_bar=True)
        
        return loss  # Return tensor to call ".backward" on
        
    def validation_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        imgs,features = batch["image"],batch["features"]

        timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (imgs.shape[0],), device=self.device).long()

        noise = torch.randn_like(imgs, device=self.device, requires_grad=False)
        noisy_imgs = self.scheduler.add_noise(original_samples=imgs, noise=noise, timesteps=timesteps)
        
        # Add segmentation to noisy images
        if self.hparams.model_hparams['with_segmentation']==True:
            segmentation = batch["segmentation"]
            noisy_imgs = torch.cat([noisy_imgs,segmentation],dim=1)
        else:
            segmentation=None

        noise_pred = self(x=noisy_imgs,
            timesteps=timesteps,
            features=features)
        
        if self.scheduler.prediction_type == 'v_prediction':
            target = self.scheduler.get_velocity(imgs, noise, timesteps)
        elif self.scheduler.prediction_type == 'sample':
            target = imgs
        elif self.scheduler.prediction_type == 'epsilon':
            target = noise

        loss = mse_loss(noise_pred, target)
        
        self.log("val/loss", loss, prog_bar=True)

        if batch_idx==0:
            imgs_subset, features_subset = imgs[:8],features[:8]
            if self.hparams.model_hparams['with_segmentation']==True:
                segmentation_subset=segmentation[:8]
            else:
                segmentation_subset=None
            
            og_img,edited_img,noisy_img = self.edit(imgs=imgs_subset,
                                                    features=features_subset,
                                                    segmentation=segmentation_subset,
                                                    guidance_scale=1,
                                                    start_t=100)
            
            image_array = torch.cat([edited_img.squeeze(1).cpu(),noisy_img.squeeze(1).cpu(),og_img.squeeze(1).cpu()],dim=1)
            image_array = torch.cat([img for img in image_array],dim=1)
            
            images = wandb.Image(torch.clip(image_array,0,1), caption="Top: Output, Middle: Input, Bottom: Original")

            wandb.log({"examples": images})
    
    def on_test_start(self):
        self.classifier_model = FeaturesPredModel.load_from_checkpoint(FEATURE_PRED_CHECKPOINT_PATH).to(self.device)
        self.classifier_model.eval()
        
        self.radnet = torch.hub.load("Warvito/radimagenet-models", model="radimagenet_resnet50", verbose=True).to(self.device)
        self.radnet.eval()
        
        self.real_rednet_features = []
        self.edited_rednet_features = []
        
        self.fid = FIDMetric()
        self.mmd = MMDMetric()
        self.ssim = SSIMMetric(spatial_dims=2,
                               data_range=1.0,
                               kernel_size=11)
        
        torch.manual_seed(0)

    @torch.no_grad    
    def test_step(self, batch, batch_idx):
        imgs,features = batch["image"],batch["features"]
        segmentation = batch["segmentation"] if self.hparams.model_hparams['with_segmentation']==True else None
        
        # Create dict to store values to log
        values = {}
        
        idx_shuffled = torch.randperm(features.shape[0])
        
        # shufle classes
        shufled_features = features[idx_shuffled]
        
        _,edited_imgs,_ = self.edit(imgs,
                                    shufled_features,
                                    # original_features=features,
                                    segmentation=segmentation,
                                    guidance_scale=self.hparams.test_params['guidance_scale'],
                                    start_t=self.hparams.test_params['steps']
                                    )

        preds = self.classifier_model(edited_imgs)

        accs,ms_errors = self.classifier_model.calc_metrics(preds=preds, targets=shufled_features)
        
        # Log Accs
        for label,metric in accs.items():
            values.update({f"test/acc_{label}": metric.item()})
        
        # Log MSEs
        for label,metric in ms_errors.items():
            values.update({f"test/mse_{label}": metric.item()})
        
        # FID radnet features calculation
        self.real_rednet_features.append(get_features(imgs,self.radnet).detach().cpu())
        self.edited_rednet_features.append(get_features(edited_imgs,self.radnet).detach().cpu())
        
        # Log MMD
        values.update({f"test/MMD": self.mmd(imgs,edited_imgs).item()})
        
        # Log SSIM
        
        values.update({f"test/SSIM_original\\edited": self.ssim(edited_imgs, imgs).mean().item()})
        values.update({f"test/SSIM_shuffled\\edited": self.ssim(edited_imgs,imgs[idx_shuffled]).mean().item()})

        self.log_dict(values)
            
        if batch_idx==0:
            imgs, edited_imgs = imgs[:8],edited_imgs[:8]
            
            image_array = torch.cat([edited_imgs.squeeze(1).cpu(),imgs.squeeze(1).cpu()],dim=1)
            image_array = torch.cat([img for img in image_array],dim=1)
            
            images = wandb.Image(torch.clip(image_array,0,1),
                                 caption=f"Top: Output, Bottom: Original. (GS:{self.hparams.test_params['guidance_scale']},steps:{self.hparams.test_params['steps']})")

            wandb.log({"test_examples": images})

    def on_test_epoch_end(self):
        # Log FID
        self.log(f"test/FID", self.fid(torch.vstack(self.edited_rednet_features).to(self.device),
                                       torch.vstack(self.real_rednet_features).to(self.device))
        )
      
        self.real_rednet_features.clear() 
        self.edited_rednet_features.clear()
        
        for feature in self.classifier_model.calc_metrics_outputs.keys():
            preds,targets = (torch.cat(self.classifier_model.calc_metrics_outputs[feature]['pred']),
                             torch.cat(self.classifier_model.calc_metrics_outputs[feature]['target'])
                             )
            
            wandb.log({f"test/{feature}" : wandb.plot.confusion_matrix(probs=None,
                        preds=preds.numpy(),
                        y_true=targets.numpy(),
                        class_names=self.classifier_model.features_labels[feature])})
                
        self.classifier_model.calc_metrics_outputs.clear()  # free memory
        del self.radnet
        del self.classifier_model
    
    
    @torch.no_grad
    def generate(self, features, segmentation=None, guidance_scale=7, inference_steps=None):
        if inference_steps is None:
            inference_steps = self.scheduler.num_train_timesteps
        
        self.scheduler.set_timesteps(num_inference_steps=inference_steps)
        
        features_conditions = torch.cat([torch.zeros_like(features), features])
        
        imgs_shape = (features.shape[0],
                1,
                self.hparams.model_hparams['spatial_size'][0],
                self.hparams.model_hparams['spatial_size'][1])
        
        noise = torch.randn(imgs_shape, device=self.device, requires_grad=False)

        for t in tqdm(self.scheduler.timesteps):
           
            noise_input = torch.cat([noise] * 2)
            
            # Add segmentation to noisy images
            if self.hparams.model_hparams['with_segmentation']==True:
                noise_input = torch.cat([noise_input, torch.cat([segmentation] * 2)],dim=1)
                
            model_output = self(noise_input,
                                    timesteps=torch.tensor((t,),device=self.device),
                                    features=features_conditions)
            
            noise_pred_uncond, noise_pred_cond = model_output.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            noise, _ = self.scheduler.step(noise_pred, t, noise)

        return noise
    
    @torch.no_grad
    def edit(self, imgs, features, original_features=None, segmentation=None, guidance_scale=7, start_t=50, inference_steps=None):
        if inference_steps is None:
            inference_steps = self.scheduler.num_train_timesteps
        
        self.scheduler.set_timesteps(num_inference_steps=inference_steps)
        
        if original_features is None:
            original_features=torch.zeros_like(features)
        
        features_conditions = torch.cat([original_features, features])
        
        noise = torch.randn_like(imgs,device=self.device,requires_grad=False)
        noise = self.scheduler.add_noise(original_samples=imgs, noise=noise,
                                    timesteps=torch.tensor((start_t,),device=self.device).long())
        
        start_imgs = noise.cpu()

        for t in tqdm(range(start_t-1,-1,-1)):

            noise_input = torch.cat([noise] * 2)

            # Add segmentation to noisy images
            if self.hparams.model_hparams['with_segmentation']==True:
                noise_input = torch.cat([noise_input, torch.cat([segmentation] * 2)],dim=1)

            model_output = self(noise_input,
                                    timesteps=torch.tensor((t,), device=self.device),
                                    features=features_conditions)

            noise_pred_uncond, noise_pred_cond = model_output.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            noise, _ = self.scheduler.step(noise_pred, t, noise)

        return imgs.cpu(),noise,start_imgs