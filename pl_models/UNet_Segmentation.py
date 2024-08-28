import torch
import torch.optim as optim
from torch.nn.functional import mse_loss
import lightning as L
import wandb
from torcheval.metrics.functional import multiclass_accuracy

from monai.metrics import DiceMetric
from monai.losses import DiceLoss, DiceCELoss
from monai.networks.nets import UNet

class UNet_Segmentation(L.LightningModule):
    def __init__(self,
                 model_hparams,
                 unet_hprams,
                 optimizer_name,
                 optimizer_hparams,
                 scheduler_name=None,
                 scheduler_hparams=None):
        super().__init__()
        self.save_hyperparameters()
            
        self.model = UNet(**unet_hprams)
        
        if model_hparams['loss'] == 'dice':
            self.loss_function = DiceLoss(include_background=model_hparams['include_background'], to_onehot_y=True, softmax=True,
                                          weight=model_hparams['weights'] if (model_hparams['include_background']==True or model_hparams['weights'] is None) else model_hparams['weights'][1:])
        elif model_hparams['loss'] == 'diceCE':
            self.loss_function = DiceCELoss(include_background=model_hparams['include_background'], to_onehot_y=True, softmax=True, weight=model_hparams['weights'])
        
        self.dice_metric = DiceMetric(include_background=False, reduction="mean_batch", num_classes=3)
        self.mean_dice_metric = DiceMetric(include_background=False, reduction="mean", num_classes=3)
    
    def forward(self, x):
        return self.model(x=x)

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
        imgs,labels = batch["image"],batch["segmentation"]
        
        outputs = self.forward(imgs)
        
        loss =  self.loss_function(outputs, labels)
        
        self.log("train/loss", loss, prog_bar=True)
        
        return loss  # Return tensor to call ".backward" on
    
    def validation_step(self, batch, batch_idx):
        imgs,labels = batch["image"],batch["segmentation"]
        
        outputs = self.forward(imgs)
        
        loss =  self.loss_function(outputs, labels)
        
        self.log("val/loss", loss, prog_bar=True)
        
        accs = multiclass_accuracy(outputs.argmax(1).unsqueeze(1).flatten().long(),labels.flatten().long(),
                                   num_classes=3, average=None)

        self.log_dict({'val/acc_background':accs[0],
                       'val/acc_breast':accs[1],
                       'val/acc_fgt':accs[2],
                       })
        
        # self.dice_metric(y_pred=outputs.argmax(1).unsqueeze(1), y=labels)
        # self.mean_dice_metric(y_pred=outputs.argmax(1).unsqueeze(1), y=labels)
        
        
        if batch_idx==0:
            label_to_color = {
                        0: [0, 0, 0],
                        1: [65/255,105/255,225/255],
                        2: [255/255,165/255,0],
                    }
            
            preds = outputs[0:8].argmax(1).unsqueeze(1).cpu()
            labels_0_8 = labels[0:8].cpu()
            
            rgb_labels = torch.zeros((labels_0_8.shape[0],labels_0_8.shape[-2], labels_0_8.shape[-1], 3), device='cpu')
            rgb_preds = torch.zeros((preds.shape[0],preds.shape[-2], preds.shape[-1], 3), device='cpu')
            
            for gray, rgb in label_to_color.items():
                rgb_preds[preds[:,0,:,:] == gray, :] = torch.tensor(rgb, dtype=rgb_preds.dtype, device='cpu')
                rgb_labels[labels_0_8[:,0,:,:] == gray, :] = torch.tensor(rgb, dtype=rgb_preds.dtype, device='cpu')
            
            image_array = torch.cat([rgb_preds.cpu(), rgb_labels.cpu(), torch.stack([imgs[0:8].squeeze(1).cpu()]*3,dim=-1)],dim=1)
            image_array = torch.cat([img for img in image_array],dim=1)
            
            images = wandb.Image(torch.clip(image_array,0,1),
                                 caption=f"Top: Pred, Middle: Target, Bottom: Original. (Black: Background, Blue: Breast, Orange: FGT)")

            wandb.log({"val_examples": images})
        
    # def on_validation_epoch_end(self):
    #     values = {}
    #     val_dice = self.dice_metric.aggregate().tolist()
    #     val_mean_dice = self.mean_dice_metric.aggregate().item()
        
    #     values.update({'val/meandice':val_mean_dice})
    #     values.update({'val/dice_breast':val_dice[0]})
    #     values.update({'val/dice_fgt':val_dice[1]})
        
    #     self.log_dict(values)
        
    #     self.dice_metric.reset()
    #     self.mean_dice_metric.reset()
    
    def on_test_start(self):
        self.test_step_outputs = {'accs':[],'preds':[],'targets':[]}
        torch.manual_seed(0)

    @torch.no_grad    
    def test_step(self, batch, batch_idx):
        imgs,labels = batch["image"],batch["segmentation"]
        
        preds = self.pred(imgs)
        
        self.dice_metric(y_pred=preds, y=labels)
        self.mean_dice_metric(y_pred=preds, y=labels)
        
        accs = multiclass_accuracy(preds.flatten().long(), labels.flatten().long(),
                                    num_classes=3, average=None)

        self.test_step_outputs['accs'].append(accs)
        self.test_step_outputs['preds'].append(preds.flatten().long().detach().cpu())
        self.test_step_outputs['targets'].append(labels.flatten().long().detach().cpu())
        
        if batch_idx==0:
            label_to_color = {
                        0: [0, 0, 0],
                        1: [65/255,105/255,225/255],
                        2: [255/255,165/255,0],
                    }
            
            preds = preds[0:8].cpu()
            labels_0_8 = labels[0:8].cpu()
            
            rgb_labels = torch.zeros((labels_0_8.shape[0],labels_0_8.shape[-2], labels_0_8.shape[-1], 3), device='cpu')
            rgb_preds = torch.zeros((preds.shape[0],preds.shape[-2], preds.shape[-1], 3), device='cpu')
            
            for gray, rgb in label_to_color.items():
                rgb_preds[preds[:,0,:,:] == gray, :] = torch.tensor(rgb, dtype=rgb_preds.dtype, device='cpu')
                rgb_labels[labels_0_8[:,0,:,:] == gray, :] = torch.tensor(rgb, dtype=rgb_preds.dtype, device='cpu')
            
            image_array = torch.cat([rgb_preds.cpu(), rgb_labels.cpu(), torch.stack([imgs[0:8].squeeze(1).cpu()]*3,dim=-1)],dim=1)
            image_array = torch.cat([img for img in image_array],dim=1)
            
            images = wandb.Image(torch.clip(image_array,0,1),
                                 caption=f"Top: Pred, Middle: Target, Bottom: Original. (Black: Background, Blue: Breast, Orange: FGT)")

            wandb.log({"test_examples": images})

    def on_test_epoch_end(self):
        values = {}
        val_dice = self.dice_metric.aggregate().tolist()
        val_mean_dice = self.mean_dice_metric.aggregate().item()
        
        accs = torch.stack(self.test_step_outputs['accs']).mean(0)
        
        values.update({'test/meandice':val_mean_dice})
        values.update({'test/dice_breast':val_dice[0]})
        values.update({'test/dice_fgt':val_dice[1]})
        values.update({'test/acc_background':accs[0]})
        values.update({'test/acc_breast':accs[1]})
        values.update({'test/acc_fgt':accs[2]})
        
        self.log_dict(values)

        preds,targets = (torch.cat(self.test_step_outputs['preds']),
                        torch.cat(self.test_step_outputs['targets'])
                        )
        wandb.log({"test/confusion_matrix" : wandb.plot.confusion_matrix(probs=None,
            preds=preds.numpy(),
            y_true=targets.numpy(),
            class_names=['Backgroung', 'Breast', 'FGT'])})
        
        self.dice_metric.reset()
        self.mean_dice_metric.reset()
        self.test_step_outputs.clear()
    
    def pred(self, x):
        return self.forward(x).argmax(1).unsqueeze(1)
