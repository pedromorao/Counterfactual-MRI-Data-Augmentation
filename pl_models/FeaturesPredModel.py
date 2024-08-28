import wandb
import torch
import torch.optim as optim
from torch.nn.functional import mse_loss, cross_entropy
from torcheval.metrics.functional import multiclass_accuracy, mean_squared_error
import lightning as L
from monai.networks.nets import ResNet
from utils.dataset import get_feature

class FeaturesPredModel(L.LightningModule):
    def __init__(self,
                 model_hparams,
                 optimizer_name,
                 optimizer_hparams,
                 features_dims,
                 features_labels,
                 features_scale,
                 catg_features_weights):
        super().__init__()
        self.save_hyperparameters()
        self.model = ResNet(block='basic',
                            layers=[2,2,2,2],
                            block_inplanes=[64,128,256,512],
                            spatial_dims=2,
                            n_input_channels=1,
                            conv1_t_stride=2,
                            num_classes=sum(features_dims.values()))

        self.features_dims = features_dims
        self.features_labels = features_labels
        self.features_scale = features_scale
        self.catg_features_weights = catg_features_weights
        self.calc_metrics_outputs = {}

    def forward(self, imgs):
        return self.model(imgs)
        
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
        imgs, targets = batch["image"],batch["features"]
        preds = self(imgs)
        
        losses = []
        
        for feature_name in self.features_dims.keys():
            pred = get_feature(preds, feature_name, self.features_dims, self.features_scale)
            target = get_feature(targets, feature_name, self.features_dims, self.features_scale)
            
            # continuous features
            if self.features_dims[feature_name] == 1:
                losses.append(mse_loss(pred, target))
                
            # categorical features  
            else:
                target = target.argmax(-1)
                losses.append(cross_entropy(pred, target,
                                            weight=torch.tensor(self.catg_features_weights[feature_name],
                                                                dtype=torch.float32,
                                                                device=self.device)))
                
        loss = torch.stack(losses).sum()
        self.log("train/loss", loss, prog_bar=True)

        return loss  # Return tensor to call ".backward" on
    
    def validation_step(self, batch, batch_idx):
        imgs, targets = batch["image"],batch["features"]
        preds = self(imgs)
        
        losses = []
        
        for feature_name in self.features_dims.keys():
            pred,pred_indices = get_feature(preds, feature_name, self.features_dims, self.features_scale, as_indices='both')
            target,target_indices = get_feature(targets, feature_name, self.features_dims, self.features_scale, as_indices='both')

            # continuous features
            if self.features_dims[feature_name] == 1:
                losses.append(mse_loss(pred, target))

                mse = mean_squared_error(pred_indices, target_indices)
                self.log(f"val/mse_{feature_name}", mse)
                
            # categorical features  
            else:
                target = target.argmax(-1)
                losses.append(cross_entropy(pred, target,
                                            weight=torch.tensor(self.catg_features_weights[feature_name],
                                                                dtype=torch.float32,
                                                                device=self.device)))
                
                acc = multiclass_accuracy(pred_indices, target_indices,
                                          num_classes=self.features_dims[feature_name],
                                          average='micro')
                self.log(f"val/acc_{feature_name}", acc)
                
                
        loss = torch.stack(losses).sum()

        self.log("val/loss", loss, prog_bar=True)
    
    def on_test_epoch_start(self):
        self.test_step_outputs = {feature_name:{'pred':[],'target':[]} for feature_name in self.features_dims.keys() if self.features_dims[feature_name] != 1}
    
    def test_step(self, batch, batch_idx):
        imgs, targets = batch["image"],batch["features"]
        preds = self(imgs)
        
        losses = []
        
        for feature_name in self.features_dims.keys():
            pred,pred_indices = get_feature(preds, feature_name, self.features_dims, self.features_scale, as_indices='both')
            target,target_indices = get_feature(targets, feature_name, self.features_dims, self.features_scale, as_indices='both')

            # continuous features
            if self.features_dims[feature_name] == 1:
                losses.append(mse_loss(pred, target))

                mse = mean_squared_error(pred_indices, target_indices)
                self.log(f"test/mse_{feature_name}", mse)
                
            # categorical features  
            else:
                target = target.argmax(-1)
                losses.append(cross_entropy(pred, target,
                                            weight=torch.tensor(self.catg_features_weights[feature_name],
                                                                dtype=torch.float32,
                                                                device=self.device)))
                
                acc = multiclass_accuracy(input=pred_indices, target=target_indices,
                                          num_classes=self.features_dims[feature_name],
                                          average='micro')
                self.log(f"test/acc_{feature_name}", acc)
                
                self.test_step_outputs[feature_name]['pred'].append(pred_indices.detach().cpu())
                self.test_step_outputs[feature_name]['target'].append(target_indices.detach().cpu())
                
        loss = torch.stack(losses).sum()

        self.log("test/loss", loss.detach())

    
    def on_test_epoch_end(self):
        for feature_name in self.test_step_outputs.keys():
            preds,targets = self.test_step_outputs[feature_name]['pred'],self.test_step_outputs[feature_name]['target']
            preds,targets = torch.cat(preds),torch.cat(targets)

            wandb.log({f"test/{feature_name}" : wandb.plot.confusion_matrix(probs=None,
                        preds=preds.numpy(),
                        y_true=targets.numpy(),
                        class_names=self.features_labels[feature_name])})
                
        self.test_step_outputs.clear()  # free memory
    
    @torch.no_grad
    def calc_metrics(self, preds, targets):
        if self.calc_metrics_outputs == {}:
            self.calc_metrics_outputs = {feature_name:{'pred':[],'target':[]} for feature_name in self.features_dims.keys() if self.features_dims[feature_name] != 1}
        
        preds = preds.to(self.device)
        targets = targets.to(self.device)

        accs = {}
        ms_errors = {}
        
        for feature_name in self.features_dims.keys():
            pred_indices = get_feature(preds, feature_name, self.features_dims, self.features_scale, as_indices=True)
            target_indices = get_feature(targets, feature_name, self.features_dims, self.features_scale, as_indices=True)

            # continuous features
            if self.features_dims[feature_name] == 1:
                
                ms_errors.update({f"{feature_name}":mean_squared_error(pred_indices, target_indices).detach().cpu()})
                
            # categorical features  
            else:
                acc = multiclass_accuracy(input=pred_indices, target=target_indices,
                                            num_classes=self.features_dims[feature_name],
                                            average='micro')
                
                accs.update({f"{feature_name}": acc.detach().cpu()})
                
                self.calc_metrics_outputs[feature_name]['pred'].append(pred_indices.detach().cpu())
                self.calc_metrics_outputs[feature_name]['target'].append(target_indices.detach().cpu())
            
        return accs,ms_errors