import argparse, os
import json
from types import SimpleNamespace 
import logging
import numpy as np
import pandas as pd
import wandb
from tqdm import tqdm
from medmnist import NoduleMNIST3D
from monai.data import DataLoader
from monai.transforms import Activations, AsDiscrete
from monai.utils import set_determinism
from monai.metrics import ROCAUCMetric
import torch
from torch import optim
from sklearn.metrics import accuracy_score, f1_score
from utils import (getDataset, creatConfutionMatrix, creatPredictionTable,
                   logBestMertics, logTraindModel, getNetworkArch,transformsType)
from evaluate import evaluate

# defaults
# default_config = SimpleNamespace(
#     framework="pytorch",
#     batch_size=512, #8 keep small in Colab to be manageable
#     augment=False, # use data augmentation
#     epochs=50, # for brevity, increase for better results :)
#     lr=5e-4,
#     mixed_precision=True, # use automatic mixed precision
#     arch="SEResNet50",
#     optimizer = "AdamW",
#     seed=42,
#     log_preds=False,
#     as_rgb = False,
#     weighted_loss = False,
#     scheduler_reducing_factor = 0.2,
#     scheduler_lr_patience = 4

# )
def t_or_f(arg):
    ua = str(arg).upper()
    if 'TRUE'.startswith(ua): return True
    else: return False

def parse_args(default_config):
    "Overriding default argments"

    argparser = argparse.ArgumentParser(description='Process hyper-parameters')
    argparser.add_argument('--batch_size', type=int, default=default_config['batch_size'], help='batch size')
    argparser.add_argument('--epochs', type=int, default=default_config['epochs'], help='number of training epochs')
    argparser.add_argument('--lr', type=float, default=default_config['lr'], help='learning rate')
    argparser.add_argument('--arch', type=str, default=default_config['arch'], help='Network architecture')
    argparser.add_argument('--optimizer', type=str, default=default_config['optimizer'], help='optimizer')
    argparser.add_argument('--weight_decay', type=float, default=default_config['weight_decay'], help='weight_decay of the optimzer')
    argparser.add_argument('--augment', type=t_or_f, default=default_config['augment'], help='Use image augmentation')
    argparser.add_argument('--seed', type=int, default=default_config['seed'], help='random seed')
    argparser.add_argument('--log_preds', type=t_or_f, default=default_config['log_preds'], help='log model predictions')
    argparser.add_argument('--log_pred_type', type=str, default=default_config['log_pred_type'], help='which split portian to log (val, test)')
    argparser.add_argument('--mixed_precision', type=t_or_f, default=default_config['mixed_precision'], help='use fp16')
    argparser.add_argument('--as_rgb', type=t_or_f, default=default_config['as_rgb'], help='use rgb dataset') 
    argparser.add_argument('--weighted_loss', type=t_or_f, default=default_config['weighted_loss'], help='use wieghted loss')
    argparser.add_argument('--scheduler_reducing_factor', type=int, default=default_config['scheduler_reducing_factor'], help='reducing learning rate factor')
    argparser.add_argument('--scheduler_lr_patience', type=float, default=default_config['scheduler_lr_patience'], help='number of epochs with no improvment to wait before reducing LR ')
    
    args = argparser.parse_args()
    default_config.update(vars(args))
    return default_config



def train(config):
  # print("COOOOOOOOOOOOOOOnfogiraiotns")
  # print(config)

  logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
  experiment = wandb.init(project='NoduleMNIST', entity='saied-salem',
                              resume='allow', job_type="training", config=config)
  config = wandb.config
  # define a metric we are interested in the maximum of
  wandb.define_metric("AUC_score", summary="max")
  wandb.define_metric("F1_score", summary="max")
  wandb.define_metric("accuracy_result", summary="max")
  device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  input_channels = 3 if config.as_rgb else 1
  model= getNetworkArch(config.arch, input_channels).to(device)
  # print(model.named_parameters())
  logging.info(f'Network:\n'
              f'\t{input_channels} input channels\n'
              )
  saved_model_path = config.arch + '_best_F1_score_model.pth'
  logging.info(f'Using device {device}')
  train_loader, val_loader= getDataset(config.augment,config.batch_size, config.as_rgb)
  n_train = len(train_loader.dataset)
  n_val = len(val_loader.dataset)
  n_malignant= np.sum(train_loader.dataset.labels)
  n_benign = n_train - n_malignant
  pos_weight = torch.tensor(n_benign/n_malignant,dtype=torch.float32) 
  pos_weight= pos_weight if config.weighted_loss else None
  logging.info(f'''Starting training:
    Epochs:          {config.epochs}
    Batch size:      {config.batch_size}
    Learning rate:   {config.lr}
    Training size:   {n_train}
    Benign sampels:   {pos_weight}
    Malignant sampels:   {n_malignant}
    Validation size: {n_val}
    Device:          {device.type}
    Images scaling:  {train_loader.dataset.imgs[0].shape}
    Mixed Precision: {config.mixed_precision}
''')
  optimizer = getattr(optim, config.optimizer)(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
  loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'max',min_lr=1e-6,
              factor= config.scheduler_reducing_factor, patience=config.scheduler_lr_patience)
  grad_scaler = torch.cuda.amp.GradScaler(enabled=config.mixed_precision)
  sigmoid = Activations(sigmoid=True)
  threshoulding = AsDiscrete(threshold=0.5)
  auc_object = ROCAUCMetric()
  global_step = 0
  best_f1_score_metric =-1
  best_AUC_metric =-1
  best_acc_metric =-1

  for epoch in range(config.epochs):
    # print('-' * 10)
    # print(f"epoch {epoch + 1}/{epoch_num}")
    model.train()
    epoch_loss = 0
    step = 0
    with tqdm(total=n_train, desc=f'Epoch {epoch}/{config.epochs}', unit='img') as pbar:
      for inputs, labels in train_loader:
          inputs, labels = inputs.to(device,dtype=torch.float32), labels.to(device,dtype=torch.float32)
          with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=config.mixed_precision):
            outputs = model(inputs)
            loss = loss_function(outputs, labels)

          optimizer.zero_grad(set_to_none=True)
          grad_scaler.scale(loss).backward()
          # torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
          grad_scaler.step(optimizer)
          grad_scaler.update()
          epoch_loss += loss.item()
          pbar.update(inputs.shape[0])
          pbar.set_postfix(**{'loss (batch)': loss.item()})
          global_step += 1
          epoch_loss += loss.item()

    auc_result, f1_score_result, accuracy_result, _ , _,_ = evaluate(model,device,config, val_loader,
                                              sigmoid, threshoulding, metrics=[auc_object,f1_score])
    scheduler.step(f1_score_result)
    logging.info(f'F1 score: {f1_score_result:.3f} \t AUC score: {auc_result:.3f} \t acc score: {accuracy_result:.3f}')
    if f1_score_result > best_f1_score_metric:
      best_f1_score_metric = f1_score_result
      best_metric_epoch = epoch + 1
      torch.save(model.state_dict(), saved_model_path)
      print('saved new best metric model')
    if auc_result > best_AUC_metric:
      best_AUC_metric = auc_result
    if accuracy_result > best_acc_metric:
      best_acc_metric = accuracy_result

    experiment.log({
        'train loss': epoch_loss,
        'epoch': epoch,
        'learning_rate':optimizer.param_groups[0]['lr'],
        'F1_score': np.round(f1_score_result,3),
        'AUC_score': np.round(auc_result,3) ,
        'accuracy_result':np.round(accuracy_result,3) ,

    })
  
  split_type = config.log_pred_type
  dataset = NoduleMNIST3D(split=split_type, download=True, as_rgb=config.as_rgb)
  model.load_state_dict(torch.load(saved_model_path))
  loader = DataLoader(dataset,batch_size=config.batch_size, num_workers=os.cpu_count() )
  auc_result, f1_score_result, accuracy_result, y, y_pred, y_pred_labels = evaluate(model,device,config, loader,sigmoid, threshoulding, metrics=[auc_object,f1_score])
  if config.log_preds:
    # pass
    # print(split_type)
    creatPredictionTable(config, loader, y, y_pred_labels)
    
  # print(y)
  # print(y_pred)

  # wandb.log({"roc_curve" : wandb.plot.roc_curve(y, y_pred, labels=['Benign','Malignant'])})
  creatConfutionMatrix(y, y_pred_labels)
  logBestMertics(best_f1_score_metric, best_AUC_metric, best_acc_metric )
  logTraindModel(saved_model_path, config)
  wandb.finish()


if __name__ == '__main__':

  default_config= None
  with open('configeration_file.json') as json_file:
    default_config = json.load(json_file)
  parse_args(default_config)
  set_determinism(default_config['seed'],use_deterministic_algorithms=False)
  train(default_config)





















