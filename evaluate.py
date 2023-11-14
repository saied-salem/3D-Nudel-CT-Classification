import torch 
import json
import argparse, os
import wandb
from sklearn.metrics import accuracy_score, f1_score, classification_report

from torch import optim
from medmnist import NoduleMNIST3D
from utils import getNetworkArch,  creatConfutionMatrix, creatPredictionTable
from monai.data import DataLoader
from monai.metrics import ROCAUCMetric
from monai.transforms import Activations, AsDiscrete
from monai.utils import set_determinism
from types import SimpleNamespace



default_config = SimpleNamespace(
    framework="pytorch",
    batch_size=256,
    mixed_precision=False,
    arch= 'resnet10',
    seed=42,
    log_preds=True,
    log_pred_type="val",
    as_rgb=True,
    scheduler_reducing_factor=0.8,
    scheduler_lr_patience=4
)

def t_or_f(arg):
    ua = str(arg).upper()
    if 'TRUE'.startswith(ua): return True
    else: return False

def parse_args():
    argparser = argparse.ArgumentParser(description='Process hyper-parameters')
    argparser.add_argument('--model_name', default=None, type=str, help='Enter the name of the model you need to evaluate')
    argparser.add_argument('--offline_mode', default=False, type=t_or_f, help='whether to use model config from wandb artifacts or local')

    args = argparser.parse_args()

    return args.model_name, args.offline_mode

@torch.inference_mode()
def evaluate(model, device, config, val_loader,sigmoid, threshoulding, metrics):
  auc_object, f1_score = metrics
  model.eval()
  y_pred = torch.tensor([], dtype=torch.float32, device=device)
  y = torch.tensor([], dtype=torch.long, device=device)
  with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=config.mixed_precision):
    for inputs, labels in val_loader:
        val_images, val_labels = inputs.to(device, dtype=torch.float32), labels.to(device, dtype=torch.float32)
        y_pred = torch.cat([y_pred, model(val_images)], dim=0)
        y = torch.cat([y, val_labels], dim=0)

    y = y.squeeze()
    y_pred = y_pred.squeeze()
    y_pred = sigmoid(y_pred)
    auc_object(y_pred,y)
    auc_result = auc_object.aggregate()
    auc_object.reset()
    
    y_pred_labels= threshoulding(y_pred)
    acc_value = torch.eq(y_pred_labels, y)
    acc_results= acc_value.sum().item() / len(acc_value)
    y = y.squeeze().cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    y_pred_labels = y_pred_labels.cpu().numpy()

    f1_score_result = f1_score(y , y_pred_labels)
    model.train()
  return auc_result, f1_score_result, acc_results, y, y_pred, y_pred_labels

def init(model_name, offline):
  model_path= None
  config= default_config
  if offline:
    curr_folder = os.getcwd()
    config = default_config
    model_name_path =  [ path for path in os.listdir(curr_folder)  if 'pth' in path][0]
    model_path = os.path.join(curr_folder,model_name_path)
    network_arch = model_name_path.split('_')[0]
    config.arch = network_arch
  else:
    run = wandb.init(project='NoduleMNIST', entity='saied-salem',
                                resume='allow', job_type="evaluation")


    artifact = run.use_artifact(model_name,type='model')
    artifact_dir =artifact.download()
    print(artifact_dir)
    model_name_path = os.listdir(artifact_dir)[0]
    model_path = os.path.join(artifact_dir,model_name_path)
    producer_run = artifact.logged_by()
    wandb.config.update(producer_run.config)
    config = wandb.config

  set_determinism(config.seed,use_deterministic_algorithms=False)

  device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  input_channels = 3 if config.as_rgb else 1
  model= getNetworkArch(config.arch, input_channels).to(device)
  print(model_path)
  model.load_state_dict(torch.load(model_path, map_location=device))

  sigmoid = Activations(sigmoid=True)
  threshoulding = AsDiscrete(threshold=0.5)
  auc_object = ROCAUCMetric()

  dataset = NoduleMNIST3D(split='test', download=True, as_rgb=config.as_rgb)
  loader = DataLoader(dataset,batch_size=config.batch_size, num_workers=os.cpu_count() )
  auc_result, f1_score_result, accuracy_result, y, y_pred, y_pred_labels = evaluate(model,device,config, loader, sigmoid, threshoulding, metrics=[auc_object,f1_score])

  creatPredictionTable(config, loader, y, y_pred_labels)
  creatConfutionMatrix(y, y_pred_labels)

  # wandb.log({"roc_curve" : wandb.plot.roc_curve(y, y_pred, labels=['Benign','Malignant'])})
  run.log({
        'AUC_score': auc_result,
        'F1_score': f1_score_result,
        'accuracy_result':accuracy_result,
  })
  wandb.define_metric("AUC_score", summary="max")
  wandb.define_metric("F1_score", summary="max")
  wandb.define_metric("accuracy_result", summary="max")
  
  wandb.finish()
  
  print(classification_report(y, y_pred_labels, target_names=['benign','maliganit'], digits=4))
  
  return auc_result, f1_score_result, accuracy_result



@torch.inference_mode()
def evaluate_working(model, device, config, val_loader,sigmoid, threshoulding, metrics):
  auc_object, f1_score = metrics
  model.eval()
  y_pred = torch.tensor([], dtype=torch.float32, device=device)
  y = torch.tensor([], dtype=torch.long, device=device)
  with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=config.mixed_precision):
    for inputs, labels in val_loader:
        val_images, val_labels = inputs.to(device, dtype=torch.float32), labels.to(device, dtype=torch.float32)
        y_pred = torch.cat([y_pred, model(val_images)], dim=0)
        y = torch.cat([y, val_labels], dim=0)

    y = y.squeeze()
    y_pred = y_pred.squeeze()
    y_pred = sigmoid(y_pred)
    auc_object(y_pred,y)
    auc_result = auc_object.aggregate()
    auc_object.reset()
    
    y_pred= threshoulding(y_pred)
    acc_value = torch.eq(y_pred, y)
    acc_results= acc_value.sum().item() / len(acc_value)
    y = y.squeeze().cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    f1_score_result = f1_score(y , y_pred)
    model.train()
  return auc_result, f1_score_result, acc_results, y, y_pred

if __name__ == '__main__':
    model_name, offline = parse_args()
    init(model_name,offline)
    




