
import os
import numpy as np
import pandas as pd
from medmnist import NoduleMNIST3D
import wandb
from sklearn.metrics import ConfusionMatrixDisplay
from monai.data import DataLoader
from monai.networks import nets
from monai.transforms import (
    Activations,
    EnsureChannelFirst,
    AsDiscrete,
    Compose,
    LoadImage,
    RandFlip,
    RandRotate,
    RandBiasField,
    RandZoom,
    ScaleIntensity,
    HistogramNormalize,
    RandAdjustContrast,
    RandBiasField,
    RandGaussianSharpen,
    RandGaussianSmooth,
    ToTensor,
)

def creatPredictionTableAndConfutionMatrix(config, loader, y, y_pred):
  pred_table = {
    'Images': [ wandb.Video(np.transpose(img*255, (3, 0, 1,2)).astype(np.uint8))  for img,_ in iter(loader.dataset)],
    "Pred_labels": [ "malignant" if label== 1 else "benign"  for label in  y_pred ],
    "True_labels": [ "malignant" if label== 1 else "benign"  for label in  y ]
}
  # print("HHHHHHHHHHHH")
  wandb_pred_table = wandb.Table(dataframe=pd.DataFrame(pred_table))
  wandb.log({"pred_table":wandb_pred_table})

  disp = ConfusionMatrixDisplay.from_predictions(y_true=y, y_pred=y_pred,
                                                   display_labels=['Benign','Malignant'],
                                                   normalize='pred')
  fig = disp.ax_.get_figure()
  fig.set_figwidth(10)
  fig.set_figheight(10) 
  disp.ax_.set_title('Confusion Matrix (by Pixels)', fontdict={'fontsize': 32, 'fontweight': 'medium'})
  fig.show()
  fig.autofmt_xdate(rotation=45)
  wandb.log({'confusion_matrix': disp.figure_})


def logBestMertics(best_f1_score_metric, best_AUC_metric, best_acc_metric ):
  wandb.summary['F1_score'] = best_f1_score_metric
  wandb.summary['AUC_score'] = best_AUC_metric
  wandb.summary['accuracy_result'] = best_acc_metric


def transformsType(augment):
  transformations = []
  if augment:
    transformations=[          
          # RandBiasField(),
          # RandAdjustContrast(prob=0.3, gamma=(0.4, 1.5)),
          RandGaussianSharpen(prob = 0.7),
          RandGaussianSmooth(prob=0.3),
          RandFlip(prob=0.3),
          RandRotate(prob=0.3,range_x =1,range_y =1,range_z =1),
          # ScaleIntensity(0,255),
          ToTensor(),]
  else:
    transformations =[ToTensor()]

  return transformations

def getDataset(augment, batch_size, as_rgb):
  transformations= transformsType(augment)
  train_transform = Compose(transformations)
  train_dataset = NoduleMNIST3D(split='train', transform=train_transform, download=True,as_rgb=as_rgb)
  val_dataset = NoduleMNIST3D(split='val', download=True, as_rgb=as_rgb)
  train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True, num_workers=os.cpu_count(), pin_memory=True )
  val_loader =  DataLoader(val_dataset,batch_size=batch_size,shuffle=True, num_workers=os.cpu_count(), pin_memory=True )

  return train_loader, val_loader

def logTraindModel(model_path,config):
  trained_model_artifact = wandb.Artifact(
            config.arch, type="model",
            description= os.path.basename(model_path),
            metadata=dict(config))

  trained_model_artifact.add_file(model_path)
  wandb.log_artifact(trained_model_artifact)

def getNetworkArch(arch_name,input_channels):
  input_channel_names = ['n_input_channels', 'in_channels' ]
  output_class_names = ['num_classes']
  network= None
  try:
    network=getattr(nets, arch_name)(spatial_dims= 3, in_channels= input_channels, num_classes=1)
  except:
    network= getattr(nets, arch_name)(spatial_dims= 3, in_channels= input_channels, num_classes=1)
  return network








