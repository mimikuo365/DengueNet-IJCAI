import tensorflow as tf
import numpy as np
import keras
import keras_tuner
from pathlib import Path
import os

from DengueNet import *
from DataPreprocess import *

def getResultLocation(args, folder, name, append_name):
    path = ""
    if args.fixed_hyper:
        path = Path(folder) / name / append_name
    else:
        path = Path(folder) / name / "tuner"
    return path

def prepareTimeSeries(features: np.array, n_in: int) -> np.array:
    """Create time series from features with given window size.
    
    Args:
        n_in: Given window size.
    """
    print(features.shape)
    return np.stack(
        [ features[idx:idx+n_in] for idx in range(0, features.shape[0] - n_in) ],
        axis=0
    )

def createModel(setting: dict, leaky_rate):
    return DengueNet(setting, leaky_rate).create()

def createTuner(setting: dict, name: str, overwrite=False) -> keras_tuner.BayesianOptimization:
    """Return Keras Tuner for model hyper-parameter tunning.
    """
    print(setting['model_folder'] / name)
    return  keras_tuner.BayesianOptimization(
              hypermodel=DengueNet(setting),
              overwrite=overwrite,
              objective=keras_tuner.Objective("val_mean_absolute_error", direction="min"),
              directory=setting['model_folder'],
              project_name=name,
              max_trials=3
            )
    
def getModels(include_vit=False, only_vit=False):
    if only_vit:
        return {
            "FengVit":          8,  # RGB images, # Radiomics
            # "Vit":              1,  # RGB images
            # "VitFengCase":      10,  # RGB images, # Radiomics, # Cases  
        }
    elif not include_vit:
        return {
            # "Mobile":           3,  # RGB images
            # "Case":             4,  #                          # Cases
            "Feng":             2,  #             # Radiomics
            # "FengMobile":       7,  # RGB images, # Radiomics
        }
        
    else:
        return {
            "Vit":              1,  # RGB images
            "FengVit":          8,  # RGB images, # Radiomics
            "VitFengCase":      10,  # RGB images, # Radiomics, # Cases  
            "Feng":             2,  #             # Radiomics
            "Case":             4,  #                          # Cases
            # "FengMobileVit":    5,  # RGB images, # Radiomics -> have error
            # "MobileVit":        6,  # RGB images
            # "Mobile":           3,  # RGB images
            # "FengMobile":       7,  # RGB images, # Radiomics
            # "Ensemble":         9,  # RGB images, # Radiomics, # Cases
        }
    return models

def getLeakyRateList():
    # return [0.1, 0.15, 0.175, 0.2, 0.2125, 0.225, 0.2375, 0.25, 0.2625, 0.275]
    return [0.1, 0.15, 0.2, 0.225, 0.25, 0.275]

def getModelList():
    """Return the list of models to explore.
    """
    return ["Vit", "DengueNet", "FeatureEng", "MobileNet", "Case", "Ensemble", "DengueNetV2", "MobileVit"]

# This function keeps the initial learning rate for the first ten epochs
# and decreases it exponentially after that.
def scheduler(epoch, lr=1e-3):
  if epoch < 20:
    return lr
  else:
    return lr * tf.math.exp(-0.1)

def scheduler3(epoch, lr=1e-3):
    if epoch < 20:
    # if epoch < 5:
        return lr
    # elif epoch < 10:
    #     return 1e-4
    else:
        return lr * tf.math.exp(-0.1)

def getCallback(patience=10):
    """Get model callback.
    """
    callback = [
        tf.keras.callbacks.LearningRateScheduler(scheduler),
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience)
    ]
    return callback

def scheduler2(epoch, lr=1e-4):
# def scheduler2(epoch, lr=1e-4):
  if epoch < 20:
    return lr
  else:
    return lr * tf.math.exp(-0.1)

def getCallback2(patience=10):
    """Get model callback.
    """
    
    callback = [
        tf.keras.callbacks.LearningRateScheduler(scheduler2),
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience)
    ]
    return callback

def getInput(model_name, data, labels, splits=['train', 'val', 'test']):
    """Generate the input and output for a certain model.
    """
    selected_input=[]
    model_idx = getModels(True)[model_name]
    if model_idx == 1:
        selected_input = ["x_vit"]
    elif model_idx == 2:
        selected_input = ['x_feng']
    elif model_idx == 3:
        selected_input = ["x_cnn"]
    elif model_idx == 4:
        selected_input = ['case']
    elif model_idx == 5:
        selected_input = ['x_feng', 'x_cnn', 'x_vit']
    elif model_idx == 6:
        selected_input = ['x_cnn', 'x_vit']
    elif model_idx == 7:
        selected_input = ['x_feng', 'x_cnn']
    elif model_idx == 8:
        selected_input = ['x_feng', 'x_vit']
    elif model_idx == 9:
        selected_input = ['x_feng', 'x_cnn', 'x_vit', 'case']
    elif model_idx == 10:
        selected_input = ['x_feng', 'x_vit', 'case']
    else:
        print(f"Error: Not existed model {model_name}")
    
    new_data = {}
    for split in splits:
        new_data[split] = {}
        new_data[split]['output'] = labels[split]
        new_data[split]['origin_output'] = labels[f'origin_{split}']
        
        if len(selected_input) == 1:
            input = selected_input[0]
            new_data[split]['input'] = data[split][input]
        else:
            new_data[split]['input'] = []
            for input in selected_input:
                new_data[split]['input'].append(data[split][input])
            
    return new_data