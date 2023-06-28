import os
from pathlib import Path
# from skimage import io
# import numpy as np
import pandas as pd
# import logging
import sys
import argparse
import tensorflow as tf
# from tensorflow import keras
import pickle 
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

tf.get_logger().setLevel('ERROR')
tf.keras.utils.disable_interactive_logging()


root_dir = Path(__file__).parents[2]
sys.path.insert(0, str(root_dir / 'code/util'))
sys.path.insert(0, str(root_dir / 'code/model'))

from DengueNet import *
from DataPreprocess import *
from ModelUtil import *
# from ModelUtil import getModelList, getInput, createTuner, getCallback, createModel, getResultLocation
from ForFeatureEngineering import getFeatureList, getThresholdPairList
from CalculateMetric import *

def evalSelectedFeature(args, model_name, city, threshold_dic, selected_feature_ls, cloud, shadow, include_top, swap, all_feature_ls):
    feature_folder_name = getFeatureFolderName(all_feature_ls, selected_feature_ls)
    setting = getSetting(args.project_folder, swap, model_name, city, 
                        include_top=include_top, threshold_dic=threshold_dic, 
                        selected_feature_ls=selected_feature_ls,
                        threshold_cloud=cloud, threshold_shadow=shadow,
                        feature_folder_name=feature_folder_name)
    leaky_rate = 0.2
    len_callback = 10
    learning_rate = 1e-4
    normalize_feature = False
    callback = 1
    
    result_location = Path(f"/mnt/usb/jupyter-mimikuo/feature_{learning_rate}_{swap}/callback{callback}/Normalize_{normalize_feature}/{city}/{selected_feature_ls}")
    # result_location = setting["result_folder"] / f"{model_name}_True/{leaky_rate}"
    print("result_location:", result_location)
    
    if os.path.exists(result_location):
        try:
          print(f"Error!! Folder {result_location} already existed!!")
          with open(f'{result_location}/performance.pkl', 'rb') as f:
              performance = pickle.load(f)
          return performance
        except:
          print(f"Error!! Folder {result_location} not existed!!")
            
    checkFolder(result_location)
    setting, data, labels, scaler = setup(setting, city, normalize_feature=normalize_feature)
    data = getInput(setting["model"], data, labels)
    
    printTime("Model Training (fixed parameter)", 1)
    model = createModel(setting, leaky_rate)
    model_parameter = DengueNet.getHyperparameter(leaky_rate=0.2, vit_trainable=False, learning_rate=learning_rate) 
    
    if callback == 1:
      history = model.fit(data['train']['input'], data['train']['output'], epochs=100, verbose=0,
                          validation_data=(data['val']['input'], data['val']['output']), 
                          callbacks=getCallback(len_callback))
    else:
      history = model.fit(data['train']['input'], data['train']['output'], epochs=100, verbose=0,
                          validation_data=(data['val']['input'], data['val']['output']), 
                          callbacks=getCallback2(len_callback))
      
    printTime("Model Training (fixed parameter)")
              
    date = datetime.now().strftime("%H:%M:%S_%d-%m-%Y") + ".log"
    f = open(result_location / str(date), "w")
    plotModelStructure(model, result_location)
    performance = evalModel(model, data, f, result_location, scaler, history, len_callback)
    performance["features"] = selected_feature_ls
    
    with open(f'{result_location}/performance.pkl', 'wb') as f:
        pickle.dump(performance, f)
    return performance

def getFeatureFolderName(all_feature_ls, selected_feature_ls):
  folder = ""
  for feature in selected_feature_ls:
    if folder == "":
      folder = str(list(all_feature_ls).index(feature))
    else:
      folder = folder + "_" + str(list(all_feature_ls).index(feature))
  print(folder)
  return folder

def main(args):
    model_name = "Feng"
    swap = True
    include_top = False
    cloud = 70
    shadow = 10
    # city_ls = getCityList()
    city_ls = ['5001', '50001', '73001', '76001', '54001']
    
    if args.city:
      print(city_ls)
      city_ls = [city_ls[args.city]]
    
    for city in city_ls:
      best_feature_ls = []
      min_performance = {
        "feature": "",
        "mae": 10000
      }
      found_better = True
      path = Path(f"/mnt/usb/jupyter-mimikuo/dataset/sorted/{city}")
      df = pd.read_csv(path / "label.csv")
      img_ls, _ = readImgs(df, path)
      all_feature_ls = readFeatureName(df, path)
      threshold_dic = getThresholds(img_ls)
      
      df = pd.DataFrame(
        data={
            "index": range(78),
            "mae": range(78),
            "feature": [list() for _ in range(78)]
        }
      )
      
      while found_better:
        found_better = False
        second_min_performance = {
            "feature": "",
            "mae": 10000
          }
        for feature in all_feature_ls:
          if feature in best_feature_ls:
            continue
          selected_feature_ls = best_feature_ls + [feature]
          print(city)
          print("selected_feature_ls:", selected_feature_ls)
          performance = evalSelectedFeature(args, model_name, city, threshold_dic, 
                                            selected_feature_ls, cloud, shadow, include_top, 
                                            swap, all_feature_ls)
          cur_mae = performance['val']['MAE']
          # cur_mae = performance['train']['MAE'] * 0.3 + performance['val']['MAE'] * 0.7
          
          # if cur_mae <= min_performance["mae"]:
          #   min_performance["mae"] = cur_mae
          #   min_performance["feature"] = feature
          #   found_better = True
            
          # if cur_mae <= second_min_performance["mae"]:
          #   second_min_performance["mae"] = cur_mae
          #   second_min_performance["feature"] = feature
            
        # if found_better:
        #   found_better = True
        #   best_feature_ls = best_feature_ls + [min_performance["feature"]]
        # elif len(best_feature_ls) < 30:
        #   found_better = True
        #   best_feature_ls = best_feature_ls + [second_min_performance["feature"]]
        #   min_performance["mae"] = second_min_performance["mae"]
        # if found_better:
        #   print("best_feature_ls:", best_feature_ls)
        #   df.at[len(best_feature_ls), "feature"] += best_feature_ls
        #   df.at[len(best_feature_ls), "mae"] += min_performance["mae"]
      
        print(df.head(10))
        checkFolder(Path(f"/mnt/usb/jupyter-mimikuo/feature2/best/"))
        df.to_csv(f'/mnt/usb/jupyter-mimikuo/feature2/best/{city}.csv', index=False)
      
if __name__ == "__main__":
    args = setParser(select_city=True)
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_index
    gpus = tf.config.list_physical_devices('GPU')
    print(gpus)
    if gpus:
      try:
        if args.limit_gpu:
          tf.config.set_logical_device_configuration(
              gpus[0],
              [tf.config.LogicalDeviceConfiguration(memory_limit=args.gpu_size*1024)])
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
      except RuntimeError as e:
        print(e)

    main(args)