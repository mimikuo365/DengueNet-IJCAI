import os
from pathlib import Path
import pandas as pd
import sys
import argparse
import tensorflow as tf
from datetime import datetime

root_dir = Path(__file__).parents[2]
sys.path.insert(0, str(root_dir / 'code/util'))
sys.path.insert(0, str(root_dir / 'code/model'))

from DengueNet import *
from DataPreprocess import *
from ModelUtil import *
from ForFeatureEngineering import getFeatureList, getThresholdPairList
from CalculateMetric import evalModel, plotModelStructure

def main(args):
    selected_feature_ls = getFeatureList()
    leaky_rate_ls = getLeakyRateList()
    vit_norm_ls = [False]
    threshold_pair_ls = list(getThresholdPairList(args))
    lstm_week_ls = [5]
    df_columns = createColumns()
    len_callback = 10
    learning_rate_ls = [1e-3]
    date = datetime.now().strftime("%H:%M:%S_%d-%m-%Y") + ".log"
    
    if args.only_vit:
      model_dic = getModels(only_vit=True)
    elif args.include_vit:
      print("Include ViT model!!")
      model_dic = getModels(True)
    else:
      model_dic = getModels(False)
    city_ls = getCityList(args.reverse_city_sequence)
    
    if args.city:
      city_ls = [city_ls[args.city]]
      
    for city in city_ls:
      path = Path(f"/mnt/usb/jupyter-mimikuo/dataset/sorted/{city}")
      df = pd.read_csv(path / "label.csv")
      all_feature_ls = readFeatureName(df, path)
      # threshold_path = path / "threshold.pickle"
    
      # if os.path.exists(threshold_path):
      #     with open(threshold_path, 'rb') as f:
      #         threshold_dic = pickle.load(f)
      # else:
      #   img_ls, _ = readImgs(df, path)
      #   threshold_dic = getThresholds(img_ls)
      
      for model_name in model_dic.keys():
        swap_bool_ls = [True, False]
        if model_name == "Case":
          swap_bool_ls = [False]
        
        for swap in swap_bool_ls:
          for vit_norm in vit_norm_ls:
            for cloud, shadow in threshold_pair_ls:
              for vit_trainable in [True]: # Use trainable ViT model
                for learning_rate in learning_rate_ls:
                # for lstm_week in lstm_week_ls: 
                  lstm_week = lstm_week_ls[0]
                  result_df = pd.DataFrame(columns=df_columns)
                  for leaky_rate in leaky_rate_ls:
                    
                    feature_folder_name = getFeatureFolderName(all_feature_ls, selected_feature_ls)
                    setting = getSetting(args.project_folder, swap, model_name, city, 
                                        include_top=False, threshold_dic=None, 
                                        selected_feature_ls=selected_feature_ls,
                                        threshold_cloud=cloud, threshold_shadow=shadow,
                                        feature_folder_name=feature_folder_name, lstm_week=lstm_week)
                    selected_model =  str(lstm_week) + "/" + setting['model']+ "_" + str(swap) + "/" + str(learning_rate) + "_" + str(len_callback) + "_callback1/" 
                    if vit_norm:
                      selected_model += "NormImg"
                    else:
                      selected_model += "FixedOriginImg2"
                      
                    df_location = Path(setting["result_folder"]) / selected_model / f"result_{learning_rate}.csv"
                    model_parameter = DengueNet.getHyperparameter(leaky_rate=leaky_rate, vit_trainable=vit_trainable, learning_rate=learning_rate) 
                    model_location = getResultLocation(args, setting["model_folder"], selected_model, model_parameter)
                    result_location = getResultLocation(args, setting["result_folder"], selected_model, model_parameter)
                    print(model_location)
                    print(result_location)
                    
                    if os.path.exists(result_location / "history.png"):
                      print(f"Error!! Folder {result_location} already existed!!")
                      continue
                    
                    checkFolder(result_location)
                    setting, data, labels, scaler = setup(setting, city, normalize_vit=vit_norm)
                    data = getInput(model_name, data, labels)
                    
                    if args.fixed_hyper:
                      printTime("Model Training (fixed parameter)", 1)
                      model = createModel(setting, leaky_rate)
                      history = model.fit(data['train']['input'], data['train']['output'], epochs=100, batch_size=8,
                                          validation_data=(data['val']['input'], data['val']['output']),
                                          callbacks=getCallback(len_callback))
                      printTime("Model Training (fixed parameter)")
                      
                      # printTime("Model Save", 1)
                      # model.save(model_location / "h5_model.h5", save_format="h5")
                      # printTime("Model Save")
                      
                      printTime("Model Evaluation", 1)
                      f = open(result_location / str(date), "w")
                      plotModelStructure(model, result_location)
                      performance = evalModel(model, data, f, result_location, scaler, history, len_callback)
                      printTime("Model Evaluation")
                      f.close()

                      tmp_dic = {"leaky_rate": leaky_rate}
                      for split in performance.keys():
                        for metric in performance[split].keys():
                          tmp_dic[f"{split}_{metric}"] = performance[split][metric]
                      result_df = result_df.append(tmp_dic, ignore_index=True)
                      
                      # if os.path.exists(df_location):
                      #   pre_result_df = pd.read_csv(df_location)
                      #   result_df = result_df.append(pre_result_df, ignore_index=True)                   
                      # result_df.to_csv(df_location, index=False)

                    else:
                      printTime("Tuner Searching", 1)
                      tuner = createTuner(setting, selected_model)
                      tuner.search(data['train']['input'], data['train']['output'], epochs=100, 
                                  #  verbose=0,
                                  validation_data=(data['val']['input'], data['val']['output']), callbacks=getCallback(len_callback))
                      printTime("Tuner Searching")
                      
                      printTime("Model Evaluation", 1)
                      f = open(result_location / str(date), "w")
                      best_hps=createTuner(setting, selected_model).get_best_hyperparameters()[0]
                      model = DengueNet(setting).build(best_hps) 
                      printTime("Model Training", 1)
                      history = model.fit(data['train']['input'], data['train']['output'], epochs=30, verbose=0,
                                          validation_data=(data['val']['input'], data['val']['output']), 
                                          callbacks=getCallback())
                      printTime("Model Training")
                      plotModelStructure(model, result_location)
                      evalModel(model, data, f, result_location, scaler, history)
                      printTime("Model Evaluation")
                  # if os.path.exists(df_location):
                  #   pre_result_df = pd.read_csv(df_location)
                  #   result_df = result_df.append(pre_result_df, ignore_index=True)                   
                  if len(result_df) != 0:
                    result_df.to_csv(df_location, index=False)

if __name__ == "__main__":
    args = setParser(True)
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_index
    gpus = tf.config.list_physical_devices('GPU')

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
    
# import os
# from pathlib import Path
# # from skimage import io
# # import numpy as np
# import pandas as pd
# # import logging
# import sys
# import argparse
# import tensorflow as tf
# # from tensorflow import keras
# import pickle 
# from datetime import datetime
# tf.get_logger().setLevel('ERROR')
# tf.keras.utils.disable_interactive_logging()


# root_dir = Path(__file__).parents[2]
# sys.path.insert(0, str(root_dir / 'code/util'))
# sys.path.insert(0, str(root_dir / 'code/model'))

# from DengueNet import *
# from DataPreprocess import *
# from ModelUtil import getModelList, getInput, createTuner, getCallback, createModel, getResultLocation
# from ForFeatureEngineering import getFeatureList, getThresholdPairList
# from CalculateMetric import *

# def evalSelectedFeature(args, model_name, city, threshold_dic, selected_feature_ls, cloud, shadow, include_top, swap, all_feature_ls):
#     feature_folder_name = getFeatureFolderName(all_feature_ls, selected_feature_ls)
#     setting = getSetting(args.project_folder, swap, model_name, city, 
#                         include_top=include_top, threshold_dic=threshold_dic, 
#                         selected_feature_ls=selected_feature_ls,
#                         threshold_cloud=cloud, threshold_shadow=shadow,
#                         feature_folder_name=feature_folder_name)
#     result_location = setting["result_folder"] / f"{model_name}_True"
#     print("result_location:", result_location)
#     if os.path.exists(result_location):
#         try:
#           print(f"Error!! Folder {result_location} already existed!!")
#           with open(f'{result_location}/performance.pkl', 'rb') as f:
#               performance = pickle.load(f)
#           return performance
#         except:
#           print(f"Error!! Folder {result_location} not existed!!")
            
#     checkFolder(result_location)
#     setting, data, labels, scaler = setup(setting, city)
#     data = getInput(setting["model"], data, labels)
    
#     printTime("Model Training (fixed parameter)", 1)
#     model = createModel(setting)
#     history = model.fit(data['train']['input'], data['train']['output'], epochs=100, verbose=0,
#                         validation_data=(data['val']['input'], data['val']['output']), 
#                         callbacks=getCallback())
#     printTime("Model Training (fixed parameter)")
              
#     date = datetime.now().strftime("%H:%M:%S_%d-%m-%Y") + ".log"
#     f = open(result_location / str(date), "w")
#     plotModelStructure(model, result_location)
#     performance = evalModel(model, data, f, result_location, scaler, history)
#     performance["features"] = selected_feature_ls
    
#     with open(f'{result_location}/performance.pkl', 'wb') as f:
#         pickle.dump(performance, f)
#     return performance

# def getFeatureFolderName(all_feature_ls, selected_feature_ls):
#   folder = ""
#   for feature in selected_feature_ls:
#     if folder == "":
#       folder = str(list(all_feature_ls).index(feature))
#     else:
#       folder = folder + "_" + str(list(all_feature_ls).index(feature))
#   print(folder)
#   return folder

# def main(args):
#     model_name = "Feng"
#     swap = True
#     include_top = False
#     cloud = 60
#     shadow = 20
#     city_ls = getCityList()
    
#     if args.city:
#       print(city_ls)
#       city_ls = [city_ls[args.city]]
    
#     for city in city_ls:
#       best_feature_ls = []
#       min_performance = {
#         "feature": "",
#         "mae": 10000
#       }
#       found_better = True
#       path = Path(f"/mnt/usb/jupyter-mimikuo/dataset/sorted/{city}")
#       df = pd.read_csv(path / "label.csv")
#       all_feature_ls = readFeatureName(df, path)
#       threshold_path = path / "threshold.pickle"
    
#       if os.path.exists(threshold_path):
#           with open(threshold_path, 'rb') as f:
#               threshold_dic = pickle.load(f)
#       else:
#         img_ls, _ = readImgs(df, path)
#         threshold_dic = getThresholds(img_ls)
      
      
#       df = pd.DataFrame(
#         data={
#             "index": range(78),
#             "mae": range(78),
#             "feature": [list() for _ in range(78)]
#         }
#       )
      
#       while found_better:
#         found_better = False
#         second_min_performance = {
#             "feature": "",
#             "mae": 10000
#           }
#         for feature in all_feature_ls:
#           if feature in best_feature_ls:
#             continue
#           selected_feature_ls = best_feature_ls + [feature]
#           print(city)
#           print("selected_feature_ls:", selected_feature_ls)
#           performance = evalSelectedFeature(args, model_name, city, threshold_dic, 
#                                             selected_feature_ls, cloud, shadow, include_top, 
#                                             swap, all_feature_ls)
#           cur_mae = performance['train']['MAE'] * 0.3 + performance['val']['MAE'] * 0.7
          
#           if cur_mae <= min_performance["mae"]:
#             min_performance["mae"] = cur_mae
#             min_performance["feature"] = feature
#             found_better = True
            
#           if cur_mae <= second_min_performance["mae"]:
#             second_min_performance["mae"] = cur_mae
#             second_min_performance["feature"] = feature
            
#         if found_better:
#           found_better = True
#           best_feature_ls = best_feature_ls + [min_performance["feature"]]
#         elif len(best_feature_ls) < 30:
#           found_better = True
#           best_feature_ls = best_feature_ls + [second_min_performance["feature"]]
#           min_performance["mae"] = second_min_performance["mae"]
#         if found_better:
#           print("best_feature_ls:", best_feature_ls)
#           df.at[len(best_feature_ls), "feature"] += best_feature_ls
#           df.at[len(best_feature_ls), "mae"] += min_performance["mae"]
      
#         print(df.head(10))
#         df.to_csv(f'/mnt/usb/jupyter-mimikuo/feature/best/{city}.csv', index=False)
      
# if __name__ == "__main__":
#     args = setParser(select_city=True)
#     os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_index
#     gpus = tf.config.list_physical_devices('GPU')
#     print(gpus)
#     if gpus:
#       try:
#         if args.limit_gpu:
#           tf.config.set_logical_device_configuration(
#               gpus[0],
#               [tf.config.LogicalDeviceConfiguration(memory_limit=args.gpu_size*1024)])
#         logical_gpus = tf.config.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#       except RuntimeError as e:
#         print(e)

#     main(args)
# import os
# from pathlib import Path
# # from skimage import io
# # import numpy as np
# import pandas as pd
# # import logging
# import sys
# import argparse
# import tensorflow as tf
# # from tensorflow import keras
# from datetime import datetime
# # tf.get_logger().setLevel('ERROR')

# root_dir = Path(__file__).parents[2]
# sys.path.insert(0, str(root_dir / 'code/util'))
# sys.path.insert(0, str(root_dir / 'code/model'))

# from DengueNet import *
# from DataPreprocess import *
# from ModelUtil import *
# from ForFeatureEngineering import getFeatureList, getThresholdPairList
# from CalculateMetric import evalModel, plotModelStructure

# def main(args):
#     selected_feature_ls = getFeatureList()
#     leaky_rate_ls = getLeakyRateList()
#     vit_norm_ls = [False]
#     threshold_pair_ls = list(getThresholdPairList(args))
#     lstm_week_ls = [5]
#     # lstm_week_ls = [5, 7, 9]
#     df_columns = createColumns()
    
#     date = datetime.now().strftime("%H:%M:%S_%d-%m-%Y") + ".log"
#     if args.include_vit:
#       print("Include ViT model!!")
#       model_dic = getModels(True)
#     else:
#       model_dic = getModels(False)
#     city_ls = getCityList(args.reverse_city_sequence)
#     len_callback = 30
#     # len_callback = 20
#     # learning_rate = 0.00005
#     learning_rate = 1e-4
    
#     if args.city:
#       city_ls = [city_ls[args.city]]
      
#     for city in city_ls:
#       path = Path(f"/mnt/usb/jupyter-mimikuo/dataset/sorted/{city}")
#       df = pd.read_csv(path / "label.csv")
#       all_feature_ls = readFeatureName(df, path)
#       threshold_path = path / "threshold.pickle"
    
#       if os.path.exists(threshold_path):
#           with open(threshold_path, 'rb') as f:
#               threshold_dic = pickle.load(f)
#       else:
#         img_ls, _ = readImgs(df, path)
#         threshold_dic = getThresholds(img_ls)
      
#       for model_name in model_dic.keys():
#         swap_bool_ls = [True, False]
#         if model_name == "Case":
#           swap_bool_ls = [False]
        
#         for swap in swap_bool_ls:
#           for vit_norm in vit_norm_ls:
#             for cloud, shadow in threshold_pair_ls:
#               for vit_trainable in [True]: # Use trainable ViT model
#                 for lstm_week in lstm_week_ls: 
#                   result_df = pd.DataFrame(columns=df_columns)
#                   for leaky_rate in leaky_rate_ls:
#                     feature_folder_name = getFeatureFolderName(all_feature_ls, selected_feature_ls)
#                     setting = getSetting(args.project_folder, swap, model_name, city, 
#                                         include_top=False, threshold_dic=threshold_dic, 
#                                         selected_feature_ls=selected_feature_ls,
#                                         threshold_cloud=cloud, threshold_shadow=shadow,
#                                         feature_folder_name=feature_folder_name, lstm_week=lstm_week)
#                     selected_model =  str(lstm_week) + "/" + setting['model']+ "_" + str(swap) + "/"
#                     if vit_norm:
#                       selected_model += "NormImg"
#                     else:
#                       selected_model += "OriginImg"
                      
#                     df_location = Path(setting["result_folder"]) / selected_model / "result_e-3.csv"
#                     model_parameter = DengueNet.getHyperparameter(leaky_rate=leaky_rate, vit_trainable=vit_trainable, learning_rate=learning_rate) + "_" + str(len_callback)
#                     model_location = getResultLocation(args, setting["model_folder"], selected_model, model_parameter)
#                     result_location = getResultLocation(args, setting["result_folder"], selected_model, model_parameter)
#                     print(model_location)
#                     print(result_location)
                    
#                     if os.path.exists(result_location / "history.png"):
#                       print(f"Error!! Folder {result_location} already existed!!")
#                       continue
                    
#                     checkFolder(result_location)
#                     # checkFolder(model_location)
                    
#                     setting, data, labels, scaler = setup(setting, city, normalize_vit=vit_norm)
#                     data = getInput(model_name, data, labels)
                    
#                     if args.fixed_hyper:
#                       printTime("Model Training (fixed parameter)", 1)
#                       model = createModel(setting, leaky_rate)
#                       history = model.fit(data['train']['input'], data['train']['output'], epochs=300, batch_size=8,
#                                           validation_data=(data['val']['input'], data['val']['output']),
#                                           callbacks=getCallback2(len_callback))
#                       printTime("Model Training (fixed parameter)")
                      
#                       # printTime("Model Save", 1)
#                       # model.save(model_location / "h5_model.h5", save_format="h5")
#                       # printTime("Model Save")
                      
#                       printTime("Model Evaluation", 1)
#                       f = open(result_location / str(date), "w")
#                       plotModelStructure(model, result_location)
#                       performance = evalModel(model, data, f, result_location, scaler, history, len_callback)
#                       printTime("Model Evaluation")
#                       f.close()

#                       tmp_dic = {"leaky_rate": leaky_rate}
#                       for split in performance.keys():
#                         for metric in performance[split].keys():
#                           tmp_dic[f"{split}_{metric}"] = performance[split][metric]
#                       result_df = result_df.append(tmp_dic, ignore_index=True)
                      
#                       # if os.path.exists(df_location):
#                       #   pre_result_df = pd.read_csv(df_location)
#                       #   result_df = result_df.append(pre_result_df, ignore_index=True)                   
#                       # result_df.to_csv(df_location, index=False)

#                     else:
#                       printTime("Tuner Searching", 1)
#                       tuner = createTuner(setting, selected_model)
#                       tuner.search(data['train']['input'], data['train']['output'], epochs=100, 
#                                   #  verbose=0,
#                                   validation_data=(data['val']['input'], data['val']['output']), callbacks=getCallback(len_callback))
#                       printTime("Tuner Searching")
                      
#                       printTime("Model Evaluation", 1)
#                       f = open(result_location / str(date), "w")
#                       best_hps=createTuner(setting, selected_model).get_best_hyperparameters()[0]
#                       model = DengueNet(setting).build(best_hps) 
#                       printTime("Model Training", 1)
#                       history = model.fit(data['train']['input'], data['train']['output'], epochs=30, verbose=0,
#                                           validation_data=(data['val']['input'], data['val']['output']), 
#                                           callbacks=getCallback())
#                       printTime("Model Training")
#                       plotModelStructure(model, result_location)
#                       evalModel(model, data, f, result_location, scaler, history)
#                       printTime("Model Evaluation")
#                   # if os.path.exists(df_location):
#                   #   pre_result_df = pd.read_csv(df_location)
#                   #   result_df = result_df.append(pre_result_df, ignore_index=True)                   
#                   result_df.to_csv(df_location, index=False)

# if __name__ == "__main__":
#     args = setParser(True)
#     os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_index
#     gpus = tf.config.list_physical_devices('GPU')

#     if gpus:
#       try:
#         if args.limit_gpu:
#           tf.config.set_logical_device_configuration(
#               gpus[0],
#               [tf.config.LogicalDeviceConfiguration(memory_limit=args.gpu_size*1024)])
#         logical_gpus = tf.config.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#       except RuntimeError as e:
#         print(e)

#     main(args)