from pathlib import Path
from skimage import io
import pandas as pd
import pickle
import sys
import os
import sklearn
from sklearn.preprocessing import MinMaxScaler
import SimpleITK as sitk
from datetime import datetime
import argparse

root_dir = Path(__file__).parents[2]
sys.path.insert(0, str(root_dir / 'code/util'))

from ForFeatureEngineering import *
from ForMobileNet import *
from ModelUtil import *
from PlotFigure import *

def createColumns():
  metric_name = ["MAE", "sMAPE", "RMSE"]
  splits = ["train", "val", "test"]
  columns = []
  for split in splits:
    columns.append(f"{split}_pred")
    for metric in metric_name:
      columns.append(f"{split}_{metric}")
      columns.append(f"{split}_{metric}_std")
  return columns

def setParser(select_city=False):
  parser = argparse.ArgumentParser()
  parser.add_argument('--limit_gpu', action=argparse.BooleanOptionalAction)
  parser.add_argument("--gpu_index", type=str)
  parser.add_argument("--gpu_size", type=int)
  parser.add_argument("--project_folder", type=str)
  parser.add_argument("--cloud_threshold", type=int)
  parser.add_argument("--shadow_threshold", type=int)
  if select_city:
    parser.add_argument("--city", type=int)

  parser.add_argument('--fixed_thres', action=argparse.BooleanOptionalAction)
  parser.add_argument('--fixed_hyper', action=argparse.BooleanOptionalAction)
  parser.add_argument('--reverse_city_sequence', action=argparse.BooleanOptionalAction)
  parser.add_argument('--include_vit', action=argparse.BooleanOptionalAction)
  parser.add_argument('--only_vit', action=argparse.BooleanOptionalAction)
  args = parser.parse_args()
  return args

def getFeatureFolderName(all_feature_ls, selected_feature_ls):
  folder = ""
  for feature in selected_feature_ls:
    if folder == "":
      folder = str(list(all_feature_ls).index(feature))
    else:
      folder = folder + "_" + str(list(all_feature_ls).index(feature))
  return folder

def printTime(name, option=0):
  if option == 1:
    print(f"[{name}] Start: ", datetime.now().strftime("%H:%M:%S"))
  else:
    print(f"[{name}] End: ", datetime.now().strftime("%H:%M:%S"))

def checkFolder(path):
  """Recursively check if folders exist.
  """
  if not path.exists():
    os.makedirs(path)
  return path

def getRadiomicSetting() -> dict:
  radiomics_setting = {
    'binWidth': 25,
    'resampledPixelSpacing': None, 
    'interpolator': sitk.sitkBSpline,
    'label': 1,
  }
  return radiomics_setting

def getCityList(reverse=False):
  city_ls = ['50001', '73001', '76001', '54001', '5001']
  if reverse:
    city_ls.reverse()
  return city_ls 

def getDataDict(setting) -> dict:
  data = {}
  for split in ["train", "val", "test"]:
    data[split] = {}
    for feature in ["x_feng", "reshape_img", "x_vit", "x_cnn", "dirty_tile_label"]:
      data[split][feature] = []
    data[split]["average_img_dic"] = createAvgImg(setting['num_tile'])
      
    for band in [1, 2, 3, 11]:
      data[split][f"dirty_tile_label_{band}"] = []
      data[split][f"average_img_dic_{band}"] = createAvgImg(setting['num_tile'])
  return data

def getCodeToNameMap():
  return {
    5001: "Medellín",
    50001: "Villavicencio",
    76001: "Cali",
    73001: "Ibagué",
    54001: "Cúcuta"
}

def setArgs(include_top, selected_feature_ls, cloud, shadow, feature_folder_name):
    top_str = ""
    if not include_top:
      top_str = "no"
    
    feature_str = "default"
    if feature_folder_name:
      feature_str = f"{len(selected_feature_ls)}/{feature_folder_name}"
    elif len(selected_feature_ls) == 1:
      feature_str = f"1/{selected_feature_ls[0]}"
    else:
      feature_str = str(len(selected_feature_ls))
        
    cloud_str = "default"
    if cloud != 0 and shadow != 0:
      cloud_str = f"{cloud}_{shadow}"
      
    return f"{feature_str}/{cloud_str}/{top_str}Top"

def getSetting(project_folder, swap_enable, model, city, threshold_cloud, threshold_shadow, threshold_dic,
               include_top, selected_feature_ls, feature_folder_name=None, result_main_folder="eval", lstm_week=10):
    project_folder = Path(project_folder)
    dataset_folder = project_folder / 'dataset/sorted'
    img_folder = dataset_folder / city / 'image'
    feature_folder = dataset_folder / city / 'feature'
    label_file = dataset_folder / city / 'label.csv'
    metadata_file = dataset_folder / city / 'metadata.csv'
    sub_folder = setArgs(include_top, selected_feature_ls, threshold_cloud, threshold_shadow, feature_folder_name) 
    model_folder = project_folder / "model" / sub_folder / city
    result_folder = project_folder / result_main_folder / sub_folder / city

    if len(selected_feature_ls) == 0:
      selected_feature_ls = ['Variance', 'ShortRunHighGrayLevelEmphasis', 'LargeAreaLowGrayLevelEmphasis', 
                             'GrayLevelVariance', 'SmallAreaLowGrayLevelEmphasis', 'SizeZoneNonUniformity', 
                             'ShortRunLowGrayLevelEmphasis', 'RunLengthNonUniformity', 'LargeAreaEmphasis', 
                             'SumSquares', 'ZoneVariance', 'DifferenceVariance', 'LongRunHighGrayLevelEmphasis', 
                             'LargeAreaHighGrayLevelEmphasis', 'ClusterProminence', 'LongRunLowGrayLevelEmphasis', 
                             'LongRunEmphasis', 'SizeZoneNonUniformityNormalized']
    
    setting = {
          # Folders
          'city': city,
          'project_folder': project_folder,
          'dataset_folder': dataset_folder,
          'img_folder': img_folder,
          'feature_folder': feature_folder,
          'label_file': label_file,
          'model_folder': model_folder,
          "result_folder": result_folder,
          "metadata_file": metadata_file,
          
          # Basic
          "model": model,
          'num_tile': 16,
          'len_tile': 0,
          'lstm_weeks': lstm_week, 
          'train_ratio': 0.8,
          'col': None,
          'swap_enable': swap_enable,
          
          # Bands: https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/bands/
          'radiomics_band': 11, 
          'cnn_band':  [1, 2, 3], # Bands R, G, B 

          # MobileNetV2
          'resized_img_shape': (224, 224, 3),
          'pretrain_model': 'MobileNetV2',
          'include_top': include_top,
          
          "threshold_cloud": threshold_cloud,
          "threshold_shadow": threshold_shadow,
          # Features
          "selected_feature_ls": selected_feature_ls
          }

    for band in [1, 2, 3, 11]:
      threshold_path = dataset_folder / city / f"threshold_{band}.pickle"
      with open(threshold_path, 'rb') as f:
          threshold_dic = pickle.load(f)
      setting[f"threshold_cloud_{band}"] = threshold_dic[threshold_cloud]       
      setting[f"threshold_shadow_{band}"] = threshold_dic[threshold_shadow]

    return setting 

def splitDataset(df, col_name, val_index, test_index, lstm_week=0):
  df_np = df[col_name].to_numpy()
  return {
        'train': df_np[:val_index],
        'val': df_np[val_index-lstm_week:test_index],
        'test': df_np[test_index-lstm_week:]
  }

def getThresholds(img_ls, step=5):
    printTime("getThresholds", 1)
    threshold_dic = {}
    img_ls = np.array(img_ls) 
    for i in range(0, 100, step):
        threshold_dic[i] = np.percentile(img_ls, i)
    printTime("getThresholds")
    return threshold_dic
    
def readImgs(df, path, train_ratio=0.8, selected_band=11):
    printTime("readImgs", 1)
    img_ls = []
    name_ls = []
    for i, row in df.iterrows():
        if i >= len(df) * train_ratio:
            break
        img_name = path / "image" / (str(row['epiweek']) + '.tiff')
        # print(img_name)
        origin_img = io.imread(img_name)
        name_ls.append(str(row['epiweek']))
        img_ls.append(origin_img[:,:,selected_band])
    printTime("readImgs")
    return img_ls, name_ls

def setup(setting, city, is_debug=False, read_vit=True, normalize_vit=False, save_img=False, normalize_feature=False):
  printTime("setup", 1)
  df = pd.read_csv(setting['label_file'])
  data = getDataDict(setting)
  num_train = int(setting['train_ratio'] * len(df))
  num_val = int((len(df) - num_train) / 2)
  partition = splitDataset(df, 'epiweek', num_train, num_train+num_val, lstm_week=setting["lstm_weeks"])
  labels = splitDataset(df, 'Cases', num_train, num_train+num_val, lstm_week=setting["lstm_weeks"])
  metadata = splitDataset(pd.read_csv(setting['metadata_file']), 'temp', num_train, num_train+num_val, lstm_week=setting["lstm_weeks"])

  path = setting['dataset_folder'] / city
  feature_np = np.load(path / "feature.npy")
  rgb_np = np.load(path / "rgb.npy")
  band_np = np.load(path / "feng.npy")
  vit_np = np.load(path / "embeddings_vit.npy")
  
  for split in ["train", "val", "test"]:
    data, setting  = getInputDataset(data, partition, split, city, setting, feature_np, rgb_np, band_np, vit_np, lstm_week=setting["lstm_weeks"])

  data, labels, scaler, setting = prepareData(data, setting, labels, metadata, read_vit, normalize_vit, save_img, city, band_np, normalize_feature)
  if is_debug:
    for key in data.keys():
      print(key, data[key]['x_feng'].shape, data[key]['x_cnn'].shape,  data[key]['case'].shape, labels[key].shape)
    
  printTime("setup")
  return setting, data, labels, scaler

def getInputDataset(data, partition, name, city, setting, feature_np, rgb_np, band_img_ls, vit_np, train_ratio=0.8, val_ratio=0.9, total_len=156, lstm_week=10) -> list:
  start, end = int(total_len * val_ratio)-lstm_week, total_len
  if name == "train":
    start, end = 0, int(total_len * train_ratio)
  elif name == "val":
    start, end = int(total_len * train_ratio)-lstm_week, int(total_len * val_ratio)
    
  data[name]["x_feng"] = feature_np[start:end]
  data[name]["reshape_img"] = rgb_np[start:end]
  # data[name]["x_vit"] = vit_np[start:end]

  if type(setting['col']) == type(None):
    with open("/mnt/usb/jupyter-mimikuo/dataset/sorted/50001/feature/201601.pickle", 'rb') as handle:
        feature_dic = pickle.load(handle)
    setting['col'] = list(feature_dic[0].keys())
    setting['col'].extend(['name', 'city', 'tile_num'])
  
  for i in range(len(partition[name])):
    for band in [1, 2, 3, 11]:
      if band == 11:
        len_tile = setting[f'len_tile_{band}'] = int(band_img_ls[0].shape[0] / setting["num_tile"])
        img = band_img_ls[i, :setting["num_tile"] * len_tile, :setting["num_tile"] * len_tile]
      else: 
        setting[f'len_tile_{band}'] = int(data[name]["reshape_img"][i].shape[0] / setting["num_tile"])
        img = data[name]["reshape_img"][i][:,:,band-1]
      
      label_ls, data[name][f'average_img_dic_{band}'] = swapLabelling(img, setting, data[name][f'average_img_dic_{band}'], 
                                                                  setting[f'threshold_cloud_{band}'], setting[f'threshold_shadow_{band}'], 
                                                                  setting[f'len_tile_{band}'])
      # print(name, i, band, np.array(label_ls).sum(), len(label_ls))
      data[name][f"dirty_tile_label_{band}"].append(label_ls)
      
  for i in data[name]:
    if type(data[name][i]) == list:
      data[name][i] = np.array(data[name][i]) 

  return data, setting

def prepareData(data, setting, labels, metadata, read_vit, normalize_vit_img, save_img, city, band_np, normalize_feature):
  def getRadiomics(data, setting, band_np, save_img, normalize_feature, splits=["train", "val", "test"]):
    def getAvgFeature(data, setting):
      average_img_dic = data['train']['average_img_dic_11']
      avg_img = obtainAvgImg(average_img_dic, setting, setting["len_tile_11"])
      avg_feature = calculateSampleFeatures(avg_img, setting, getRadiomicSetting())
      return avg_feature, avg_img

    avg_feature, avg_img = getAvgFeature(data, setting)
    band_np = band_np[:,:avg_img.shape[0], :avg_img.shape[1]]
    start = 2016
    counter = 1
    img_start_idx = 0
    
    for split in splits:
      sample_feature = data[split]['x_feng']
      dirty_label = data[split]['dirty_tile_label_11']
      
      if setting['swap_enable']:
        # print("Swapping tiles......")
        sample_feature = performTileSwapping(sample_feature, dirty_label, avg_feature)
        
        if save_img:
          img_end_idx = img_start_idx + dirty_label.shape[0]
          # print(avg_img.shape, band_np[img_start_idx : img_end_idx,:,:].shape)
          # print(dirty_label.shape, dirty_label)
          origin_img = band_np[img_start_idx:img_end_idx,:,:].copy()
          swapped_img = swapImgTile(origin_img.copy(), dirty_label, avg_img, setting["len_tile_11"], setting[f"num_tile"])
          result_folder = Path(f"/mnt/usb/jupyter-mimikuo/demo/{city}/swap_gray_resized/{setting['threshold_cloud']}_{setting['threshold_shadow']}")
          checkFolder(result_folder/"comparison")
          counter, start = plotSwapResultsGray(origin_img, swapped_img, counter, start, result_folder, avg_img)
          img_start_idx = img_end_idx 

      sample_feature = selectFeature(sample_feature, setting['col'][:-3], setting['selected_feature_ls'])
      sample_feature = sample_feature.reshape(sample_feature.shape[0], -1)
      sample_feature = prepareTimeSeries(sample_feature, setting['lstm_weeks'])
      data[split]['x_feng'] = sample_feature
      # print("Radiomics:", data[split]['x_feng'].shape) # (n_sample, window_size, n_feature)
    
    if normalize_feature:
      for feature_idx in range(sample_feature.shape[2]):
        tmp_data = data['train']['x_feng'][:,:,feature_idx]
        scaler = MinMaxScaler(feature_range=(0, 1)).fit(tmp_data)
        
        for split in splits:
          tmp_data = data[split]['x_feng'][:,:,feature_idx]
          data[split]['x_feng'][:,:,feature_idx] = scaler.transform(tmp_data)
    
    return data
  
  def getCnn(data, setting, splits=["train", "val", "test"]):
    for split in splits:
      sample_img = data[split]['reshape_img']
      sample_img = cnnPreprocess(sample_img, setting["pretrain_model"])
      # print("Cnn before time series:", sample_img.shape)
      # for i in range(3):
      #   print("Std and Avg:", np.std(sample_img[:,:,i].flatten()), np.average(sample_img[:,:,i].flatten()))
      sample_img = prepareTimeSeries(sample_img, setting['lstm_weeks'])
      data[split]['x_cnn'] = sample_img
      # print("Cnn:", data[split]['x_cnn'].shape) # (n_sample, window_size, 224, 224, 3)
    return data
  
  def getVit(data, setting, read_vit, normalize_vit, save_img, city, splits=["train", "val", "test"]):
    start = 2016
    counter = 1
    for split in splits:
      origin_img = data[split]['reshape_img'].copy() # Before time series: (124, 224, 224, 3)
      sample_img = data[split]['reshape_img'].copy() # Before time series: (124, 224, 224, 3)
      # print("ViT before time series:", sample_img.shape)
      
      if setting['swap_enable']:
        avg_img_rgb = []
        dirty_label = []
        for band in [1, 2, 3]:
          dirty_label.append(data[split][f'dirty_tile_label_{band}'])
        dirty_label = np.average(np.array(dirty_label), axis=0)
        # print(dirty_label.shape)
        # print(dirty_label)
        dirty_label = (dirty_label > 0.5).astype(int)
        # print(dirty_label)
        
        for band in [1, 2, 3]:
          # print("Swapping tiles......")
          # print(data[split][f'dirty_tile_label_{band}'].shape)
          # print(data[split][f'dirty_tile_label_{band}'])
          # dirty_label = data[split][f'dirty_tile_label_{band}']
          average_img_dic = data['train'][f'average_img_dic_{band}']
          avg_img = obtainAvgImg(average_img_dic, setting, setting[f"len_tile_{band}"], return_int=False)
          avg_img_rgb.append(avg_img)
          sample_img[:,:,:,band-1] = swapImgTile(sample_img[:,:,:,band-1], dirty_label, avg_img, 
                                   setting[f"len_tile_{band}"], setting[f"num_tile"])
        if save_img:
          result_folder = Path(f"/mnt/usb/jupyter-mimikuo/demo/{city}/swap_rgb_resized/{setting['threshold_cloud']}_{setting['threshold_shadow']}")
          avg_img_rgb = np.array(avg_img_rgb)
          # print(avg_img_rgb.shape)
          checkFolder(result_folder/"single")
          checkFolder(result_folder/"comparison")
          counter, start = plotSwapResults(data[split]['reshape_img'], sample_img, counter, start, result_folder, np.transpose(avg_img_rgb, (1, 2, 0)))

      if normalize_vit:
        sample_img = (sample_img - np.average(sample_img)) / np.std(sample_img)
        # sample_img = (sample_img - 127.5) / 127.5
            
      sample_img = prepareTimeSeries(sample_img, setting['lstm_weeks'])
      data[split]['x_vit'] = sample_img
      # print("Vit:", data[split]['x_vit'].shape)
    return data
  
  def getCase(data, setting, labels, splits=["train", "val", "test"]):
    window_size = setting['lstm_weeks']
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(labels['train'].reshape(-1, 1))
    
    for split in splits:
      case = labels[split]
      labels["origin_"+split] = case[window_size:]
      
      case = scaler.transform(case.reshape(-1, 1))
      labels[split] = case[window_size:]
      
      case = prepareTimeSeries(case.reshape(-1, 1), window_size)
      data[split]['case'] = case 
      
      print("Case:", data[split]['case'].shape)
    return data, labels, scaler
  
  # def getMetadata(data, setting, metadata, splits=["train", "val", "test"]):
    #   window_size = setting['lstm_weeks']
    #   scaler = MinMaxScaler(feature_range=(0, 1))
    #   scaler = scaler.fit(labels['train'].reshape(-1, 1))
      
      
    #   for split in splits:
    #     case = labels[split]
    #     labels["origin_"+split] = case[window_size:]
        
    #     case = scaler.transform(case.reshape(-1, 1))
    #     labels[split] = case[window_size:]
        
    #     case = prepareTimeSeries(case.reshape(-1, 1), window_size)
    #     data[split]['case'] = case 
        
    #     print("Case:", data[split]['case'].shape)
    #   return data, labels, scaler
  
  data = getRadiomics(data, setting, band_np, save_img, normalize_feature)
  data = getCnn(data, setting)
  data = getVit(data, setting, read_vit, normalize_vit_img, save_img, city)
  data, labels, scaler = getCase(data, setting, labels)
  if read_vit:
    setting['len_feature'] = data['train']['x_feng'].shape[2]
  else:
    setting['len_feature'] = 768
  return data, labels, scaler, setting
