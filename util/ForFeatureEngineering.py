import numpy as np
import SimpleITK as sitk
import six
import logging
import pickle
# import radiomics
from radiomics import firstorder, glcm, glrlm, glszm
logger = logging.getLogger("radiomics") # set level for all classes
logger.setLevel(logging.ERROR)

def getThresholdPairList(args=None):
  if args and args.fixed_thres:
    return zip([args.cloud_threshold], [args.shadow_threshold])
  else:
    # cloud = [60, 65, 60]
    # shadow = [20, 15, 15]
    # cloud = [60]
    # shadow = [15]
    cloud = [70]
    shadow = [10]
    # cloud = [60, 60, 60, 65, 70, 75]
    # shadow = [20, 15, 10, 10, 10, 10]
    return zip(cloud, shadow)

def getFeatureList():
  # return ["ClusterShade", "ClusterProminence", "ClusterTendency", "GrayLevelVariance"]
  return ["ClusterShade", "ClusterProminence", "Skewness", "RunPercentage", 
          "JointAverage", "ClusterTendency", "MeanAbsoluteDeviation", "Idm",
          "Skewness", "MCC"]

def createAvgImg(num_tile):
  average_img_dic = {}
  for i in range(num_tile * num_tile):
    average_img_dic[i] = {
        "tiles": [],
        "numbers": 0
    }
  return average_img_dic

def getImageRadiomicFeatureAsList(feature_dic, columns):
  all_tile_ls = []
  
  for tile_num in feature_dic.keys():
    tile_ls = [feature_dic[tile_num][col] for col in columns]
    all_tile_ls.append(tile_ls)

  return np.array(all_tile_ls)

def swapLabelling(origin_img, setting, average_img_dic, thres_cloud, thres_shad, len_tile):
  num_tile = setting['num_tile']
  band_img = origin_img
    
  label_ls = []
  counter = 0
  for r_idx in range(num_tile):
    for c_idx in range(num_tile):
      r, c = r_idx * len_tile, c_idx * len_tile
      tile = band_img[r:r+len_tile, c:c+len_tile]

      cloud_avg = np.sum(tile > thres_cloud) / (len_tile*len_tile)
      shad_avg = np.sum(tile < thres_shad) / (len_tile*len_tile)
      
      if cloud_avg >= 0.5 or shad_avg >= 0.5:
        label_ls.append(1)
      else:
        label_ls.append(0)

      average_img_dic[counter]['tiles'].append(tile)
      average_img_dic[counter]['numbers'] += 1
      counter += 1

  return label_ls, average_img_dic

def obtainAvgImg(average_img_dic, setting, len_tile, return_int=True):
  img_len = setting["num_tile"] * len_tile
  # print(setting["num_tile"], len_tile, img_len)
  # print(average_img_dic[0]['tiles'])
  # print(average_img_dic[0]['numbers'])
  avg_img = np.zeros((img_len, img_len))
  avg_img_int = np.zeros((img_len, img_len))
  counter = 0
  
  for r_idx in range(setting["num_tile"]):
    for c_idx in range(setting["num_tile"]):
      r = r_idx * len_tile
      c = c_idx * len_tile
      # print(np.array(average_img_dic[counter]['tiles']).shape)
      # print(max(np.array(average_img_dic[counter]['tiles']).flatten()), min(np.array(average_img_dic[counter]['tiles']).flatten()))
      # print(np.array(average_img_dic[counter]['tiles'])[0])
      
      sum_tiles = np.sum(np.array(average_img_dic[counter]['tiles']), axis=0)
      # print(max(sum_tiles.flatten()), min(sum_tiles.flatten()))
      
      avg_tiles = np.divide(sum_tiles, average_img_dic[counter]['numbers'])
      # print(max(avg_tiles.flatten()), min(avg_tiles.flatten()))
      
      int_tiles = np.rint(avg_tiles)
      avg_img[r : r + len_tile, c : c + len_tile] = avg_tiles
      avg_img_int[r : r + len_tile, c : c + len_tile] = int_tiles
      counter += 1

  if return_int:
    return avg_img_int
  return avg_img

def calculateRadiomics(image, radiomics_setting):
  mask = sitk.GetImageFromArray(np.full(image.shape, 1))
  image = sitk.GetImageFromArray(image)
  feature_ls = {}

  firstOrderFeatures = firstorder.RadiomicsFirstOrder(image, mask, **radiomics_setting)
  firstOrderFeatures.enableAllFeatures()
  results = firstOrderFeatures.execute()
  for (key, val) in six.iteritems(results):
    feature_ls[key] = val

  # Show GLCM features
  glcmFeatures = glcm.RadiomicsGLCM(image, mask, **radiomics_setting)
  glcmFeatures.enableAllFeatures()
  results = glcmFeatures.execute()
  for (key, val) in six.iteritems(results):
    feature_ls[key] = val

  # Show GLRLM features
  glrlmFeatures = glrlm.RadiomicsGLRLM(image, mask, **radiomics_setting)
  glrlmFeatures.enableAllFeatures()
  results = glrlmFeatures.execute()
  for (key, val) in six.iteritems(results):
    feature_ls[key] = val

  # Show GLSZM features
  glszmFeatures = glszm.RadiomicsGLSZM(image, mask, **radiomics_setting)
  glszmFeatures.enableAllFeatures()
  results = glszmFeatures.execute()
  for (key, val) in six.iteritems(results):
    feature_ls[key] = val

  return feature_ls

def calculateSampleFeatures(img, setting, radiomics_setting):
  counter = 0
  len_tile = setting["len_tile_11"]
  feature_dic = {}
  
  for r_idx in range(setting["num_tile"]):
    for c_idx in range(setting["num_tile"]):
      r = r_idx * len_tile
      c = c_idx * len_tile
        
      tile = img[r : r + len_tile, c : c + len_tile]
      features = calculateRadiomics(tile, radiomics_setting)
      feature_dic[counter] = features
      counter += 1
  
  return getImageRadiomicFeatureAsList(feature_dic, setting['col'][:-3])

def performTileSwapping(sample_ls, dirty_tile_label, average_feature):
  num_sample, num_tile, _ = sample_ls.shape
  
  for i in range(num_sample):
    for j in range(num_tile):
      if dirty_tile_label[i][j] == 1:
        sample_ls[i, j] = average_feature[j]
  
  return sample_ls

def swapImgTile(sample_img, dirty_tile_label, avg_img, len_tile, num_tile):
  num_sample = sample_img.shape[0]

  for i in range(num_sample):
    for r_idx in range(num_tile):
      for c_idx in range(num_tile):
        if dirty_tile_label[i][r_idx * num_tile + c_idx] == 1:
          r = r_idx * len_tile
          c = c_idx * len_tile
          avg_tile = avg_img[r : r + len_tile, c : c + len_tile]
          sample_img[i][r : r + len_tile, c : c + len_tile] = avg_tile
  
  return sample_img  

def selectFeature(sample_ls, feature_ls, selected_ls):
  print('Original dimension:', sample_ls.shape)
  delete_index_ls = []
  for i, col in enumerate(feature_ls):
    if col not in selected_ls: 
      delete_index_ls.append(i)
  sample_ls = np.delete(sample_ls, delete_index_ls, axis=2)
  print('New dimension:', sample_ls.shape)
  return sample_ls

def readFeatureName(df, path):
    cols = []
    for i, row in df.iterrows():
        feature_name = path / "feature" / (str(row['epiweek']) + '.pickle')
        with open(feature_name, 'rb') as handle:
            feature_dic = pickle.load(handle)
        cols = list(feature_dic[0].keys())
        break
    return np.array(cols)