import os
import argparse
import numpy as np
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
import tensorflow.keras as keras
from keras import layers
from keras.layers import Input, Permute, Dense
from tensorflow.keras.models import Model
from transformers import TFViTModel, ViTFeatureExtractor
# import warnings
# warnings.filterwarnings('ignore')
from pathlib import Path
import pandas as pd
import sys

root_dir = Path(__file__).parents[2]
sys.path.insert(0, str(root_dir / 'code/util'))
sys.path.insert(0, str(root_dir / 'code/model'))
from DataPreprocess import *
from CalculateMetric import evalModel, plotModelStructure

def getBackbone(target_size, backbone):
    base_model = TFViTModel.from_pretrained(backbone)
    base_model.trainable = False
    inputs = Input(target_size)
    channel_fist_inputs = Permute((3,1,2))(inputs)
    embeddings = base_model.vit(channel_fist_inputs)[0][:,0,:]
    vit_model = Model(inputs=inputs, outputs=embeddings)
    vit_model.compile()
    return vit_model

def readImage(path, normalized_img=False, target_size=(224, 224, 3)):
    image_test = io.imread(path)
    image_test = image_test[:,:, [1,2,3]]
    image_test = skimage.transform.resize(image_test, (target_size[0], target_size[1]),
                                          anti_aliasing=True, preserve_range=True)
    if normalized_img:
        image_test = (image_test - 127.5) / 127.5
    image_test = np.expand_dims(image_test,axis=0)  # Add Batch dimension
    return image_test

def predict(model, image):
    embedding = model.predict(image)
    embedding = np.squeeze(embedding, axis=0)
    return embedding

def split_columns(df):
    df_aux = pd.DataFrame(df['embedding'].tolist())
    df_aux = pd.concat([df[['epiweek']], df_aux], axis=1)
    return df_aux

def getEmbeddings(path, use_np=True, normalized_img=False):
    df = pd.read_csv(path / "label.csv")
    img_np = np.load(path / "rgb.npy")
    model_path = "google/vit-base-patch16-224-in21k"
    target_size = (224, 224, 3)
    
    feature_df = pd.DataFrame(columns=['epiweek', 'embedding'])
    np_location = path / "embeddings_vit.npy"
    df_location = path / "embeddings_vit.csv"
    feature_np = []
    
    model = getBackbone(target_size, model_path)
    for i, row in df.iterrows():
        if not use_np:
            image = readImage(path / f"image/{epiweek}.tiff")
        else:
            image = np.expand_dims(img_np[i,:,:,:],axis=0) # Add Batch dimension

        epiweek = str(row["epiweek"])
        embedding = predict(model, image)
        print(embedding.shape)
        feature_np.append(embedding)
        feature_df = feature_df.append({'embedding': embedding, 'epiweek': epiweek}, ignore_index=True)

    feature_df = split_columns(feature_df)
    feature_df.to_csv(df_location, index=False)
    feature_np = np.array(feature_np)
    print(feature_np.shape)
    np.save(np_location, feature_np)

def printEmbeddings(path):
    np_location = path / "embeddings_vit.npy"
    np_vit = np.load(np_location)
        
    for i in range(np_vit.shape[0]):
        sample = np_vit[i]
        print(np.mean(sample), np.std(sample))
    
    for i in range(np_vit.shape[1]):
        sample = np_vit[:, i]
        print(np.mean(sample), np.std(sample))

def scaleEmbeddings(path):
    np_location = path / "embeddings_vit.npy"
    np_vit = np.load(np_location)
    
    train_ratio = 0.8
    size = np_vit.shape[0]
    val_start = int(train_ratio * size)
    test_start = int((size - val_start) / 2) + val_start
    
    for feature_idx in range(np_vit.shape[1]):
        print(np_vit[:val_start, feature_idx].shape)
        print(np_vit[:val_start, feature_idx].reshape(-1, 1).shape)
        scaler = StandardScaler().fit(np_vit[:val_start, feature_idx].reshape(-1, 1))
        for start, end in zip([0, val_start, test_start], [val_start, test_start, size]):
            print(np_vit[start:end, feature_idx].shape)
            np_vit[start:end, feature_idx] = scaler.transform(np_vit[start:end, feature_idx].reshape(-1, 1)).flatten()
    
    np.save(np_location, np_vit)

def main():
    for city in getCityList():
        path = Path(f"/mnt/usb/jupyter-mimikuo/dataset/sorted/{city}/")
        getEmbeddings(path)
        # # scaleEmbeddings(path)
        # printEmbeddings(path)
    
if __name__ == "__main__":
    args = setParser(True)
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_index
    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    main()