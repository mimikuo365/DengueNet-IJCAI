import os
import argparse
import numpy as np
from sklearn.preprocessing import StandardScaler

import matplotlib.cm as cm
import tensorflow as tf
import tensorflow.keras as keras
from keras import layers
from keras.layers import Input, Permute, Dense
from tensorflow.keras.models import Model
from transformers import TFViTModel, ViTFeatureExtractor
from pathlib import Path
import pandas as pd
import sys
import itertools
import cv2
import matplotlib.pyplot as plt
# from PIL import Image
from IPython.display import Image, display

root_dir = Path(__file__).parents[2]
sys.path.insert(0, str(root_dir / 'code/util'))
sys.path.insert(0, str(root_dir / 'code/model'))
from DataPreprocess import *
from CalculateMetric import evalModel, plotModelStructure
from ModelUtil import getModelList, getInput, createTuner, getCallback, createModel, getResultLocation

def getVitCaseModel():
    backbone = "google/vit-base-patch16-224-in21k"
    target_size = (224, 224, 3)
    inputs = Input(target_size)
    
    base_model = TFViTModel.from_pretrained(backbone)
    channel_fist_inputs = Permute((3,1,2))(inputs)
    # base_model.trainable = False
    
    # print(base_model.vit.embeddings.patch_embeddings)
    # base_model.vit.embeddings.trainable = True              # 758,057
    base_model.vit.embeddings.trainable = False
    
    # base_model.vit.encoder.trainable = True                 # 85,069,865
    base_model.vit.encoder.trainable = False
    
    # base_model.vit.layernorm.trainable = True               # 16,937
    base_model.vit.layernorm.trainable = False
    
    # base_model.vit.pooler.trainable = True                  # 605,993
    base_model.vit.pooler.trainable = False

    # layer = base_model.vit(channel_fist_inputs)[0]
    layer = base_model.vit(channel_fist_inputs)[0][:,0,:]
    # layer = tf.keras.layers.Dense(units=20, activation=tf.keras.layers.LeakyReLU(alpha=0.3))(layer)
    # layer = tf.keras.layers.Dense(1, activation=tf.keras.layers.LeakyReLU(alpha=0.3), name="output_layer")(layer)
    vit_model = Model(inputs=[inputs], outputs=[layer])
    
    loss=tf.keras.losses.MeanAbsoluteError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    metrics=[
        tf.keras.metrics.MeanAbsoluteError(name="mae")
    ]
    vit_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return vit_model

def prepareLabel(labels, val_start, test_start):
    new_label = []
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(labels[:val_start].reshape(-1, 1))
    for start, end in zip([0, val_start, test_start], [val_start, test_start, len(labels)]):
        new_label.append(scaler.transform(labels[start:end].reshape(-1, 1)).flatten())
    new_label = np.array(list(itertools.chain.from_iterable(new_label)))
    print(len(new_label))
    print(new_label)
    return new_label, scaler

def prepareData(path):
    df = pd.read_csv(path / "label.csv")
    img_np = np.load(path / "rgb.npy")
    val_start = int(len(df) * 0.8)
    test_start = int((len(df) - val_start) / 2) + val_start
    labels, scaler = prepareLabel(df['Cases'].to_numpy(), val_start, test_start)

    data = {}
    splits = ["train", "val", "test"]
    for start, end, split in zip([0, val_start, test_start], [val_start, test_start, len(df)], splits):
        data[split] = {
            "input": img_np[start:end],
            "output": labels[start:end],
            "origin_output": df['Cases'].to_numpy()[start:end]
        }
    for split in splits:
        for name in ["input", "output", "origin_output"]:
            print(split, name, type(data[split][name]))
            print(data[split][name])
    return data, scaler
    
def runVitCase(path, city):
    model = getVitCaseModel()
    print(model.summary())
    alpha = 0.2
    result_location = getResultLocation(args, "/mnt/usb/jupyter-mimikuo/eval_vit", "Predict_Case", city)
    checkFolder(result_location)
    data, scaler = prepareData(path)
    history = model.fit(data['train']['input'], data['train']['output'], epochs=100, 
                        batch_size=8,
                        validation_data=(data['val']['input'], data['val']['output']),                         
                        callbacks=getCallback())
    
    printTime("Model Evaluation", 1)
    # date = datetime.now().strftime("%H:%M:%S_%d-%m-%Y") + ".log"
    # f = open(result_location / str(date), "w")
    
    plotModelStructure(model, result_location)
    scaled_pred = model.predict(data['test']['input'])
    print(scaled_pred.shape)
    # evalModel(model, data, f, result_location, scaler, history)
    # printTime("Model Evaluation")
    
    for i, origin_img in enumerate(data['test']['input']):
      # origin_img = data['test']['input'][0]

      heatmap = scaled_pred[0].reshape(16,16,3)
      # heatmap = cv2.resize(heatmap, ())
      norm_heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
      
      # Rescale heatmap to a range 0-255
      heatmap = np.average(norm_heatmap, axis=2)
      heatmap = np.uint8(255 * heatmap)
      # Use jet colormap to colorize heatmap
      jet = cm.get_cmap("jet")
      
      # Use RGB values of the colormap
      jet_colors = jet(np.arange(256))[:, :3]
      jet_heatmap = jet_colors[heatmap]

      # # Create an image with RGB colorized heatmap
      jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
      jet_heatmap = jet_heatmap.resize((origin_img.shape[0], origin_img.shape[1]))
      jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)
      heatmap_img = keras.preprocessing.image.array_to_img(jet_heatmap)
      heatmap_img.save(result_location / f"{i}_jet_heatmap.png")

      # Superimpose the heatmap on original image
      superimposed_img = jet_heatmap * alpha + origin_img
      # superimposed_img = jet_heatmap * alpha + origin_img
      superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
      superimposed_img.save(result_location / f"{i}_jet_merge.png")

      # Superimpose the heatmap on original image
      print( np.array(norm_heatmap).shape)
      no_jet_heatmap = cv2.resize(np.array(norm_heatmap), (origin_img.shape[0], origin_img.shape[1]))
      print(no_jet_heatmap.shape)
      print(max(no_jet_heatmap.flatten()), min(no_jet_heatmap.flatten()), no_jet_heatmap.dtype)
      heatmap_img = keras.preprocessing.image.array_to_img(no_jet_heatmap)
      heatmap_img.save(result_location / f"{i}_no_jet_heatmap.png")
      superimposed_img = no_jet_heatmap * 256 * alpha + origin_img
      superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
      superimposed_img.save(result_location / f"{i}_no_jet_merge.png")


    # # Display Grad CAM
    # display(Image(cam_path))

    
    # im = Image.fromarray(heatmap.astype("uint8"))
    # im.save(str(result_location / "heatmap.png"))

    # result = (heatmap * origin_img).astype("uint8")
    # im = Image.fromarray(result)
    # im.save(str(result_location / "merge.png"))

def readImage(path, target_size=(224, 224, 3)):
    # Read Image
    image_test = io.imread(path)
    # Select RGB Bands
    image_test = image_test[:,:, [1,2,3]]
    print(max(image_test.flatten()), min(image_test.flatten()))
    image_test = resize(image_test, (target_size[0], target_size[1]),
                           anti_aliasing=True, preserve_range=True)
    
    print("Before adjust:", np.mean(image_test.flatten()), np.std(image_test.flatten())) 
    image_test = (image_test - 127.5) 
    print("After adjust:", np.mean(image_test.flatten()), np.std(image_test.flatten())) 
    image_test = image_test / 127.5
    print("After adjust:", np.mean(image_test.flatten()), np.std(image_test.flatten())) 

    print(image_test.shape)
    print(max(image_test.flatten()), min(image_test.flatten()))
    # Add Batch dimension
    image_test = np.expand_dims(image_test,axis=0)
    print(image_test.shape)
    return image_test

def predict(model, image):
    embedding = model.predict(image)
    embedding = np.squeeze(embedding, axis=0)
    return embedding

def runVitLstmModel(city):
    len_callback = 300
    setting = getSetting("/mnt/usb/jupyter-mimikuo/", False, "Vit", city, 
                        include_top=False, threshold_dic=None, 
                        selected_feature_ls="",
                        threshold_cloud=0, threshold_shadow=0,
                        feature_folder_name="")
    result_location = getResultLocation(args, setting["result_folder"], selected_model, model_parameter)
    if os.path.exists(result_location / "history.png"):
        print(f"Error!! Folder {result_location} already existed!!")
        return
    checkFolder(result_location)
    setting, data, labels, scaler = setup(setting, city)
    data = getInput("Vit", data, labels)
    setting["model"] = "Feng"
    model = createVit()
    history = model.fit(data['train']['input'], data['train']['output'], epochs=300, 
                        batch_size=2,
                        validation_data=(data['val']['input'], data['val']['output']), 
                        callbacks=getCallback(len_callback))

    printTime("Model Evaluation", 1)
    date = datetime.now().strftime("%H:%M:%S_%d-%m-%Y") + ".log"
    f = open(result_location / str(date), "w")
    plotModelStructure(model, result_location)
    evalModel(model, data, f, result_location, scaler, history)
    printTime("Model Evaluation")

def main():
    model_path = "google/vit-base-patch16-224-in21k"
    target_size = (224, 224, 3)
    
    city_ls = getCityList()
    for city in city_ls:
        path = Path(f"/mnt/usb/jupyter-mimikuo/dataset/sorted/{city}/")
        runVitCase(path, city)
        # runVitLstmModel(city)
        break
if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--gpu_index", type=str)
    # args = parser.parse_args()
    args = setParser(True)
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_index
    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    main()