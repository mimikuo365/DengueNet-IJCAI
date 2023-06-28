# Get ViT layer descriptions
# https://discuss.huggingface.co/t/keras-fine-tune-vision-transformer-model/21385/3
# https://github.com/huggingface/transformers/issues/18282#issuecomment-1201554944

# Keras implementation
# https://www.philschmid.de/image-classification-huggingface-transformers-keras

# Heatmap
# https://www.kaggle.com/code/basu369victor/covid19-detection-with-vit-and-heatmap
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
from pathlib import Path
import pandas as pd
import sys
import itertools


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
    
    # vit_classifier.layers[-1].activation = None
    
    base_model = TFViTModel.from_pretrained(backbone)
    channel_fist_inputs = Permute((3,1,2))(inputs)
    # base_model.trainable = False
    # print(base_model.vit.embeddings.patch_embeddings)
    # base_model.vit.embeddings.trainable = True              # 758,057
    base_model.vit.embeddings.trainable = False
    # Construct the CLS token, position and patch embeddings.
    
    # base_model.vit.encoder.trainable = True                 # 85,069,865
    base_model.vit.encoder.trainable = False
    
    # base_model.vit.layernorm.trainable = True               # 16,937
    base_model.vit.layernorm.trainable = False
    
    # base_model.vit.pooler.trainable = True                  # 605,993
    base_model.vit.pooler.trainable = False
    # base_model.trainable = True
    layer = base_model.vit(channel_fist_inputs)[0][:,0,:]
    layer = tf.keras.layers.Dense(units=20, activation=tf.keras.layers.LeakyReLU(alpha=0.3))(layer)
    layer = tf.keras.layers.Dense(1, activation=tf.keras.layers.LeakyReLU(alpha=0.3), name="output_layer")(layer)
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
    
    # result_location = getResultLocation(args, "/mnt/usb/jupyter-mimikuo/eval", "Vit_Predict_Case", city)
    # # if os.path.exists(result_location / "history.png"):
    # #     print(f"Error!! Folder {result_location} already existed!!")
    # #     return
    # checkFolder(result_location)
    # data, scaler = prepareData(path)
    # history = model.fit(data['train']['input'], data['train']['output'], epochs=300, 
    #                     batch_size=4,
    #                     validation_data=(data['val']['input'], data['val']['output']),                         
    #                     callbacks=getCallback())
    
    # printTime("Model Evaluation", 1)
    # date = datetime.now().strftime("%H:%M:%S_%d-%m-%Y") + ".log"
    # f = open(result_location / str(date), "w")
    # plotModelStructure(model, result_location)
    # evalModel(model, data, f, result_location, scaler, history)
    # printTime("Model Evaluation")

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