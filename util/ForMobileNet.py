from skimage import io
from skimage.transform import resize
import numpy as np
import tensorflow as tf


def resizeToRgbImg(origin_img, selected_three_bands, resized_img_shape):
    selected_img = np.stack(
        (
            origin_img[:, :, selected_three_bands[0]],
            origin_img[:, :, selected_three_bands[1]],
            origin_img[:, :, selected_three_bands[2]],
        )
    )
    selected_img = np.transpose(selected_img, (1, 2, 0))
    selected_img = resize(
        selected_img, resized_img_shape, preserve_range=True, anti_aliasing=True
    )
    return selected_img


def cnnPreprocess(img_ls, pretrain_model):
    if pretrain_model == "MobileNetV2":  # minimum size & parameters
        img_ls = tf.keras.applications.mobilenet_v2.preprocess_input(img_ls)
    elif pretrain_model == "EfficientNetV2L":  # best top 5 acc
        img_ls = tf.keras.applications.efficientnet_v2.preprocess_input(img_ls)
    elif pretrain_model == "MobileNetV2":  # min depth
        img_ls = tf.keras.applications.vgg16.preprocess_input(img_ls)
    elif pretrain_model == "ResNet50V2":
        img_ls = tf.keras.applications.resnet_v2.preprocess_input(img_ls)
    return img_ls
