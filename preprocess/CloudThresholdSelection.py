import os
from skimage import io
from pathlib import Path
import pandas as pd  # Needed to avoid "GLIBCXX_3.4.26' not found" error
import numpy as np
import pickle
import sys
import argparse
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2 as cv

root_dir = Path(__file__).parents[2]
sys.path.insert(0, str(root_dir / "code/util"))
sys.path.insert(0, str(root_dir / "code/model"))

from DataPreprocess import (
    splitDataset,
    getCityList,
    getCodeToNameMap,
    getSetting,
    setup,
    checkFolder,
)


def getThresholds(img_ls, step=5):
    print("Calculating thresholds...")
    threshold_dic = {}
    img_ls = np.array(img_ls)
    print(img_ls.shape)  # sample_num, height, width, bands
    for i in range(0, 100, step):
        threshold_dic[i] = np.percentile(img_ls, i)
    return threshold_dic


def readImgs(df, path, train_ratio=0.8, selected_band=11):
    print("Reading images...")
    img_ls = []
    name_ls = []
    for i, row in df.iterrows():
        if i >= len(df) * train_ratio:
            break
        img_name = path / "image" / (str(row["epiweek"]) + ".tiff")
        origin_img = io.imread(img_name)
        name_ls.append(str(row["epiweek"]))
        img_ls.append(origin_img[:, :, selected_band])
    return img_ls, name_ls


def plotThreshold():
    for city in getCityList():
        path = Path(f"/mnt/usb/jupyter-mimikuo/dataset/sorted/{city}")
        df = pd.read_csv(path / "label.csv")
        img_ls, name_ls = readImgs(df, path)
        threshold_dic = getThresholds(img_ls)
        folder = checkFolder(Path(f"/mnt/usb/jupyter-mimikuo/cloud/{city}"))

        for name, img in zip(name_ls, img_ls):
            if Path(f"{folder}/{name}.png").exists():
                continue
            fig, axs = plt.subplots(
                2, 10, figsize=(16, 5), dpi=300, constrained_layout=True
            )
            plt.subplots_adjust(top=0.9, wspace=0.1, hspace=0.1)

            for i, ax in zip(threshold_dic.keys(), fig.get_axes()):
                if i == 0:
                    ax.set_title("Origin")
                    ax.imshow(img.astype(np.uint8), cmap="gray")
                else:
                    if i < 50:
                        masked_img = img < i
                    else:
                        masked_img = img > i

                    print(city, name, i, ":", threshold_dic[i])
                    ax.imshow(masked_img.astype(np.uint8), cmap="gray")
                    ax.set_title(str(f"{i}: {int(threshold_dic[i])}"))
                ax.set_axis_off()
            plt.savefig(f"{folder}/{name}.png")


def plotOpening():
    for city in getCityList():
        path = Path(f"/mnt/usb/jupyter-mimikuo/dataset/sorted/{city}")
        df = pd.read_csv(path / "label.csv")
        img_ls, name_ls = readImgs(df, path)
        threshold_path = path / "threshold.pickle"
        folder = checkFolder(Path(f"/mnt/usb/jupyter-mimikuo/cloud_opening/{city}"))

        if os.path.exists(threshold_path):
            with open(threshold_path, "rb") as f:
                threshold_dic = pickle.load(f)
        else:
            img_ls, _ = readImgs(df, path)
            threshold_dic = getThresholds(img_ls)

        for name, img in zip(name_ls, img_ls):
            if Path(f"{folder}/{name}.png").exists():
                continue
            fig, axs = plt.subplots(
                2, 10, figsize=(16, 5), dpi=300, constrained_layout=True
            )
            plt.subplots_adjust(top=0.9, wspace=0.1, hspace=0.1)

            for i, ax in zip(threshold_dic.keys(), fig.get_axes()):
                if i == 0:
                    ax.set_title("Origin")
                    ax.imshow(img.astype(np.uint8), cmap="gray")
                else:
                    if i < 50:
                        masked_img = img < i
                    else:
                        masked_img = img > i
                    print(type(masked_img))
                    print(masked_img.shape)

                    kernel = np.ones((5, 5), np.uint8)
                    opening = cv.morphologyEx(
                        np.float32(masked_img), cv.MORPH_OPEN, kernel
                    )
                    print(city, name, i, ":", threshold_dic[i])
                    ax.imshow(opening.astype(np.uint8), cmap="gray")
                    ax.set_title(str(f"{i}: {int(threshold_dic[i])}"))
                ax.set_axis_off()
            plt.savefig(f"{folder}/{name}.png")


def main(option=1) -> None:
    if option == 0:
        plotThreshold()
    else:
        plotOpening()


if __name__ == "__main__":
    main()
