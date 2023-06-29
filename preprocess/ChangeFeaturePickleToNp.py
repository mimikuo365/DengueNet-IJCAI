import numpy as np
from pathlib import Path
import pandas as pd
import pickle
import os
import sys
import argparse

root_dir = Path(__file__).parents[2]
sys.path.insert(0, str(root_dir / "code/util"))
sys.path.insert(0, str(root_dir / "code/model"))
from DataPreprocess import *


def createNumpy(path):
    df = pd.read_csv(path / "label.csv")
    feature_np = []
    np_location = path / "feature.npy"

    if np_location.exists():
        feature_np = np.load(np_location)
        print(feature_np.shape)
        return

    for i, row in df.iterrows():
        epiweek = str(row["epiweek"])
        print(os.path.basename(path), epiweek)
        feature_path = path / f"feature/{epiweek}.pickle"
        with open(feature_path, "rb") as handle:
            feature_dic = pickle.load(handle)

        tmp_np = []
        for tile in feature_dic.keys():
            tile_np = []
            for feature in feature_dic[tile].keys():
                tile_np.append(feature_dic[tile][feature])
            tmp_np.append(tile_np)
        feature_np.append(tmp_np)

    feature_np = np.array(feature_np)
    print(feature_np.shape)
    np.save(np_location, feature_np)


def createImgLs(path):
    df = pd.read_csv(path / "label.csv")
    rgb_np, feng_np = [], []
    rgb_location, feng_location = path / "rgb.npy", path / "feng.npy"

    if rgb_location.exists():
        rgb_np = np.load(rgb_location)
        feng_np = np.load(feng_location)
        print(rgb_np.shape, feng_np.shape)
        return

    for i, row in df.iterrows():
        epiweek = str(row["epiweek"])
        print(os.path.basename(path), epiweek)
        img_path = path / f"image/{epiweek}.tiff"
        origin_img = io.imread(img_path)

        feng_img = origin_img[:, :, 11]
        resized_rgb_img = resizeToRgbImg(origin_img, [1, 2, 3], (224, 224, 3))

        feng_np.append(feng_img)
        rgb_np.append(resized_rgb_img)

    print(np.array(feng_np).shape)
    print(np.array(rgb_np).shape)
    np.save(feng_location, feng_np)
    np.save(rgb_location, rgb_np)


def main(args):
    city_ls = getCityList()
    if args.city:
        createImgLs(
            Path(f"/mnt/usb/jupyter-mimikuo/dataset/sorted/{city_ls[args.city]}/")
        )
    else:
        for city in city_ls:
            createImgLs(Path(f"/mnt/usb/jupyter-mimikuo/dataset/sorted/{city}/"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", type=int)
    args = parser.parse_args()
    main(args)
