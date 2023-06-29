import os
from pathlib import Path
import pandas as pd
import sys
import argparse
import tensorflow as tf
from IPython.display import Image, display
import matplotlib.pyplot as plt  # import figure
from scipy.stats.stats import pearsonr
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

root_dir = Path(__file__).parents[2]
sys.path.insert(0, str(root_dir / "code/util"))
sys.path.insert(0, str(root_dir / "code/model"))

from DengueNet import *
from DataPreprocess import *
from ModelUtil import *
from ForFeatureEngineering import getFeatureList, getThresholdPairList
from CalculateMetric import evalModel, plotModelStructure, plotHeatmap


def plotPredictions(city_ls):
    root_folder = Path("/mnt/usb/jupyter-mimikuo/demo/")
    df = pd.read_csv(root_folder / "predictions.csv")
    print(df.head(4))
    for city in city_ls:
        path = Path(f"/mnt/usb/jupyter-mimikuo/dataset/sorted/{city}")
        ground_true = pd.read_csv(path / "label.csv")["Cases"][5:]
        print("Ground Truth:", len(ground_true))
        selected_df = df.loc[df["City"] == int(city)].copy()

        model_ls = ["FengVit"]
        swap_ls = [True, False]
        name = getCodeToNameMap()[int(city)]
        plotSelected(
            model_ls, swap_ls, selected_df, ground_true, root_folder / city, name
        )

        model_ls = ["VitFengCase", "FengVit", "Case"]
        swap_ls = [True]
        name = getCodeToNameMap()[int(city)]
        plotSelected(
            model_ls, swap_ls, selected_df, ground_true, root_folder / city, name
        )

        model_ls = ["FEng", "ViT", "VitFeng"]
        swap_ls = [True]
        name = getCodeToNameMap()[int(city)]
        plotSelected(
            model_ls, swap_ls, selected_df, ground_true, root_folder / city, name
        )

        model_ls = ["FengVit", "Case"]
        swap_ls = [True]
        name = getCodeToNameMap()[int(city)]
        plotSelected(
            model_ls, swap_ls, selected_df, ground_true, root_folder / city, name
        )


def plotSelected(model_ls, swap_ls, df, ground_true, folder, city_name):
    color_ls = ["blue", "orange", "cyan", "green", "olive", "purple", "brown"]
    name = "_".join(model_ls)
    plt.figure(figsize=(8, 5))
    length = range(len(ground_true))
    plt.rcParams.update({"font.size": 14})
    plt.plot(length, ground_true, label="Ground Truth", color="tab:red")
    plt.plot(
        [156 - 5 - 16, 156 - 5 - 16],
        [min(ground_true), max(ground_true)],
        linestyle="--",
        color="tab:gray",
    )
    plt.plot(
        [156 - 5 - 32, 156 - 5 - 32],
        [min(ground_true), max(ground_true)],
        linestyle="--",
        color="tab:gray",
    )

    counter = 0
    for i, row in df.iterrows():
        label = str(row["Model"])
        swap = bool(row["Swap"])

        pred = str(row["Prediction"]).replace(",", "").split()
        pred = [int(x) for x in pred]
        corr = pearsonr(pred[-16:], ground_true[-16:])
        df.at[i, "coorelation"] = round(corr[0], 3)
        df.at[i, "p_value"] = round(corr[1], 3)
        if (label == "Case" and "Case" in model_ls) or (
            label in model_ls and swap in swap_ls
        ):
            text = getMap(label)
            if swap:
                text += " (w/ CSR)"
            plt.plot(pred, label=text, color=color_ls[counter])
            counter += 1

    plt.ylabel("Dengue Cases")
    plt.xlabel("Week")
    plt.legend()
    plt.savefig(folder / f"{name}.png", dpi=300)
    df = df.drop(columns=["Prediction"])
    df.to_csv(folder / "coorelation.csv")


def getMap(label):
    if label == "Case":
        return "Case"
    elif label == "FengVit":
        return "ViT+FEng"
    elif label == "VitFengCase":
        return "ViT+FEng+Case"
    return label


def plotVitHeatmap(args, city_ls):
    selected_feature_ls = getFeatureList()
    cloud = 70
    shadow = 10
    lstm_week = 5
    len_callback = 10
    leaky_ls = [0.15, 0.275, 0.275, 0.2, 0.2]
    city_ls = ["5001", "50001", "54001", "73001", "76001"]
    if args.city:
        city_ls = [city_ls[args.city]]

    for city, leaky_rate in zip(city_ls, leaky_ls):
        path = Path(f"/mnt/usb/jupyter-mimikuo/dataset/sorted/{city}")
        df = pd.read_csv(path / "label.csv")
        all_feature_ls = readFeatureName(df, path)

        for model_name in getModels(only_vit=True).keys():
            for swap in [True, False]:
                print(model_name)
                feature_folder_name = getFeatureFolderName(
                    all_feature_ls, selected_feature_ls
                )
                setting = getSetting(
                    args.project_folder,
                    swap,
                    model_name,
                    city,
                    include_top=False,
                    threshold_dic=None,
                    selected_feature_ls=selected_feature_ls,
                    threshold_cloud=cloud,
                    threshold_shadow=shadow,
                    feature_folder_name=feature_folder_name,
                    lstm_week=lstm_week,
                )

                model_parameter = (
                    DengueNet.getHyperparameter(
                        leaky_rate=leaky_rate, vit_trainable=True
                    )
                    + "_"
                    + str(len_callback)
                )
                result_location = Path(f"/mnt/usb/jupyter-mimikuo/demo/{city}")
                checkFolder(result_location)

                setting, data, labels, scaler = setup(setting, city)
                data = getInput(model_name, data, labels)

                printTime("Training ", 1)
                model = createModel(setting, leaky_rate)
                history = model.fit(
                    data["train"]["input"],
                    data["train"]["output"],
                    epochs=10,
                    batch_size=8,
                    validation_data=(data["val"]["input"], data["val"]["output"]),
                    verbose=0,
                    callbacks=getCallback2(len_callback),
                )
                printTime("Training")
                encoder = keras.Sequential()
                for idx, layer in enumerate(model.layers):
                    if idx == 0 or 2:
                        encoder.add(layer)
                    if idx == 2:
                        break

                printTime("Inference", 1)
                setting, data, labels, scaler = setup(setting, city)
                data = getInput("Vit", data, labels)

                scaled_pred = encoder.predict(data["test"]["input"])
                print(scaled_pred.shape)
                printTime("Inference")

                compared_data = []
                all_heatmap_data = []
                for i in range(4, scaled_pred.shape[0]):
                    heatmap_data = []
                    for j in range(4):
                        heatmap_data.append(scaled_pred[i - j, j])
                    heatmap_data = np.average(heatmap_data, axis=0)
                    all_heatmap_data.append(heatmap_data)
                    compared_data.append(data["test"]["input"][i, 0, :, :, :])

                compared_data = np.array(compared_data)
                all_heatmap_data = np.array(all_heatmap_data)
                print(result_location)
                plotHeatmap(compared_data, all_heatmap_data, result_location)


def main(args):
    city_ls = getCityList(args.reverse_city_sequence)
    plotVitHeatmap(args, city_ls)


if __name__ == "__main__":
    args = setParser(True)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_index
    gpus = tf.config.list_physical_devices("GPU")

    if gpus:
        try:
            if args.limit_gpu:
                tf.config.set_logical_device_configuration(
                    gpus[0],
                    [
                        tf.config.LogicalDeviceConfiguration(
                            memory_limit=args.gpu_size * 1024
                        )
                    ],
                )
            logical_gpus = tf.config.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)

    main(args)
