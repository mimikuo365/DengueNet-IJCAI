import os
from pathlib import Path
import pandas as pd
import sys
import argparse
import tensorflow as tf
from datetime import datetime

root_dir = Path(__file__).parents[2]
sys.path.insert(0, str(root_dir / "code/util"))
sys.path.insert(0, str(root_dir / "code/model"))

from DengueNet import *
from DataPreprocess import *
from ModelUtil import *
from ForFeatureEngineering import getFeatureList, getThresholdPairList
from CalculateMetric import evalModel, plotModelStructure


def main(args):
    selected_feature_ls = getFeatureList()
    leaky_rate_ls = getLeakyRateList()
    threshold_pair_ls = list(getThresholdPairList(args))
    lstm_week_ls = [3, 5, 7, 9]
    vit_norm_ls = [False]

    date = datetime.now().strftime("%H:%M:%S_%d-%m-%Y") + ".log"
    if args.include_vit:
        print("Include ViT model!!")
        model_dic = getModels(True)
    else:
        model_dic = getModels(False)
    city_ls = getCityList(args.reverse_city_sequence)
    len_callback = 10

    if args.city:
        city_ls = [city_ls[args.city]]

    for city in city_ls:
        path = Path(f"/mnt/usb/jupyter-mimikuo/dataset/sorted/{city}")
        df = pd.read_csv(path / "label.csv")
        all_feature_ls = readFeatureName(df, path)
        threshold_path = path / "threshold.pickle"

        if os.path.exists(threshold_path):
            with open(threshold_path, "rb") as f:
                threshold_dic = pickle.load(f)
        else:
            img_ls, _ = readImgs(df, path)
            threshold_dic = getThresholds(img_ls)

        for model_name in model_dic.keys():
            swap_bool_ls = [True, False]
            if model_name == "Case":
                swap_bool_ls = [False]

            for swap in swap_bool_ls:
                for vit_norm in vit_norm_ls:
                    for cloud, shadow in threshold_pair_ls:
                        for vit_trainable in [True]:  # Use trainable ViT model
                            for lstm_week in lstm_week_ls:
                                df_columns = [
                                    "train_MAE",
                                    "val_MAE",
                                    "test_MAE",
                                    "train_sMAPE",
                                    "val_sMAPE",
                                    "test_sMAPE",
                                    "train_RMSE",
                                    "val_RMSE",
                                    "test_RMSE",
                                ]
                                result_df = pd.DataFrame(columns=df_columns)

                                for leaky_rate in leaky_rate_ls:
                                    feature_folder_name = getFeatureFolderName(
                                        all_feature_ls, selected_feature_ls
                                    )
                                    setting = getSetting(
                                        args.project_folder,
                                        swap,
                                        model_name,
                                        city,
                                        include_top=False,
                                        threshold_dic=threshold_dic,
                                        selected_feature_ls=selected_feature_ls,
                                        threshold_cloud=cloud,
                                        threshold_shadow=shadow,
                                        feature_folder_name=feature_folder_name,
                                        lstm_week=lstm_week,
                                    )

                                    selected_model = (
                                        str(lstm_week)
                                        + "/"
                                        + setting["model"]
                                        + "_"
                                        + str(swap)
                                        + "/"
                                    )
                                    if vit_norm:
                                        selected_model += "NormImg"
                                    else:
                                        selected_model += "OriginImg"

                                    df_location = (
                                        Path(setting["result_folder"])
                                        / selected_model
                                        / "result.csv"
                                    )
                                    model_parameter = (
                                        DengueNet.getHyperparameter(
                                            leaky_rate=leaky_rate,
                                            vit_trainable=vit_trainable,
                                        )
                                        + "_"
                                        + str(len_callback)
                                    )
                                    model_location = getResultLocation(
                                        args,
                                        setting["model_folder"],
                                        selected_model,
                                        model_parameter,
                                    )
                                    result_location = getResultLocation(
                                        args,
                                        setting["result_folder"],
                                        selected_model,
                                        model_parameter,
                                    )
                                    print(model_location)

                                    if os.path.exists(result_location / "history.png"):
                                        print(
                                            f"Error!! Folder {result_location} already existed!!"
                                        )
                                        continue

                                    checkFolder(result_location)
                                    checkFolder(model_location)
                                    setting, data, labels, scaler = setup(
                                        setting, city, normalize_vit=vit_norm
                                    )
                                    data = getInput(model_name, data, labels)

                                    printTime("Model Training (fixed parameter)", 1)
                                    model = createModel(setting, leaky_rate)
                                    history = model.fit(
                                        data["train"]["input"],
                                        data["train"]["output"],
                                        epochs=150,
                                        batch_size=8,
                                        validation_data=(
                                            data["val"]["input"],
                                            data["val"]["output"],
                                        ),
                                        callbacks=getCallback(len_callback),
                                    )
                                    printTime("Model Training (fixed parameter)")

                                    printTime("Model Evaluation", 1)
                                    f = open(result_location / str(date), "w")
                                    plotModelStructure(model, result_location)
                                    performance = evalModel(
                                        model, data, f, result_location, scaler, history
                                    )
                                    printTime("Model Evaluation")

                                    tmp_dic = {"leaky_rate": leaky_rate}
                                    for split in performance.keys():
                                        for metric in performance[split].keys():
                                            print(
                                                f"{split}_{metric}",
                                                performance[split][metric],
                                            )
                                            tmp_dic[f"{split}_{metric}"] = performance[
                                                split
                                            ][metric]
                                    result_df = result_df.append(
                                        tmp_dic, ignore_index=True
                                    )
                                result_df.to_csv(df_location, index=False)


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
