import sklearn
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import itertools
import matplotlib.cm as cm

from ModelUtil import *


def plotHeatmap(data, scaled_pred, result_location, alpha=0.1):
    """
    Ref:
      https://keras.io/examples/vision/grad_cam/#setup
      https://www.kaggle.com/code/basu369victor/covid19-detection-with-vit-and-heatmap#Model-Performance-Visulization
      https://github.com/jeonsworld/ViT-pytorch/blob/main/visualize_attention_map.ipynb
      https://www.kaggle.com/code/piantic/vision-transformer-vit-visualize-attention-map/notebook
    """
    for i, origin_img in enumerate(data):
        heatmap = scaled_pred[i].reshape(16, 16, 3)
        norm_heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = np.average(norm_heatmap, axis=2)  # Rescale heatmap to a range 0-255
        heatmap = np.uint8(255 * heatmap)

        origin_img = tf.maximum(origin_img, 0) / tf.math.reduce_max(origin_img)
        origin_img = np.uint8(255 * origin_img)

        jet = cm.get_cmap("jet")  # Use jet colormap to colorize heatmap
        jet_colors = jet(np.arange(256))[:, :3]  # Use RGB values of the colormap
        jet_heatmap = jet_colors[heatmap]

        # Create an image with RGB colorized heatmap
        jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((origin_img.shape[0], origin_img.shape[1]))
        print(1, jet_heatmap.size, origin_img.shape)
        jet_heatmap.save(result_location / f"{i}_jet_heatmap.png")

        # Superimpose the heatmap on original image
        jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

        print(2, jet_heatmap.shape, origin_img.shape)
        print(tf.math.reduce_max(jet_heatmap), tf.math.reduce_min(jet_heatmap))
        print(tf.math.reduce_max(origin_img), tf.math.reduce_min(origin_img))

        superimposed_img = jet_heatmap * alpha + origin_img
        superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
        keras.preprocessing.image.array_to_img(origin_img).save(
            result_location / f"{i}_jet_norm.png"
        )
        superimposed_img.save(result_location / f"{i}_jet_merge.png")


def plotModelStructure(model, path, file_name="model_summary.log") -> None:
    f = open(path / file_name, "w")
    try:
        model.summary(print_fn=lambda x: f.write(x + "\n"))
    except:
        f.write("Model not built error.")
    f.close()


def symmetricMeanAbsolutePercentageError(actual, predicted) -> float:
    actual, predicted = actual.flatten(), predicted.flatten()
    return round(
        100
        * np.mean(
            (2 * np.abs(predicted - actual)) / (np.abs(actual) + np.abs(predicted))
        ),
        3,
    )


def computeMetric(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    smape = symmetricMeanAbsolutePercentageError(actual, predicted)

    # Sqpuare parameter: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
    rmse = mean_squared_error(actual, predicted, squared=False)

    return mae, smape, rmse


def evalModel(
    origin_model, data, f, result_path, scaler, history, len_callback, trials=3
):
    total_normal_pred_ls = []
    total_scaled_pred_ls = []

    for i in range(trials):
        normal_pred_ls = []
        scaled_pred_ls = []
        model = origin_model
        _ = model.fit(
            data["train"]["input"],
            data["train"]["output"],
            epochs=10,
            batch_size=8,
            validation_data=(data["val"]["input"], data["val"]["output"]),
            callbacks=getCallback(len_callback),
        )

        for split in data.keys():
            scaled_pred = model.predict(data[split]["input"])
            scaled_pred_ls.append(scaled_pred)
            normal_pred = np.around(scaler.inverse_transform(scaled_pred).flatten())
            normal_pred_ls.append(normal_pred)

        for pred, total_ls in zip(
            [normal_pred_ls, scaled_pred_ls],
            [total_normal_pred_ls, total_scaled_pred_ls],
        ):
            pred = list(itertools.chain.from_iterable(pred))
            total_ls.append(pred)

    plotHistory(history, result_path)
    avg_total_normal_pred_ls = np.average(np.array(total_normal_pred_ls), axis=0)
    performance = writeMetric(
        data,
        avg_total_normal_pred_ls,
        np.array(total_normal_pred_ls),
        f,
        y_true_label="origin_output",
    )
    plotPrediction(
        data, avg_total_normal_pred_ls, result_path, y_true_label="origin_output"
    )
    return performance


def writeMetric(data, avg_pred, all_pred, f, y_true_label="origin_output") -> None:
    start = 0
    performance = {}
    all_performance = {}
    metric_name = ["MAE", "sMAPE", "RMSE"]
    print("Writing metrics...")

    for split in data.keys():
        performance[split] = {}
        all_performance[split] = {}
        for metric in metric_name:
            all_performance[split][metric] = []

        f.write(f"{split}================================\n")
        print(np.array(all_pred).shape)
        length = data[split][y_true_label].shape[0]
        for i in range(len(all_pred)):
            pred = all_pred[i]
            print(pred)
            print(
                length,
                data[split][y_true_label].shape,
                pred.shape,
                pred[start : start + length].shape,
            )
            mae, smape, rmse = computeMetric(
                data[split][y_true_label], pred[start : start + length]
            )
            for metric, val in zip(metric_name, [mae, smape, rmse]):
                all_performance[split][metric].append(val)
        start = start + length

        for metric in metric_name:
            performance_ls = np.array(all_performance[split][metric])
            print(performance_ls.shape, performance_ls)
            performance[split][metric] = round(np.average(performance_ls), 2)
            performance[split][f"{metric}_std"] = round(np.std(performance_ls), 2)
            std_name = metric + "_std"
            f.write(
                f"  {metric}:   {performance[split][metric]}±{performance[split][std_name]}\n"
            )
            print(
                f"  {metric}:   {performance[split][metric]}±{performance[split][std_name]}\n"
            )

        performance[split]["pred"] = avg_pred
        f.write((",".join(["%d "] * pred.size) + "\n") % tuple(avg_pred))

    return performance


def plotHistory(history, folder):
    path = Path(folder) / f"history.png"
    plt.figure()
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="validation")
    plt.title("History")
    plt.legend()
    plt.savefig(path)


def plotPrediction(data, y_pred, folder, y_true_label):
    plt.figure()
    path = Path(folder) / f"predict_{y_true_label}.png"

    y_true = []
    for i, key in enumerate(data.keys()):
        true = data[key][y_true_label]
        y_true.append(true)

    y_true = list(itertools.chain.from_iterable(y_true))
    max_y, min_y = max(y_true) + 5, min(y_true) - 5

    pre = 0
    for i, key in enumerate(data.keys()):
        true = data[key][y_true_label]
        plt.plot(
            [pre + len(true), pre + len(true)],
            [min_y, max_y],
            label="_nolegend_",
            color="g",
        )
        pre += len(true)

    for y, name, color in zip(
        [y_true, y_pred], ["GroundTruth", "Prediction"], ["b", "r"]
    ):
        plt.plot(range(len(y)), y, label=name, color=color)

    plt.legend(loc="upper left")
    plt.savefig(path)
