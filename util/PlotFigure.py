from skimage import io
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


def plotSwappedImg(sample_img, result_folder, start, counter):
    resized_rgb_img = keras.preprocessing.image.array_to_img(sample_img)
    resized_rgb_img.save(result_folder / f"{start}{str(counter).zfill(2)}.png")


def plotComparison(swapped_img, origin_img, result_folder, start, counter):
    save_path = result_folder / f"{start}{str(counter).zfill(2)}.png"
    fig, axs = plt.subplots(2, 4, figsize=(12, 5), dpi=300, constrained_layout=True)
    plt.subplots_adjust(top=0.9, wspace=0.1, hspace=0.1)

    for i, ax in enumerate(fig.get_axes()):
        color_map = ["Red", "Green", "Blue"]
        if i == 3:
            ax.set_title("Original Image")
            ax.imshow(origin_img.astype(np.uint8))
        elif i == 7:
            ax.set_title("Swapped Image")
            ax.imshow(swapped_img.astype(np.uint8))
        elif i < 3:
            selected_img = origin_img[:, :, i]
            ax.set_title("Origin " + color_map[i])
            ax.imshow(selected_img.astype(np.uint8), cmap="gray")
        else:
            selected_img = swapped_img[:, :, i % 4]
            ax.set_title("Swapped " + color_map[i % 4])
            ax.imshow(selected_img.astype(np.uint8), cmap="gray")

        ax.set_axis_off()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plotSwapResults(
    origin_img_ls, sample_img_ls, counter, start, result_folder, avg_img
):
    avg_img = np.uint8(
        (avg_img - tf.math.reduce_min(avg_img))
        * 255
        / (tf.math.reduce_max(avg_img) - tf.math.reduce_min(avg_img))
    )

    fig, axs = plt.subplots(1, 4, figsize=(10, 2), dpi=300, constrained_layout=True)
    plt.subplots_adjust(top=0.9, wspace=0.1, hspace=0.1)
    for i, ax in enumerate(fig.get_axes()):
        if i < 3:
            ax.imshow(avg_img[:, :, i].astype(np.uint8), cmap="gray")
            ax.set_title("Band " + str(i + 1))
            ax.set_axis_off()
        else:
            ax.imshow(avg_img.astype(np.uint8))
            ax.set_title("Combined Image")
            ax.set_axis_off()

    plt.savefig(result_folder / "average_rgb.png", dpi=300, bbox_inches="tight")
    plt.close()

    for i in range(sample_img_ls.shape[0]):
        print(f"{start}{str(counter).zfill(2)}: {result_folder}")
        origin_img = origin_img_ls[i, :, :, :]
        sample_img = sample_img_ls[i, :, :, :]

        origin_img = np.uint8(
            (origin_img - tf.math.reduce_min(origin_img))
            * 255
            / (tf.math.reduce_max(origin_img) - tf.math.reduce_min(origin_img))
        )
        sample_img = np.uint8(
            (sample_img - tf.math.reduce_min(sample_img))
            * 255
            / (tf.math.reduce_max(sample_img) - tf.math.reduce_min(sample_img))
        )

        plotSwappedImg(sample_img, result_folder / "single", start, counter)
        plotComparison(
            sample_img, origin_img, result_folder / "comparison", start, counter
        )

        counter += 1
        if counter == 53:
            start += 1
            counter = 1
    return counter, start


def plotComparisonGray(swapped_img, origin_img, result_folder, start, counter):
    save_path = result_folder / f"{start}{str(counter).zfill(2)}.png"
    fig, axs = plt.subplots(1, 2, figsize=(8, 16), dpi=300, constrained_layout=True)
    plt.subplots_adjust(top=0.9, wspace=0.1, hspace=0.1)

    for i, ax in enumerate(fig.get_axes()):
        if i == 0:
            ax.set_title("Original Image")
            ax.imshow(origin_img.astype(np.uint8), cmap="gray")
        else:
            ax.set_title("Swapped Image")
            ax.imshow(swapped_img.astype(np.uint8), cmap="gray")
        ax.set_axis_off()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plotSwapResultsGray(
    origin_img_ls, sample_img_ls, counter, start, result_folder, avg_img
):
    fig, axs = plt.subplots(1, 1, figsize=(10, 10), dpi=300, constrained_layout=True)
    plt.subplots_adjust(top=0.9, wspace=0.1, hspace=0.1)
    for i, ax in enumerate(fig.get_axes()):
        ax.imshow(avg_img.astype(np.uint8), cmap="gray")
        ax.set_title("Average Image")
        ax.set_axis_off()
    plt.savefig(result_folder / "average_gray.png", dpi=300, bbox_inches="tight")
    plt.close()
    for i in range(sample_img_ls.shape[0]):
        print(f"{start}{str(counter).zfill(2)}: {result_folder}")
        origin_img = origin_img_ls[i, :, :]
        sample_img = sample_img_ls[i, :, :]
        plotComparisonGray(
            sample_img, origin_img, result_folder / "comparison", start, counter
        )

        counter += 1
        if counter == 53:
            start += 1
            counter = 1
    return counter, start
