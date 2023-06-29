import sys
import pandas as pd
from skimage import io
import os
from pathlib import Path

root_dir = Path(__file__).parents[2]
sys.path.insert(0, str(root_dir / "code/util"))
sys.path.insert(0, str(root_dir / "code/model"))
from DataPreprocess import *


def orderingFiles():
    # Previous ordering code
    root_dir = Path(__file__).parent.parent
    dataset_folder = root_dir / "dataset"
    for code in city_dic:
        folder = dataset_folder / "sorted" / str(code) / "image"
        if not folder.exists():
            folder.mkdir()

    for code in city_dic:
        print(code, city_dic[code])
        source_img_folder = dataset_folder / "image" / str(code)
        target_img_folder = dataset_folder / "sorted" / str(code) / "image"
        if source_img_folder.exists():
            print(source_img_folder, target_img_folder)
            os.system(f"mv {source_img_folder} {target_img_folder}")


def getOriginImg(main_folder, city_ls):
    for city in city_ls:
        path = main_folder / f"dataset/sorted/{city}/image"
        result_folder = main_folder / f"demo/{city}/origin_rgb_resized"
        checkFolder(result_folder)

        for img_name in os.listdir(path):
            print(img_name)
            origin_img = io.imread(path / img_name)
            resized_rgb_img = resizeToRgbImg(origin_img, [1, 2, 3], (224, 224, 3))
            print(resized_rgb_img.shape)
            resized_rgb_img = keras.preprocessing.image.array_to_img(resized_rgb_img)
            resized_rgb_img.save(result_folder / str(img_name.split(".")[0] + ".png"))


def main():
    main_folder = Path("/mnt/usb/jupyter-mimikuo/")
    city_ls = getCityList()
    selected_feature_ls = getFeatureList()
    leaky_rate_ls = getLeakyRateList()
    vit_norm_ls = [False]
    threshold_pair_ls = list(getThresholdPairList())
    lstm_week_ls = [5]
    df_columns = createColumns()
    city_ls = getCityList()

    for city in city_ls:
        path = Path(f"/mnt/usb/jupyter-mimikuo/dataset/sorted/{city}")
        df = pd.read_csv(path / "label.csv")
        all_feature_ls = readFeatureName(df, path)

        for cloud, shadow in threshold_pair_ls:
            print(cloud, shadow)
            if cloud != 70 and shadow != 10:
                continue
            for lstm_week in lstm_week_ls:
                result_df = pd.DataFrame(columns=df_columns)
                feature_folder_name = getFeatureFolderName(
                    all_feature_ls, selected_feature_ls
                )
                setting = getSetting(
                    "/mnt/usb/jupyter-mimikuo/",
                    True,
                    "No_model",
                    city,
                    include_top=False,
                    threshold_dic=None,
                    selected_feature_ls=selected_feature_ls,
                    threshold_cloud=cloud,
                    threshold_shadow=shadow,
                    feature_folder_name=feature_folder_name,
                    lstm_week=lstm_week,
                )

                setting, data, labels, scaler = setup(setting, city, save_img=True)
                for split in data.keys():
                    print(data[split]["x_vit"].shape)


if __name__ == "__main__":
    main()
