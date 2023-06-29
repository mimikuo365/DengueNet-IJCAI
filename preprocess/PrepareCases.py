import pandas as pd
import os
from pathlib import Path
from epiweeks import Week
from datetime import date


def get_epi_week(origin_str):
    process_str = "".join(filter(lambda c: (c.isdigit() or c == "-"), origin_str))
    date_ls = process_str.split("-")
    return Week.fromdate(date(int(date_ls[0]), int(date_ls[1]), int(date_ls[2])))


def main():
    root_dir = Path(__file__).parent.parent.parent
    dataset_folder = root_dir / "dataset"
    df = pd.read_csv(dataset_folder / "csv/1029_Municipalities_cases.csv")

    city_dic = {
        5001: "Medellín",
        50001: "Villavicencio",
        76001: "Cali",
        73001: "Ibagué",
        54001: "Cúcuta",
    }

    for code in city_dic:
        folder = dataset_folder / "sorted" / str(code)
        if not folder.exists():
            folder.mkdir()

    for code in city_dic:
        print(code, city_dic[code])

        # cases
        selected_df = (
            df[df["Municipality code"] == code]
            .sort_values(by=["epiweek"])
            .reset_index()
        )
        selected_df = selected_df.drop(
            columns=["index", "Municipality code", "Municipality"]
        )
        selected_df["epiweek"] = selected_df["epiweek"].str.replace("/w", "")
        selected_df = selected_df.set_index("epiweek")

        df_folder = dataset_folder / "sorted" / str(code)
        selected_df.to_csv(df_folder / "label.csv")

        # images
        img_folder = dataset_folder / "sorted" / str(code) / "image"
        for img in img_folder.glob("*"):
            if img.stem.startswith("image"):
                newName = str(get_epi_week(str(img.stem))) + img.suffix
                os.system(f"mv {img} {str(img.parent / newName)}")
            elif img.stem.startswith("2015"):
                os.system(f"rm {img}")

        # features
        img_folder = dataset_folder / "sorted" / str(code) / "feature"
        for img in img_folder.glob("*"):
            if img.stem.startswith("image"):
                newName = str(get_epi_week(str(img.stem))) + img.suffix
                os.system(f"mv {img} {str(img.parent / newName)}")


if __name__ == "__main__":
    main()
