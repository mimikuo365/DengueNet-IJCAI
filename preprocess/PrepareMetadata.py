import pandas as pd
import os
from pathlib import Path
from epiweeks import Week
from datetime import date

def get_epi_week(origin_str):
    process_str = ''.join(filter(lambda c: (c.isdigit() or c == '-'), origin_str)) 
    date_ls = process_str.split('-')
    return Week.fromdate(date(int(date_ls[0]), int(date_ls[1]), int(date_ls[2])))

def main():
    dataset_folder = Path('/mnt/usb/jupyter-mimikuo/dataset')
    temp_df = pd.read_csv(dataset_folder / 'csv/temperature_all_no_missing.csv')
    pred_df = pd.read_csv(dataset_folder / 'csv/precipitation_all.csv')
    df = temp_df.merge(pred_df, on='LastDayWeek')
    # print(df.isnull().values.sum())

    city_dic = {
        5001: "Medellín",
        50001: "Villavicencio",
        76001: "Cali",
        73001: "Ibagué",
        54001: "Cúcuta"
    }

    for code in city_dic:
        new_df_path = dataset_folder / 'sorted' / str(code) / "metadata.csv"
        new_df = pd.DataFrame(columns=["temp", "prec"])

        selected_df = df.copy(deep=True)
        drop_cols = []
        for column in selected_df.columns:
            if column != "LastDayWeek" and str(code) not in column:
                drop_cols.append(column)
        selected_df = selected_df.drop(columns=drop_cols)
        print(selected_df.head(5))
        
        for i, row in selected_df.iterrows():
            tmp_df = {}
            epiweek = str(get_epi_week(str(row["LastDayWeek"])))
            if "2016" in epiweek or "2017" in epiweek or "2018" in epiweek:
                tmp_df["epiweek"] = epiweek
                tmp_df["temp"] = row[f"temperature_{code}"]
                tmp_df["prec"] = row[f"precipitation_{code}"]
                new_df = new_df.append(tmp_df, ignore_index=True)


        new_df.to_csv(new_df_path, index=False)

if __name__ == "__main__":
    main()