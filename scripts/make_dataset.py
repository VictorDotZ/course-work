import json
import os
import sys

import pandas as pd


if __name__ == "__main__":
    source_path = sys.argv[1]
    save_path = sys.argv[2]

    filenames = os.listdir(source_path)

    df = pd.DataFrame()

    for filename in filenames:
        feature_name = filename.split(".")[0]
        with open(source_path + filename, "r", encoding="utf-8") as f:
            data = json.load(f)
            values = data[feature_name]
            timestamps = list(map(lambda observation: observation["x"], values))
            df.index = df.index.append(pd.Index(timestamps))

    df.index = pd.to_datetime(df.index, unit="ms").date
    df.index = df.index.unique()
    df = df.sort_index()

    for filename in filenames:
        feature_name = filename.split(".")[0]
        with open(source_path + filename, "r", encoding="utf-8") as f:
            data = json.load(f)
            values = data[feature_name]
            timestamps = list(map(lambda observation: observation["x"], values))
            features = list(map(lambda observation: observation["y"], values))

            for timestamp, feature in zip(
                pd.to_datetime(timestamps, unit="ms").date, features
            ):
                df.loc[timestamp, feature_name] = feature

    df.to_csv(save_path + "btc.csv", index_label="timestamp")
