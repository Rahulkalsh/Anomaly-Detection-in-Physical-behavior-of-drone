import numpy as np
import pandas as pd
from pyulog import ULog
from pyulog.px4 import PX4ULog
from sklearn.preprocessing import RobustScaler

def preprocess_ulog(ulog_path):
    # Define the required columns for each dataset
    Columns_require = {
        "vehicle_attitude": ["roll", "pitch", "yaw"],
        "vehicle_local_position": ["vx", "vy", "vz"],
        "sensor_combined": ["gyro_rad[0]", "gyro_rad[1]", "gyro_rad[2]"],
        "vehicle_magnetometer": ["magnetometer_ga[0]", "magnetometer_ga[1]", "magnetometer_ga[2]"],
        "sensor_baro": ["pressure", "temperature"],
        "battery_status": ["voltage_v", "current_a"],
    }

    def extract_data(ulog):
        cols = []
        for dataset, attrs in Columns_require.items():
            try:
                # Get the dataset
                dataset_data = ulog.get_dataset(dataset)
                data = dataset_data.data
                timestamp = data["timestamp"]
            except (KeyError, IndexError, ValueError) as e:
                return f"Error extracting dataset {dataset}: {e}"

            for attr in attrs:
                values = data.get(attr)
                if values is None:
                    return f"Attribute {attr} not found in dataset {dataset}"
                cols.append({
                    "dataset": dataset,
                    "attr": attr,
                    "timestamp": timestamp,
                    "values": values,
                })
        return cols

    def compress(col1, col2):
        if len(col1["timestamp"]) < len(col2["timestamp"]):
            col1, col2 = col2, col1
        idx = np.searchsorted(col1["timestamp"], col2["timestamp"], side="right")
        start = 0
        final = np.zeros_like(col2["timestamp"], dtype=np.float32)
        for j, i in enumerate(idx):
            if i > start:
                final[j] = col1["values"][start:i].mean()
            else:
                final[j] = col1["values"][start if start < len(col1["values"]) else -1]
            start = i
        col1["timestamp"] = col2["timestamp"]
        col1["values"] = final

    def expand(col1, col2):
        if len(col1["timestamp"]) < len(col2["timestamp"]):
            col1, col2 = col2, col1
        idx = np.searchsorted(col2["timestamp"], col1["timestamp"], side="left")
        final = np.zeros_like(col1["timestamp"], dtype=np.float32)
        for j, i in enumerate(idx):
            final[j] = col2["values"][i if i < len(col2["values"]) else -1]
        col2["timestamp"] = col1["timestamp"]
        col2["values"] = final

    def align_cols(cols):
        # Use the longest timestamp series as the reference
        idx = 0
        for i in range(1, len(cols)):
            if len(cols[i]["timestamp"]) > len(cols[idx]["timestamp"]):
                idx = i
        reference_col = cols[idx]
        for col in cols:
            if col is not reference_col:
                if len(col["timestamp"]) > len(reference_col["timestamp"]):
                    compress(col, reference_col)
                else:
                    expand(reference_col, col)

    def cols_to_df(cols):
        return pd.DataFrame(
            np.vstack([cols[0]["timestamp"]] + [col["values"] for col in cols]).T,
            columns=["timestamp"] + [f'{col["dataset"]}.{col["attr"]}' for col in cols],
        )

    def interpolate_missing_values(df):
        for column in df.columns:
            df[column] = df[column].interpolate(method='linear', limit_direction='both')
        return df

    def scale_data(df):
        scaler = RobustScaler()
        df = df.iloc[:,1:]
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        return df

    try:
        # Initialize ULog and PX4ULog
        ulog = ULog(ulog_path)
        px4ulog = PX4ULog(ulog)
        px4ulog.add_roll_pitch_yaw()
    except ValueError as e:
        return f"Error reading ULog file: {e}"
    except Exception as e:
        return f"Unexpected error: {e}"

    # Extract data
    cols = extract_data(ulog)
    if isinstance(cols, str):
        return f"Failed to extract data: {cols}"
    if min(len(col["timestamp"]) for col in cols) < 20:
        return "Data is too short."

    # Align columns and preprocess data
    align_cols(cols)
    df = cols_to_df(cols)
    df = interpolate_missing_values(df)
    if df.shape[0] < 100:
        return "Data too short after interpolation."
    df = scale_data(df)
    return df
