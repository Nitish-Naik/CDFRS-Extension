import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os


def generate_50mb_first_10min(
    n_machines=5,
    target_rows=500_000,
    start_time="2026-02-24 11:00:00",
    anomaly_prob=0.1,
    output_file="generated_10min_windowed.csv"
):

    start_time = pd.to_datetime(start_time)
    ten_min_end = start_time + timedelta(minutes=10)

    rows = []

    # Distribute rows evenly across machines
    rows_per_machine = target_rows // n_machines

    for machine_id in range(n_machines):

        base_temp = np.random.uniform(48, 55)
        base_vibration = np.random.uniform(0.48, 0.6)
        base_pressure = np.random.uniform(101.25, 101.35)
        base_humidity = np.random.uniform(29.5, 30.8)

        for i in range(rows_per_machine):

            # Spread timestamps evenly within first 10 mins
            seconds_offset = np.random.uniform(0, 600)
            current_time = start_time + timedelta(seconds=seconds_offset)

            anomaly = 0
            anomaly_type = "none"

            temperature = base_temp + np.random.normal(0, 0.5)
            vibration = base_vibration + np.random.normal(0, 0.02)
            pressure = base_pressure + np.random.normal(0, 0.02)
            humidity = base_humidity + np.random.normal(0, 0.5)

            if random.random() < anomaly_prob:
                anomaly = 1
                anomaly_type = random.choice(["drift", "stuck"])

                if anomaly_type == "drift":
                    temperature += np.random.uniform(5, 8)
                    vibration += np.random.uniform(0.1, 0.15)
                else:
                    vibration = base_vibration

            rows.append([
                current_time,
                machine_id,
                round(temperature, 6),
                round(vibration, 6),
                round(pressure, 6),
                round(humidity, 6),
                anomaly,
                anomaly_type
            ])

    df = pd.DataFrame(rows, columns=[
        "timestamp",
        "machine_id",
        "temperature",
        "vibration",
        "pressure",
        "humidity",
        "anomaly",
        "anomaly_type"
    ])

    df.to_csv(output_file, index=False)

    size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"Generated {len(df)} rows")
    print(f"File size: {size_mb:.2f} MB")
    print(f"Saved as: {output_file}")


if __name__ == "__main__":
    generate_50mb_first_10min()