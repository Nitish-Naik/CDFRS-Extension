import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random


def generate_1hour_data(
    n_machines=5,
    start_time="2026-02-24 11:00:00",
    anomaly_prob=0.1
):
    start_time = pd.to_datetime(start_time)
    end_time = start_time + timedelta(hours=1)

    all_rows = []

    for machine_id in range(n_machines):

        current_time = start_time

        # Base normal operating values per machine
        base_temp = np.random.uniform(48, 55)
        base_vibration = np.random.uniform(0.48, 0.6)
        base_pressure = np.random.uniform(101.25, 101.35)
        base_humidity = np.random.uniform(29.5, 30.8)

        last_vibration = base_vibration

        while current_time < end_time:

            anomaly = 0
            anomaly_type = "none"

            # Normal behavior
            temperature = base_temp + np.random.normal(0, 0.5)
            vibration = base_vibration + np.random.normal(0, 0.02)
            pressure = base_pressure + np.random.normal(0, 0.02)
            humidity = base_humidity + np.random.normal(0, 0.5)

            # Inject anomaly
            if random.random() < anomaly_prob:
                anomaly = 1
                anomaly_type = random.choice(["drift", "stuck"])

                if anomaly_type == "drift":
                    temperature += np.random.uniform(5, 8)
                    vibration += np.random.uniform(0.1, 0.15)

                elif anomaly_type == "stuck":
                    vibration = last_vibration  # freeze sensor

            last_vibration = vibration

            row = {
                "timestamp": current_time,
                "machine_id": machine_id,
                "temperature": round(temperature, 6),
                "vibration": round(vibration, 6),
                "pressure": round(pressure, 6),
                "humidity": round(humidity, 6),
                "anomaly": anomaly,
                "anomaly_type": anomaly_type,
            }

            all_rows.append(row)

            # advance 5–15 seconds
            current_time += timedelta(seconds=np.random.randint(5, 15))

    df = pd.DataFrame(all_rows)
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


if __name__ == "__main__":
    df = generate_1hour_data()
    df.to_csv("iot_11_to_12.csv", index=False)
    print("Generated iot_11_to_12.csv")