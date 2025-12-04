# generate_sample_data.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

def generate_csv(path="data/raw_weather.csv", days=365*2, start_date="2023-01-01"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    start = datetime.fromisoformat(start_date)
    dates = [start + timedelta(days=i) for i in range(days)]

    # create realistic seasonal temperature pattern + noise
    day_of_year = np.array([d.timetuple().tm_yday for d in dates])
    # mean daily temp: sine wave for seasons
    temps_mean = 20 + 10 * np.sin(2 * np.pi * (day_of_year - 172) / 365)  # peak ~ day 172 (June)
    temps = temps_mean + np.random.normal(0, 3, size=len(dates))  # daily mean
    temp_max = temps + np.random.uniform(2, 8, size=len(dates))
    temp_min = temps - np.random.uniform(2, 8, size=len(dates))

    # rainfall: random with seasonal bias
    rainfall = np.clip(np.random.exponential(scale=2.0, size=len(dates)) * (0.5 + 0.5*np.cos(2*np.pi*(day_of_year-200)/365)), 0, None)
    # humidity correlated with rainfall
    humidity = np.clip(60 + (rainfall > 0) * np.random.uniform(10, 25, size=len(dates)) - (temps-20)*0.5 + np.random.normal(0,5,len(dates)), 0, 100)

    # some missing values inserted randomly
    df = pd.DataFrame({
        "date": [d.strftime("%Y-%m-%d") for d in dates],
        "temp_max": np.round(temp_max, 1),
        "temp_min": np.round(temp_min, 1),
        "temp_mean": np.round(temps, 1),
        "rainfall_mm": np.round(rainfall, 1),
        "humidity": np.round(humidity, 1)
    })

    # insert NaNs randomly
    for col in ["temp_max", "temp_min", "rainfall_mm", "humidity"]:
        mask = np.random.rand(len(df)) < 0.03  # 3% missing
        df.loc[mask, col] = np.nan

    df.to_csv(path, index=False)
    print(f"Generated sample data at {path}")

if __name__ == "__main__":
    generate_csv()