import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


DATA_IN = Path("data/raw_weather.csv")
CLEAN_OUT = Path("output/cleaned_data.csv")
PLOTS_DIR = Path("output/plots")
SUMMARY_MONTH = Path("output/monthly_summary.csv")
SUMMARY_YEAR = Path("output/yearly_summary.csv")
REPORT_MD = Path("report.md")

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
Path("output").mkdir(parents=True, exist_ok=True)


def load_and_inspect(path=DATA_IN):
    df = pd.read_csv(path)
    print("\nHEAD:\n", df.head())
    print("\nINFO:\n")
    print(df.info())
    print("\nDESCRIBE:\n", df.describe(include="all"))
    return df


def clean_data(df):
    print("\nCleaning data...")

    
    df = df.rename(columns=lambda x: x.strip().lower())

    
    df['date'] = pd.to_datetime(df['date'], errors="coerce")
    df = df.dropna(subset=["date"])

    df = df.sort_values("date").reset_index(drop=True)

    
    df = df.set_index("date")

    
    numeric_cols = ["temp_mean", "temp_max", "temp_min", "rainfall_mm", "humidity"]

    for col in numeric_cols:
        if col not in df.columns:
            df[col] = np.nan  
        df[col] = pd.to_numeric(df[col], errors='coerce')


    df[numeric_cols] = df[numeric_cols].interpolate(method='time', limit_direction='both')


    df["rainfall_mm"] = df["rainfall_mm"].clip(lower=0).fillna(0)

    
    df = df.reset_index()

    return df



def compute_stats(df):
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["month_name"] = df["date"].dt.strftime("%b")


    monthly = df.set_index("date").resample("M").agg({
        "temp_mean": ["mean", "min", "max", "std"],
        "rainfall_mm": "sum",
        "humidity": ["mean", "std"]
    })
    monthly.columns = ["_".join(col) for col in monthly.columns]


    yearly = df.groupby("year").agg({
        "temp_mean": ["mean", "min", "max", "std"],
        "rainfall_mm": "sum",
        "humidity": ["mean", "std"]
    })
    yearly.columns = ["_".join(col) for col in yearly.columns]

    return df, monthly, yearly


def plot_daily_temperature(df):
    plt.figure(figsize=(10,4))
    plt.plot(df["date"], df["temp_mean"], label="Mean")
    plt.plot(df["date"], df["temp_max"], label="Max", alpha=0.5)
    plt.plot(df["date"], df["temp_min"], label="Min", alpha=0.5)
    plt.title("Daily Temperature Trends")
    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.tight_layout()
    out = PLOTS_DIR / "daily_temperature.png"
    plt.savefig(out)
    plt.close()
    print("Saved:", out)

def plot_monthly_rainfall(monthly):
    plt.figure(figsize=(10,4))
    x = monthly.index.strftime("%Y-%m")
    plt.bar(x, monthly["rainfall_mm_sum"])
    plt.xticks(rotation=45)
    plt.title("Monthly Rainfall")
    plt.xlabel("Month")
    plt.ylabel("Rainfall (mm)")
    plt.tight_layout()
    out = PLOTS_DIR / "monthly_rainfall.png"
    plt.savefig(out)
    plt.close()
    print("Saved:", out)

def plot_humidity_vs_temp(df):
    plt.figure(figsize=(6,6))
    plt.scatter(df["temp_mean"], df["humidity"], alpha=0.6)
    plt.title("Humidity vs Temperature")
    plt.xlabel("Mean Temperature (°C)")
    plt.ylabel("Humidity (%)")
    plt.tight_layout()
    out = PLOTS_DIR / "humidity_vs_temp.png"
    plt.savefig(out)
    plt.close()
    print("Saved:", out)

def plot_combined(monthly):
    x = monthly.index.strftime("%Y-%m")
    rain = monthly["rainfall_mm_sum"]
    temp = monthly["temp_mean_mean"]

    fig, ax1 = plt.subplots(figsize=(10,5))
    ax1.bar(x, rain, alpha=0.6, label="Rainfall")
    ax1.set_ylabel("Rainfall (mm)")
    ax1.tick_params(axis='x', rotation=45)

    ax2 = ax1.twinx()
    ax2.plot(x, temp, color="red", marker="o", label="Temperature")
    ax2.set_ylabel("Temperature (°C)")

    plt.title("Monthly Rainfall vs Temperature")
    fig.tight_layout()
    out = PLOTS_DIR / "combined_monthly_rain_temp.png"
    plt.savefig(out)
    plt.close()
    print("Saved:", out)


def save_outputs(df, monthly, yearly):
    df.to_csv(CLEAN_OUT, index=False)
    monthly.to_csv(SUMMARY_MONTH)
    yearly.to_csv(SUMMARY_YEAR)
    print("\nSaved cleaned data and summaries.")



def write_report(df, monthly, yearly):
    hottest = monthly["temp_mean_mean"].idxmax()
    wettest = monthly["rainfall_mm_sum"].idxmax()

    text = f"""
# Weather Data Report

## Overview
- Total days: {len(df)}
- Date range: {df['date'].min().date()} to {df['date'].max().date()}

## Insights
- **Hottest Month:** {hottest.strftime('%Y-%m')} (avg temp {monthly['temp_mean_mean'].max():.2f}°C)
- **Wettest Month:** {wettest.strftime('%Y-%m')} ({monthly['rainfall_mm_sum'].max():.1f} mm rain)

## Files Generated
- Cleaned CSV (`output/cleaned_data.csv`)
- Monthly summary (`output/monthly_summary.csv`)
- Yearly summary (`output/yearly_summary.csv`)
- PNG Plots in `output/plots/`
    """

    REPORT_MD.write_text(text)
    print("\nReport saved:", REPORT_MD)



def main():
    print("\nLoading data...")
    df = load_and_inspect(DATA_IN)

    clean = clean_data(df)

    daily, monthly, yearly = compute_stats(clean)

    
    plot_daily_temperature(clean)
    plot_monthly_rainfall(monthly)
    plot_humidity_vs_temp(clean)
    plot_combined(monthly)

    
    save_outputs(clean, monthly, yearly)
    write_report(clean, monthly, yearly)

    print("\n✔ Finished Successfully!")


if __name__ == "__main__":
    main()