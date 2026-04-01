"""
analyze.py
----------
Data cleaning and analysis module.
Simulates the role of Apache Hadoop batch processing in the case study —
aggregating and summarizing large volumes of COVID-19 data.
"""

import pandas as pd
import numpy as np


def clean_data(df):
    """
    Cleans the raw COVID-19 dataset.
    - Removes rows with missing dates or locations
    - Fills numeric NaN with 0
    - Sorts by country and date
    """
    print("🧹 Cleaning data...")
    df = df.dropna(subset=["date", "location"])
    df["date"] = pd.to_datetime(df["date"])
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    df = df.sort_values(["location", "date"]).reset_index(drop=True)
    print(f"   ✅ Clean dataset: {df.shape[0]:,} rows")
    return df


def get_summary_stats(df):
    """
    Returns a summary table: total cases, deaths, and vaccination % per country.
    This mimics a Hive query result in Hadoop — aggregating across partitions.
    """
    print("📊 Computing summary statistics...")
    summary = df.groupby("location").agg(
        total_cases=("total_cases", "max"),
        total_deaths=("total_deaths", "max"),
        peak_daily_cases=("new_cases", "max"),
        avg_daily_cases=("new_cases", "mean"),
        max_vaccinated_pct=("people_vaccinated_per_hundred", "max"),
    ).reset_index()

    summary["case_fatality_rate_%"] = (
        summary["total_deaths"] / summary["total_cases"] * 100
    ).round(2)

    summary["avg_daily_cases"] = summary["avg_daily_cases"].round(0).astype(int)
    summary = summary.sort_values("total_cases", ascending=False).reset_index(drop=True)

    return summary


def get_monthly_aggregates(df):
    """
    Aggregates daily data into monthly totals per country.
    This is the kind of batch job that would run on Hadoop/HDFS.
    """
    df = df.copy()
    df["year_month"] = df["date"].dt.to_period("M")
    monthly = df.groupby(["location", "year_month"]).agg(
        monthly_cases=("new_cases", "sum"),
        monthly_deaths=("new_deaths", "sum"),
    ).reset_index()
    monthly["year_month"] = monthly["year_month"].astype(str)
    return monthly


def get_wave_periods(df, country):
    """
    Identifies COVID wave peaks for a specific country.
    Uses a rolling 7-day average to smooth noise — similar to
    Spark Streaming window functions.
    """
    country_df = df[df["location"] == country].copy()
    country_df = country_df.sort_values("date")
    country_df["rolling_7day"] = country_df["new_cases"].rolling(7).mean()
    return country_df


def get_top_countries_by_cases(df, n=6):
    """Returns top N countries by total cases."""
    return (
        df.groupby("location")["total_cases"].max()
        .sort_values(ascending=False)
        .head(n)
        .reset_index()
    )


if __name__ == "__main__":
    from fetch_data import fetch_covid_data
    df = fetch_covid_data()
    df = clean_data(df)

    print("\n📋 Summary Statistics:")
    print(get_summary_stats(df).to_string(index=False))

    print("\n📅 Monthly Aggregates (first 10 rows):")
    print(get_monthly_aggregates(df).head(10).to_string(index=False))
