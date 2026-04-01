"""
fetch_data.py
-------------
Fetches COVID-19 data from Our World in Data (OWID) public dataset.
This simulates the API/ETL pipeline described in the case study.
"""

import pandas as pd
import requests
import os

# URL for the OWID COVID-19 dataset (public, no API key needed)
OWID_URL = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/sample_covid_data.csv")

# Countries we want to focus on
COUNTRIES = ["India", "United States", "Brazil", "United Kingdom", "Germany", "France"]

# Columns we need for our analysis
USEFUL_COLUMNS = [
    "location", "date", "total_cases", "new_cases",
    "total_deaths", "new_deaths", "total_vaccinations",
    "people_vaccinated_per_hundred", "population"
]


def fetch_covid_data(use_cache=True):
    """
    Downloads the OWID COVID-19 dataset and filters to selected countries.
    
    Args:
        use_cache (bool): If True, use locally saved data if it exists.
    
    Returns:
        pd.DataFrame: Cleaned and filtered COVID-19 data
    """

    if use_cache and os.path.exists(DATA_PATH):
        print("✅ Loading cached data from disk...")
        df = pd.read_csv(DATA_PATH, parse_dates=["date"])
        return df

    print("🌐 Fetching data from Our World in Data...")
    print(f"   Source: {OWID_URL}")

    try:
        df = pd.read_csv(OWID_URL, parse_dates=["date"])
        print(f"   Raw dataset size: {df.shape[0]:,} rows × {df.shape[1]} columns")

        # Filter to selected countries only
        df = df[df["location"].isin(COUNTRIES)]

        # Keep only the columns we need
        available_cols = [c for c in USEFUL_COLUMNS if c in df.columns]
        df = df[available_cols]

        # Fill missing values with 0 for numeric columns
        numeric_cols = df.select_dtypes(include="number").columns
        df[numeric_cols] = df[numeric_cols].fillna(0)

        # Save to disk
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
        print(f"✅ Data saved to {DATA_PATH}")
        print(f"   Filtered dataset size: {df.shape[0]:,} rows × {df.shape[1]} columns")

        return df

    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        print("   Using built-in sample data instead...")
        return _generate_sample_data()


def _generate_sample_data():
    """
    Generates a realistic sample dataset when internet is unavailable.
    This represents what would come from the API pipeline.
    """
    import numpy as np
    from datetime import datetime, timedelta

    print("🔧 Generating sample data...")
    start_date = datetime(2020, 3, 1)
    dates = [start_date + timedelta(days=i) for i in range(900)]

    records = []
    country_params = {
        "India":         {"peak": 400000, "vax_rate": 60},
        "United States": {"peak": 300000, "vax_rate": 75},
        "Brazil":        {"peak": 100000, "vax_rate": 80},
        "United Kingdom":{"peak": 50000,  "vax_rate": 85},
        "Germany":       {"peak": 70000,  "vax_rate": 78},
        "France":        {"peak": 80000,  "vax_rate": 77},
    }

    for country, params in country_params.items():
        total_cases = 0
        total_deaths = 0
        for i, date in enumerate(dates):
            # Simulate wave patterns
            wave = (np.sin(i / 80) + 1) * 0.5
            new_cases = max(0, int(params["peak"] * wave * np.random.uniform(0.8, 1.2)))
            new_deaths = int(new_cases * np.random.uniform(0.005, 0.015))
            total_cases += new_cases
            total_deaths += new_deaths
            vax = min(100, (i / 900) * params["vax_rate"])

            records.append({
                "location": country,
                "date": date,
                "new_cases": new_cases,
                "total_cases": total_cases,
                "new_deaths": new_deaths,
                "total_deaths": total_deaths,
                "people_vaccinated_per_hundred": round(vax, 2),
                "population": 1_000_000
            })

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"✅ Sample data saved: {len(df):,} rows")
    return df


if __name__ == "__main__":
    df = fetch_covid_data(use_cache=False)
    print("\n📋 Sample of fetched data:")
    print(df.head(10))
    print("\n📊 Data info:")
    print(df.describe())
