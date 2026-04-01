"""
predict.py
----------
Machine Learning module for COVID-19 case prediction.
Uses Linear Regression (as a simplified time-series model) to forecast
future daily case counts — representing the ML techniques described in the case study
(LSTM, Prophet, etc.) in a beginner-friendly way.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


def prepare_features(df, country):
    """
    Converts time-series data into features for ML.
    Uses day number as the feature (X) and new_cases as target (y).
    """
    country_df = df[df["location"] == country].copy()
    country_df = country_df.sort_values("date").reset_index(drop=True)

    # Add 7-day rolling average to smooth the signal
    country_df["rolling_avg"] = country_df["new_cases"].rolling(7, min_periods=1).mean()

    # Feature: how many days since outbreak started
    country_df["day_number"] = np.arange(len(country_df))

    # Drop rows with NaN
    country_df = country_df.dropna(subset=["rolling_avg"])

    X = country_df[["day_number"]].values
    y = country_df["rolling_avg"].values
    dates = country_df["date"].values

    return X, y, dates, country_df


def train_model(X, y):
    """
    Trains a Linear Regression model and returns:
    - model: trained model
    - metrics: MAE and R² score
    - X_train, X_test, y_train, y_test for plotting
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False  # No shuffle — time series!
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 0, None)  # Cases can't be negative

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "MAE": round(mae, 2),
        "R2_Score": round(r2, 4),
        "interpretation": "Lower MAE = better | R² closer to 1.0 = better fit"
    }

    return model, metrics, X_train, X_test, y_train, y_test, y_pred


def predict_future(model, X, days_ahead=30):
    """
    Uses the trained model to predict the next N days.
    
    Args:
        model: Trained LinearRegression model
        X: Original feature array (to know the last day number)
        days_ahead: How many days to forecast
    
    Returns:
        future_days: Array of future day numbers
        future_preds: Predicted case counts
    """
    last_day = X[-1][0]
    future_days = np.arange(last_day + 1, last_day + 1 + days_ahead).reshape(-1, 1)
    future_preds = model.predict(future_days)
    future_preds = np.clip(future_preds, 0, None)
    return future_days.flatten(), future_preds


def run_prediction_pipeline(df, country="India"):
    """
    Full prediction pipeline for a given country.
    Returns everything needed for the dashboard prediction tab.
    """
    print(f"\n🤖 Running ML prediction for: {country}")

    X, y, dates, country_df = prepare_features(df, country)
    model, metrics, X_train, X_test, y_train, y_test, y_pred = train_model(X, y)
    future_days, future_preds = predict_future(model, X, days_ahead=30)

    print(f"   📉 MAE  : {metrics['MAE']:,.0f} cases")
    print(f"   📈 R²   : {metrics['R2_Score']}")

    return {
        "country_df": country_df,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "future_days": future_days,
        "future_preds": future_preds,
        "metrics": metrics,
        "dates": dates,
    }


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from fetch_data import fetch_covid_data
    from analyze import clean_data

    df = fetch_covid_data()
    df = clean_data(df)
    result = run_prediction_pipeline(df, country="India")
    print("\n✅ Prediction complete!")
    print(f"   Next 30 days forecast range: "
          f"{result['future_preds'].min():,.0f} – {result['future_preds'].max():,.0f} cases/day")
