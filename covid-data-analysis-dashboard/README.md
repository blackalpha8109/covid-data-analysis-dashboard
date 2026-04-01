# 🦠 COVID-19 Big Data Analytics Dashboard

> **Big Data Analytics – TA-1 Project | Topic 10**  
> *Big Data in COVID-19 Tracking and Prediction*

---

## 👥 Team Members
| Name | Roll No |
|------|---------|
| Arjun Jaiswal | 28 |
| Armaan Ahmed | 29 |
| Arnav Milmile | 30 |

---

## 📌 Project Overview

This project demonstrates the application of **Big Data Analytics** techniques to real-world COVID-19 data. It includes:

- 📊 **Interactive Dashboard** — Visualize global COVID-19 trends
- 🤖 **ML Prediction Model** — Forecast future case counts using time-series analysis
- 🔄 **Data Pipeline** — Automated data fetching from public APIs
- 📓 **Jupyter Notebook** — Step-by-step analysis walkthrough

---

## 🗂️ Project Structure

```
covid19-dashboard/
│
├── README.md                   ← You are here
├── requirements.txt            ← Python dependencies
│
├── data/
│   └── sample_covid_data.csv   ← Sample dataset (auto-downloaded)
│
├── notebooks/
│   └── covid_analysis.ipynb    ← Full analysis notebook
│
└── src/
    ├── fetch_data.py           ← Fetches data from Our World in Data API
    ├── analyze.py              ← Data cleaning & analysis
    ├── visualize.py            ← Generates charts & graphs
    ├── predict.py              ← ML model for case prediction
    └── dashboard.py            ← Runs the interactive dashboard
```

---

## 🛠️ Tools & Technologies Used

| Category | Tools |
|----------|-------|
| Language | Python 3.x |
| Big Data Processing | Pandas, NumPy (simulating Spark/Hadoop pipeline) |
| Machine Learning | Scikit-learn, statsmodels |
| Visualization | Matplotlib, Seaborn, Plotly |
| Dashboard | Streamlit |
| Data Source | Our World in Data (OWID) API |

---

## 📦 Installation & Setup

### Step 1: Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/covid19-dashboard.git
cd covid19-dashboard
```

### Step 2: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the dashboard
```bash
streamlit run src/dashboard.py
```

### Step 4: Or open the notebook
```bash
jupyter notebook notebooks/covid_analysis.ipynb
```

---

## 📊 Features

### 1. Global Case Tracker
- Total confirmed cases, deaths, and recoveries
- Country-wise comparison bar charts
- Daily new cases trend line

### 2. Heatmap & Spread Visualization
- Monthly case heatmap across top countries
- Vaccination progress tracker

### 3. ML Prediction Module
- 30-day case count forecast using Linear Regression
- Model accuracy metrics (MAE, R² Score)

### 4. Data Pipeline
- Automated data fetch from [Our World in Data](https://ourworldindata.org/covid-cases)
- Data cleaning and preprocessing steps

---

## 📚 Data Sources

| Source | Description |
|--------|-------------|
| [Our World in Data (OWID)](https://ourworldindata.org/covid-cases) | Global cases, deaths, vaccinations |
| [WHO COVID-19 Dashboard](https://covid19.who.int) | Official WHO statistics |
| [Johns Hopkins CSSE](https://coronavirus.jhu.edu) | Historical aggregated data |

---

## 🔬 How It Relates to Big Data

| Big Data Concept | How This Project Uses It |
|-----------------|--------------------------|
| **Volume** | Millions of daily records from 200+ countries |
| **Velocity** | Real-time API data fetching |
| **Variety** | Epidemiological + Mobility + Genomic data types |
| **Hadoop/HDFS** | Simulated batch processing pipeline in `analyze.py` |
| **Spark** | In-memory processing concept demonstrated with Pandas |
| **ML** | Time-series forecasting with regression models |

---

## 📈 Sample Output

The dashboard shows:
- 📉 Wave patterns across countries
- 🗺️ Geographic spread analysis
- 🔮 30-day prediction with confidence intervals

---

## 📝 License
This project is for educational purposes as part of the Big Data Analytics course.
