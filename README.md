#  Time Series Forecasting Projects

*A research-driven, practice-focused exploration of time-dependent patterns, forecasting models, and data-driven insights.*

This repository contains my hands-on journey into **Time Series Forecasting**, where I explore different forecasting techniques, experiment with real datasets, and apply both classical and modern modeling approaches.
The goal is to build an understanding of time-dependent behavior, develop forecasting intuition, and refine machine learning skills through *learning by doing*.

Each notebook in this repo represents a standalone experiment — from small practice exercises using Prophet to large-scale forecasting with millions of rows using LightGBM.

---

##  Repository Structure

Every project folder / notebook includes:

* **Exploratory Time Series Analysis (TSA)**
* **Preprocessing & Feature Engineering**
* **Model Training & Evaluation**
* **Experimentation, Observations & Learning Notes**

Current notebooks in the repository:

| Notebook                                           | Description                                                                                          | Status   |
| -------------------------------------------------- | ---------------------------------------------------------------------------------------------------- | -------- |
| **myPractice.ipynb**                               | First practice notebook using Prophet on the classic AirPassengers dataset.                          | Complete |
| **forecastingbyOrderds.ipynb**                     | Exploration of forecasting using Prophet with custom configurations.                                 | Complete |
| **Time Series data_ASDA.ipynb**                    | Time series analysis of ASDA supermarket pricing data.                                               | Complete |
| **All_Data_Morrisons TimeSeriesForecasting.ipynb** | Large-scale forecasting with 1.7M+ Morrisons SKU-level records using LightGBM + engineered features. | Complete |

---

## 🚀 Project Highlights

### 1️ **AirPassengers Forecasting (myPractice.ipynb)**

*My first step into time series modeling.*

**Objective:**
Understand Prophet’s basics and structure a simple forecasting workflow.

**Key Learnings:**

* Converting raw data into Prophet's `ds` / `y` format
* Splitting training & testing with time-aware boundaries
* Visualizing predictions alongside true test values
* Identifying trend, seasonality, and change points
* Understanding Prophet’s strengths & limitations

**Techniques Used:**

* Prophet modeling
* Future dataframe creation
* Seasonal decomposition
* Forecast visualization with test overlay

---

### 2️ **Supermarket Pricing Forecasting – Morrisons Dataset (1.7M+ rows)**

 *The most advanced notebook in the repository.*
Found in: **All_Data_Morrisons TimeSeriesForecasting.ipynb**

**Objective:**
Forecast the **price per unit** of supermarket products (SKUs) using high-volume, structured retail data.

**Dataset Summary:**

* **1.79 million rows**
* Features: product name, category, price, brand type, date, unit type
* High cardinality and noisy real-world pricing data

**What I Did:**
✔ Detailed EDA with missing value analysis
✔ Handled missing values using grouped forward/backward fill
✔ Engineered lag & rolling statistical features
✔ Derived calendar features (dow, month, quarter, weekend, year, etc.)
✔ Applied **TimeSeriesSplit CV**
✔ Used **LightGBM** inside a `TransformedTargetRegressor` (log transform)
✔ Hyperparameter tuning with **GridSearchCV**
✔ Full model evaluation on test set with RMSE, MAPE, SMAPE
✔ Exported final model with joblib

**Key Concepts Practiced:**

* Large-scale time series preprocessing
* Memory-aware feature engineering
* Handling categorical variables in LightGBM
* Avoiding leakage with grouped lag features
* Understanding CV instability in real-world retail scenarios
* Model tuning tradeoffs (speed vs generalization)

**Results Snapshot:**

* Test RMSE: **~11.96**
* SMAPE: **~1.41%**
* Learned importance of outliers, category cardinality, and model stability

---

### 3️ **ASDA Retail Time Series (Time Series data_ASDA.ipynb)**

A mid-sized practice notebook analyzing supermarket pricing dynamics.

**Focus Areas:**

* EDA on time-dependent pricing
* Temporal grouping
* Seasonal patterns
* Baseline forecasting experiments

---

### 4️ **Prophet Forecasting with Custom Orders (forecastingbyOrderds.ipynb)**

**Practice notebook** investigating more customization with Prophet.

---

##  Learning Focus & Research Angle

This repository reflects an ongoing learning journey into:

###  Time Series Fundamentals

Trends, seasonality, autocorrelation, lagged dependencies, data leakage awareness.

###  Classical vs Modern Approaches

From **Prophet**, a business-friendly model, to **LightGBM**, a gradient boosting model suitable for massive data and engineered features.

###  Real-World Challenges

High-cardinality categorical data, missing values, noisy retail pricing, and huge datasets.

###  Experimentation Mindset

Trying, observing, evaluating, correcting, and iterating — the core of applied machine learning.

---

##  Tools & Technologies

**Languages:**

* Python

**Libraries:**

* Prophet
* LightGBM
* pandas, numpy
* matplotlib, seaborn
* statsmodels
* scikit-learn (Modeling, TTSplit, TransformedTargetRegressor, GridSearchCV)
* missingno
* joblib

---

##  Future Work

* Experimenting with multivariate forecasting (VAR, LSTM)
* Calendar / holiday adjustments
* Outlier detection strategies
* Hierarchical retail forecasting
* MLflow experiment tracking

---

## 📄 About

A collection of experimental and research-oriented time series projects built as part of my “learn by doing” machine learning journey.
The goal is not just to build models, but to understand them, question them, and iterate toward better insights.

---

