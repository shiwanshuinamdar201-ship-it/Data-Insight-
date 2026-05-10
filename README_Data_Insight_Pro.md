# 🧠 Data Insight Pro

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

**AI-powered analytics platform — upload any dataset, get instant cleaning, visualizations & ML predictions**

*Mentor: Mr. Yadnesh Bhagwat · Team of 2*

</div>

---

## 📌 The Problem

Data analysts spend 60–80% of their time on data cleaning and exploration before they can do any real work. Non-technical users can't do it at all. Data Insight Pro collapses that gap — upload a CSV, and within seconds you have a clean dataset, smart charts, and trained ML models.

---

## ✨ Features

| Module | What it does |
|---|---|
| 🧹 **Auto Data Cleaner** | Detects and handles nulls, duplicates, type mismatches, and outliers automatically |
| 📊 **Smart Visualizer** | Recommends and renders the right chart type based on column dtypes |
| 🤖 **One-Click ML** | Trains classification or regression models with automatic feature selection |
| 📋 **Data Profiler** | Generates a full statistical summary — distributions, correlations, cardinality |
| 💾 **Export** | Download cleaned data and model predictions as CSV |

---

## 🏗️ Architecture

```
User uploads CSV
      │
      ▼
┌─────────────────┐
│  Data Profiler  │  → dtype inference, null %, cardinality, skewness
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Auto Cleaner   │  → imputation strategy auto-selected per column
│                 │    (mean/median for numeric, mode for categorical)
│                 │  → outlier detection via IQR
│                 │  → duplicate removal
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────────┐
│  Smart Charts   │     │   ML Engine          │
│                 │     │                      │
│  • Histogram    │     │  Auto-detects task:  │
│  • Box plot     │     │  → Classification    │
│  • Scatter      │     │  → Regression        │
│  • Heatmap      │     │                      │
│  • Bar chart    │     │  Models tried:       │
└─────────────────┘     │  → Random Forest     │
                        │  → Logistic/Linear   │
                        │  → KNN               │
                        │                      │
                        │  Returns: accuracy,  │
                        │  feature importance, │
                        │  predictions CSV     │
                        └─────────────────────┘
```

---

## 🔍 Core Logic

### Auto Cleaning Strategy
```python
def auto_clean(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Remove duplicates
    df = df.drop_duplicates()
    
    for col in df.columns:
        # 2. Smart imputation based on dtype
        if df[col].dtype in ['float64', 'int64']:
            skewness = df[col].skew()
            # Use median for skewed distributions, mean for normal
            fill_value = df[col].median() if abs(skewness) > 1 else df[col].mean()
            df[col].fillna(fill_value, inplace=True)
            
            # 3. IQR-based outlier capping (not removal — preserves data)
            Q1, Q3 = df[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            df[col] = df[col].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    return df
```

### Smart Chart Recommendation
```python
def recommend_chart(df: pd.DataFrame, col: str) -> str:
    dtype = df[col].dtype
    cardinality = df[col].nunique()
    
    if dtype in ['float64', 'int64']:
        return 'histogram' if cardinality > 20 else 'bar'
    elif dtype == 'object':
        return 'bar' if cardinality <= 15 else 'wordcloud'
    elif dtype == 'datetime64':
        return 'line'
```

### Auto ML Task Detection
```python
def detect_ml_task(target: pd.Series) -> str:
    if target.dtype == 'object' or target.nunique() <= 10:
        return 'classification'
    return 'regression'
```

---

## 🚀 Getting Started

```bash
# Clone the repo
git clone https://github.com/shiwanshuinamdar201-ship-it/Data-Insight-.git
cd Data-Insight-

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app.py
# Opens at http://localhost:8501
```

### Requirements
```
streamlit>=1.20.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.1.0
plotly>=5.10.0
scipy>=1.9.0
```

---

## 📁 Project Structure

```
Data-Insight-/
├── app.py                      # Main Streamlit app entry point
├── modules/
│   ├── profiler.py             # Dataset statistical profiling
│   ├── cleaner.py              # Auto cleaning engine
│   ├── visualizer.py           # Smart chart recommender + renderer
│   └── ml_engine.py            # Auto ML — task detection + model training
├── utils/
│   └── helpers.py              # Shared utilities
├── sample_data/
│   ├── titanic.csv             # Sample classification dataset
│   └── house_prices.csv        # Sample regression dataset
└── requirements.txt
```

---

## 💡 Key Learnings

- **Skewness-aware imputation** (median vs mean) produces significantly cleaner distributions than blind mean-fill
- IQR-based **outlier capping** (vs removal) is safer for small datasets — you don't lose rows, just clip extremes
- Auto-detecting ML task type from target column cardinality works reliably for 90%+ of real-world tabular datasets
- Streamlit's session state is essential for multi-step pipelines — without it, every widget interaction reruns everything

---

## 🧠 Skills Demonstrated

`Machine Learning` `Data Cleaning` `Feature Engineering` `Scikit-learn` `Streamlit` `Plotly` `Pandas` `EDA` `Python` `Software Architecture`

---

<div align="center">
Made by <a href="https://github.com/shiwanshuinamdar201-ship-it">Shiwanshu Inamdar</a> & team · B.Tech CSE Data Science · D.Y. Patil International University
</div>
