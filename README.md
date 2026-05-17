# 🫀 Heart Attack Risk Prediction

A machine learning project that predicts heart attack risk in patients using a rich set of clinical and lifestyle features. The project covers exploratory data analysis, domain-aware feature engineering, and benchmarking of three classification models with hyperparameter tuning.

----

## 📊 Dataset

- **Source:** [Kaggle — Heart Attack Risk Prediction Dataset](https://www.kaggle.com/datasets/iamsouravbanerjee/heart-attack-prediction-dataset)
- **Size:** 8,763 patients × 26 features
- **Target:** Heart Attack Risk (1: Yes, 0: No)
- **Class distribution:** 5,624 no risk (64.2%) / 3,139 at risk (35.8%) — imbalanced dataset
- **No missing values**

**Key features include:** Age, Sex, Cholesterol, Blood Pressure, Heart Rate, Diabetes, Family History, Smoking, Obesity, Alcohol Consumption, Exercise Hours Per Week, Diet, BMI, Triglycerides, Stress Level, Sedentary Hours Per Day, Sleep Hours Per Day, Country, Continent

---

## 🔍 Project Pipeline

### 1. Exploratory Data Analysis (EDA)
- Target distribution bar chart — flagged class imbalance (64% vs 36%)
- Gender distribution and crosstab analysis of heart attack risk by Sex
- Crosstab breakdown by Country and Continent
- Smoking status analysis with bar chart comparing risk vs non-risk patients
- Correlation heatmap across all numerical features

### 2. Feature Engineering
- **Blood Pressure parsing:** Split the `Blood Pressure` string column (e.g. "120/80") into separate `Systolic` and `Diastolic` numeric columns
- **Mean Arterial Pressure (MAP):** Computed as `Diastolic + (Systolic - Diastolic) / 3` — a clinically meaningful cardiovascular metric
- **Diet ordinal encoding:** Mapped to numeric values (Healthy=2, Average=1, Unhealthy=0) preserving the natural order
- **One-Hot Encoding:** Applied to `Sex`
- **Dropped:** `Patient ID`, `Blood Pressure` (raw), `Country`, `Continent`, `Hemisphere` — geolocation features dropped after analysis showed no strong predictive signal

### 3. Modeling

**Baseline comparison — 3 models evaluated in one loop:**

| Model | Metrics Reported |
|---|---|
| Logistic Regression | Accuracy, Precision, Recall, F1 |
| Random Forest Classifier | Accuracy, Precision, Recall, F1 |
| K-Nearest Neighbors | Accuracy, Precision, Recall, F1 |

**Hyperparameter tuning with GridSearchCV (5-fold CV):**

- **Random Forest:** tuned `n_estimators` [100, 200, 500], `max_depth`, `max_features`, `min_samples_split`, `min_samples_leaf`
- **Logistic Regression:** tuned `C` across 30 values (log scale from 1e-4 to 1e4), solver=liblinear

---

## 📁 Project Structure

```
heart-attack-prediction/
│
├── heart_attack_EDA_and_prediction_.ipynb   # Main notebook
├── heart_attack_dataset.csv                  # Dataset
└── README.md
```

---

## ▶️ How to Run

1. Clone the repository
```bash
git clone https://github.com/your-username/heart-attack-prediction.git
```

2. Install the required libraries
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

3. Open the notebook in Jupyter
```bash
jupyter notebook heart_attack_EDA_and_prediction_.ipynb
```

4. Make sure `heart_attack_dataset.csv` is in the same folder as the notebook, then run all cells

---

## 📦 Libraries Used

| Library | Purpose |
|---|---|
| `pandas` / `numpy` | Data loading and manipulation |
| `matplotlib` / `seaborn` | Data visualization |
| `scikit-learn` | Preprocessing, Logistic Regression, Random Forest, KNN, GridSearchCV, metrics |

---

## 💡 Key Findings

- The dataset is imbalanced — over 64% of patients show no heart attack risk, meaning accuracy alone is a misleading metric; Precision, Recall, and F1 are more meaningful here
- **Smoking, Family History, Previous Heart Problems, and Cholesterol** are among the most clinically relevant features in the dataset
- **Diet, Exercise, and Sedentary Hours** reflect lifestyle factors that contribute to risk alongside purely clinical indicators
- Splitting Blood Pressure into Systolic and Diastolic components provides more granular signal than the raw combined string
- Random Forest with tuned hyperparameters outperformed Logistic Regression and KNN on this dataset

---

## ⚠️ Disclaimer

This project is for educational purposes only. The predictions made by these models are not a substitute for professional medical advice, diagnosis, or treatment.

---

## 👤 Author

**Khaled Abdulaziz**
