# RetailMind AI – Customer Churn Prediction

An end-to-end machine learning project that predicts customer churn in the telecommunications industry. RetailMind AI uses customer demographics, account information, service subscriptions, and billing details to identify customers who are likely to discontinue their services.

The objective is to help businesses proactively identify at-risk customers and develop targeted customer-retention strategies.

---

## Project Objectives

- Analyse customer data to identify churn patterns.
- Build machine learning models to predict customer churn.
- Address class imbalance in the dataset.
- Perform feature engineering to improve predictive performance.
- Compare multiple classification algorithms.
- Apply hyperparameter tuning and cross-validation.
- Explore ensemble learning techniques.
- Develop a reusable prediction pipeline.

## Dataset

**Dataset:** IBM Telco Customer Churn Dataset  
**Source:** [Kaggle – Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

| Attribute | Description |
|---|---|
| Initial records | 7,043 |
| Initial features | 21 |
| Target variable | Churn |
| Target classes | Yes / No |
| Prediction type | Binary Classification |

The target variable was encoded as:

- `0` – Customer did not churn.
- `1` – Customer churned.

The `customerID` column was excluded from model training because it is an identifier rather than a predictive feature.

## Technology Stack

- **Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Visualisation:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Ensemble Learning:** Random Forest, XGBoost
- **Model Persistence:** Joblib
- **Development Environment:** Jupyter Notebook, VS Code
- **Version Control:** Git and GitHub

---

## Project Development

### Day 1 – Data Preprocessing and Baseline Model

The initial stage focused on understanding the dataset, handling missing values, and establishing a baseline machine learning model.

**Key activities:**
- Exploratory Data Analysis (EDA).
- Identification and treatment of missing values.
- Conversion of `TotalCharges` into a numeric feature.
- Encoding the target variable.
- Stratified train-test split (80:20).
- Development of a reusable preprocessing pipeline.

**Baseline model:** Logistic Regression

The preprocessing pipeline included:

| Feature Type | Preprocessing |
|---|---|
| Numerical | Median imputation and StandardScaler |
| Categorical | Most-frequent imputation and OneHotEncoder |

Class imbalance was addressed using `class_weight="balanced"`.

**Saved artifacts:**
- `models/churn_pipeline.joblib`
- `data/processed/test_data.csv`
- `cleaneddata.csv`

### Day 2 – Model Comparison

Three additional classification algorithms were implemented and evaluated against the Logistic Regression baseline.

**Models implemented:**
1. Decision Tree
2. Random Forest
3. Support Vector Machine (SVM)

Each model used the reusable preprocessing pipeline to ensure consistent data preparation.

**Evaluation metrics:**
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

Random Forest emerged as the best-performing model in the Day 2 comparison.

**Saved artifacts:**
- `models/decision_tree.joblib`
- `models/random_forest.joblib`
- `models/svm.joblib`

### Day 3 – Hyperparameter Tuning and Model Optimisation

The objective of Day 3 was to optimise the models and improve their generalisation performance.

**Key activities:**

- Implemented GridSearchCV.
- Applied StratifiedKFold cross-validation with five folds.
- Tuned Decision Tree, Random Forest, and SVM.
- Used recall as the primary GridSearchCV scoring metric.
- Compared training and validation performance.
- Analysed potential overfitting and underfitting.
- Generated learning curves.
- Evaluated the selected model on the test dataset.

**Cross-validation results:**

| Model | Best CV Recall |
|---|---:|
| Random Forest | 0.8502 |
| SVM | 0.8355 |
| Decision Tree | 0.7826 |

Random Forest achieved the highest cross-validation recall.

**Training-validation gap analysis:**

| Model | Training Recall | Validation Recall | Gap |
|---|---:|---:|---:|
| Decision Tree | 0.8134 | 0.7826 | 0.0308 |
| Random Forest | 0.8515 | 0.8502 | 0.0013 |
| SVM | 0.8355 | 0.8355 | 0.0000 |

The tuned Random Forest demonstrated strong consistency between training and validation recall.

**Selected model:** Random Forest

**Saved artifact:**
- `models/churn_v3.joblib`

**Tuning report:**
- `reports/tuning_results.json`

### Day 4 – Feature Engineering and Ensemble Learning

Day 4 focused on improving the predictive capabilities of the existing model through domain-specific feature engineering and ensemble learning.

#### 1. Feature Engineering

New features were derived from existing customer attributes.

| Feature | Description |
|---|---|
| TenureGroup | Categorises customers based on tenure |
| IsNewCustomer | Identifies customers with tenure of 12 months or less |
| IsLongTermCustomer | Identifies customers with tenure of at least 48 months |
| AverageMonthlyCharge | Total charges divided by tenure |
| ChargeRatio | Compares actual total charges with expected charges |
| ServiceCount | Counts subscribed services |
| UsesMultipleServices | Identifies customers using at least three services |
| IsMonthToMonth | Identifies month-to-month contracts |
| HasLongTermContract | Identifies one-year and two-year contracts |
| UsesElectronicBilling | Identifies customers using paperless billing |

#### 2. Feature Importance Analysis

Random Forest feature importance was used to investigate the contribution of transformed features.

The tuned Day 3 Random Forest was reused for this analysis rather than unnecessarily training another importance-estimation model.

#### 3. Ensemble Learning

Two ensemble models were trained using the engineered dataset:

**Random Forest**
- Bootstrap-based ensemble learning.
- Multiple decision trees.
- Class-weight balancing.
- Aggregated predictions.

**XGBoost**
- Gradient boosting framework.
- Sequential tree construction.
- Learning-rate regularisation.
- Subsampling and feature subsampling.

#### 4. Model Comparison

The following models were compared:

- Day 3 tuned Random Forest.
- Day 4 Random Forest with engineered features.
- Day 4 XGBoost with engineered features.

Evaluation metrics included accuracy, precision, recall, F1-score, and ROC-AUC.

**Final Day 4 comparison:**

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Day 3 Random Forest | TBD | TBD | TBD | TBD | TBD |
| Day 4 Random Forest | TBD | TBD | TBD | TBD | TBD |
| Day 4 XGBoost | TBD | TBD | TBD | TBD | TBD |

Replace the placeholders with the actual values from `day4_comparison_df`.

**Saved artifacts:**
- `models/churn_v4.joblib`
- `reports/day4_model_comparison.csv`

---

## Evaluation Metrics

The project uses multiple evaluation metrics to assess model performance.

| Metric | Purpose |
|---|---|
| Accuracy | Measures overall prediction correctness |
| Precision | Measures how many predicted churners actually churn |
| Recall | Measures how many actual churners are identified |
| F1-score | Balances precision and recall |
| ROC-AUC | Measures the model's ability to distinguish between churn classes |

Recall is particularly important because failing to identify a customer who is likely to churn can result in a missed retention opportunity.

---

## Project Structure

```text
RetailMind/
│
├── data/
│   └── processed/
│       └── test_data.csv
│
├── models/
│   ├── churn_pipeline.joblib
│   ├── decision_tree.joblib
│   ├── random_forest.joblib
│   ├── svm.joblib
│   ├── churn_v3.joblib
│   └── churn_v4.joblib
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_baseline_model.ipynb
│   ├── 06_model_comparison.ipynb
│   ├── 07_hyperparameter_tuning.ipynb
│   └── 08_feature_engineering_ensembles.ipynb
│
├── reports/
│   ├── tuning_results.json
│   └── day4_model_comparison.csv
│
├── src/
│   └── retailmind/
│       └── pipeline.py
│
├── tests/
│
├── cleaneddata.csv
├── README.md
└── requirements.txt
```

*Note: The structure represents the intended project organisation. Adjust filenames if your actual notebook names differ.*

---

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/SChandra2003/RetailMind.git
cd RetailMind
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open the notebooks directory and execute the notebooks in the appropriate order.

---

## Future Improvements

- Further hyperparameter optimisation.
- Experiment with SMOTE and other class-imbalance techniques.
- Perform systematic feature selection inside cross-validation pipelines.
- Explore probability calibration and decision-threshold optimisation.
- Evaluate model explainability using SHAP.
- Build an interactive churn prediction dashboard.
- Develop a REST API using FastAPI.
- Deploy the final model for real-time customer churn predictions.
- Monitor model drift and prediction performance.

---

## Important Considerations

The current model is an experimental machine learning solution built using a public telecom churn dataset.

The test set has been used for evaluation across multiple development stages. Therefore, the Day 4 comparison should be treated as an experimental benchmark rather than an independent estimate of final production performance.

A future production version should use training-only cross-validation for model selection and a separate untouched holdout dataset for final evaluation.

---

## Author

**Subhojit Chandra**

GitHub: [SChandra2003](https://github.com/SChandra2003)

---

*RetailMind AI – Turning customer data into actionable churn insights.*