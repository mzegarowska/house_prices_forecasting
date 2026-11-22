# Real Estate Price Prediction & Analysis (Python)

A Python-based academic project focused on exploratory data analysis, real estate price prediction, and evaluating the impact of feature engineering, regularization, and data preprocessing on machine learning models.

The project combines regression and classification tasks, including a custom implementation of Ridge Regression using Gradient Descent and an SVM classifier for identifying high-value properties.

## Tech Stack & Concepts

- **Language:** Python
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
- **Core Concepts:** Exploratory Data Analysis (EDA), Data Preprocessing, Feature Engineering, Statistical Analysis, Data Visualization, Supervised Machine Learning
- **Algorithms:** Ordinary Least Squares (OLS), Ridge Regression, Gradient Descent, Support Vector Machine (SVM)
- **Evaluation Metrics:**
  - Regression: MAE, RMSE, R²
  - Classification: Accuracy, Precision, Recall, ROC AUC

## Individual Academic Project

This project was completed independently as part of a university course focused on data exploration and analysis.

**My Responsibilities:**
- Conducted exploratory data analysis and identified potential data quality issues.
- Designed and implemented data preprocessing procedures.
- Implemented Ridge Regression with Gradient Descent.
- Developed and evaluated regression models with different feature sets and target transformations.
- Performed feature engineering to investigate factors affecting property prices.
- Trained and evaluated SVM classifiers for high-value property identification.
- Designed experiments, compared results, and formulated conclusions.

## Features

- **Exploratory Data Analysis:** Statistical examination of property transactions, price distributions, correlations, and potentially inconsistent records.
- **Data Preprocessing:** Handling missing values, encoding categorical variables, feature scaling, and processing suspicious observations.
- **Time-Based Data Splitting:** Chronological training, validation, and test sets with property ID separation to prevent data leakage.
- **Custom Ridge Regression:** Implementation of a regularized linear regression model trained using Gradient Descent.
- **Feature Engineering:** Creation of additional variables describing property age, basement proportion, and bedroom-to-bathroom ratio.
- **Model Comparison:** Evaluation of baseline regression, regularized models, engineered features, and logarithmic target transformation.
- **SVM Classification:** Identification of high-value properties using Support Vector Machines.
- **Experimental Analysis:** Investigation of the impact of location features, regularization, and removing high-priced observations from training data.

## Dataset & Methodology

The project uses a real estate transaction dataset stored in `domy.csv`, containing information about property prices, location, living area, building characteristics, and transaction dates.

To simulate predictions for future transactions involving previously unseen properties, the dataset was divided chronologically. Property identifiers were kept separate across datasets to prevent data leakage.

All preprocessing parameters were fitted using training data only. Model selection was performed on validation data, while the test set was reserved for final evaluation.

## Model Performance

The final regression model uses a custom Ridge Regression implementation with engineered features and a logarithmic transformation of property prices.

**Final Regression Model (M5):**
- Regularization parameter: λ = 0.0001
- Validation MAE: approximately $76,677
- Test MAE: approximately $85,368
- Test R²: approximately 0.85
- Test MAE improvement over the median baseline: 62.3%

**Final Classification Model (Linear SVM):**
- Regularization parameter: C = 10
- High-value property threshold: $648,000
- Test Accuracy: approximately 90.91%
- Test Precision: approximately 91.0%
- Test Recall: approximately 72.4%
- Test ROC AUC: approximately 0.97

## Project Structure

- `real_estate_prices_forecasting.ipynb` — Jupyter Notebook containing exploratory analysis, preprocessing, custom regression implementation, classification models, experiments, and conclusions.
- `domy.csv` — Input dataset required to reproduce the analysis (subject to redistribution permissions).

## How to Run Locally

1. Place `domy.csv` in the same directory as the notebook.
2. Launch Jupyter Notebook:

```bash
jupyter notebook
```

3. Open `real_estate_prices_forecasting.ipynb`.
4. Select Run All Cells to reproduce the analysis and model evaluations.

## Key Findings

- Property prices exhibit a right-skewed distribution, making logarithmic target transformation beneficial for regression accuracy.
- Feature engineering improved prediction performance by incorporating additional property characteristics.
- Logarithmic target transformation significantly reduced validation MAE compared with regression on the original price scale.
- Location-related features were particularly important; removing them substantially increased prediction error.
- Excluding the most expensive transactions from training reduced overall validation MAE but worsened predictions for high-value properties.
- The final SVM classifier substantially outperformed the majority-class baseline, although some expensive properties remained undetected.

## Project Context

This project was completed as an individual university assignment and demonstrates practical experience in exploratory data analysis, implementing machine learning algorithms, feature engineering, model evaluation, and experimental methodology.

Particular emphasis was placed on preventing data leakage, comparing modeling decisions, and interpreting results on previously unseen data.
