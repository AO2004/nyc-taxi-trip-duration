# NYC Taxi Trip Duration Prediction

Predicting New York City taxi trip durations using machine learning and feature engineering.

## Overview

This project builds a regression model to estimate taxi trip duration from historical NYC taxi trip data. The workflow covers the complete machine learning pipeline, including:

- Exploratory Data Analysis (EDA)
- Data cleaning
- Feature engineering
- Preprocessing
- Model training and evaluation
- Model serialization for inference

The final model is trained using **Ridge Regression** with carefully engineered spatial and temporal features.

---

## Dataset

The project uses the **NYC Taxi Trip Duration** dataset from Kaggle.

**Target**
- `trip_duration` (log-transformed during training)

---

## Feature Engineering

Engineered features include:

- Haversine distance
- Manhattan distance
- Trip bearing (sin & cos encoding)
- Pickup hour
- Day of week
- Day of month
- Month
- Weekend indicator
- rush hour
- busy hour
- Combined weekday-hour feature
- Pickup & dropoff K-Means clusters

---

## Preprocessing

- handling outliers
- Missing value handling
- One-Hot Encoding for categorical features
- Standard Scaling for numerical features
- Polynomial interaction features
- Kmeans clustering(discretization)

---

## Model

- Ridge Regression

---

## Results

| Dataset | RMSE | R² |
|---------|------:|------:|
| Train | **0.4088** | **0.7207** |
| Validation | **0.4554** | **0.6759** |
| Test | **0.4460** | **0.6861** |

The close train, validation, and test scores indicate good generalization with minimal overfitting.

---

## Repository Structure

```
nyc-taxi-trip-duration/
├── notebooks/
│   ├── EDA.ipynb
│   |__ trainer.ipynb
├── src/
│   ├── feature_engineering.py
│   |__ inference.py
├── models/
│   ├── kmeans_model.joblib
|   ├── main_model.joblib
|   ├── metadata.json   
│   |__ preprocessor.joblib
└── README.md
```

---

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

---

## Future Improvements

- Gradient Boosting models (LightGBM, XGBoost, CatBoost)
- Hyperparameter optimization
- Cross-validation
- Additional geospatial features

---

## Demo
<p align="center">
  <img width="1911" height="1017" alt="Image" src="https://github.com/user-attachments/assets/03a2f025-285a-4360-aab2-422020dd125b" />
</p>
---


## Resources
* eda template: https://www.kaggle.com/code/bextuychiev/my-6-part-powerful-eda-template
* feature engineering tips: https://medium.com/@bijit211987/10-advanced-feature-engineering-methods-46b63a1ee92e
