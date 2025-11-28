# Airfare Prediction with Exogenous Variables Using Supervised Learning and Deep Learning Techniques
This repository is for the MSc Data Science & Society thesis.

## Project Overview
This project explores airfare prediction using machine learning and deep learning models. Airfares are highly volatile, influenced by factors such as seasonality, route popularity, and airline operations, making reliable prediction challenging. The study compares model performance with and without exogenous variables, including airline-specific operational scale and (seat) capacity utilization. By incorporating these factors, the project aims to help travelers understand which elements most impact airfares and make more cost-effective decisions.

## Dataset Used
The dataset used for this thesis is the Airfare ML dataset, which consists of webscraped airfare observations from EaseMyTrip and is obtained from [Kaggle](https://www.kaggle.com/datasets/yashdharme36/airfare-ml-predicting-flight-fares?select=Cleaned_dataset.csv).
The exogenous variables containing airline-level operational scale and (seat) capacity measures are obtained from the Directorate General of Civil Aviation of India ([DGCA](https://www.dgca.gov.in/digigov-portal/))

## Language
- Python (version 3.11.13)

## Algorithms/Models used
- [Random Forest (Regressor)](https://doi.org/10.1023/A:1010933404324)
- [CatBoost (Regressor)](https://doi.org/10.48550/arXiv.1706.09516)
- [XGBoost (Regressor)](https://doi.org/10.1145/2939672.2939785)
- [Multilayer Perceptron](https://doi.org/10.1038/323533a0)

## Packages, Libraries, and Frameworks
- CarbonTracker (version 2.3.1)
- CatBoost (version 1.2.8)
- Matplotlib (version 3.10.7)
- NumPy (version 1.26.4)
- Optuna (version 4.5.0)
- Pandas (version 2.3.3)
- PyTorch (version 2.5.1)
- Scikit-learn (version 1.7.2)
- Scipy (version 1.16.2)
- Seaborn (version 0.13.2)
- SHAP (version 0.49.1)
- XGBoost (version 3.1.1)
