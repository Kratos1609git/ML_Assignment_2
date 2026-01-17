# Cancer Risk Classification – ML Assignment 2

## a. Problem Statement
The objective of this project is to build and compare multiple machine learning
classification models to predict cancer risk based on patient health and lifestyle
factors. The project demonstrates a complete end-to-end ML pipeline including
model training, evaluation, deployment, and interactive prediction using Streamlit.

## b. Dataset Description
The dataset used is "Cancer Risk Factors Dataset" obtained from a public repository.
It contains more than 500 instances and over 12 features related to demographic,
behavioral, and medical risk factors.  
The target variable indicates whether a patient is at risk of cancer.

## c. Models Used and Evaluation Metrics

All models were trained on the same dataset and evaluated using:
Accuracy, AUC Score, Precision, Recall, F1 Score, and MCC.


                     Accuracy       AUC  Precision  Recall        F1       MCC
Logistic Regression    0.7550  0.948262   0.753290  0.7550  0.753585  0.689268
Decision Tree          0.6750  0.887697   0.681833  0.6750  0.674854  0.589178
KNN                    0.6375  0.883218   0.622925  0.6375  0.623759  0.539780
Naive Bayes            0.5575  0.873400   0.667794  0.5575  0.546051  0.487605
Random Forest          0.7700  0.948388   0.765993  0.7700  0.764498  0.708471
XGBoost                0.7750  0.963688   0.772590  0.7750  0.772032  0.714781

## Model Performance Observations

| Model | Observation |
|------|------------|
| Logistic Regression |			Performs well on linearly separable data but struggles with complex interactions. |
| Decision Tree | 				Captures non-linear patterns but prone to overfitting. |
| KNN | 						Sensitive to feature scaling and choice of neighbors. |
| Naive Bayes | 				Simple and fast but assumes feature independence. |
| Random Forest | 				Improves stability and generalization by aggregating multiple trees. |
| XGBoost | 					Achieves the best overall performance due to boosting and regularization. |

## Deployment
The trained models were deployed using Streamlit Community Cloud with an
interactive interface allowing dataset upload, model selection, and performance
visualization.
