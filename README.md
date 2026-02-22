# Diabetes Risk Prediction Project
This project implements a machine learning classification system designed to predict diabetes risk using health indicator data.

### Methodology
The project involved the training and evaluation of four distinct base models: Logistic Regression, Linear Support Vector Machine (LinearSVM), Random Forest, and K-Nearest Neighbors (KNN). Two ensemble techniques were implemented on the fine tuned models: Hard Voting and Soft Voting classifiers.

### Performance and Model Selection
Random Forest and the Soft Voting Ensemble yielded the highest performance metrics. Both models achieved an accuracy of approximately 75% and an F-score of approximately 76%.

The Random Forest model was selected for the final deployment  due to its robust performance and consistency.

Detailed data analysis and the complete model training process are documented in the `notebook/` directory. The final model is deployed as an interactive web application using Streamlit.
