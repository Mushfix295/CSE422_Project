# CSE422_Project
Artificial Intelligence

This project addresses student academic outcome prediction (Dropout / Graduate / Enrolled) using a multi-class classification framework on an educational dataset with 17 categorical and 7 numerical features.
​
The dataset undergoes comprehensive exploratory data analysis (EDA), revealing class imbalance, strong feature-feature correlations (e.g., parental occupations ~0.95), and meaningful target relationships (previous grades ~+0.40 with outcome).
​
A robust preprocessing pipeline combines median imputation and standard scaling for numerical features, plus mode imputation and one-hot encoding for categorical variables, ensuring reproducibility.
​
Six machine learning models are implemented: Neural Network (128→64 dense layers with dropout and batch normalization), Random Forest (500 trees, balanced weighting), XGBoost (GPU-accelerated, 600 estimators), K-Nearest Neighbors (k=5), Decision Tree (Gini criterion, unlimited depth), and Logistic Regression (one-vs-rest).
​
Each model is wrapped in a scikit-learn pipeline to seamlessly apply preprocessing, trained on 70% stratified splits, and evaluated on held-out test data (30%).
​
Performance is assessed using classification reports, confusion matrices, and micro-average ROC curves with AUC scores, providing a unified multi-class evaluation framework.
​
The Neural Network leverages EarlyStopping (patience=8, validation split=0.15) to prevent overfitting while training for up to 100 epochs with batch size 64.
​
A comparative analysis aggregates accuracy, precision, recall, and F1-score (macro-averaged) across all six models in grouped bar charts and overlaid ROC plots for direct visual comparison.
​
Redundant features (Father's qualification/occupation, Mother's occupation) are dropped post-correlation analysis to reduce multicollinearity and improve model generalization.
​
The entire workflow—from data loading and cleaning to model training and visualization—is implemented in Python using pandas, scikit-learn, TensorFlow/Keras, XGBoost, and matplotlib/seaborn.
​
