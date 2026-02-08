import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix, f1_score, classification_report
from scipy.stats import entropy
import os

def calculate_gini(y_true, y_pred_prob):
    # Simplified Gini for regression/binary
    # Usually Gini is for classification; here we can use it for binarized price
    pass

def train_and_evaluate():
    file_path = "/Users/lakshminarasimhan.santhanamgigkri.com/Workspace/HSR Houseprice/cleaned_hsr_house_prices.csv"
    df = pd.read_csv(file_path)
    
    # 1. Preprocessing
    # Drop the original 'House Price' if we use 'Price_Cleaned'
    X = df.drop(['House Price', 'Price_Cleaned'], axis=1)
    y = df['Price_Cleaned']
    
    # One-hot encoding for property_type
    X = pd.get_dummies(X, columns=['property_type'], drop_first=True)
    X = X.astype(float) # Ensure all numeric
    
    # 2. Split 60:40
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    
    # 3. Multiple Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 4. Predictions
    y_pred = model.predict(X_test)
    
    # --- Regression Metrics ---
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # --- Advanced / Classification Metrics (for demonstration) ---
    # Binarize price at median to show classification metrics (e.g., predicting "Premium" house)
    median_price = y.median()
    y_test_bin = (y_test > median_price).astype(int)
    y_pred_bin = (y_pred > median_price).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_test_bin, y_pred_bin).ravel()
    f1 = f1_score(y_test_bin, y_pred_bin)
    
    # KL Divergence (Rough estimation of distribution difference)
    # We normalize to probabilities for KL
    p = np.histogram(y_test, bins=10, density=True)[0] + 1e-10
    q = np.histogram(y_pred, bins=10, density=True)[0] + 1e-10
    kl = entropy(p, q)
    
    print("--- Regression Performance Metrics ---")
    print(f"R2 Score: {r2:.4f}")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    
    print("\n--- Classification Metrics (Price > Median binarization) ---")
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    print(f"F1 Score: {f1:.4f}")
    print(f"KL Divergence: {kl:.4f}")
    
    # Output to file for Walkthrough
    with open("/Users/lakshminarasimhan.santhanamgigkri.com/Workspace/HSR Houseprice/model_results.txt", "w") as f:
        f.write(f"R2: {r2}\nMAE: {mae}\nMSE: {mse}\nRMSE: {rmse}\nTP: {tp}\nTN: {tn}\nFP: {fp}\nFN: {fn}\nF1: {f1}\nKL: {kl}")

if __name__ == "__main__":
    train_and_evaluate()
