import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import os

def run_prediction_sample():
    file_path = "/Users/lakshminarasimhan.santhanamgigkri.com/Workspace/HSR Houseprice/cleaned_hsr_house_prices.csv"
    df = pd.read_csv(file_path)
    
    # 1. Preprocessing (matching previous step)
    X = df.drop(['House Price', 'Price_Cleaned'], axis=1)
    y = df['Price_Cleaned']
    X = pd.get_dummies(X, columns=['property_type'], drop_first=True)
    X = X.astype(float)
    
    # 2. Split (matching previous step)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    
    # 3. Re-train / Or load if saved (re-training here for simplicity)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 4. Predictions on the test set
    y_pred = model.predict(X_test)
    
    # 5. Create a comparison table for the first 10 test records
    comparison_df = pd.DataFrame({
        'Actual Price': y_test.values,
        'Predicted Price': np.round(y_pred, 2)
    })
    comparison_df['Absolute Error'] = np.round(np.abs(comparison_df['Actual Price'] - comparison_df['Predicted Price']), 2)
    
    # Add some key features for context
    # Note: X_test indices match y_test indices
    # We'll pull total_sqft and num_bedrooms for display
    test_features = df.iloc[X_test.index][['total_sqft', 'num_bedrooms', 'property_type']].reset_index(drop=True)
    comparison_df = pd.concat([test_features, comparison_df], axis=1)
    
    # 6. Visualization: Actual vs Predicted Scatter Plot
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=y_test.values, y=y_pred, alpha=0.5)
    
    # Add identity line (perfect prediction)
    max_val = max(y_test.max(), y_pred.max())
    min_val = min(y_test.min(), y_pred.min())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Prediction')
    
    plt.xlabel('Actual Price (₹ Lakhs)')
    plt.ylabel('Predicted Price (₹ Lakhs)')
    plt.title('Actual vs Predicted House Prices (Test Set)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plot_path = os.path.join(os.path.dirname(file_path), 'actual_vs_predicted.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Scatter plot saved to: {plot_path}")

    # Save to CSV for user
    output_path = "/Users/lakshminarasimhan.santhanamgigkri.com/Workspace/HSR Houseprice/prediction_comparison.csv"
    comparison_df.to_csv(output_path, index=False)
    print(f"\nComparison saved to: {output_path}")

if __name__ == "__main__":
    run_prediction_sample()
