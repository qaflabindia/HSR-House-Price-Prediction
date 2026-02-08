import pandas as pd
import numpy as np

def identify_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

def main():
    file_path = "/Users/lakshminarasimhan.santhanamgigkri.com/Workspace/HSR Houseprice/hsr_house_prices.csv"
    df = pd.read_csv(file_path)
    
    print("--- 1. Outlier Investigation for House Price ---")
    outliers_price, lb_p, ub_p = identify_outliers(df, 'House Price')
    print(f"Lower Bound: {lb_p:.2f}, Upper Bound: {ub_p:.2f}")
    print(f"Number of Outliers found: {len(outliers_price)}")
    
    if not outliers_price.empty:
        print("\nSummary of Outlier Houses (Price):")
        print(outliers_price[['total_sqft', 'property_type', 'is_gated_community', 'num_bedrooms', 'House Price']].sort_values(by='House Price', ascending=False))
        
        print("\n--- Distribution of Property Types in Outliers ---")
        print(outliers_price['property_type'].value_counts())
        
        print("\n--- Average Comparison ---")
        print(f"Avg price of outliers: {outliers_price['House Price'].mean():.2f}")
        print(f"Avg price of whole dataset: {df['House Price'].mean():.2f}")

    print("\n--- 2. Outlier Investigation for total_sqft ---")
    outliers_sqft, lb_s, ub_s = identify_outliers(df, 'total_sqft')
    print(f"Lower Bound: {lb_s:.2f}, Upper Bound: {ub_s:.2f}")
    print(f"Number of Outliers found: {len(outliers_sqft)}")

if __name__ == "__main__":
    main()
