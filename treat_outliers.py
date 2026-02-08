import pandas as pd
import numpy as np
import os

def winsorize_price(file_path, output_path):
    df = pd.read_csv(file_path)
    
    # Calculate IQR
    Q1 = df['House Price'].quantile(0.25)
    Q3 = df['House Price'].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    
    print(f"Applying Winsorization at Upper Bound: ₹{upper_bound:.2f}L")
    
    # Cap the values
    df['Price_Cleaned'] = np.where(df['House Price'] > upper_bound, upper_bound, df['House Price'])
    
    # Identify how many were capped
    capped_count = (df['House Price'] > upper_bound).sum()
    print(f"Number of records capped: {capped_count}")
    
    # Save the cleaned data
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to: {output_path}")

if __name__ == "__main__":
    input_file = "/Users/lakshminarasimhan.santhanamgigkri.com/Workspace/HSR Houseprice/hsr_house_prices.csv"
    output_file = "/Users/lakshminarasimhan.santhanamgigkri.com/Workspace/HSR Houseprice/cleaned_hsr_house_prices.csv"
    winsorize_price(input_file, output_file)
