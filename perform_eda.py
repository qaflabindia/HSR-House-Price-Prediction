import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def perform_eda(file_path):
    # Load data
    df = pd.read_csv(file_path)
    
    print("--- 1. Data Shape ---")
    print(df.shape)
    print("\n--- 2. Variable Types ---")
    print(df.dtypes)
    print("\n--- 3. Null Values ---")
    print(df.isnull().sum())
    print("\n--- 4. Descriptive Statistics ---")
    print(df.describe())
    
    # Visualizations
    output_dir = os.path.dirname(file_path)
    
    # Correlation Heatmap
    plt.figure(figsize=(12, 10))
    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
    plt.close()
    
    # Outlier Detection with Boxplots for key variables
    key_vars = ['total_sqft', 'House Price', 'distance_to_mrt', 'house_age']
    plt.figure(figsize=(15, 10))
    for i, var in enumerate(key_vars):
        plt.subplot(2, 2, i+1)
        sns.boxplot(y=df[var])
        plt.title(f'Boxplot of {var}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'outlier_detection.png'))
    plt.close()

    # Filter variables by correlation threshold > 0.1 with 'House Price'
    numeric_df = df.select_dtypes(include=[np.number])
    correlations = numeric_df.corr()['House Price'].abs()
    threshold = 0.1
    high_corr_vars = correlations[correlations > threshold].index.tolist()
    
    print(f"\n--- 5. Variables with Correlation > {threshold} with House Price ---")
    print(high_corr_vars)

    # Full Pairplot (Filtered)
    if len(high_corr_vars) > 1:
        sns.pairplot(df[high_corr_vars])
        plt.savefig(os.path.join(output_dir, 'filtered_pair_plot.png'))
        plt.close()

    # Feature vs Price "Stacked" plot (Filtered)
    cols_to_plot = [c for c in high_corr_vars if c != 'House Price']
    if cols_to_plot:
        n_cols = 3
        n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols
        
        plt.figure(figsize=(20, n_rows * 5))
        for i, var in enumerate(cols_to_plot):
            plt.subplot(n_rows, n_cols, i + 1)
            sns.regplot(x=df[var], y=df['House Price'], scatter_kws={'alpha':0.3})
            plt.title(f'{var} vs House Price (corr={numeric_df.corr().loc[var, "House Price"]:.2f})')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_vs_price_filtered.png'))
        plt.close()

    print(f"\nEDA completed. Filtered plots saved in: {output_dir}")

if __name__ == "__main__":
    csv_path = "/Users/lakshminarasimhan.santhanamgigkri.com/Workspace/HSR Houseprice/hsr_house_prices.csv"
    perform_eda(csv_path)
