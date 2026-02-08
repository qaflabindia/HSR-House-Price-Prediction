import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def compare_treatment(original_path, cleaned_path, output_dir):
    df_orig = pd.read_csv(original_path)
    df_clean = pd.read_csv(cleaned_path)
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    sns.boxplot(y=df_orig['House Price'])
    plt.title('Original Price (with outliers)')
    
    plt.subplot(1, 2, 2)
    sns.boxplot(y=df_clean['Price_Cleaned'])
    plt.title('Cleaned Price (Winsorized)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'outlier_comparison.png'))
    plt.close()
    
    print(f"Comparison plot saved to: {output_dir}")

if __name__ == "__main__":
    orig = "/Users/lakshminarasimhan.santhanamgigkri.com/Workspace/HSR Houseprice/hsr_house_prices.csv"
    clean = "/Users/lakshminarasimhan.santhanamgigkri.com/Workspace/HSR Houseprice/cleaned_hsr_house_prices.csv"
    out = "/Users/lakshminarasimhan.santhanamgigkri.com/Workspace/HSR Houseprice"
    compare_treatment(orig, clean, out)
