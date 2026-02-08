# Walkthrough - Synthetic House Price Data Generation

I have successfully generated the synthetic dataset for house price prediction in HSR Layout, Bangalore.

## Changes Made
- Created [generate_data.py](file:///Users/lakshminarasimhan.santhanamgigkri.com/Workspace/HSR%20Houseprice/generate_data.py) to script the data generation using `pandas` and `numpy`.
- Generated [hsr_house_prices.csv](file:///Users/lakshminarasimhan.santhanamgigkri.com/Workspace/HSR%20Houseprice/hsr_house_prices.csv) containing 1000 records.

## Data Schema
The dataset includes 16 variables:
1. `total_sqft`
2. `num_bedrooms`
3. `num_bathrooms`
4. `num_balconies`
5. `num_stories`
6. `house_age`
7. `distance_to_mrt`
8. `num_convenience_stores`
9. `latitude`
10. `longitude`
11. `is_gated_community`
12. `parking_slots`
13. `property_type`
14. `floor_number`
15. `distance_to_main_road`
16. **House Price** (Target Variable)

## Validation Results
- **Record Count**: 1000 records.
- **Pricing Logic**: Includes base price per sqft with adjustments for age, gated community, MRT distance, and property type.
- **Location**: Latidue/Longitude centered around HSR Layout, Bangalore.

### Data Sample
```csv
total_sqft,num_bedrooms,num_bathrooms,num_balconies,num_stories,house_age,distance_to_mrt,num_convenience_stores,latitude,longitude,is_gated_community,parking_slots,property_type,floor_number,distance_to_main_road,House Price
1743,3,4,0,2,3,1.56,11,12.9117,77.6368,1,1,Apartment,1,432,291.58
662,1,2,2,2,30,3.01,9,12.9175,77.6404,0,2,Villa,0,72,61.82
2474,3,3,1,2,3,1.43,9,12.9153,77.6437,1,2,Apartment,18,57,355.98
```

## Exploratory Data Analysis (EDA)

I performed an initial exploratory analysis on the dataset to check for patterns, outliers, and correlations.

### 1. Data Summary
- **Shape**: (1000, 16)
- **Missing Values**: 0 (Clean dataset)
- **Target Variable**: `House Price` ranges from ~₹38L to ~₹816L, with a mean of ~₹308L.

### 2. Correlation Analysis
The heatmap shows strong correlations between `total_sqft`, `num_bedrooms`, and `House Price`, which is expected.
![Correlation Heatmap](/Users/lakshminarasimhan.santhanamgigkri.com/.gemini/antigravity/brain/ee337058-fd49-42cb-a914-b27bbba5fdc7/correlation_heatmap.png)

### 3. Filtered Pair Plots (Correlation > 0.1)
I have filtered the variables to only include those with an absolute correlation greater than **0.1** with the `House Price`. This makes the visualization much cleaner and focuses on the most impactful features:
`['total_sqft', 'num_bedrooms', 'num_bathrooms', 'house_age', 'distance_to_mrt']`.

![Filtered Pair Plot](/Users/lakshminarasimhan.santhanamgigkri.com/.gemini/antigravity/brain/ee337058-fd49-42cb-a914-b27bbba5fdc7/filtered_pair_plot.png)

### 4. Feature vs House Price (Filtered Stacked View)
This view shows regression plots for only the variables meeting the 0.1 correlation threshold. You can clearly see the strong linear relationship with `total_sqft` and the negative correlation with `house_age`.
![Feature vs Price Filtered](/Users/lakshminarasimhan.santhanamgigkri.com/.gemini/antigravity/brain/ee337058-fd49-42cb-a914-b27bbba5fdc7/feature_vs_price_filtered.png)

### 5. Outlier Detection & Investigation
Boxplots indicate the presence of outliers in the `House Price` column. I performed a detailed investigation using the Interquartile Range (IQR) method.

- **Thresholds**: 
    - Q1: ₹174.4L, Q3: ₹416.3L
    - Upper Bound (Q3 + 1.5*IQR): **₹779.2L**
- **Findings**: 
    - We found **4 outliers** with prices exceeding ₹779L.
    - **Villas** represent 75% of these outliers.
    - All outliers have a `total_sqft` greater than 4000.
    - These are "legitimate" premium properties in the synthetic data, representing the luxury segment of HSR Layout.

### 6. Outlier Treatment (Winsorization)
I applied **Winsorization (Capping)** to the dataset.
- **Method**: Any `House Price` above the upper bound of ₹779.2L was capped at exactly ₹779.2L.
- **Result**: The 4 extreme outliers have been pulled back to the edge of the distribution, which helps prevent them from disproportionately biasing future regression models.
- **File**: [`cleaned_hsr_house_prices.csv`](file:///Users/lakshminarasimhan.santhanamgigkri.com/Workspace/HSR%20Houseprice/cleaned_hsr_house_prices.csv)

![Outlier Comparison](/Users/lakshminarasimhan.santhanamgigkri.com/.gemini/antigravity/brain/ee337058-fd49-42cb-a914-b27bbba5fdc7/outlier_comparison.png)

## Model Training & Performance

I trained a **Multiple Linear Regression** model using a **60:40** split on the cleaned dataset.

### 1. Regression Metrics
These metrics evaluate how accurately the model predicts the numerical house price:
- **R2 Score**: **0.8107** (The model explains ~81% of the variance in prices)
- **MAE (Mean Absolute Error)**: **₹55.30L**
- **RMSE (Root Mean Squared Error)**: **₹67.33L**
- **MSE (Mean Squared Error)**: **4533.58**

### 2. Classification Metrics (Price > Median)
To provide the requested classification metrics (TP, FP, etc.), I binarized the target variable at its median value (predicting if a house is "Above Median Price"):
- **True Positives (TP)**: 193
- **True Negatives (TN)**: 166
- **False Positives (FP)**: 30
- **False Negatives (FN)**: 11
- **F1 Score**: **0.9040**
- **KL Divergence**: **0.2549**

### 3. Conclusion
The model shows strong predictive power for synthetic house prices in HSR Layout, with particularly high accuracy in distinguishing "Premium" vs "Standard" properties (High F1 score).

### 4. Sample Predictions vs Actual (Test Set)
Here is a sample of 10 records from the test dataset (60:40 split) showing the actual price vs the model's prediction:

| Total Sqft | Bedrooms | Property Type | Actual Price (L) | Predicted Price (L) | Absolute Error |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 2460 | 3 | Villa | ₹225.86 | ₹334.78 | ₹108.92 |
| 4351 | 5 | Villa | ₹511.96 | ₹467.12 | ₹44.84 |
| 1239 | 2 | Apartment | ₹124.47 | ₹100.79 | ₹23.68 |
| 2688 | 4 | Villa | ₹361.41 | ₹288.95 | ₹72.46 |
| 2135 | 3 | Apartment | ₹315.44 | ₹250.98 | ₹64.46 |
| 1022 | 2 | Villa | ₹103.67 | ₹78.73 | ₹24.94 |
| 936 | 2 | Independent House | ₹73.96 | ₹62.59 | ₹11.37 |
| 3711 | 5 | Villa | ₹367.08 | ₹438.72 | ₹71.64 |
| 1382 | 2 | Independent House | ₹149.68 | ₹146.79 | ₹2.89 |
| 726 | 1 | Apartment | ₹83.15 | ₹31.58 | ₹51.57 |

You can find the full comparison for the entire test set in: [`prediction_comparison.csv`](file:///Users/lakshminarasimhan.santhanamgigkri.com/Workspace/HSR%20Houseprice/prediction_comparison.csv).

#### Visualizing Predictions
The scatter plot below shows the relationship between the actual house prices and the model's predictions. The **red dashed line** represents the "Perfect Prediction" (Actual = Predicted).

![Actual vs Predicted](/Users/lakshminarasimhan.santhanamgigkri.com/.gemini/antigravity/brain/ee337058-fd49-42cb-a914-b27bbba5fdc7/actual_vs_predicted.png)

### 5. Interactive Project Notebook
I have consolidated all the above steps into a comprehensive, professional Jupyter Notebook. This notebook is designed for both technical stakeholders and business users.

- **File**: [**HSR_Real_Estate_Analysis.ipynb**](file:///Users/lakshminarasimhan.santhanamgigkri.com/Workspace/HSR%20Houseprice/HSR_Real_Estate_Analysis.ipynb)
- **Features**: Includes executive summaries, behind-the-hood math explanations, and interactive code blocks.

---
*All results, scripts, and the final notebook are available in the [HSR Houseprice](file:///Users/lakshminarasimhan.santhanamgigkri.com/Workspace/HSR%20Houseprice) folder.*
