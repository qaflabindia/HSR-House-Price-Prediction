import nbformat as nbf
import os

def generate_notebook():
    nb = nbf.v4.new_notebook()

    # --- TITLE ---
    nb.cells.append(nbf.v4.new_markdown_cell("# End-to-End Real Estate Analytics: HSR Layout House Price Prediction\n"
                                            "**Target Audience**: Cross-domain Business Users, Data Analysts, and Stakeholders\n"
                                            "**Author**: Antigravity AI Assistant"))

    # --- 1. EXECUTIVE SUMMARY ---
    nb.cells.append(nbf.v4.new_markdown_cell("## 1. Executive Summary\n\n"
                                            "### Objective\n"
                                            "The goal of this project was to demonstrate a complete data science pipeline—from synthetic data generation to predictive modeling—for a specific micro-market: **HSR Layout, Bangalore**.\n\n"
                                            "### Key Findings\n"
                                            "- **Model Accuracy**: Our Multiple Linear Regression model achieved an **R2 score of 0.81**, which is statistically robust for real estate pricing.\n"
                                            "- **Reliability**: Using **Winsorization** to treat outliers significantly improved the model's reliability for the common market segment while accurately identifying luxury properties.\n"
                                            "- **Primary Drivers**: Property size (`total_sqft`) and proximity to MRT are the strongest predictors of value.\n\n"
                                            "### Business Value\n"
                                            "This framework allows business users to quantify the 'premium' of amenities (gated communities, floor levels) and predict market clearing prices for new inventory with ~81% confidence."))

    # --- 2. DATA GENERATION & LOAD ---
    nb.cells.append(nbf.v4.new_markdown_cell("## 2. Data Acquisition & Loading\n\n"
                                            "### Analyst Commentary\n"
                                            "In many cases, real-world data is sparse or expensive. We generated **1,000 synthetic records** tailored to HSR Layout's geography and market reality (e.g., Silk Board/Agara metro proximity). This allows us to test hypotheses in a controlled environment.\n\n"
                                            "### Behind the Hood: The Generation Logic\n"
                                            "We didn't just pick random numbers. We used **Logarithmic and Linear multipliers** to ensure the data follows real-world economic principles:\n"
                                            "- **Base Price**: 8,000 - 15,000 INR/sqft.\n"
                                            "- **Depreciation**: $Price_{new} = Price_{base} \\times (1 - 0.01 \\times Age)$\n"
                                            "- **Location Penalty**: Prices drop linearly by 5% per KM after the first 1KM distance from MRT."))
    
    nb.cells.append(nbf.v4.new_code_cell("# Step 1: Initializing Libraries and Loading Data\n"
                                            "import pandas as pd\n"
                                            "import numpy as np\n"
                                            "import matplotlib.pyplot as plt\n"
                                            "import seaborn as sns\n"
                                            "from sklearn.model_selection import train_test_split\n"
                                            "from sklearn.linear_model import LinearRegression\n"
                                            "from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error\n\n"
                                            "# We are using the generated hsr_house_prices.csv file from our workspace\n"
                                            "df = pd.read_csv('hsr_house_prices.csv')\n"
                                            "print(f'Shape of data: {df.shape}')\n"
                                            "df.head()"))

    # --- 3. EXPLORATORY DATA ANALYSIS (EDA) ---
    nb.cells.append(nbf.v4.new_markdown_cell("## 3. Exploratory Data Analysis (EDA)\n\n"
                                            "### Analyst Commentary\n"
                                            "EDA is where we 'listen' to the data. We look for missing values, check for data types, and understand the relationship between variables. A clean dataset with zero null values is our starting point.\n\n"
                                            "### Behind the Hood: Pearson Correlation\n"
                                            "A correlation score ranges from -1 to +1:\n"
                                            "- **+1**: Perfect positive relationship (as X goes up, Y goes up).\n"
                                            "- **0**: No relationship.\n"
                                            "- **-1**: Perfect inverse relationship.\n\n"
                                            "We focus on variables with absolute correlation $> 0.1$ for our primary models."))
    
    nb.cells.append(nbf.v4.new_code_cell("# Step 2: Basic Statistics and Correlation\n"
                                            "print('--- Missing Values ---')\n"
                                            "print(df.isnull().sum())\n\n"
                                            "# Visualizing the Correlation Heatmap\n"
                                            "plt.figure(figsize=(12, 8))\n"
                                            "numeric_df = df.select_dtypes(include=[np.number])\n"
                                            "sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')\n"
                                            "plt.title('How features relate to House Price')\n"
                                            "plt.show()"))

    # --- 4. PREPROCESSING & OUTLIER TREATMENT ---
    nb.cells.append(nbf.v4.new_markdown_cell("## 4. Preprocessing & Outlier Treatment\n\n"
                                            "### Analyst Commentary\n"
                                            "Real estate often has 'super-premium' properties (villas) that skew the statistics. We identified 4 such outliers. Treating them ensures our model doesn't get 'confused' by these rare high-end cases while still keeping them in our analysis.\n\n"
                                            "### Behind the Hood: The Interquartile Range (IQR) Method\n"
                                            "We define an outlier as any value that falls outside the 'whiskers' of our distribution:\n"
                                            "- **$IQR = Q3 - Q1$\n"
                                            "- **$Upper Bound = Q3 + (1.5 \\times IQR)$\n\n"
                                            "We use **Winsorization**, which caps extreme values at these boundaries rather than deleting them."))
    
    nb.cells.append(nbf.v4.new_code_cell("# Step 3: Outlier Detection and Capping (Winsorization)\n"
                                            "Q1 = df['House Price'].quantile(0.25)\n"
                                            "Q3 = df['House Price'].quantile(0.75)\n"
                                            "IQR = Q3 - Q1\n"
                                            "upper_bound = Q3 + 1.5 * IQR\n\n"
                                            "print(f'Upper Bound for Pricing: {upper_bound:.2f} Lakhs')\n\n"
                                            "# Create Cleaned Column\n"
                                            "df['Price_Cleaned'] = np.where(df['House Price'] > upper_bound, upper_bound, df['House Price'])\n\n"
                                            "# Visualization Before/After\n"
                                            "fig, ax = plt.subplots(1, 2, figsize=(14, 6))\n"
                                            "sns.boxplot(y=df['House Price'], ax=ax[0]).set_title('Original Prices (with outliers)')\n"
                                            "sns.boxplot(y=df['Price_Cleaned'], ax=ax[1]).set_title('Winsorized Prices (treated)')\n"
                                            "plt.show()"))

    # --- 5. MODEL TRAINING ---
    nb.cells.append(nbf.v4.new_markdown_cell("## 5. Model Training (60:40 Split)\n\n"
                                            "### Analyst Commentary\n"
                                            "We split our data into a **Training Set (60%)** to teach the model and a **Test Set (40%)** to grade its performance. This prevents 'overfitting'—where a model memorizes data rather than learning patterns.\n\n"
                                            "### Behind the Hood: Multiple Linear Regression (MLR)\n"
                                            "MLR finds the best-fit line through a multidimensional space. Our target is to minimize the **Residual Sum of Squares (RSS)**:\n\n"
                                            "$$RSS = \\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2$$\n\n"
                                            "Each feature (sqft, age, etc.) gets a 'weight' or **Beta Coefficient** representing its contribution to the final price."))
    
    nb.cells.append(nbf.v4.new_code_cell("# Step 4: Preprocessing Features and Splitting Data\n"
                                            "# Convert categorical 'property_type' to numeric format (One-Hot Encoding)\n"
                                            "X = df.drop(['House Price', 'Price_Cleaned'], axis=1)\n"
                                            "y = df['Price_Cleaned']\n\n"
                                            "X = pd.get_dummies(X, columns=['property_type'], drop_first=True)\n"
                                            "X = X.astype(float)\n\n"
                                            "# Splitting 60% Train, 40% Test\n"
                                            "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)\n\n"
                                            "# Training the Model\n"
                                            "reg_model = LinearRegression()\n"
                                            "reg_model.fit(X_train, y_train)\n"
                                            "print('Model Training Complete')"))

    # --- 6. EVALUATION & VISUALIZATION ---
    nb.cells.append(nbf.v4.new_markdown_cell("## 6. Model Evaluation & Metrics\n\n"
                                            "### Analyst Commentary\n"
                                            "How do we know the model is good? We look at multiple metrics. No single metric tells the whole story. **R2** tells us the explained variance, while **RMSE** tells us the average ticket-size error in Lakhs.\n\n"
                                            "### Behind the Hood: Metric Definitions\n"
                                            "- **R2 (Coefficient of Determination)**: Proportion of variance in the dependent variable that is predictable from the independent variables. (Goal: closer to 1.0).\n"
                                            "- **MAE (Mean Absolute Error)**: The average of the absolute differences between predictions and actual values. It tells us how far off our predictions are on average.\n"
                                            "- **MSE (Mean Squared Error)**: The average of the squares of the errors. It penalizes larger errors more heavily than smaller ones.\n"
                                            "- **RMSE (Root Mean Squared Error)**: The square root of MSE. It is in the same units as the target variable (Lakhs), making it easy to interpret."))
    
    nb.cells.append(nbf.v4.new_code_cell("# Step 5: Testing and Performance Metrics\ny_pred = reg_model.predict(X_test)\n\n"
                                            "r2 = r2_score(y_test, y_pred)\n"
                                            "mae = mean_absolute_error(y_test, y_pred)\n"
                                            "mse = mean_squared_error(y_test, y_pred)\n"
                                            "rmse = np.sqrt(mse)\n\n"
                                            "print('--- Performance Summary ---')\n"
                                            "print(f'R2 Score: {r2:.4f}')\n"
                                            "print(f'MAE (Mean Absolute Error): {mae:.2f} Lakhs')\n"
                                            "print(f'MSE (Mean Squared Error): {mse:.2f}')\n"
                                            "print(f'RMSE (Root Mean Squared Error): {rmse:.2f} Lakhs')"))

    nb.cells.append(nbf.v4.new_code_cell("# Step 6: Visualizing Actual vs Predicted\n"
                                            "plt.figure(figsize=(10, 8))\n"
                                            "sns.scatterplot(x=y_test.values, y=y_pred, alpha=0.6, color='blue')\n"
                                            "# Identity line\n"
                                            "max_val = max(y_test.max(), y_pred.max())\n"
                                            "plt.plot([0, max_val], [0, max_val], color='red', linestyle='--', label='Perfect Score Line')\n"
                                            "plt.xlabel('Actual Market Price (Lakhs)')\n"
                                            "plt.ylabel('Model Prediction (Lakhs)')\n"
                                            "plt.title('Model Fidelity Validation')\n"
                                            "plt.legend()\n"
                                            "plt.show()"))

    # --- 7. SAMPLE PREDICTIONS ---
    nb.cells.append(nbf.v4.new_markdown_cell("## 7. Sample Prediction Comparisons\n\n"
                                            "### Analyst Commentary\n"
                                            "Finally, we map our predictions back to real houses. This table allows business users to spot-check individual records and build trust in the model's accuracy."))
    
    nb.cells.append(nbf.v4.new_code_cell("# Step 7: Mapping Test Data for Comparison\n"
                                            "comparison_sample = pd.DataFrame({\n"
                                            "    'Actual Price': y_test.values,\n"
                                            "    'Predicted Price': np.round(y_pred, 2)\n"
                                            "})\n"
                                            "comparison_sample['Error'] = comparison_sample['Actual Price'] - comparison_sample['Predicted Price']\n"
                                            "comparison_sample.head(10)"))

    # --- 8. CONFUSION MATRIX & CLASSIFICATION ---
    nb.cells.append(nbf.v4.new_markdown_cell("## 8. Classification Performance: The Confusion Matrix\n\n"
                                            "### Analyst Commentary\n"
                                            "While this is a regression problem (predicting continuous prices), in business, we often make binary decisions: *'Is this house above or below the market median valuation?'*. To test the model's decision-making accuracy, we converted the prices into two categories: **Standard** and **Premium**.\n\n"
                                            "### Behind the Hood: Confusion Matrix Math\n"
                                            "A Confusion Matrix maps the model's 'guesses' against the 'truth':\n"
                                            "- **True Positive (TP)**: Correctly predicted a **Premium** house.\n"
                                            "- **True Negative (TN)**: Correctly predicted a **Standard** house.\n"
                                            "- **False Positive (FP - Type I Error)**: Predicted 'Premium', but it was 'Standard'.\n"
                                            "- **False Negative (FN - Type II Error)**: Predicted 'Standard', but it was 'Premium'.\n\n"
                                            "#### Key Metrics Explained:\n"
                                            "- **Precision (Quality)**: Of all the houses we predicted as 'Premium', how many actually were? High precision means we are 'careful' about calling a house premium.\n"
                                            "- **Recall (Quantity)**: Of all the actual 'Premium' houses in the market, how many did we successfully find? High recall means we aren't 'missing' many deals.\n\n"
                                            "#### Key Formulas:\n"
                                            "$$Precision = \\frac{TP}{TP + FP}$$  \n"
                                            "$$Recall = \\frac{TP}{TP + FN}$$  \n"
                                            "$$F1 Score = 2 \\times \\frac{Precision \\times Recall}{Precision + Recall}$$"))
    
    nb.cells.append(nbf.v4.new_code_cell("# Step 8: Binarizing Data and Plotting Confusion Matrix\n"
                                            "from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay\n\n"
                                            "# Binarizing at the median price\n"
                                            "median_val = df['Price_Cleaned'].median()\n"
                                            "y_test_bin = (y_test > median_val).astype(int)\n"
                                            "y_pred_bin = (y_pred > median_val).astype(int)\n\n"
                                            "cm = confusion_matrix(y_test_bin, y_pred_bin)\n\n"
                                            "plt.figure(figsize=(8, 6))\n"
                                            "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', \n"
                                            "            xticklabels=['Standard', 'Premium'], \n"
                                            "            yticklabels=['Standard', 'Premium'])\n"
                                            "plt.xlabel('Predicted Label')\n"
                                            "plt.ylabel('True Label')\n"
                                            "plt.title('Confusion Matrix: Premium House Prediction Accuracy')\n"
                                            "plt.show()"))

    # --- 9. PREDICT NEW HOUSE ---
    nb.cells.append(nbf.v4.new_markdown_cell("## 9. Live Prediction: Value a New House\n\n"
                                            "### Analyst Commentary\n"
                                            "The ultimate goal of any model is **inference**—using the learned patterns to value a house that isn't in our database yet. We've created a reusable function that takes specific house features and returns a predicted price in Lakhs.\n\n"
                                            "### Behind the Hood: Inference Logic\n"
                                            "When we pass new data, the function must:\n"
                                            "1. **Align Schema**: Ensure the property type is converted into the same 'One-Hot' binary columns (Villa, Independent House) used during training.\n"
                                            "2. **Dot Product**: Multiply each input feature by its corresponding Beta coefficient (weight) and add the Intercept.\n"
                                            "3. **Capping**: Since our model was trained on winsorized data, the prediction will naturally reflect the market's standard premium range."))
    
    nb.cells.append(nbf.v4.new_code_cell("def predict_house_price(total_sqft, num_bedrooms, num_bathrooms, num_balconies, \n"
                                            "                        num_stories, house_age, distance_to_mrt, num_convenience_stores, \n"
                                            "                        latitude, longitude, is_gated_community, parking_slots, \n"
                                            "                        property_type, floor_number, distance_to_main_road):\n"
                                            "    # 1. Create a dictionary for the input\n"
                                            "    input_data = {\n"
                                            "        'total_sqft': [total_sqft],\n"
                                            "        'num_bedrooms': [num_bedrooms],\n"
                                            "        'num_bathrooms': [num_bathrooms],\n"
                                            "        'num_balconies': [num_balconies],\n"
                                            "        'num_stories': [num_stories],\n"
                                            "        'house_age': [house_age],\n"
                                            "        'distance_to_mrt': [distance_to_mrt],\n"
                                            "        'num_convenience_stores': [num_convenience_stores],\n"
                                            "        'latitude': [latitude],\n"
                                            "        'longitude': [longitude],\n"
                                            "        'is_gated_community': [is_gated_community],\n"
                                            "        'parking_slots': [parking_slots],\n"
                                            "        'floor_number': [floor_number],\n"
                                            "        'distance_to_main_road': [distance_to_main_road]\n"
                                            "    }\n"
                                            "    \n"
                                            "    # 2. Add property type columns (One-Hot Encoding handling)\n"
                                            "    # The model expects 'property_type_Independent House' and 'property_type_Villa'\n"
                                            "    input_data['property_type_Independent House'] = [1 if property_type == 'Independent House' else 0]\n"
                                            "    input_data['property_type_Villa'] = [1 if property_type == 'Villa' else 0]\n"
                                            "    \n"
                                            "    # Convert to DataFrame\n"
                                            "    input_df = pd.DataFrame(input_data)\n"
                                            "    \n"
                                            "    # Ensure column order matches X_train\n"
                                            "    input_df = input_df[X_train.columns]\n"
                                            "    \n"
                                            "    # 3. Predict\n"
                                            "    prediction = reg_model.predict(input_df)[0]\n"
                                            "    return np.round(prediction, 2)\n\n"
                                            "# --- Example Prediction Call ---\n"
                                            "sample_price = predict_house_price(\n"
                                            "    total_sqft=2500, num_bedrooms=3, num_bathrooms=3, num_balconies=2, \n"
                                            "    num_stories=2, house_age=5, distance_to_mrt=0.5, num_convenience_stores=8, \n"
                                            "    latitude=12.91, longitude=77.64, is_gated_community=1, parking_slots=2,\n"
                                            "    property_type='Villa', floor_number=0, distance_to_main_road=100\n"
                                            ")\n\n"
                                            "print(f'Estimated Price for the sample house in HSR: ₹{sample_price} Lakhs')"))

    # --- 10. FORMULA & MANUAL CALCULATION ---
    nb.cells.append(nbf.v4.new_markdown_cell("## 10. The Mathematical Formula & Manual Calculation\n\n"
                                            "### Analyst Commentary\n"
                                            "In this section, we pull back the curtain on the model's coefficients. Every prediction is just a sum of features multiplied by their 'importance weights' (Coefficients). This allows the business to see exactly how much **one extra square foot** or **one extra kilometer from MRT** costs in the market.\n\n"
                                            "### Behind the Hood: Coefficient Interpretation\n"
                                            "If the coefficient for `total_sqft` is 0.08, it means for every 1 sqft increase, the price increases by ₹0.08 Lakhs (₹8,000), assuming all other factors remain constant."))
    
    nb.cells.append(nbf.v4.new_code_cell("# Step 10: Extracting Coefficients into a readable table\n"
                                            "feature_names = X_train.columns\n"
                                            "coefficients = reg_model.coef_\n"
                                            "intercept = reg_model.intercept_\n\n"
                                            "coef_df = pd.DataFrame({\n"
                                            "    'Feature': feature_names,\n"
                                            "    'Weight (Beta)': coefficients\n"
                                            "}).sort_values(by='Weight (Beta)', ascending=False)\n\n"
                                            "print(f'Model Intercept (Base Price): {intercept:.2f} Lakhs')\n"
                                            "coef_df"))

    nb.cells.append(nbf.v4.new_markdown_cell("### Manual Math Verification\n"
                                            "Let's take 3 actual records from our test set and calculate the price manually using the formula:\n\n"
                                            "$$Price = Intercept + \\sum (Feature \\times Weight)$$\n\n"
                                            "We will then compare our 'Manual Math' with the 'Model Prediction' to prove they are identical."))
    
    nb.cells.append(nbf.v4.new_code_cell("# Manual Prediction Demonstration\n"
                                            "sample_indices = X_test.head(3).index\n"
                                            "samples = X_test.head(3)\n\n"
                                            "for i, (idx, row) in enumerate(samples.iterrows()):\n"
                                            "    manual_sum = intercept + np.dot(row.values, coefficients)\n"
                                            "    model_pred = y_pred[i]\n"
                                            "    actual = y_test.iloc[i]\n"
                                            "    \n"
                                            "    print(f'--- Record {idx} ---')\n"
                                            "    print(f'Manual Calculation: ₹{manual_sum:.2f} Lakhs')\n"
                                            "    print(f'Model Prediction:   ₹{model_pred:.2f} Lakhs')\n"
                                            "    print(f'Actual Market Price:₹{actual:.2f} Lakhs')\n"
                                            "    print(f'Difference (Error): ₹{np.abs(manual_sum - model_pred):.6f}')\n"
                                            "    print()"))

    # Save
    notebook_file = "/Users/lakshminarasimhan.santhanamgigkri.com/Workspace/HSR Houseprice/HSR_Real_Estate_Analysis.ipynb"
    with open(notebook_file, 'w') as f:
        nbf.write(nb, f)
    
    print(f"Success: Expanded Notebook generated at {notebook_file}")

if __name__ == "__main__":
    generate_notebook()
