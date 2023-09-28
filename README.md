# Car_price_analysis
**Car Price Analysis Using Regression**
This project provides a Python script for car price analysis using regression techniques. The script follows a series of steps to preprocess data, train a regression model, and evaluate its performance. Below is an overview of each part of the code:

Function Definitions
The code defines several functions:

DistributionPlot: This function creates distribution plots (kernel density estimates) to compare two datasets.
plots: It generates Seaborn box plots and scatter plots for data visualization.
pearsoncorelation: This function calculates the Pearson correlation coefficient and p-value between two variables and displays the results.
Data Loading and Preprocessing
The script loads a dataset from a CSV file and performs preprocessing:

Replaces missing values marked as '?' with zeros.
Scales numeric features using Min-Max scaling.
Replaces zero values in the "price" column with the mean and multiplies the price values by a factor (80 in this case).
Encodes categorical variables using label encoding.
Data Splitting
The data is split into training and testing sets using scikit-learn's train_test_split function.

Polynomial Feature Transformation
Polynomial features are created from the input features using scikit-learn's PolynomialFeatures. This step captures more complex relationships between variables.

Ridge Regression Model
The script trains a Ridge regression model on the transformed data. Ridge regression is used to prevent overfitting by adding a regularization term to the cost function.

Model Evaluation
The code evaluates the Ridge regression model by calculating:

Pearson correlation coefficients and p-values for numeric features, with the results visualized.
Mean of coefficients and the intercept of the regression model.
R-squared value, a metric for assessing model performance.
Cross-validation scores and predictions.
Distribution Plot
The final step involves creating a distribution plot to visualize the actual and predicted values from the model.

Please note that this code assumes proper data formatting and handling of any missing dependencies. Make sure to adjust the code as needed for your specific dataset and environment.

