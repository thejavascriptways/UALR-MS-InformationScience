#Step - Import libraries required for various operations
# Data manipulation
import pandas as pd
import numpy as np

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats # For statistics like skewness and kurtosis

# visualizations
#%matplotlib inline
sns.set_style('darkgrid') # grid style setting other options 'whitegrid', 'dark', 'white', and 'ticks'
plt.style.use('fivethirtyeight') # Another popular style

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Optional: Display settings for pandas
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

print("Python libraries imported successfully")



# Step - Read the data, for this project data is loaded in collab local repo
try:
    df = pd.read_csv('D:\documents\ms\ifsc 7320\AmesHousing.csv')
    print("Data loaded successfully.")
    print(f"Dataset shape: {df.shape}")
except FileNotFoundError:
    print("Dataset not found. Please ensure the file is in the correct directory.")
    # Handle error or exit if the file is essential
    df = None # Set to None if loading failed


# Step - Data Inspection
if  df is not None:
    # Display the first 5 rows
    print("\nFirst 5 rows of the dataset:")
    print(df.head())

    # Display basic information (non-null counts, data types)
    print("\nDataset Info:")
    df.info()

    # Display descriptive statistics for numerical columns
    print("\nDescriptive Statistics (Numerical Features):")
    # Include 'all' to get stats for object columns too (like count, unique, top, freq)
    print(df.describe(include='all'))
else:
    print("\nSkipping initial inspection due to data loading issues.")


#STEP Data clean up for projecting plots

def clean_data_for_histplot(df):
    """
    Cleans a dataframe type input array to ensure compatibility with seaborn.histplot, handling common data issues.

    Args:
        pandas.DataFrame

    Returns:
        pandas.DataFrame: A cleaned DataFrame.  Returns None if an error occurs.
    """
    try:
        
       # 1. Drop the 'Order' and 'PID' columns, as they are not needed for plotting
        columns_to_drop = ['Order', 'PID']
        df = df.drop(columns=columns_to_drop, errors='ignore')  # errors='ignore' prevents errors if the columns don't exist

        # 2. Handle missing values
        # a. Identify numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        # b. Impute missing values in numeric columns with the median
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())

        # c. Identify non-numeric columns
        non_numeric_cols = df.select_dtypes(exclude=['number']).columns
        # d. Impute missing values in non-numeric columns with the most frequent value
        for col in non_numeric_cols:
            df[col] = df[col].fillna(df[col].mode()[0])

        # 3. Convert year columns to datetime objects
        year_cols = ['Year Built', 'Year Remod/Add', 'Garage Yr Blt', 'Yr Sold']
        for col in year_cols:
            if col in df.columns: #check if the column exists
                df[col] = pd.to_datetime(df[col], errors='coerce') #errors='coerce' will replace invalid values with NaT

        
        # 4. Remove any infinite values (which can cause problems with plotting)
        df = df.replace([float('inf'), float('-inf')], pd.NA)

        # 5. Drop any rows where the target variable has missing values
        if 'SalePrice' in df.columns:
            df = df.dropna(subset=['SalePrice'])

        # 6. Drop any remaining rows with missing values
        df = df.dropna()

        # 7. Print info about the DataFrame *before* returning it
        print("Cleaned DataFrame Info:")
        df.info()

        # 8. Return the cleaned DataFrame
        return df

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


print('Cleaning data')

#user-defined method to clean up data
#such as missing values, non numeric values, dattime fields getting used in plots
df_cleaned = clean_data_for_histplot(df)

print('Cleaning finish')

# #STEP Target Variable Analysis*********************************************************
# #Step - Explore the data and target variable price using a histogram plot

# Distribution Plot

if  df is not None:

    print(df_cleaned.info())
    plt.figure(figsize=(10,6))
    sns.histplot(data=df_cleaned, x='SalePrice')
    #plt.show()
    plt.title('Distribution of SalePrice')
    plt.xlabel('Sale Price ($)')
    plt.ylabel('Frequency1')
    plt.show()
    print('\nDistribution plot complete')
else:
    print("\nSalePrice analysis skipped.")

# Log Transformation
print('Log Transformation')
if df_cleaned is not None:
    # Apply log1p transformation
    train_df = np.log1p(df_cleaned['SalePrice'])

    plt.figure(figsize=(10, 6))
    sns.histplot(train_df, kde=True)
    plt.title('Distribution of Log-Transformed SalePrice')
    plt.xlabel('Log(1 + Sale Price)')
    plt.ylabel('Frequency')
    plt.show()
    
else:
    print("\nLog-transformed SalePrice analysis skipped.")


#Combined Plots

if df_cleaned is not None:
    print("\nFirst 5 rows of the cleaned DataFrame:")
    print(df_cleaned.head())
    # Apply log1p transformation to SalePrice *after* cleaning
    if 'SalePrice' in df_cleaned.columns:
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        log1p_SalePrice = np.log1p(df_cleaned['SalePrice'])

        # Create a figure with two subplots
        plt.figure(figsize=(15, 6))

        # Plot the original SalePrice distribution
        plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
        sns.histplot(df_cleaned['SalePrice'], kde=True)
        plt.title('Original Distribution of SalePrice')
        plt.xlabel('Sale Price')
        plt.ylabel('Frequency')

        # Plot the log-transformed SalePrice distribution
        plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
        sns.histplot(log1p_SalePrice, kde=True)
        plt.title('Distribution of Log-Transformed SalePrice')
        plt.xlabel('Log(1 + Sale Price)')
        plt.ylabel('Frequency')

        plt.tight_layout()  # Adjust layout to prevent overlapping
        plt.show()
    else:
        print("Error: 'SalePrice' column not found in the cleaned DataFrame.")
else:
    print("Error: Unable to process the data.  Check the file path and format.")


# STEP Numerical Value Analysis


#Corelation Heatmap

if df_cleaned is not None:
    # Select only numerical columns for correlation analysis
    numerical_cols = df_cleaned.select_dtypes(include=np.number).columns
    correlation_matrix = df_cleaned[numerical_cols].corr()

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False, fmt=".1f") # annot=True can be slow for many features
    plt.title('Correlation Heatmap of Numerical Features')
    plt.show()

    # Display top N correlations with SalePrice (using the log-transformed version)
    print("\nTop 10 Features Correlated with Log(SalePrice):")
    # Ensure 'SalePrice_Log' exists before using it
    if 'SalePrice_Log' in correlation_matrix.columns:
        print(correlation_matrix.sort_values(ascending=False).head(11))
    else:
        print("Log-transformed SalePrice not available for correlation ranking.")
else:
    print("\nCorrelation heatmap skipped.")



#STEP Missing Data Analysis

if df_cleaned is not None:
    # Calculate missing values
    missing_values = df_cleaned.isnull().sum()
    missing_percent = (missing_values / len(df_cleaned)) * 100
    missing_data_summary = pd.DataFrame({'Missing Count': missing_values, 'Missing Percentage': missing_percent})

    # Filter to show only columns with missing values and sort
    missing_data_summary = missing_data_summary[missing_data_summary['Missing Count'] > 0].sort_values(by='Missing Percentage', ascending=False)

    print("\nFeatures with Missing Values:")
    print(missing_data_summary)

    # Optional: Visualize missing percentages
    if not missing_data_summary.empty:
        plt.figure(figsize=(12, 6))
        sns.barplot(x=missing_data_summary.index, y=missing_data_summary['Missing Percentage'])
        plt.xticks(rotation=90)
        plt.title('Percentage of Missing Values by Feature')
        plt.ylabel('Percentage (%)')
        plt.show()
else:
    print("\nMissing data analysis skipped.")
