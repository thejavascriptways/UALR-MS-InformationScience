print('STEP - Import libraries required for various operations')
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


print('STEP - Read the data, for this project data is loaded in collab local repo')
try:
    df = pd.read_csv('/Users/deepaksingla/Documents/finalProjectReportDataset.csv')
    print("Training data loaded successfully.")
    print(f"Dataset shape: {df.shape}")
except FileNotFoundError:
    print("Dataset not found. Please ensure the file is in the correct directory.")
    # Handle error or exit if the file is essential
    df = None # Set to None if loading failed
    

print('STEP - Data Inspection')
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


print('STEP - Explore the data and targetbvariable price using a histogram plot')
if  df is not None:
    plt.figure(figsize=(4,2))
    sns.histplot(df, kde=True)
    plt.title('Distribution of SalePrice')
    plt.xlabel('Sale Price ($)')
    plt.ylabel('Frequency')
    plt.show()

    # Calculate and print skewness and kurtosis
    print(f"Skewness: {df.skew():.2f}")
    print(f"Kurtosis: {df.kurt():.2f}")
else:
    print("\nSalePrice analysis skipped.")
