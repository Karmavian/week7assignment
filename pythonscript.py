# Assignment: Data Loading, Analysis, and Visualization
# Objective: Load a dataset, perform basic analysis, and visualize using matplotlib & seaborn

# -------------------------------
# Task 0: Import Libraries
# -------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# For inline plots in Jupyter Notebook
%matplotlib inline  

# -------------------------------
# Task 1: Load and Explore Dataset
# -------------------------------
# Example: Using Iris dataset
try:
    df = pd.read_csv("iris.csv")  # replace with your dataset path
except FileNotFoundError:
    print("File not found. Please check the file path.")
    
# Inspect first few rows
print("First 5 rows of the dataset:")
print(df.head())

# Check data types and info
print("\nDataset Info:")
print(df.info())

# Check missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Fill or drop missing values if any
df = df.dropna()  # or df.fillna(value, inplace=True)

# -------------------------------
# Task 2: Basic Data Analysis
# -------------------------------

# Statistical summary of numerical columns
print("\nStatistical summary:")
print(df.describe())

# Example: Grouping by categorical column 'species' and computing mean of numerical columns
if 'species' in df.columns:
    group_mean = df.groupby('species').mean()
    print("\nMean values per species:")
    print(group_mean)

# Identify patterns or interesting findings
print("\nObservations:")
print("Check differences in means among species or categories.")

# -------------------------------
# Task 3: Data Visualization
# -------------------------------

# Set Seaborn style for aesthetics
sns.set(style="whitegrid")

# 1. Line chart (example: time-series if your dataset has a date column)
# Here, we simulate a trend by plotting first numerical column
plt.figure(figsize=(8,5))
plt.plot(df.index, df[df.columns[0]], marker='o')
plt.title("Line Chart of First Numerical Column")
plt.xlabel("Index")
plt.ylabel(df.columns[0])
plt.show()

# 2. Bar chart (comparison across categories)
if 'species' in df.columns:
    plt.figure(figsize=(8,5))
    sns.barplot(x='species', y=df.columns[0], data=df)
    plt.title(f"Average {df.columns[0]} per Species")
    plt.xlabel("Species")
    plt.ylabel(f"Average {df.columns[0]}")
    plt.show()

# 3. Histogram (distribution of a numerical column)
plt.figure(figsize=(8,5))
sns.histplot(df[df.columns[0]], bins=10, kde=True)
plt.title(f"Histogram of {df.columns[0]}")
plt.xlabel(df.columns[0])
plt.ylabel("Frequency")
plt.show()

# 4. Scatter plot (relationship between two numerical columns)
plt.figure(figsize=(8,5))
sns.scatterplot(x=df.columns[0], y=df.columns[1], hue='species' if 'species' in df.columns else None)
plt.title(f"Scatter Plot: {df.columns[0]} vs {df.columns[1]}")
plt.xlabel(df.columns[0])
plt.ylabel(df.columns[1])
plt.show()

# -------------------------------
# End of Assignment
# -------------------------------