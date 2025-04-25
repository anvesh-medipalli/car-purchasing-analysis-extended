import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("car_purchasing.csv",encoding='latin1')

# Data Overview
print("Data Info:")
print(df.info())
print("\nStatistics:")
print(df.describe())

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Pairplot of numeric features
sns.pairplot(df.select_dtypes(include=['float64', 'int64']))
plt.suptitle("Numeric Features Pairplot", y=1.02)
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
