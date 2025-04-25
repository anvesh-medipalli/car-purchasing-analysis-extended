import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

# Load data
df = pd.read_csv("car_purchasing.csv", encoding="latin1")

# Remove any leading/trailing spaces from column names
df.columns = df.columns.str.strip()

# Print column names to check them
print("Available Columns:", df.columns)

# Title
st.title("🚗 Car Purchasing Dashboard")

# Show data
if st.checkbox("Show raw data"):
    st.write(df.head())

# Correlation heatmap
st.subheader("Correlation Heatmap")
fig, ax = plt.subplots()

# Select only numeric columns for the heatmap
numeric_df = df.select_dtypes(include='number')

# Plot the heatmap
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# Predict car purchase amount
st.subheader("Predict Car Purchase Amount")

# Define features and target columns based on actual column names
required_columns = ['age', 'annual Salary', 'credit card debt', 'net worth']

# Check if the required columns exist
if all(col in df.columns for col in required_columns):
    features = df[required_columns]
else:
    st.error(f"Some columns are missing. Available columns are: {df.columns}")

# Target column
target = df['car purchase amount']

# Input sliders and number inputs
age = st.slider("Age", 18, 70, 30)
salary = st.number_input("Annual Salary", 20000, 200000, 50000)
debt = st.number_input("Credit Card Debt", 0, 50000, 5000)
worth = st.number_input("Net Worth", 1000, 1000000, 100000)

# Initialize and train the RandomForestRegressor model
model = RandomForestRegressor()
model.fit(features, target)

# Create input data for prediction
input_data = pd.DataFrame([[age, salary, debt, worth]],
                          columns=['age', 'annual Salary', 'credit card debt', 'net worth'])

# Predict car purchase amount
prediction = model.predict(input_data)[0]
st.success(f"Estimated Car Purchase Amount: ${prediction:,.2f}")
