import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load the data
file_path = 'converted_data.csv'  # Ensure this is the correct path to your CSV file
data = pd.read_csv(file_path)

# 2. Data Preprocessing

# Handling missing values by filling with mean (or other strategies as needed)
data.fillna(data.mean(), inplace=True)

# Set ACD110 as the target column
target_column = 'ACD110'
X = data.drop(target_column, axis=1)
y = data[target_column]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the feature data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Combine the processed training data into a DataFrame for saving
processed_data_train = pd.DataFrame(X_train, columns=X.columns)
processed_data_train[target_column] = y_train.values

# Save the processed data to a new CSV file
processed_file_path = 'processed_train_data.csv'
processed_data_train.to_csv(processed_file_path, index=False)

print(f"Processed data has been saved to {processed_file_path}")
