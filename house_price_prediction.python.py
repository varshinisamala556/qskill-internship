import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ==================== 1. Create Sample Dataset ====================
print("Creating sample house dataset...\n")

np.random.seed(42)
n_samples = 500

data = {
    'Rooms': np.random.randint(2, 7, n_samples),
    'Size_sqft': np.random.randint(800, 4000, n_samples),
    'Age_Years': np.random.randint(1, 40, n_samples),
    'Location': np.random.choice(['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai'], n_samples),
    'Price_Lakhs': None  # Will calculate below
}

df = pd.DataFrame(data)

# Generate realistic Price based on features
df['Price_Lakhs'] = (
    df['Size_sqft'] * 0.08 +
    df['Rooms'] * 15 +
    np.random.normal(0, 20, n_samples) -
    df['Age_Years'] * 1.2
).clip(lower=40)

df.to_csv('house_price_dataset.csv', index=False)
print("✅ Dataset saved as 'house_price_dataset.csv'")

# ==================== 2. Data Preprocessing & Model ====================
X = df.drop('Price_Lakhs', axis=1)
y = df['Price_Lakhs']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create preprocessing + model pipeline
categorical_features = ['Location']
numerical_features = ['Rooms', 'Size_sqft', 'Age_Years']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
    ],
    remainder='passthrough'
)

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# ==================== 3. Model Evaluation ====================
print("\n" + "="*50)
print("MODEL PERFORMANCE")
print("="*50)
print(f"Mean Absolute Error: ₹{mean_absolute_error(y_test, y_pred):.2f} Lakhs")
print(f"Root Mean Squared Error: ₹{np.sqrt(mean_squared_error(y_test, y_pred)):.2f} Lakhs")
print(f"R² Score: {r2_score(y_test, y_pred):.4f} ({r2_score(y_test, y_pred)*100:.1f}% accuracy)")

# ==================== 4. Example Predictions ====================
print("\nSample Predictions:")
sample = X_test.head(5).copy()
sample['Actual Price'] = y_test.head(5).values
sample['Predicted Price'] = y_pred[:5]
print(sample.round(2))

# ==================== 5. Visualization ====================
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.7, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Price (Lakhs)')
plt.ylabel('Predicted Price (Lakhs)')
plt.title('Actual vs Predicted House Prices')

plt.subplot(1, 2, 2)
plt.hist(y_test - y_pred, bins=20, color='green', alpha=0.7)
plt.title('Prediction Error Distribution')
plt.xlabel('Error (Lakhs)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.savefig('house_price_prediction_results.png')
plt.show()

print("\n✅ Task 3 Completed!")
print("Files created: house_price_dataset.csv and house_price_prediction_results.png")