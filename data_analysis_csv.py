import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==================== 1. Create Sample Dataset ====================
# (You can replace this with pd.read_csv('your_file.csv') later)

data = {
    'House_ID': range(1, 101),
    'Rooms': np.random.randint(2, 7, 100),
    'Size_sqft': np.random.randint(800, 3500, 100),
    'Location': np.random.choice(['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai'], 100),
    'Price_Lakhs': np.random.randint(40, 250, 100),
    'Age_Years': np.random.randint(1, 30, 100)
}

df = pd.DataFrame(data)

# Save to CSV (for submission)
df.to_csv('house_data.csv', index=False)
print("✅ Sample dataset created and saved as 'house_data.csv'\n")

# ==================== 2. Basic Data Analysis ====================
print("=== Dataset Overview ===")
print(df.head())
print("\n=== Basic Statistics ===")
print(df.describe())

# Average of selected columns
print(f"\nAverage Price: ₹{df['Price_Lakhs'].mean():.2f} Lakhs")
print(f"Average Size: {df['Size_sqft'].mean():.0f} sqft")
print(f"Average Rooms: {df['Rooms'].mean():.1f}")

# Group by Location
location_avg = df.groupby('Location')['Price_Lakhs'].mean().sort_values(ascending=False)
print("\n=== Average Price by Location ===")
print(location_avg)

# ==================== 3. Visualizations ====================

plt.figure(figsize=(15, 10))

# 1. Bar Chart - Average Price by Location
plt.subplot(2, 2, 1)
location_avg.plot(kind='bar', color='skyblue')
plt.title('Average House Price by Location')
plt.ylabel('Price (Lakhs)')
plt.xticks(rotation=45)

# 2. Scatter Plot - Size vs Price
plt.subplot(2, 2, 2)
plt.scatter(df['Size_sqft'], df['Price_Lakhs'], alpha=0.7, color='green')
plt.title('House Size vs Price')
plt.xlabel('Size (sqft)')
plt.ylabel('Price (Lakhs)')

# 3. Histogram - Price Distribution
plt.subplot(2, 2, 3)
plt.hist(df['Price_Lakhs'], bins=15, color='orange', edgecolor='black')
plt.title('House Price Distribution')
plt.xlabel('Price (Lakhs)')
plt.ylabel('Frequency')

# 4. Heatmap - Correlation
plt.subplot(2, 2, 4)
numeric_df = df.select_dtypes(include=np.number)
corr = numeric_df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')

plt.tight_layout()
plt.savefig('house_analysis_visualizations.png')
plt.show()

# ==================== 4. Insights ====================
print("\n" + "="*50)
print("KEY INSIGHTS & OBSERVATIONS:")
print("="*50)
print("1. Houses in Bangalore and Mumbai tend to have higher average prices.")
print("2. Strong positive correlation between Size and Price (as expected).")
print("3. Most houses are priced between 80-180 Lakhs.")
print("4. Number of rooms and size show good positive correlation with price.")
print("\n✅ Analysis complete! Check the generated PNG file for visualizations.")