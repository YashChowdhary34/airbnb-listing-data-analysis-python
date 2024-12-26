# Airbnb Listing Data Analysis

<div align="center">
    <img src="https://github.com/user-attachments/assets/39c2fcbc-b024-4376-b5e3-3676b579755f" alt="Zomato Logo" height="300" width="500">
</div>

## Overview
This project analyzes Airbnb listings to uncover market trends, user behaviors, and property insights using Python. The analysis involves dataset exploration, data cleaning, univariate and bivariate analysis, feature engineering, and deriving insights from visualizations. 

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset Information](#dataset-information)
3. [Exploration](#exploration)
4. [Data Cleaning](#data-cleaning)
5. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
   - [Univariate Analysis](#univariate-analysis)
   - [Bivariate Analysis](#bivariate-analysis)
6. [Feature Engineering](#feature-engineering)
7. [Future Work](#future-work)
8. [Conclusion](#conclusion)

## Introduction
The Airbnb market is competitive, with various factors influencing property listings' success. This project dives deep into data analysis to identify:
- Popular property types and locations
- Pricing strategies
- Key features contributing to higher ratings or bookings

---

## Dataset Information
The dataset provides detailed information about Airbnb listings. Key columns include:
- **`id`**: Unique identifier for listings
- **`name`**: Name of the listing
- **`host_name`**: Name of the host
- **`neighbourhood_group`**: Broad geographical area of the listing
- **`neighbourhood`**: Specific location
- **`room_type`**: Type of room offered (e.g., entire home, private room)
- **`price`**: Nightly price of the listing
- **`availability_365`**: Number of available days in a year
- **`reviews_per_month`**: Average monthly reviews

```python
import pandas as pd

# Load the dataset
file_path = "dataset/airbnb_listings.csv"
data = pd.read_csv(file_path)

# Display basic information
data.info()
```

---

## Exploration
We begin by exploring the dataset to understand its structure, identify missing values, and check for inconsistencies.

```python
# Display the first few rows of the dataset
data.head()

# Check for missing values
data.isnull().sum()

# Describe numerical columns
data.describe()
```

**Insights:**
- Missing values in `reviews_per_month` and `host_name`.
- High variability in `price` with some outliers.

---

## Data Cleaning
Data cleaning ensures a consistent and analyzable dataset. This step involves:
1. Filling missing values
2. Removing duplicates
3. Addressing outliers

```python
# Fill missing values
data['reviews_per_month'].fillna(0, inplace=True)

# Drop duplicate rows
data.drop_duplicates(inplace=True)

# Handle outliers in the price column
data = data[data['price'] < data['price'].quantile(0.95)]
```

**Insights:**
- Missing values in `reviews_per_month` replaced with 0.
- Outliers in the price column capped at the 95th percentile.

---

## Exploratory Data Analysis (EDA)

### Univariate Analysis
Analyzing individual variables to identify distribution and patterns.

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Distribution of prices
sns.histplot(data['price'], bins=50, kde=True)
plt.title("Price Distribution")
plt.show()

# Room type frequency
sns.countplot(data=data, x='room_type')
plt.title("Room Types")
plt.show()
```

**Insights:**
- Most properties are priced under $200.
- "Entire home/apartment" is the most common room type.

### Bivariate Analysis
Exploring relationships between two variables.

```python
# Price vs. neighbourhood group
sns.boxplot(data=data, x='neighbourhood_group', y='price')
plt.title("Price by Neighbourhood Group")
plt.show()

# Correlation heatmap
corr = data.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()
```

**Insights:**
- Listings in some neighbourhood groups are priced significantly higher.
- Moderate correlation between `availability_365` and `price`.

---

## Feature Engineering
Creating new features to improve data analysis and predictive modeling.

```python
# Create a price category feature
data['price_category'] = pd.cut(
    data['price'], bins=[0, 100, 200, 300, float('inf')], 
    labels=['Low', 'Medium', 'High', 'Very High']
)

# Extract host activity level based on availability
data['host_activity'] = data['availability_365'].apply(lambda x: 'Active' if x > 150 else 'Inactive')
```

**Insights:**
- Categorizing price helps identify market segments.
- Host activity can highlight differences in listing strategies.

---

## Future Work
1. Incorporate additional datasets to analyze seasonal trends.
2. Use machine learning models for price prediction.
3. Explore sentiment analysis of reviews for qualitative insights.

---

## Conclusion
This analysis provided valuable insights into the Airbnb market:
- Popular room types and locations.
- Price distribution and trends.
- Potential features like `price_category` and `host_activity` for further analysis.

Future work will focus on predictive modeling and integrating user feedback to enhance decision-making for stakeholders.
