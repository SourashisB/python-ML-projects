import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set the style for our visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")

def load_data():
    """
    Load the CSV data file from the current directory
    """
    # Get all CSV files in the current directory
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    
    if not csv_files:
        print("No CSV files found in the current directory.")
        return None
    
    # If there are multiple, use the first one
    if len(csv_files) > 1:
        print(f"Multiple CSV files found. Using: {csv_files[0]}")
    else:
        print(f"Loading data from: {csv_files[0]}")
        
    # Load the data
    try:
        df = pd.read_csv(csv_files[0])
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def explore_basic_stats(df):
    """
    Explore basic statistics and information about the dataset
    """
    print("\n=== BASIC DATASET INFORMATION ===")
    
    # Display basic info
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nBasic information:")
    print(df.info())
    
    print("\nSummary statistics:")
    print(df.describe().round(2))
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("\nMissing values by column:")
        print(missing[missing > 0])
    else:
        print("\nNo missing values found in the dataset.")
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    print(f"\nNumber of duplicate rows: {duplicates}")
    
    return

def analyze_demographics(df):
    """
    Analyze demographic information
    """
    print("\n=== DEMOGRAPHIC ANALYSIS ===")
    
    # Create a figure with multiple subplots
    plt.figure(figsize=(16, 12))
    
    # Age distribution
    plt.subplot(2, 3, 1)
    sns.histplot(df['Age'], bins=20, kde=True)
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Count')
    
    # Gender distribution
    plt.subplot(2, 3, 2)
    gender_counts = df['Gender'].value_counts()
    plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%')
    plt.title('Gender Distribution')
    
    # Income level
    plt.subplot(2, 3, 3)
    income_order = sorted(df['Income_Level'].unique())
    sns.countplot(data=df, x='Income_Level', order=income_order)
    plt.title('Income Level Distribution')
    plt.xticks(rotation=45)
    
    # Marital status
    plt.subplot(2, 3, 4)
    sns.countplot(data=df, x='Marital_Status')
    plt.title('Marital Status Distribution')
    plt.xticks(rotation=45)
    
    # Education level
    plt.subplot(2, 3, 5)
    sns.countplot(data=df, x='Education_Level')
    plt.title('Education Level Distribution')
    plt.xticks(rotation=45)
    
    # Location
    plt.subplot(2, 3, 6)
    location_counts = df['Location'].value_counts().head(10)  # Top 10 locations
    sns.barplot(x=location_counts.index, y=location_counts.values)
    plt.title('Top 10 Locations')
    plt.xticks(rotation=90)
    
    plt.tight_layout()
    plt.show()
    
    # Demographics correlation with purchase amount
    print("\nAverage Purchase Amount by Demographics:")
    
    plt.figure(figsize=(18, 12))
    
    # Age vs Purchase Amount
    plt.subplot(2, 3, 1)
    sns.scatterplot(data=df, x='Age', y='Purchase_Amount', alpha=0.6)
    plt.title('Age vs Purchase Amount')
    
    # Gender vs Purchase Amount
    plt.subplot(2, 3, 2)
    sns.boxplot(data=df, x='Gender', y='Purchase_Amount')
    plt.title('Gender vs Purchase Amount')
    
    # Income Level vs Purchase Amount
    plt.subplot(2, 3, 3)
    sns.boxplot(data=df, x='Income_Level', y='Purchase_Amount', order=income_order)
    plt.title('Income Level vs Purchase Amount')
    plt.xticks(rotation=45)
    
    # Marital Status vs Purchase Amount
    plt.subplot(2, 3, 4)
    sns.boxplot(data=df, x='Marital_Status', y='Purchase_Amount')
    plt.title('Marital Status vs Purchase Amount')
    plt.xticks(rotation=45)
    
    # Education Level vs Purchase Amount
    plt.subplot(2, 3, 5)
    sns.boxplot(data=df, x='Education_Level', y='Purchase_Amount')
    plt.title('Education Level vs Purchase Amount')
    plt.xticks(rotation=45)
    
    # Occupation vs Purchase Amount (top 10)
    plt.subplot(2, 3, 6)
    occupation_avg = df.groupby('Occupation')['Purchase_Amount'].mean().sort_values(ascending=False).head(10)
    sns.barplot(x=occupation_avg.index, y=occupation_avg.values)
    plt.title('Top 10 Occupations by Avg Purchase Amount')
    plt.xticks(rotation=90)
    
    plt.tight_layout()
    plt.show()
    
    return

def analyze_purchase_behavior(df):
    """
    Analyze purchase behavior and patterns
    """
    print("\n=== PURCHASE BEHAVIOR ANALYSIS ===")
    
    plt.figure(figsize=(16, 12))
    
    # Purchase Amount Distribution
    plt.subplot(2, 3, 1)
    sns.histplot(df['Purchase_Amount'], bins=30, kde=True)
    plt.title('Purchase Amount Distribution')
    plt.xlabel('Purchase Amount')
    plt.ylabel('Count')
    
    # Purchase Category Distribution
    plt.subplot(2, 3, 2)
    category_counts = df['Purchase_Category'].value_counts().head(10)  # Top 10 categories
    sns.barplot(x=category_counts.index, y=category_counts.values)
    plt.title('Top 10 Purchase Categories')
    plt.xticks(rotation=90)
    
    # Average Purchase Amount by Category
    plt.subplot(2, 3, 3)
    category_avg = df.groupby('Purchase_Category')['Purchase_Amount'].mean().sort_values(ascending=False).head(10)
    sns.barplot(x=category_avg.index, y=category_avg.values)
    plt.title('Avg Purchase Amount by Category (Top 10)')
    plt.xticks(rotation=90)
    
    # Purchase Channel Distribution
    plt.subplot(2, 3, 4)
    sns.countplot(data=df, x='Purchase_Channel')
    plt.title('Purchase Channel Distribution')
    plt.xticks(rotation=45)
    
    # Purchase Frequency Distribution
    plt.subplot(2, 3, 5)
    sns.histplot(df['Frequency_of_Purchase'], bins=20, kde=True)
    plt.title('Purchase Frequency Distribution')
    
    # Payment Method Distribution
    plt.subplot(2, 3, 6)
    payment_counts = df['Payment_Method'].value_counts()
    plt.pie(payment_counts, labels=payment_counts.index, autopct='%1.1f%%')
    plt.title('Payment Method Distribution')
    
    plt.tight_layout()
    plt.show()
    
    # Time-related analysis
    plt.figure(figsize=(16, 8))
    
    # Time of Purchase Distribution (if it's in a time format)
    plt.subplot(1, 2, 1)
    try:
        # Assuming Time_of_Purchase is in some recognizable format
        # This may need adjustment based on the actual format
        df['Purchase_Hour'] = pd.to_datetime(df['Time_of_Purchase']).dt.hour
        sns.countplot(data=df, x='Purchase_Hour')
        plt.title('Purchase Distribution by Hour of Day')
        plt.xlabel('Hour of Day')
    except:
        print("Could not parse Time_of_Purchase into hour format")
        sns.countplot(data=df, x='Time_of_Purchase')
        plt.title('Purchase Distribution by Time')
        plt.xticks(rotation=90)
    
    # Time to Decision Distribution
    plt.subplot(1, 2, 2)
    sns.histplot(df['Time_to_Decision'], bins=20, kde=True)
    plt.title('Time to Decision Distribution')
    
    plt.tight_layout()
    plt.show()
    
    return

def analyze_customer_satisfaction(df):
    """
    Analyze customer satisfaction and related metrics
    """
    print("\n=== CUSTOMER SATISFACTION ANALYSIS ===")
    
    plt.figure(figsize=(16, 12))
    
    # Customer Satisfaction Distribution
    plt.subplot(2, 3, 1)
    sns.countplot(data=df, x='Customer_Satisfaction')
    plt.title('Customer Satisfaction Distribution')
    
    # Product Rating Distribution
    plt.subplot(2, 3, 2)
    sns.countplot(data=df, x='Product_Rating')
    plt.title('Product Rating Distribution')
    
    # Return Rate Analysis
    plt.subplot(2, 3, 3)
    sns.countplot(data=df, x='Return_Rate')
    plt.title('Return Rate Distribution')
    
    # Brand Loyalty Distribution
    plt.subplot(2, 3, 4)
    sns.countplot(data=df, x='Brand_Loyalty')
    plt.title('Brand Loyalty Distribution')
    
    # Customer Loyalty Program Membership
    plt.subplot(2, 3, 5)
    loyalty_counts = df['Customer_Loyalty_Program_Member'].value_counts()
    plt.pie(loyalty_counts, labels=loyalty_counts.index, autopct='%1.1f%%')
    plt.title('Customer Loyalty Program Membership')
    
    # Satisfaction vs Purchase Amount
    plt.subplot(2, 3, 6)
    sns.boxplot(data=df, x='Customer_Satisfaction', y='Purchase_Amount')
    plt.title('Satisfaction vs Purchase Amount')
    
    plt.tight_layout()
    plt.show()
    
    # Correlation between satisfaction and other metrics
    satisfaction_corr = df[['Customer_Satisfaction', 'Product_Rating', 'Brand_Loyalty', 
                         'Return_Rate', 'Purchase_Amount', 'Frequency_of_Purchase']].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(satisfaction_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Between Satisfaction and Other Metrics')
    plt.tight_layout()
    plt.show()
    
    return

def analyze_marketing_influence(df):
    """
    Analyze marketing and external influence factors
    """
    print("\n=== MARKETING INFLUENCE ANALYSIS ===")
    
    plt.figure(figsize=(16, 12))
    
    # Social Media Influence Distribution
    plt.subplot(2, 3, 1)
    sns.countplot(data=df, x='Social_Media_Influence')
    plt.title('Social Media Influence Distribution')
    plt.xticks(rotation=45)
    
    # Engagement with Ads Distribution
    plt.subplot(2, 3, 2)
    sns.countplot(data=df, x='Engagement_with_Ads')
    plt.title('Engagement with Ads Distribution')
    
    # Discount Sensitivity Distribution
    plt.subplot(2, 3, 3)
    sns.countplot(data=df, x='Discount_Sensitivity')
    plt.title('Discount Sensitivity Distribution')
    
    # Discount Used Distribution
    plt.subplot(2, 3, 4)
    discount_counts = df['Discount_Used'].value_counts()
    plt.pie(discount_counts, labels=discount_counts.index, autopct='%1.1f%%')
    plt.title('Discount Used Distribution')
    
    # Social Media Influence vs Purchase Amount
    plt.subplot(2, 3, 5)
    sns.boxplot(data=df, x='Social_Media_Influence', y='Purchase_Amount')
    plt.title('Social Media Influence vs Purchase Amount')
    plt.xticks(rotation=45)
    
    # Discount Sensitivity vs Purchase Amount
    plt.subplot(2, 3, 6)
    sns.boxplot(data=df, x='Discount_Sensitivity', y='Purchase_Amount')
    plt.title('Discount Sensitivity vs Purchase Amount')
    
    plt.tight_layout()
    plt.show()
    
    return

def analyze_correlations(df):
    """
    Analyze correlations between numerical variables
    """
    print("\n=== CORRELATION ANALYSIS ===")
    
    # Select numerical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Create correlation matrix
    correlation_matrix = df[numerical_cols].corr()
    
    # Plot heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
    plt.title('Correlation Matrix of Numerical Variables')
    plt.tight_layout()
    plt.show()
    
    # Top 5 positive correlations (excluding self-correlations)
    positive_corr = correlation_matrix.unstack().sort_values(ascending=False)
    positive_corr = positive_corr[positive_corr < 1]  # Remove self-correlations
    print("\nTop 5 Positive Correlations:")
    print(positive_corr.head(5))
    
    # Top 5 negative correlations
    negative_corr = correlation_matrix.unstack().sort_values()
    print("\nTop 5 Negative Correlations:")
    print(negative_corr.head(5))
    
    return

def segmentation_analysis(df):
    """
    Perform basic customer segmentation analysis
    """
    print("\n=== CUSTOMER SEGMENTATION ANALYSIS ===")
    
    # RFM (Recency, Frequency, Monetary) analysis proxy
    # Since we don't have recency, we'll use Frequency_of_Purchase and Purchase_Amount
    
    # Create segments based on purchase frequency
    frequency_bins = [0, 2, 5, 10, float('inf')]
    frequency_labels = ['Low', 'Medium', 'High', 'Very High']
    df['Frequency_Segment'] = pd.cut(df['Frequency_of_Purchase'], bins=frequency_bins, labels=frequency_labels)
    
    # Create segments based on purchase amount
    amount_bins = [0, 100, 500, 1000, float('inf')]
    amount_labels = ['Low', 'Medium', 'High', 'Very High']
    df['Amount_Segment'] = pd.cut(df['Purchase_Amount'], bins=amount_bins, labels=amount_labels)
    
    # Plot segment distributions
    plt.figure(figsize=(16, 10))
    
    # Frequency segment distribution
    plt.subplot(2, 2, 1)
    sns.countplot(data=df, x='Frequency_Segment')
    plt.title('Frequency Segment Distribution')
    
    # Amount segment distribution
    plt.subplot(2, 2, 2)
    sns.countplot(data=df, x='Amount_Segment')
    plt.title('Amount Segment Distribution')
    
    # Average purchase amount by frequency segment
    plt.subplot(2, 2, 3)
    sns.barplot(data=df, x='Frequency_Segment', y='Purchase_Amount')
    plt.title('Average Purchase Amount by Frequency Segment')
    
    # Customer satisfaction by amount segment
    plt.subplot(2, 2, 4)
    sns.boxplot(data=df, x='Amount_Segment', y='Customer_Satisfaction')
    plt.title('Customer Satisfaction by Amount Segment')
    
    plt.tight_layout()
    plt.show()
    
    # Create a combined segment
    df['Customer_Segment'] = df['Frequency_Segment'].astype(str) + ' Frequency / ' + df['Amount_Segment'].astype(str) + ' Amount'
    
    # Plot key metrics by the top combined segments
    top_segments = df['Customer_Segment'].value_counts().head(5).index
    
    plt.figure(figsize=(15, 10))
    
    # Customer satisfaction by segment
    plt.subplot(2, 1, 1)
    segment_satisfaction = df[df['Customer_Segment'].isin(top_segments)].groupby('Customer_Segment')['Customer_Satisfaction'].mean().sort_values(ascending=False)
    sns.barplot(x=segment_satisfaction.index, y=segment_satisfaction.values)
    plt.title('Average Customer Satisfaction by Top 5 Segments')
    plt.xticks(rotation=45)
    
    # Return rate by segment
    plt.subplot(2, 1, 2)
    segment_return = df[df['Customer_Segment'].isin(top_segments)].groupby('Customer_Segment')['Return_Rate'].mean().sort_values()
    sns.barplot(x=segment_return.index, y=segment_return.values)
    plt.title('Average Return Rate by Top 5 Segments')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    return

def main():
    # Load the data
    df = load_data()
    if df is None:
        return
    
    # Basic exploratory analysis
    explore_basic_stats(df)
    
    # Demographic analysis
    analyze_demographics(df)
    
    # Purchase behavior analysis
    analyze_purchase_behavior(df)
    
    # Customer satisfaction analysis
    analyze_customer_satisfaction(df)
    
    # Marketing influence analysis
    analyze_marketing_influence(df)
    
    # Correlation analysis
    analyze_correlations(df)
    
    # Segmentation analysis
    segmentation_analysis(df)
    
    print("\nExploratory data analysis complete.")

if __name__ == "__main__":
    main()