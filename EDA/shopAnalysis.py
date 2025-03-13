import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# Set styling for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')

def load_data(file_path='Shop_Analysis.csv'):
    """Load the CSV data and perform initial cleaning."""
    try:
        # Load the data
        df = pd.read_csv(file_path)
        
        # Clean Purchase_Amount (remove $ and convert to float)
        df['Purchase_Amount'] = df['Purchase_Amount'].str.replace('$', '').astype(float)
        
        # Convert time fields to proper format if they exist as datetime strings
        time_columns = ['Time_of_Purchase']
        for col in time_columns:
            if col in df.columns and df[col].dtype == 'object':
                try:
                    df[col] = pd.to_datetime(df[col])
                except:
                    print(f"Could not convert {col} to datetime.")
        
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def generate_basic_stats(df):
    """Generate basic statistics about the dataset."""
    # Basic info
    info_buffer = []
    info_buffer.append("Dataset Overview:")
    info_buffer.append(f"Number of records: {df.shape[0]}")
    info_buffer.append(f"Number of features: {df.shape[1]}")
    info_buffer.append("\nMissing values by column:")
    missing_data = df.isnull().sum()
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    missing_info = pd.DataFrame({'Missing Values': missing_data, 
                                'Percentage': missing_percentage})
    missing_info = missing_info[missing_info['Missing Values'] > 0]
    
    if not missing_info.empty:
        info_buffer.append(missing_info.to_string())
    else:
        info_buffer.append("No missing values found.")
    
    # Numeric columns statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if not numeric_cols.empty:
        info_buffer.append("\nNumeric Columns Statistics:")
        info_buffer.append(df[numeric_cols].describe().to_string())
    
    # Categorical columns statistics
    categorical_cols = df.select_dtypes(include=['object']).columns
    if not categorical_cols.empty:
        info_buffer.append("\nCategorical Columns Statistics:")
        for col in categorical_cols:
            info_buffer.append(f"\n{col} - Top 5 values:")
            info_buffer.append(df[col].value_counts().head(5).to_string())
    
    # Save the basic stats to a text file
    with open('basic_statistics.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(info_buffer))
    
    print("Basic statistics saved to 'basic_statistics.txt'")

def generate_visualizations(df):
    """Generate various visualizations for the dataset."""
    # Create a directory for visualizations if it doesn't exist
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')
    
    # 1. Distribution of Purchase Amounts
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Purchase_Amount'], kde=True)
    plt.title('Distribution of Purchase Amounts')
    plt.xlabel('Purchase Amount')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('visualizations/purchase_amount_distribution.png')
    plt.close()
    
    # 2. Purchase Amount by Category
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Purchase_Category', y='Purchase_Amount', data=df)
    plt.title('Purchase Amount by Category')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('visualizations/purchase_by_category.png')
    plt.close()
    
    # 3. Purchase Amount by Gender
    if 'Gender' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Gender', y='Purchase_Amount', data=df)
        plt.title('Purchase Amount by Gender')
        plt.tight_layout()
        plt.savefig('visualizations/purchase_by_gender.png')
        plt.close()
    
    # 4. Purchase Amount by Age Group
    if 'Age' in df.columns:
        # Create age groups
        df['Age_Group'] = pd.cut(df['Age'], bins=[0, 18, 25, 35, 45, 55, 65, 100], 
                                  labels=['<18', '18-25', '26-35', '36-45', '46-55', '56-65', '65+'])
        
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Age_Group', y='Purchase_Amount', data=df)
        plt.title('Purchase Amount by Age Group')
        plt.tight_layout()
        plt.savefig('visualizations/purchase_by_age_group.png')
        plt.close()
    
    # 5. Correlation Heatmap for Numeric Variables
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        plt.figure(figsize=(12, 10))
        correlation = df[numeric_cols].corr()
        mask = np.triu(correlation)
        sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5, mask=mask)
        plt.title('Correlation Matrix of Numeric Features')
        plt.tight_layout()
        plt.savefig('visualizations/correlation_heatmap.png')
        plt.close()
    
    # 6. Purchase Channel Distribution
    if 'Purchase_Channel' in df.columns:
        plt.figure(figsize=(10, 6))
        channel_counts = df['Purchase_Channel'].value_counts()
        sns.barplot(x=channel_counts.index, y=channel_counts.values)
        plt.title('Distribution of Purchase Channels')
        plt.xlabel('Purchase Channel')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('visualizations/purchase_channel_distribution.png')
        plt.close()
    
    # 7. Customer Satisfaction Distribution
    if 'Customer_Satisfaction' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.countplot(x='Customer_Satisfaction', data=df)
        plt.title('Distribution of Customer Satisfaction')
        plt.xlabel('Satisfaction Level')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig('visualizations/customer_satisfaction.png')
        plt.close()
    
    # 8. Purchase Amount by Income Level
    if 'Income_Level' in df.columns:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Income_Level', y='Purchase_Amount', data=df)
        plt.title('Purchase Amount by Income Level')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('visualizations/purchase_by_income.png')
        plt.close()
    
    # 9. Brand Loyalty vs. Discount Sensitivity
    if 'Brand_Loyalty' in df.columns and 'Discount_Sensitivity' in df.columns:
        plt.figure(figsize=(10, 8))
        df_agg = df.groupby(['Brand_Loyalty', 'Discount_Sensitivity'])['Purchase_Amount'].mean().reset_index()
        pivot_table = df_agg.pivot(index='Brand_Loyalty', columns='Discount_Sensitivity', values='Purchase_Amount')
        sns.heatmap(pivot_table, annot=True, cmap='YlGnBu', fmt='.2f')
        plt.title('Average Purchase Amount by Brand Loyalty and Discount Sensitivity')
        plt.tight_layout()
        plt.savefig('visualizations/loyalty_vs_discount.png')
        plt.close()
    
    print("Visualizations saved to 'visualizations' directory.")

def analyze_customer_segments(df):
    """Perform customer segmentation analysis."""
    if 'Age' in df.columns and 'Income_Level' in df.columns and 'Purchase_Amount' in df.columns:
        # Create a copy to avoid warnings
        segment_df = df.copy()
        
        # Convert categorical income to numeric if needed
        if segment_df['Income_Level'].dtype == 'object':
            # This assumes income levels are categories like 'Low', 'Medium', 'High'
            # Adjust this mapping according to your actual data
            income_map = {'Low': 1, 'Medium': 2, 'High': 3, 'Very High': 4}
            if all(level in income_map for level in segment_df['Income_Level'].unique()):
                segment_df['Income_Level_Numeric'] = segment_df['Income_Level'].map(income_map)
            else:
                # If income levels are not as expected, skip this part of the analysis
                print("Income level format not as expected. Skipping income-based segmentation.")
                return
        
        # Select features for segmentation
        features = ['Age', 'Income_Level_Numeric', 'Purchase_Amount']
        
        # Simple segmentation based on percentiles
        segment_df['Age_Segment'] = pd.qcut(segment_df['Age'], q=3, labels=['Young', 'Middle-aged', 'Senior'])
        segment_df['Spending_Segment'] = pd.qcut(segment_df['Purchase_Amount'], q=3, labels=['Low', 'Medium', 'High'])
        
        # Create a summary of customer segments
        segment_summary = segment_df.groupby(['Age_Segment', 'Spending_Segment']).agg({
            'Customer_ID': 'count',
            'Purchase_Amount': ['mean', 'median', 'std'],
            'Frequency_of_Purchase': 'mean' if 'Frequency_of_Purchase' in df.columns else 'count'
        }).reset_index()
        
        # Save the segmentation summary
        segment_summary.columns = ['Age_Segment', 'Spending_Segment', 'Customer_Count', 
                                  'Avg_Purchase', 'Median_Purchase', 'Std_Purchase', 
                                  'Avg_Purchase_Frequency']
        segment_summary.to_csv('customer_segments.csv', index=False, encoding='utf-8')
        
        # Visualize the segments
        plt.figure(figsize=(12, 8))
        for age_segment, color in zip(['Young', 'Middle-aged', 'Senior'], ['blue', 'green', 'red']):
            subset = segment_df[segment_df['Age_Segment'] == age_segment]
            plt.scatter(subset['Purchase_Amount'], subset['Income_Level_Numeric'], 
                        alpha=0.6, label=age_segment, color=color)
        
        plt.title('Customer Segments by Age, Income, and Purchase Amount')
        plt.xlabel('Purchase Amount')
        plt.ylabel('Income Level')
        plt.legend()
        plt.tight_layout()
        plt.savefig('visualizations/customer_segments.png')
        plt.close()
        
        print("Customer segmentation analysis saved to 'customer_segments.csv'")

def analyze_purchase_patterns(df):
    """Analyze purchase patterns over time and channels."""
    # Time-based analysis
    if 'Time_of_Purchase' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Time_of_Purchase']):
        # Extract time components
        df['Purchase_Hour'] = df['Time_of_Purchase'].dt.hour
        df['Purchase_Day'] = df['Time_of_Purchase'].dt.day_name()
        df['Purchase_Month'] = df['Time_of_Purchase'].dt.month_name()
        
        # Hourly purchase distribution
        plt.figure(figsize=(12, 6))
        hourly_purchases = df.groupby('Purchase_Hour')['Purchase_Amount'].mean()
        sns.lineplot(x=hourly_purchases.index, y=hourly_purchases.values)
        plt.title('Average Purchase Amount by Hour of Day')
        plt.xlabel('Hour of Day')
        plt.ylabel('Average Purchase Amount')
        plt.xticks(range(0, 24))
        plt.tight_layout()
        plt.savefig('visualizations/purchase_by_hour.png')
        plt.close()
        
        # Daily purchase distribution
        plt.figure(figsize=(12, 6))
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_purchases = df.groupby('Purchase_Day')['Purchase_Amount'].mean()
        daily_purchases = daily_purchases.reindex(day_order)
        sns.barplot(x=daily_purchases.index, y=daily_purchases.values)
        plt.title('Average Purchase Amount by Day of Week')
        plt.xlabel('Day of Week')
        plt.ylabel('Average Purchase Amount')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('visualizations/purchase_by_day.png')
        plt.close()
    
    # Channel analysis
    if 'Purchase_Channel' in df.columns and 'Purchase_Category' in df.columns:
        # Channel performance by category
        channel_category = df.groupby(['Purchase_Channel', 'Purchase_Category'])['Purchase_Amount'].agg(['mean', 'count'])
        channel_category.columns = ['Average_Purchase', 'Number_of_Purchases']
        channel_category.reset_index().to_csv('channel_category_analysis.csv', index=False, encoding='utf-8')
        
        # Visualize top channels by category
        top_categories = df['Purchase_Category'].value_counts().head(5).index
        channel_data = df[df['Purchase_Category'].isin(top_categories)]
        
        plt.figure(figsize=(14, 8))
        sns.countplot(x='Purchase_Category', hue='Purchase_Channel', data=channel_data)
        plt.title('Purchase Channels by Top 5 Categories')
        plt.xlabel('Purchase Category')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Purchase Channel')
        plt.tight_layout()
        plt.savefig('visualizations/channel_by_category.png')
        plt.close()
    
    print("Purchase pattern analysis completed and saved.")

def generate_summary_report(df):
    """Generate a summary report with key insights."""
    # Create a summary buffer
    summary = []
    summary.append("# Customer Purchase Data Analysis Summary Report")
    summary.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary.append(f"\n## Dataset Overview")
    summary.append(f"Total Records: {df.shape[0]}")
    summary.append(f"Total Features: {df.shape[1]}")
    
    # Purchase statistics
    summary.append("\n## Purchase Statistics")
    summary.append(f"Average Purchase Amount: ${df['Purchase_Amount'].mean():.2f}")
    summary.append(f"Median Purchase Amount: ${df['Purchase_Amount'].median():.2f}")
    summary.append(f"Minimum Purchase Amount: ${df['Purchase_Amount'].min():.2f}")
    summary.append(f"Maximum Purchase Amount: ${df['Purchase_Amount'].max():.2f}")
    
    # Top purchase categories
    if 'Purchase_Category' in df.columns:
        summary.append("\n## Top Purchase Categories")
        top_categories = df['Purchase_Category'].value_counts().head(5)
        for category, count in top_categories.items():
            summary.append(f"- {category}: {count} purchases")
    
    # Customer demographics
    summary.append("\n## Customer Demographics")
    if 'Age' in df.columns:
        summary.append(f"Average Customer Age: {df['Age'].mean():.1f} years")
    
    if 'Gender' in df.columns:
        gender_distribution = df['Gender'].value_counts(normalize=True) * 100
        for gender, percentage in gender_distribution.items():
            summary.append(f"- {gender}: {percentage:.1f}%")
    
    # Channel effectiveness
    if 'Purchase_Channel' in df.columns:
        summary.append("\n## Channel Effectiveness")
        channel_performance = df.groupby('Purchase_Channel')['Purchase_Amount'].agg(['mean', 'count'])
        channel_performance.columns = ['Average Purchase', 'Number of Purchases']
        channel_performance = channel_performance.sort_values('Average Purchase', ascending=False)
        
        for channel, data in channel_performance.head(3).iterrows():
            summary.append(f"- {channel}: ${data['Average Purchase']:.2f} average purchase, {data['Number of Purchases']} purchases")
    
    # Key correlations
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        correlation = df[numeric_cols].corr()
        # Get the top 5 highest absolute correlations (excluding self-correlations)
        corr_pairs = []
        for i in range(len(correlation.columns)):
            for j in range(i+1, len(correlation.columns)):
                corr_pairs.append((correlation.columns[i], correlation.columns[j], correlation.iloc[i, j]))
        
        corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        if corr_pairs:
            summary.append("\n## Key Correlations")
            for var1, var2, corr_value in corr_pairs[:5]:
                summary.append(f"- {var1} and {var2}: {corr_value:.3f}")
    
    # Save the summary report
    with open('analysis_summary_report.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary))
    
    print("Summary report saved to 'analysis_summary_report.md'")

def main():
    """Main function to run the EDA."""
    print("Starting Exploratory Data Analysis...")
    
    # Load the data
    df = load_data()
    if df is None:
        return
    
    # Perform the analyses
    generate_basic_stats(df)
    generate_visualizations(df)
    analyze_customer_segments(df)
    analyze_purchase_patterns(df)
    generate_summary_report(df)
    
    print("Exploratory Data Analysis completed successfully!")

if __name__ == "__main__":
    main()