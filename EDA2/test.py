import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from unittest.mock import patch, MagicMock
import io
import sys

# Import functions from main.py
from main import (
    load_data, 
    explore_basic_stats, 
    analyze_demographics, 
    analyze_purchase_behavior,
    analyze_customer_satisfaction,
    analyze_marketing_influence,
    analyze_correlations,
    segmentation_analysis,
    main
)

@pytest.fixture
def sample_df():
    """Create a sample DataFrame that resembles your expected data structure"""
    np.random.seed(42)  # For reproducibility
    
    # Generate sample data
    n_samples = 100
    
    data = {
        'Age': np.random.randint(18, 70, n_samples),
        'Gender': np.random.choice(['Male', 'Female', 'Other'], n_samples),
        'Income_Level': np.random.choice(['Low', 'Medium', 'High'], n_samples),
        'Marital_Status': np.random.choice(['Single', 'Married', 'Divorced'], n_samples),
        'Education_Level': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
        'Location': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Miami', 'Seattle'], n_samples),
        'Occupation': np.random.choice(['Engineer', 'Doctor', 'Teacher', 'Student', 'Manager'], n_samples),
        'Purchase_Amount': np.random.uniform(10, 1000, n_samples),
        'Purchase_Category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books', 'Sports'], n_samples),
        'Purchase_Channel': np.random.choice(['Online', 'In-store', 'Mobile App'], n_samples),
        'Frequency_of_Purchase': np.random.randint(1, 20, n_samples),
        'Payment_Method': np.random.choice(['Credit Card', 'Cash', 'PayPal', 'Bank Transfer'], n_samples),
        'Time_of_Purchase': np.random.choice(['Morning', 'Afternoon', 'Evening'], n_samples),
        'Time_to_Decision': np.random.randint(1, 60, n_samples),
        'Customer_Satisfaction': np.random.randint(1, 6, n_samples),
        'Product_Rating': np.random.randint(1, 6, n_samples),
        'Return_Rate': np.random.randint(1, 6, n_samples),
        'Brand_Loyalty': np.random.randint(1, 6, n_samples),
        'Customer_Loyalty_Program_Member': np.random.choice(['Yes', 'No'], n_samples),
        'Social_Media_Influence': np.random.randint(1, 6, n_samples),
        'Engagement_with_Ads': np.random.randint(1, 6, n_samples),
        'Discount_Sensitivity': np.random.randint(1, 6, n_samples),
        'Discount_Used': np.random.choice(['Yes', 'No'], n_samples)
    }
    
    return pd.DataFrame(data)

@pytest.fixture
def mock_csv_file(sample_df, tmp_path):
    """Create a temporary CSV file with sample data"""
    csv_path = tmp_path / "sample_data.csv"
    sample_df.to_csv(csv_path, index=False)
    return csv_path

def test_load_data(mock_csv_file, monkeypatch):
    """Test the load_data function"""
    # Mock os.listdir to return our test CSV file
    def mock_listdir(_):
        return [mock_csv_file.name]
    
    monkeypatch.setattr(os, 'listdir', mock_listdir)
    
    # Mock current directory to be the temp directory
    monkeypatch.chdir(mock_csv_file.parent)
    
    # Capture output
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    # Test the function
    df = load_data()
    
    # Reset stdout
    sys.stdout = sys.__stdout__
    
    # Assertions
    assert df is not None
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] > 0
    assert f"Loading data from: {mock_csv_file.name}" in captured_output.getvalue()
    assert "Data loaded successfully" in captured_output.getvalue()

def test_load_data_no_csv(monkeypatch):
    """Test load_data function when no CSV files are present"""
    # Mock os.listdir to return no CSV files
    monkeypatch.setattr(os, 'listdir', lambda _: ["file.txt", "image.png"])
    
    # Capture output
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    # Test the function
    df = load_data()
    
    # Reset stdout
    sys.stdout = sys.__stdout__
    
    # Assertions
    assert df is None
    assert "No CSV files found in the current directory" in captured_output.getvalue()

def test_explore_basic_stats(sample_df):
    """Test the explore_basic_stats function"""
    # Capture output
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    # Test the function
    explore_basic_stats(sample_df)
    
    # Reset stdout
    sys.stdout = sys.__stdout__
    
    # Assertions
    assert "BASIC DATASET INFORMATION" in captured_output.getvalue()
    assert "First 5 rows" in captured_output.getvalue()
    assert "Basic information" in captured_output.getvalue()
    assert "Summary statistics" in captured_output.getvalue()
    assert "Missing values" in captured_output.getvalue()
    assert "Number of duplicate rows" in captured_output.getvalue()

@patch('matplotlib.pyplot.show')
def test_analyze_demographics(mock_show, sample_df):
    """Test the analyze_demographics function"""
    # Test the function
    analyze_demographics(sample_df)
    
    # Assertions
    assert mock_show.call_count == 2

@patch('matplotlib.pyplot.show')
def test_analyze_purchase_behavior(mock_show, sample_df):
    """Test the analyze_purchase_behavior function"""
    # Test the function
    analyze_purchase_behavior(sample_df)
    
    # Assertions
    assert mock_show.call_count == 2

@patch('matplotlib.pyplot.show')
def test_analyze_customer_satisfaction(mock_show, sample_df):
    """Test the analyze_customer_satisfaction function"""
    # Test the function
    analyze_customer_satisfaction(sample_df)
    
    # Assertions
    assert mock_show.call_count == 2

@patch('matplotlib.pyplot.show')
def test_analyze_marketing_influence(mock_show, sample_df):
    """Test the analyze_marketing_influence function"""
    # Test the function
    analyze_marketing_influence(sample_df)
    
    # Assertions
    assert mock_show.call_count == 1

@patch('matplotlib.pyplot.show')
def test_analyze_correlations(mock_show, sample_df):
    """Test the analyze_correlations function"""
    # Capture output
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    # Test the function
    analyze_correlations(sample_df)
    
    # Reset stdout
    sys.stdout = sys.__stdout__
    
    # Assertions
    assert mock_show.call_count == 1
    assert "Top 5 Positive Correlations" in captured_output.getvalue()
    assert "Top 5 Negative Correlations" in captured_output.getvalue()

@patch('matplotlib.pyplot.show')
def test_segmentation_analysis(mock_show, sample_df):
    """Test the segmentation_analysis function"""
    # Test the function
    segmentation_analysis(sample_df)
    
    # Assertions
    assert mock_show.call_count == 2
    assert 'Frequency_Segment' in sample_df.columns
    assert 'Amount_Segment' in sample_df.columns
    assert 'Customer_Segment' in sample_df.columns

@patch('main.load_data')
@patch('main.explore_basic_stats')
@patch('main.analyze_demographics')
@patch('main.analyze_purchase_behavior')
@patch('main.analyze_customer_satisfaction')
@patch('main.analyze_marketing_influence')
@patch('main.analyze_correlations')
@patch('main.segmentation_analysis')
def test_main_function(mock_segmentation, mock_correlations, mock_marketing, 
                      mock_satisfaction, mock_purchase, mock_demographics, 
                      mock_basic_stats, mock_load_data, sample_df):
    """Test the main function"""
    # Setup mock to return sample DataFrame
    mock_load_data.return_value = sample_df
    
    # Capture output
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    # Test the function
    main()
    
    # Reset stdout
    sys.stdout = sys.__stdout__
    
    # Assertions
    mock_load_data.assert_called_once()
    mock_basic_stats.assert_called_once_with(sample_df)
    mock_demographics.assert_called_once_with(sample_df)
    mock_purchase.assert_called_once_with(sample_df)
    mock_satisfaction.assert_called_once_with(sample_df)
    mock_marketing.assert_called_once_with(sample_df)
    mock_correlations.assert_called_once_with(sample_df)
    mock_segmentation.assert_called_once_with(sample_df)
    assert "Exploratory data analysis complete" in captured_output.getvalue()

@patch('main.load_data')
def test_main_function_no_data(mock_load_data):
    """Test the main function when no data is loaded"""
    # Setup mock to return None (no data loaded)
    mock_load_data.return_value = None
    
    # Test the function
    main()
    
    # Assertions
    mock_load_data.assert_called_once()