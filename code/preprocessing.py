import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder, StandardScaler

def clean_dataset(df):
    df = df.drop(columns=['ID', 'Customer_ID', 'Name', 'SSN'], errors='ignore')

    num_cols = ['Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 'Num_Credit_Card', 
                'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date', 'Num_of_Delayed_Payment', 
                'Changed_Credit_Limit', 'Num_Credit_Inquiries', 'Outstanding_Debt', 
                'Credit_Utilization_Ratio', 'Total_EMI_per_month', 'Amount_invested_monthly', 
                'Monthly_Balance', 'Credit_History_Age'] # Credit_History_Age will be handled separately
    
    # Clean numeric columns specified, converting non-numeric placeholders to NaN
    for col in num_cols:
        if col in df.columns and col != 'Credit_History_Age': # Exclude Credit_History_Age from this specific regex cleaning
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^0-9.-]', '', regex=True), errors='coerce')

    if 'Age' in df.columns: # Clean Age: ensure it's a positive number
        df['Age'] = df['Age'].apply(lambda x: x if pd.notnull(x) and isinstance(x, (int, float)) and x > 0 else np.nan)


    def age_to_months(val): # Convert 'Credit_History_Age' (e.g., "X Years and Y Months") to total months
        if isinstance(val, str):
            m = re.search(r'(\d+)\s+Years?\s+and\s+(\d+)\s+Months?', val, re.IGNORECASE)
            if m:
                return int(m.group(1)) * 12 + int(m.group(2))
            # Handle cases like "_ Years and __ Months" or just numbers if they are already months
            m_years_only = re.search(r'(\d+)\s+Years?', val, re.IGNORECASE)
            if m_years_only and "months" not in val.lower(): # e.g. "7 Years"
                 return int(m_years_only.group(1)) * 12
            m_months_only = re.search(r'(\d+)\s+Months?', val, re.IGNORECASE)
            if m_months_only and "years" not in val.lower(): # e.g. "8 Months"
                 return int(m_months_only.group(1))
        elif isinstance(val, (int, float)) and pd.notnull(val): # If it's already a number
            return val
        return np.nan # Return NaN if format is not recognized or value is missing
    
    if 'Credit_History_Age' in df.columns:
        df['Credit_History_Age'] = df['Credit_History_Age'].apply(age_to_months)
        df['Credit_History_Age'] = pd.to_numeric(df['Credit_History_Age'], errors='coerce')


    # Impute categorical columns with mode
    cat_cols_to_impute = ['Occupation', 'Type_of_Loan', 'Credit_Mix', 
                          'Payment_of_Min_Amount', 'Payment_Behaviour', 'Month']
    for col in cat_cols_to_impute:
        if col in df.columns:
            # Handle potential non-string values or mixed types before mode
            df[col] = df[col].astype(str).replace(['nan', '_', '-', '!', '#NAME?'], np.nan) # Standardize NaNs
            mode_val = df[col].mode()
            if not mode_val.empty:
                df[col] = df[col].fillna(mode_val[0])
            else: # If all values were NaN for some reason
                df[col] = df[col].fillna('Unknown')


    # Impute numeric columns (that were processed or Credit_History_Age) with median
    numeric_cols_to_impute = [col for col in num_cols if col in df.columns] # Re-check existence
    # Ensure Credit_History_Age is treated as numeric for imputation if present
    if 'Credit_History_Age' not in numeric_cols_to_impute and 'Credit_History_Age' in df.columns:
        numeric_cols_to_impute.append('Credit_History_Age')

    for col in numeric_cols_to_impute:
        if df[col].isnull().any(): # Only impute if there are NaNs
            df[col] = df[col].fillna(df[col].median())
        
    return df

def encode_scale_data(train_df, test_df):
    cat_cols_to_encode = ['Occupation','Type_of_Loan','Credit_Mix',
                          'Payment_of_Min_Amount','Payment_Behaviour']
    encoders = {}
    
    for col in cat_cols_to_encode:
        if col in train_df.columns and col in test_df.columns:
            le = LabelEncoder()
            # Fit on training data and transform both train and test
            train_df[col] = le.fit_transform(train_df[col].astype(str)) # Ensure string type for LE
            test_df[col] = le.transform(test_df[col].astype(str))
            encoders[col] = le

    target_le = LabelEncoder()
    if 'Credit_Score' in train_df.columns:
        train_df['Credit_Score'] = target_le.fit_transform(train_df['Credit_Score'])
    # Note: target_le is not applied to test_df as it doesn't have 'Credit_Score'

    # Identify numeric columns for scaling (excluding the target if it was numeric before encoding)
    numeric_cols_train = train_df.select_dtypes(include=np.number).columns.tolist()
    if 'Credit_Score' in numeric_cols_train: # This 'Credit_Score' is the encoded version
        numeric_cols_train.remove('Credit_Score')
    
    # Scale only columns present in both train and test to avoid issues
    common_numeric_cols = [col for col in numeric_cols_train if col in test_df.columns and col in train_df.columns]

    scaler = StandardScaler()
    if common_numeric_cols:
        train_df[common_numeric_cols] = scaler.fit_transform(train_df[common_numeric_cols])
        test_df[common_numeric_cols] = scaler.transform(test_df[common_numeric_cols])
    else:
        print("Warning: No common numeric columns found for scaling between train and test sets.")
    
    print("Encoding and scaling completed.")
    return train_df, test_df, encoders, target_le, scaler, common_numeric_cols