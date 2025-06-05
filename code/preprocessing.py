import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder, StandardScaler

def clean_dataset(df):
    df = df.drop(columns=['ID', 'Customer_ID', 'Name', 'SSN'], errors='ignore')

    num_cols = ['Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_of_Loan',
                'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 'Num_Credit_Inquiries',
                'Outstanding_Debt', 'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance']
    
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].replace(r'[^0-9.-]', '', regex=True), errors='coerce')

    if 'Age' in df.columns:
        df['Age'] = df['Age'].apply(lambda x: x if pd.notnull(x) and x > 0 else np.nan)

    def age_to_months(val):
        if isinstance(val, str):
            m = re.search(r'(\d+)\s+Years?\s+and\s+(\d+)\s+Months?', val)
            if m:
                return int(m.group(1))*12 + int(m.group(2))
        return np.nan
    
    if 'Credit_History_Age' in df.columns:
        df['Credit_History_Age'] = df['Credit_History_Age'].apply(age_to_months)

    cat_cols_to_impute = ['Occupation', 'Type_of_Loan', 'Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour', 'Month']
    for col in cat_cols_to_impute:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    numeric_cols_to_impute = [col for col in num_cols if col in df.columns]
    if 'Credit_History_Age' in df.columns:
        numeric_cols_to_impute.append('Credit_History_Age')

    for col in numeric_cols_to_impute:
        df[col] = df[col].fillna(df[col].median())
        
    return df

def encode_scale_data(train_df, test_df):
    cat_cols_to_encode = ['Occupation','Type_of_Loan','Credit_Mix','Payment_of_Min_Amount','Payment_Behaviour']
    encoders = {}
    
    for col in cat_cols_to_encode:
        if col in train_df.columns and col in test_df.columns:
            le = LabelEncoder()
            train_df[col] = le.fit_transform(train_df[col])
            test_df[col] = le.transform(test_df[col])
            encoders[col] = le

    target_le = LabelEncoder()
    if 'Credit_Score' in train_df.columns:
        train_df['Credit_Score'] = target_le.fit_transform(train_df['Credit_Score'])

    scale_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Credit_Score' in scale_cols: # Ensure target is not scaled
        scale_cols.remove('Credit_Score')
    
    # Ensure only columns present in both train and test are scaled (or handle missing ones)
    valid_scale_cols_train = [col for col in scale_cols if col in train_df.columns]
    valid_scale_cols_test = [col for col in scale_cols if col in test_df.columns]
    
    # Only scale columns present in both, or ensure alignment if some are missing in test
    # For simplicity, using intersection of columns that are candidates for scaling
    common_scale_cols = list(set(valid_scale_cols_train) & set(valid_scale_cols_test))


    scaler = StandardScaler()
    if common_scale_cols: # only scale if there are common numeric columns
        train_df[common_scale_cols] = scaler.fit_transform(train_df[common_scale_cols])
        test_df[common_scale_cols] = scaler.transform(test_df[common_scale_cols])
    
    print("Encoding and scaling completed.")
    return train_df, test_df, encoders, target_le, scaler, common_scale_cols