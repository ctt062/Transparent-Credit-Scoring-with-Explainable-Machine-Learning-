import pandas as pd
import numpy as np

def engineer_features(df):
    month_map = {'January':1,'February':2,'March':3,'April':4,'May':5,'June':6,
                 'July':7,'August':8,'September':9,'October':10,'November':11,'December':12}
    
    if 'Month' in df.columns:
        # Ensure 'Month' column is string type before mapping
        df['Month'] = df['Month'].astype(str).str.strip().map(month_map)
        # If any month was not in map (e.g. already numeric or bad format), it becomes NaN
        # Fill these NaNs, e.g., with mode or median if appropriate, or handle earlier.
        # For now, let's assume valid month strings or it was handled in cleaning.
        if df['Month'].isnull().any():
            # print(f"Warning: NaNs found in 'Month' column after mapping. Imputing with mode.")
            month_mode = df['Month'].mode()
            if not month_mode.empty:
                 df['Month'] = df['Month'].fillna(month_mode[0])
            else: # All were NaN
                 df['Month'] = df['Month'].fillna(1) # Default to January


    # Derived features - ensure columns exist and are numeric
    # Ensure denominators are not zero by adding a small epsilon
    if 'Outstanding_Debt' in df.columns and 'Annual_Income' in df.columns:
        df['Debt_to_Income'] = df['Outstanding_Debt'].astype(float) / (df['Annual_Income'].astype(float) + 1e-6)
    
    if 'Num_of_Loan' in df.columns and 'Num_Bank_Accounts' in df.columns:
        # Ensure Num_Bank_Accounts is not zero for the ratio
        df['Loan_to_BankAccount_Ratio'] = df['Num_of_Loan'].astype(float) / (df['Num_Bank_Accounts'].astype(float) + 1e-6)
    
    if 'Total_EMI_per_month' in df.columns and 'Monthly_Inhand_Salary' in df.columns:
        df['EMI_to_Salary'] = df['Total_EMI_per_month'].astype(float) / (df['Monthly_Inhand_Salary'].astype(float) + 1e-6)

    if 'Num_Credit_Inquiries' in df.columns and 'Num_of_Loan' in df.columns:
        # Assuming Num_of_Loan can be 0, add epsilon
        df['Credit_Inquiry_per_Loan'] = df['Num_Credit_Inquiries'].astype(float) / (df['Num_of_Loan'].astype(float) + 1e-6)
    
    # Replace any potential inf values that might arise from division by near-zero (if epsilon was too small or numbers are huge)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Impute NaNs created in derived features (e.g., with median or 0)
    # Example: Impute with 0 if that makes sense for ratios, or median
    for col in ['Debt_to_Income', 'Loan_to_BankAccount_Ratio', 'EMI_to_Salary', 'Credit_Inquiry_per_Loan']:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median()) # Or fillna(0)

    print("Feature engineering applied.")
    return df