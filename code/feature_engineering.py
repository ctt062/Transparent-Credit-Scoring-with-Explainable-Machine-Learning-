import pandas as pd

def engineer_features(df):
    month_map = {'January':1,'February':2,'March':3,'April':4,'May':5,'June':6,
                 'July':7,'August':8,'September':9,'October':10,'November':11,'December':12}
    
    if 'Month' in df.columns:
        df['Month'] = df['Month'].map(month_map)
    
    # Derived features - ensure columns exist before creating new ones
    if 'Outstanding_Debt' in df.columns and 'Annual_Income' in df.columns:
        df['Debt_to_Income'] = df['Outstanding_Debt'] / (df['Annual_Income'] + 1e-5)
    if 'Num_of_Loan' in df.columns and 'Num_Bank_Accounts' in df.columns:
        df['Loan_to_BankAccount_Ratio'] = df['Num_of_Loan'] / (df['Num_Bank_Accounts'] + 1e-5)
    if 'Total_EMI_per_month' in df.columns and 'Monthly_Inhand_Salary' in df.columns:
        df['EMI_to_Salary'] = df['Total_EMI_per_month'] / (df['Monthly_Inhand_Salary'] + 1e-5)
    if 'Num_Credit_Inquiries' in df.columns and 'Num_of_Loan' in df.columns:
        df['Credit_Inquiry_per_Loan'] = df['Num_Credit_Inquiries'] / (df['Num_of_Loan'] + 1e-5)
    
    print("Feature engineering applied.")
    return df