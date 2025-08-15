"""
Balance verification module for cashbook processing.
Handles verification and correction of opening and closing balances.
"""

import pandas as pd
import re
import os
from datetime import datetime

def identify_opening_balance(df, statement_month, statement_year, expected_opening_balance=None):
    """
    Identify the opening balance for a specific month in a fiscal period.
    
    Args:
        df: DataFrame with transaction data
        statement_month: Month number (1-12)
        statement_year: Year (e.g., 2023)
        expected_opening_balance: Expected opening balance if known
        
    Returns:
        tuple: (opening_balance, first_transaction_index)
    """
    # Ensure date column is datetime
    if 'date' not in df.columns:
        raise ValueError("DataFrame must have a 'date' column")
    
    # Filter to the specific month and year
    month_df = df[(df['date'].dt.month == statement_month) & (df['date'].dt.year == statement_year)]
    
    if month_df.empty:
        return None, None
    
    # Sort by date and then by original order in the statement
    if 'statement_order' in month_df.columns:
        month_df = month_df.sort_values(['date', 'statement_order'])
    else:
        month_df = month_df.sort_values('date')
    
    # Get the first transaction
    first_transaction = month_df.iloc[0]
    first_idx = first_transaction.name
    
    # Get the opening balance from the first transaction
    if 'Balance' in first_transaction:
        # Calculate the opening balance by adding debits and subtracting credits from the first transaction's balance
        opening_balance = first_transaction['Balance']
        if 'Debits' in first_transaction and first_transaction['Debits'] > 0:
            opening_balance += first_transaction['Debits']
        if 'Credits' in first_transaction and first_transaction['Credits'] > 0:
            opening_balance -= first_transaction['Credits']
    else:
        # If no balance column, use the expected opening balance
        opening_balance = expected_opening_balance
    
    # If expected opening balance is provided, use it for verification
    if expected_opening_balance is not None:
        if opening_balance is not None:
            difference = abs(opening_balance - expected_opening_balance)
            if difference > 0.01:  # Allow for small rounding differences
                print(f"Warning: Opening balance discrepancy for {statement_month}/{statement_year}")
                print(f"  Calculated: {opening_balance}")
                print(f"  Expected: {expected_opening_balance}")
                print(f"  Difference: {difference}")
                
                # Use the expected opening balance
                opening_balance = expected_opening_balance
    
    return opening_balance, first_idx

def recalculate_balances(df, opening_balance):
    """
    Recalculate all balances based on an opening balance and transaction amounts.
    
    Args:
        df: DataFrame with transaction data
        opening_balance: Opening balance to start with
        
    Returns:
        DataFrame: DataFrame with recalculated balances
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Ensure numeric columns are properly formatted
    for col in ['Debits', 'Credits', 'Balance']:
        if col in df.columns:
            # Convert to numeric, coercing errors to NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Replace NaN with 0
            df[col] = df[col].fillna(0)
    
    # Sort by date
    if 'date' in df.columns:
        if 'statement_order' in df.columns:
            df = df.sort_values(['date', 'statement_order'])
        else:
            df = df.sort_values('date')
    
    # Set the opening balance for the first transaction
    if len(df) > 0:
        df.iloc[0, df.columns.get_loc('Balance')] = opening_balance
        
        # Recalculate all subsequent balances
        current_balance = opening_balance
        
        for i in range(len(df)):
            # Get debits and credits
            debit = df.iloc[i]['Debits'] if 'Debits' in df.columns else 0
            credit = df.iloc[i]['Credits'] if 'Credits' in df.columns else 0
            
            # Handle NaN values
            debit = 0 if pd.isna(debit) else debit
            credit = 0 if pd.isna(credit) else credit
            
            # Apply transaction to balance
            if i > 0:  # Skip first row as we already set its balance
                current_balance = current_balance - debit + credit
                df.iloc[i, df.columns.get_loc('Balance')] = current_balance
    
    return df

def verify_closing_balance(df, expected_closing_balance=None):
    """
    Verify the closing balance of a DataFrame.
    
    Args:
        df: DataFrame with transaction data
        expected_closing_balance: Expected closing balance if known
        
    Returns:
        tuple: (closing_balance, difference)
    """
    if len(df) == 0:
        return None, None
    
    # Get the closing balance from the last transaction
    closing_balance = df.iloc[-1]['Balance']
    
    # If expected closing balance is provided, verify it
    if expected_closing_balance is not None:
        difference = abs(closing_balance - expected_closing_balance)
        if difference > 0.01:  # Allow for small rounding differences
            print(f"Warning: Closing balance discrepancy")
            print(f"  Calculated: {closing_balance}")
            print(f"  Expected: {expected_closing_balance}")
            print(f"  Difference: {difference}")
            return closing_balance, difference
    
    return closing_balance, 0
