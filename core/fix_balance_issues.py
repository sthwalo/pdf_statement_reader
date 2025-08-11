#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Balance Issue Fixer

This script analyzes and fixes balance calculation issues in extracted bank statement CSVs.
It addresses the following problems:
1. Missing balance values
2. Balance calculation mismatches
3. Transaction ordering issues
4. Statement boundary reconciliation
"""

import os
import sys
import pandas as pd
import numpy as np
import json
import logging
import argparse
from datetime import datetime

# Add parent directory to path to find modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def load_csv(csv_path):
    """Load CSV file into pandas DataFrame"""
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} transactions from {csv_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading {csv_path}: {e}")
        return None

def preprocess_dataframe(df):
    """Preprocess DataFrame for analysis and fixing"""
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Convert date strings to datetime objects
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # Convert numeric columns
    for col in ['Debits', 'Credits', 'Balance']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Sort by date and source (if available)
    sort_cols = ['Date']
    if 'Source' in df.columns:
        sort_cols.append('Source')
    
    df = df.sort_values(by=sort_cols).reset_index(drop=True)
    
    return df

def identify_statement_boundaries(df):
    """Identify statement boundaries in the transactions"""
    boundaries = []
    
    # Look for "BALANCE BROUGHT FORWARD" entries
    if 'Details' in df.columns:
        for idx, row in df.iterrows():
            details = str(row.get('Details', '')).upper()
            if 'BALANCE BROUGHT FORWARD' in details:
                boundaries.append(idx)
    
    logger.info(f"Identified {len(boundaries)} statement boundaries")
    return boundaries

def split_by_statement(df, boundaries):
    """Split transactions into separate statements"""
    statements = []
    
    start_idx = 0
    for boundary in boundaries:
        if boundary > start_idx:
            statements.append(df.iloc[start_idx:boundary].copy())
            start_idx = boundary
    
    # Add the last statement
    if start_idx < len(df):
        statements.append(df.iloc[start_idx:].copy())
    
    logger.info(f"Split transactions into {len(statements)} statements")
    return statements

def fix_balance_within_statement(statement_df):
    """Fix balance calculations within a single statement"""
    df = statement_df.copy()
    
    # Skip empty dataframes
    if len(df) == 0:
        return df
    
    # Reset index to avoid KeyError with loc
    df = df.reset_index(drop=True)
    
    # Find the first valid balance to use as starting point
    first_valid_idx = None
    for idx in df.index:
        if not pd.isna(df.at[idx, 'Balance']):
            first_valid_idx = idx
            break
    
    if first_valid_idx is None:
        logger.warning("No valid balance found in statement")
        return df
    
    # Start with the first valid balance
    starting_balance = df.at[first_valid_idx, 'Balance']
    
    # Create a new calculated balance column
    df['Calculated_Balance'] = np.nan
    df.at[first_valid_idx, 'Calculated_Balance'] = starting_balance
    
    # Calculate running balance forward from the first valid balance
    for i in range(first_valid_idx + 1, len(df)):
        prev_balance = df.at[i-1, 'Calculated_Balance']
        debit = df.at[i, 'Debits'] if not pd.isna(df.at[i, 'Debits']) else 0
        credit = df.at[i, 'Credits'] if not pd.isna(df.at[i, 'Credits']) else 0
        
        # Calculate new balance (debits are negative, credits are positive)
        df.at[i, 'Calculated_Balance'] = prev_balance + credit + debit
    
    # Calculate running balance backward from the first valid balance
    for i in range(first_valid_idx - 1, -1, -1):
        next_balance = df.at[i+1, 'Calculated_Balance']
        debit = df.at[i, 'Debits'] if not pd.isna(df.at[i, 'Debits']) else 0
        credit = df.at[i, 'Credits'] if not pd.isna(df.at[i, 'Credits']) else 0
        
        # Calculate previous balance (reverse the calculation)
        df.at[i, 'Calculated_Balance'] = next_balance - credit - debit
    
    # Calculate discrepancy between original and calculated balance
    df['Balance_Discrepancy'] = df['Balance'] - df['Calculated_Balance']
    
    # Replace missing or highly discrepant balance values with calculated ones
    balance_threshold = 1.0  # Threshold for acceptable discrepancy
    
    df['Balance_Fixed'] = False
    for i in df.index:
        if pd.isna(df.at[i, 'Balance']) or abs(df.at[i, 'Balance_Discrepancy']) > balance_threshold:
            df.at[i, 'Balance'] = df.at[i, 'Calculated_Balance']
            df.at[i, 'Balance_Fixed'] = True
    
    # Count fixed balances
    fixed_count = df['Balance_Fixed'].sum()
    logger.info(f"Fixed {fixed_count} balance values in statement")
    
    return df

def fix_statement_transitions(statements):
    """Fix balance transitions between statements"""
    if len(statements) <= 1:
        return statements
    
    fixed_statements = []
    fixed_statements.append(statements[0])
    
    for i in range(1, len(statements)):
        current_stmt = statements[i].copy()
        prev_stmt = fixed_statements[i-1]
        
        # Get the last balance from previous statement
        if len(prev_stmt) > 0:
            last_balance = prev_stmt['Balance'].iloc[-1]
            
            # If the current statement has a "BALANCE BROUGHT FORWARD" entry at the start,
            # update its balance to match the last balance from previous statement
            if len(current_stmt) > 0:
                first_row = current_stmt.iloc[0]
                details = str(first_row.get('Details', '')).upper()
                
                if 'BALANCE BROUGHT FORWARD' in details:
                    idx = current_stmt.index[0]
                    current_stmt.at[idx, 'Balance'] = last_balance
                    logger.info(f"Fixed statement transition balance: {last_balance}")
        
        fixed_statements.append(current_stmt)
    
    return fixed_statements

def recalculate_all_balances(df):
    """Recalculate all balances based on debits and credits"""
    # Reset index to avoid KeyError with loc
    df = df.reset_index(drop=True)
    
    # Find the first valid balance to use as starting point
    first_valid_idx = None
    for idx in df.index:
        if not pd.isna(df.at[idx, 'Balance']):
            first_valid_idx = idx
            break
    
    if first_valid_idx is None:
        logger.warning("No valid balance found in dataframe")
        return df
    
    # Start with the first valid balance
    starting_balance = df.at[first_valid_idx, 'Balance']
    
    # Create a new calculated balance column
    df['Recalculated_Balance'] = np.nan
    df.at[first_valid_idx, 'Recalculated_Balance'] = starting_balance
    
    # Calculate running balance forward
    for i in range(first_valid_idx + 1, len(df)):
        prev_balance = df.at[i-1, 'Recalculated_Balance']
        debit = df.at[i, 'Debits'] if not pd.isna(df.at[i, 'Debits']) else 0
        credit = df.at[i, 'Credits'] if not pd.isna(df.at[i, 'Credits']) else 0
        
        # Calculate new balance
        df.at[i, 'Recalculated_Balance'] = prev_balance + credit + debit
    
    # Calculate running balance backward
    for i in range(first_valid_idx - 1, -1, -1):
        next_balance = df.at[i+1, 'Recalculated_Balance']
        debit = df.at[i, 'Debits'] if not pd.isna(df.at[i, 'Debits']) else 0
        credit = df.at[i, 'Credits'] if not pd.isna(df.at[i, 'Credits']) else 0
        
        # Calculate previous balance
        df.at[i, 'Recalculated_Balance'] = next_balance - credit - debit
    
    return df

def fix_balance_issues(csv_path, output_path=None, debug=False):
    """Main function to fix balance issues in a CSV file"""
    # Load CSV
    df = load_csv(csv_path)
    if df is None:
        return False
    
    # Preprocess dataframe
    df = preprocess_dataframe(df)
    
    # Identify statement boundaries
    boundaries = identify_statement_boundaries(df)
    
    # Split into statements
    statements = split_by_statement(df, boundaries)
    
    # Fix balances within each statement
    fixed_statements = []
    for i, stmt in enumerate(statements):
        logger.info(f"Fixing statement {i+1}/{len(statements)}")
        fixed_stmt = fix_balance_within_statement(stmt)
        fixed_statements.append(fixed_stmt)
    
    # Fix transitions between statements
    fixed_statements = fix_statement_transitions(fixed_statements)
    
    # Combine fixed statements
    fixed_df = pd.concat(fixed_statements, ignore_index=True)
    
    # Recalculate all balances for consistency
    fixed_df = recalculate_all_balances(fixed_df)
    
    # Generate statistics
    stats = {
        'original_rows': len(df),
        'fixed_rows': len(fixed_df),
        'missing_balance_before': df['Balance'].isna().sum(),
        'missing_balance_after': fixed_df['Balance'].isna().sum(),
        'fixed_balance_count': fixed_df['Balance_Fixed'].sum() if 'Balance_Fixed' in fixed_df.columns else 0,
    }
    
    logger.info(f"Fixed {stats['fixed_balance_count']} balance values")
    logger.info(f"Reduced missing balances from {stats['missing_balance_before']} to {stats['missing_balance_after']}")
    
    # Save fixed dataframe
    if output_path is None:
        base_name = os.path.splitext(csv_path)[0]
        output_path = f"{base_name}_fixed.csv"
    
    # Remove debug columns if not in debug mode
    if not debug:
        cols_to_drop = ['Calculated_Balance', 'Balance_Discrepancy', 'Balance_Fixed', 'Recalculated_Balance']
        for col in cols_to_drop:
            if col in fixed_df.columns:
                fixed_df = fixed_df.drop(columns=[col])
    
    fixed_df.to_csv(output_path, index=False)
    logger.info(f"Saved fixed CSV to {output_path}")
    
    # Save debug info if requested
    if debug:
        debug_path = f"{os.path.splitext(output_path)[0]}_debug.json"
        with open(debug_path, 'w') as f:
            json.dump({
                'stats': stats,
                'boundaries': boundaries,
                'statement_counts': [len(stmt) for stmt in statements]
            }, f, indent=2, default=str)
        logger.info(f"Saved debug info to {debug_path}")
    
    return True

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Fix balance issues in bank statement CSV files')
    parser.add_argument('csv_path', help='Path to CSV file to fix')
    parser.add_argument('--output', '-o', help='Output path for fixed CSV')
    parser.add_argument('--debug', '-d', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    fix_balance_issues(args.csv_path, args.output, args.debug)

if __name__ == "__main__":
    main()
