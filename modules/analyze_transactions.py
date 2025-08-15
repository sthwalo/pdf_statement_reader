#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Transaction Analysis Module

Provides functions to analyze CSV transaction files and generate reports
on transaction patterns, date distributions, and balance trends.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import glob
import re
from collections import Counter, defaultdict

def clean_numeric(value):
    """Convert string numeric values to float, handling various formats"""
    if pd.isna(value) or value == '':
        return 0.0
    
    if isinstance(value, (int, float)):
        return float(value)
    
    # Remove currency symbols, commas, and other non-numeric characters
    if isinstance(value, str):
        # Keep negative sign and decimal point
        value = re.sub(r'[^\d.-]', '', value)
        
        try:
            return float(value)
        except ValueError:
            return 0.0
    
    return 0.0

def parse_date(date_str, default_month=None, default_year=None):
    """Parse date strings with various formats, inferring year if needed"""
    if pd.isna(date_str) or date_str == '':
        return None
    
    if isinstance(date_str, datetime):
        return date_str
    
    # Try various date formats
    formats = [
        '%d/%m/%Y', '%d-%m-%Y', '%d.%m.%Y',  # Full dates
        '%d/%m/%y', '%d-%m-%y', '%d.%m.%y',  # Short year dates
        '%d/%m', '%d-%m', '%d.%m'            # Day/month only
    ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            
            # If the format doesn't include year, use the default or current year
            if '%y' not in fmt and '%Y' not in fmt:
                if default_year is not None:
                    dt = dt.replace(year=default_year)
                else:
                    dt = dt.replace(year=datetime.now().year)
            
            # If the format doesn't include month, use the default month
            if '%m' not in fmt and default_month is not None:
                dt = dt.replace(month=default_month)
                
            return dt
        except ValueError:
            continue
    
    # If all parsing attempts fail
    return None

def analyze_csv_file(file_path):
    """Analyze a single CSV transaction file"""
    print(f"\n{'='*80}")
    print(f"Analyzing file: {os.path.basename(file_path)}")
    print(f"{'='*80}")
    
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Basic file info
        print(f"Total rows: {len(df)}")
        print(f"Columns: {', '.join(df.columns)}")
        
        # Check for date column
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        if date_columns:
            # Convert date column to datetime
            date_col = date_columns[0]
            
            # Try to extract statement month/year from filename
            filename = os.path.basename(file_path)
            month_match = re.search(r'\((\d{2})\)', filename)
            statement_month = None
            statement_year = None
            
            if month_match:
                statement_month = int(month_match.group(1))
                statement_year = 2023 if statement_month >= 3 else 2024
            
            # Parse dates
            df['parsed_date'] = df[date_col].apply(
                lambda x: parse_date(x, statement_month, statement_year)
            )
            
            # Analyze date distribution
            valid_dates = df['parsed_date'].dropna()
            if not valid_dates.empty:
                print(f"\nDate range: {valid_dates.min().date()} to {valid_dates.max().date()}")
                
                # Group by month
                df['month'] = valid_dates.dt.strftime('%Y-%m')
                month_counts = df['month'].value_counts().sort_index()
                
                print("\nTransactions by month:")
                for month, count in month_counts.items():
                    print(f"  {month}: {count} transactions")
        
        # Analyze numeric columns
        for col in df.columns:
            if any(term in col.lower() for term in ['debit', 'credit', 'amount', 'balance']):
                # Convert to numeric
                df[f'{col}_numeric'] = df[col].apply(clean_numeric)
                
                # Calculate statistics
                values = df[f'{col}_numeric'].dropna()
                if not values.empty:
                    print(f"\n{col} statistics:")
                    print(f"  Count: {len(values)}")
                    print(f"  Sum: {values.sum():.2f}")
                    print(f"  Mean: {values.mean():.2f}")
                    print(f"  Min: {values.min():.2f}")
                    print(f"  Max: {values.max():.2f}")
        
        # Analyze description patterns if available
        desc_columns = [col for col in df.columns if any(term in col.lower() for term in ['desc', 'narration', 'particular'])]
        if desc_columns:
            desc_col = desc_columns[0]
            
            # Extract common words/patterns
            all_words = []
            for desc in df[desc_col].dropna():
                words = re.findall(r'\b[A-Za-z]{3,}\b', str(desc))
                all_words.extend([w.upper() for w in words])
            
            word_counts = Counter(all_words).most_common(10)
            
            print("\nCommon terms in descriptions:")
            for word, count in word_counts:
                print(f"  {word}: {count} occurrences")
        
        print("\nAnalysis complete.")
        
    except Exception as e:
        print(f"Error analyzing file: {e}")
        import traceback
        traceback.print_exc()

def analyze_csv_directory(directory):
    """Analyze all CSV files in a directory"""
    print(f"{'='*80}")
    print(f"TRANSACTION ANALYSIS REPORT")
    print(f"Directory: {directory}")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    
    # Find all CSV files
    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {directory}")
        return
    
    print(f"Found {len(csv_files)} CSV files")
    
    # Analyze each file
    for file_path in sorted(csv_files):
        analyze_csv_file(file_path)
    
    # If there's a combined file, analyze it separately
    combined_file = os.path.join(directory, "combined_transactions.csv")
    if os.path.exists(combined_file):
        print(f"\n{'='*80}")
        print(f"COMBINED TRANSACTIONS ANALYSIS")
        print(f"{'='*80}")
        analyze_csv_file(combined_file)
    
    print(f"\n{'='*80}")
    print(f"END OF ANALYSIS REPORT")
    print(f"{'='*80}")

if __name__ == "__main__":
    # If run directly, analyze the directory provided as argument
    if len(sys.argv) > 1:
        directory = sys.argv[1]
        analyze_csv_directory(directory)
    else:
        print("Please provide a directory path containing CSV files")
