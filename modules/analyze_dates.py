#!/usr/bin/env python3
import os
import pandas as pd
import re
from datetime import datetime

def analyze_dates():
    """
    Analyze date formats in the combined CSV file and test date parsing strategies.
    """
    input_dir = os.path.join("data", "output", "camelot")
    combined_file = os.path.join(input_dir, "combined_transactions.csv")
    
    if not os.path.exists(combined_file):
        print(f"Combined CSV file not found: {combined_file}")
        return
    
    print(f"Analyzing dates in {combined_file}...")
    
    # Read the combined CSV
    df = pd.read_csv(combined_file)
    
    # Basic statistics
    total_rows = len(df)
    unique_dates = df['Date'].nunique()
    print(f"Total rows: {total_rows}")
    print(f"Unique date values: {unique_dates}")
    
    # Sample of date formats
    print("\nSample date formats:")
    for i, date in enumerate(sorted(df['Date'].unique())[:20]):
        print(f"{i+1}. '{date}'")
    
    # Analyze date patterns
    date_patterns = {}
    for date in df['Date'].unique():
        if not isinstance(date, str):
            pattern = "non-string"
        elif re.match(r'^\d{1,2}/\d{1,2}$', date):
            pattern = "DD/MM"
        elif re.match(r'^\d{1,2}/\d{1,2}/\d{4}$', date):
            pattern = "DD/MM/YYYY"
        elif re.match(r'^\d{4}-\d{2}-\d{2}$', date):
            pattern = "YYYY-MM-DD"
        elif re.match(r'^\d{1,2}-\d{1,2}-\d{4}$', date):
            pattern = "DD-MM-YYYY"
        elif re.match(r'^\d{1,2}$', date):
            pattern = "DD"
        else:
            pattern = "other"
        
        date_patterns[pattern] = date_patterns.get(pattern, 0) + 1
    
    print("\nDate patterns found:")
    for pattern, count in date_patterns.items():
        print(f"{pattern}: {count} ({count/unique_dates*100:.1f}%)")
    
    # Test date parsing with different strategies
    print("\nTesting date parsing strategies...")
    
    # Strategy 1: Parse with DD/MM format and infer year from source file
    def parse_date_strategy1(row):
        try:
            date_str = row['Date']
            source_file = row['source_file']
            
            if not isinstance(date_str, str):
                return None
                
            # Extract month from source file
            month_match = re.search(r'\((\d{2})\)', source_file)
            if not month_match:
                return None
                
            statement_month = int(month_match.group(1))
            statement_year = 2024 if statement_month >= 3 else 2025
            
            # Parse DD/MM format
            if '/' in date_str:
                day, month = date_str.split('/')
                return f"{int(day):02d}/{int(month):02d}/{statement_year}"
            
            # Handle just DD format
            if re.match(r'^\d{1,2}$', date_str):
                day = int(date_str)
                return f"{day:02d}/{statement_month:02d}/{statement_year}"
                
            return None
        except:
            return None
    
    # Apply strategy 1
    df['parsed_date1'] = df.apply(parse_date_strategy1, axis=1)
    valid_dates1 = df['parsed_date1'].notna().sum()
    print(f"Strategy 1 (DD/MM + source file month): {valid_dates1}/{total_rows} valid dates ({valid_dates1/total_rows*100:.1f}%)")
    
    # Print sample of parsed dates
    print("\nSample of parsed dates (Strategy 1):")
    sample = df[['Date', 'source_file', 'parsed_date1']].dropna().head(10)
    for _, row in sample.iterrows():
        print(f"Original: '{row['Date']}', Source: '{row['source_file']}', Parsed: '{row['parsed_date1']}'")
    
    # Check fiscal year coverage
    start_date = "2024-03-01"
    end_date = "2025-02-28"
    
    # Convert parsed dates to datetime for filtering
    df['datetime1'] = pd.to_datetime(df['parsed_date1'], format='%d/%m/%Y', errors='coerce')
    
    # Filter by fiscal year
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    mask = (df['datetime1'] >= start_dt) & (df['datetime1'] <= end_dt)
    fiscal_year_rows = df[mask]
    
    print(f"\nFiscal year coverage (2024-03-01 to 2025-02-28):")
    print(f"Total rows in fiscal year: {len(fiscal_year_rows)}/{total_rows} ({len(fiscal_year_rows)/total_rows*100:.1f}%)")
    
    # Check month distribution
    if not fiscal_year_rows.empty:
        print("\nMonth distribution in fiscal year:")
        month_counts = fiscal_year_rows['datetime1'].dt.strftime('%Y-%m').value_counts().sort_index()
        for month, count in month_counts.items():
            print(f"{month}: {count} transactions")
    
    # Check for missing months
    all_months = pd.date_range(start=start_date, end=end_date, freq='MS').strftime('%Y-%m')
    missing_months = [m for m in all_months if m not in month_counts.index]
    if missing_months:
        print("\nMissing months in fiscal year:")
        for month in missing_months:
            print(f"{month}")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    analyze_dates()
