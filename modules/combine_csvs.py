#!/usr/bin/env python3
import os
import pandas as pd
import glob
import sys
from datetime import datetime

def combine_csv_files(input_dir, output_file):
    """
    Combine all CSV files in the input directory into a single CSV file.
    """
    print(f"Combining CSV files from {input_dir} to {output_file}")
    
    # Get all CSV files in the directory
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    csv_files = [f for f in csv_files if not os.path.basename(f).startswith("combined_")]
    
    if not csv_files:
        print("No CSV files found in the directory.")
        return False
    
    print(f"Found {len(csv_files)} CSV files to combine.")
    
    # List to store individual dataframes
    dfs = []
    
    # Process each CSV file
    for file in csv_files:
        try:
            print(f"Processing {os.path.basename(file)}")
            df = pd.read_csv(file)
            
            # Add source filename
            df['source_file'] = os.path.basename(file)
            
            # Add to list
            dfs.append(df)
            print(f"Added {len(df)} rows from {os.path.basename(file)}")
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
    
    if not dfs:
        print("No data could be read from the CSV files.")
        return False
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Save to output file
    combined_df.to_csv(output_file, index=False)
    
    print(f"Successfully combined {len(dfs)} CSV files with a total of {len(combined_df)} rows.")
    print(f"Output saved to {output_file}")
    
    return True

def analyze_combined_csv(combined_csv_path):
    """
    Analyze the combined CSV file and print statistics.
    """
    print(f"\nAnalyzing combined CSV file: {combined_csv_path}")
    
    try:
        df = pd.read_csv(combined_csv_path)
        
        # Basic statistics
        print(f"Total rows: {len(df)}")
        print(f"Columns: {', '.join(df.columns)}")
        
        # Count rows by source file
        if 'source_file' in df.columns:
            source_counts = df['source_file'].value_counts()
            print("\nRows by source file:")
            for source, count in source_counts.items():
                print(f"  {source}: {count} rows")
        
        # Date analysis
        if 'Date' in df.columns:
            print("\nDate analysis:")
            # Count unique dates
            unique_dates = df['Date'].nunique()
            print(f"  Unique dates: {unique_dates}")
            
            # Show date range if possible
            try:
                df['parsed_date'] = pd.to_datetime(df['Date'], errors='coerce')
                valid_dates = df['parsed_date'].dropna()
                if not valid_dates.empty:
                    min_date = valid_dates.min()
                    max_date = valid_dates.max()
                    print(f"  Date range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
                    print(f"  Valid dates: {len(valid_dates)} ({len(valid_dates)/len(df)*100:.1f}%)")
                else:
                    print("  No valid dates found")
            except:
                print("  Could not parse dates")
        
        # Numeric columns analysis
        numeric_cols = ['Debits', 'Credits', 'Balance']
        for col in numeric_cols:
            if col in df.columns:
                print(f"\n{col} analysis:")
                # Convert to numeric, handling errors
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
                
                # Calculate statistics
                non_zero = (df[col] != 0).sum()
                non_na = df[col].notna().sum()
                
                print(f"  Non-zero values: {non_zero} ({non_zero/len(df)*100:.1f}%)")
                print(f"  Valid values: {non_na} ({non_na/len(df)*100:.1f}%)")
                if non_na > 0:
                    print(f"  Min: {df[col].min()}")
                    print(f"  Max: {df[col].max()}")
                    print(f"  Sum: {df[col].sum()}")
        
        # Check for potential duplicates
        potential_dupes = df.duplicated(subset=['Date', 'Details', 'Debits', 'Credits'], keep=False)
        dupe_count = potential_dupes.sum()
        if dupe_count > 0:
            print(f"\nFound {dupe_count} potential duplicate transactions")
        
        return True
    except Exception as e:
        print(f"Error analyzing combined CSV: {str(e)}")
        return False

if __name__ == "__main__":
    # Default paths
    input_dir = os.path.join("data", "output", "camelot")
    output_file = os.path.join(input_dir, "combined_transactions.csv")
    
    # Allow command-line overrides
    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Combine CSV files
    success = combine_csv_files(input_dir, output_file)
    
    if success:
        # Analyze the combined CSV
        analyze_combined_csv(output_file)
