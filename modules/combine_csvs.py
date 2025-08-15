#!/usr/bin/env python3
import os
import pandas as pd
import glob
import sys
import re
import shutil
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

def combine_csv_files(input_dir, output_file, fiscal_year_sorting=False, fiscal_start_month=3, fiscal_start_day=1):
    """
    Combine all CSV files in the input directory into a single CSV file.
    
    Args:
        input_dir: Directory containing CSV files to combine
        output_file: Path to save the combined CSV file
        fiscal_year_sorting: If True, sort by fiscal year instead of calendar year
        fiscal_start_month: Month when fiscal year starts (1-12)
        fiscal_start_day: Day when fiscal year starts (1-31)
    """
    logger.info(f"Combining CSV files from {input_dir} to {output_file}")
    
    # Get all CSV files in the directory
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    csv_files = [f for f in csv_files if not os.path.basename(f).startswith("combined_")]
    
    if not csv_files:
        logger.warning("No CSV files found in the directory.")
        return False
    
    logger.info(f"Found {len(csv_files)} CSV files to combine.")
    
    # List to store individual dataframes
    all_dfs = []
    file_dates = []
    
    # First pass: Read all CSVs and extract their statement periods
    for file_path in csv_files:
        try:
            logger.info(f"Processing {os.path.basename(file_path)}")
            df = pd.read_csv(file_path)
            
            if df.empty:
                logger.warning(f"Empty CSV file: {file_path}")
                continue
                
            # Add source filename
            df['source_file'] = os.path.basename(file_path)
            
            # Check if the CSV has the required columns
            if 'StatementPeriod' not in df.columns:
                logger.warning(f"CSV file missing StatementPeriod column: {file_path}")
                # Try to continue anyway
                all_dfs.append(df)
                file_dates.append(datetime.now())  # Use current date as fallback
                continue
                
            # Extract the start date from the statement period
            # Format example: "16 February 2024 - 16 March 2024"
            statement_period = df['StatementPeriod'].iloc[0]
            
            # Extract start date using regex
            start_date_match = re.search(r'(\d+\s+\w+\s+\d{4})', statement_period)
            if not start_date_match:
                logger.warning(f"Could not extract date from statement period: {statement_period}")
                all_dfs.append(df)
                file_dates.append(datetime.now())  # Use current date as fallback
                continue
                
            start_date_str = start_date_match.group(1)
            
            try:
                # Parse the date
                start_date = datetime.strptime(start_date_str, '%d %B %Y')
                
                # Store the dataframe and its start date
                all_dfs.append(df)
                file_dates.append(start_date)
                logger.info(f"Added {len(df)} rows from {os.path.basename(file_path)} with period starting {start_date_str}")
                
            except ValueError as e:
                logger.warning(f"Error parsing date '{start_date_str}': {e}")
                all_dfs.append(df)
                file_dates.append(datetime.now())  # Use current date as fallback
                
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
    
    if not all_dfs:
        logger.error("No valid CSV files found to combine")
        return False
    
    # Sort dataframes by their statement period start dates
    sorted_pairs = sorted(zip(file_dates, range(len(all_dfs))))
    sorted_dfs = [all_dfs[i] for _, i in sorted_pairs]
    
    # Combine all dataframes
    combined_df = pd.concat(sorted_dfs, ignore_index=True)
    
    # Ensure the combined CSV has the expected columns
    expected_columns = ['Date', 'Details', 'Debits', 'Credits', 'Balance', 'ServiceFee', 'Source', 'AccountNumber', 'StatementPeriod']
    
    # Check which expected columns are missing
    missing_columns = [col for col in expected_columns if col not in combined_df.columns]
    
    # Add missing columns with empty values
    for col in missing_columns:
        logger.info(f"Adding missing column: {col}")
        combined_df[col] = ''
    
    # Ensure all columns are in the right order, including any additional columns
    # First get all columns that are not in expected_columns
    additional_columns = [col for col in combined_df.columns if col not in expected_columns and col != 'source_file']
    
    # Create the final column order
    final_columns = expected_columns + additional_columns
    if 'source_file' in combined_df.columns:
        final_columns.append('source_file')
    
    # Reorder columns
    combined_df = combined_df[final_columns]
    
    # Sort transactions by date within each statement period
    # First convert Date column to datetime if possible
    try:
        # Try different date formats
        date_formats = ['%d/%m', '%d/%m/%Y', '%Y-%m-%d']
        converted = False
        
        for date_format in date_formats:
            try:
                # Add a dummy year if the format doesn't include year
                if date_format == '%d/%m':
                    # Extract year from StatementPeriod
                    combined_df['TempYear'] = combined_df['StatementPeriod'].str.extract(r'(\d{4})').iloc[:, 0]
                    # Combine with Date
                    combined_df['FullDate'] = combined_df.apply(
                        lambda row: f"{row['Date']}/{row['TempYear']}" if pd.notna(row['Date']) else None, 
                        axis=1
                    )
                    combined_df['DateObj'] = pd.to_datetime(combined_df['FullDate'], format='%d/%m/%Y', errors='coerce')
                    combined_df.drop(['TempYear', 'FullDate'], axis=1, inplace=True)
                else:
                    combined_df['DateObj'] = pd.to_datetime(combined_df['Date'], format=date_format, errors='coerce')
                    
                if not combined_df['DateObj'].isna().all():
                    converted = True
                    break
            except Exception as e:
                logger.debug(f"Failed to parse dates with format {date_format}: {e}")
        
        # Add fiscal period based on transaction date if requested
        if fiscal_year_sorting and converted:
            # Create a function to determine fiscal year for each transaction
            def get_fiscal_period(date_obj):
                if pd.isna(date_obj):
                    return None
                    
                # Get month and day from date
                month = date_obj.month
                day = date_obj.day
                year = date_obj.year
                
                # If date is before fiscal year start, it belongs to previous fiscal year
                if month < fiscal_start_month or (month == fiscal_start_month and day < fiscal_start_day):
                    fiscal_year = year - 1
                else:
                    fiscal_year = year
                    
                fiscal_year_end = fiscal_year + 1
                return f"FY{fiscal_year}-{fiscal_year_end}"
            
            # Apply fiscal period to each transaction
            combined_df['FiscalPeriod'] = combined_df['DateObj'].apply(get_fiscal_period)
            logger.info(f"Added fiscal periods to transactions based on individual transaction dates")
            
            # Sort by fiscal period, then by date
            combined_df.sort_values(['FiscalPeriod', 'DateObj'], inplace=True)
        elif converted:
            # Just sort by date within statement period
            combined_df.sort_values(['StatementPeriod', 'DateObj'], inplace=True)
        else:
            logger.warning("Could not convert dates to datetime for sorting, using original order")
            
        # Clean up temporary columns
        if 'DateObj' in combined_df.columns:
            combined_df.drop('DateObj', axis=1, inplace=True)
            
    except Exception as e:
        logger.warning(f"Error processing dates: {e}")
    
    # Save combined CSV
    combined_df.to_csv(output_file, index=False)
    logger.info(f"✅ Combined {len(all_dfs)} CSV files with {len(combined_df)} total transactions")
    logger.info(f"✅ Saved to: {output_file}")
    
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
                # Try different date formats
                date_formats = ['%d/%m/%Y', '%Y-%m-%d', '%d-%m-%Y', '%d/%m']
                converted = False
                
                for date_format in date_formats:
                    try:
                        # For formats without year, add current year
                        if date_format == '%d/%m':
                            current_year = datetime.now().year
                            df['temp_date'] = df['Date'].astype(str) + f'/{current_year}'
                            df['parsed_date'] = pd.to_datetime(df['temp_date'], format='%d/%m/%Y', errors='coerce')
                            df.drop('temp_date', axis=1, inplace=True)
                        else:
                            df['parsed_date'] = pd.to_datetime(df['Date'], format=date_format, errors='coerce')
                        
                        valid_dates = df['parsed_date'].dropna()
                        if not valid_dates.empty:
                            converted = True
                            min_date = valid_dates.min()
                            max_date = valid_dates.max()
                            print(f"  Date range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
                            print(f"  Valid dates: {len(valid_dates)} ({len(valid_dates)/len(df)*100:.1f}%)")
                            break
                    except Exception as e:
                        logger.debug(f"Failed to parse dates with format {date_format}: {e}")
                
                if not converted:
                    # Last resort: try pandas' flexible parser
                    df['parsed_date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)
                    valid_dates = df['parsed_date'].dropna()
                    if not valid_dates.empty:
                        min_date = valid_dates.min()
                        max_date = valid_dates.max()
                        print(f"  Date range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
                        print(f"  Valid dates: {len(valid_dates)} ({len(valid_dates)/len(df)*100:.1f}%)")
                    else:
                        print("  No valid dates found")
            except Exception as e:
                print(f"  Could not parse dates: {e}")
                logger.warning(f"Date parsing error: {e}")
        
        # Numeric columns analysis
        numeric_cols = ['Debits', 'Credits', 'Balance']
        for col in numeric_cols:
            if col in df.columns:
                print(f"\n{col} analysis:")
                # Convert to numeric, handling errors
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '').str.replace('R', ''), errors='coerce')
                
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
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Combine multiple statement CSV files')
    parser.add_argument('--input-dir', default=os.path.join("data", "output", "camelot"),
                        help='Directory containing CSV files to combine')
    parser.add_argument('--output', default=None,
                        help='Output CSV file (default: combined_transactions.csv in input directory)')
    parser.add_argument('--files', nargs='+', help='Specific CSV files to combine (overrides input-dir)')
    parser.add_argument('--fiscal-year', action='store_true', help='Sort by fiscal year')
    parser.add_argument('--fiscal-start-month', type=int, default=3, help='Month when fiscal year starts (1-12)')
    parser.add_argument('--fiscal-start-day', type=int, default=1, help='Day when fiscal year starts (1-31)')
    parser.add_argument('--analyze', action='store_true', help='Analyze the combined CSV after combining')
    
    args = parser.parse_args()
    
    # Set default output file if not specified
    if args.output is None:
        args.output = os.path.join(args.input_dir, "combined_transactions.csv")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # If specific files are provided, use those instead of the input directory
    if args.files:
        # Create a temporary directory to copy the files to
        temp_dir = os.path.join(os.path.dirname(args.output), "temp_combine")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Copy the files to the temporary directory
        for file in args.files:
            if os.path.exists(file):
                shutil.copy(file, os.path.join(temp_dir, os.path.basename(file)))
            else:
                logger.warning(f"File not found: {file}")
        
        # Use the temporary directory as input
        input_dir = temp_dir
    else:
        input_dir = args.input_dir
    
    # Combine CSV files
    success = combine_csv_files(
        input_dir, 
        args.output, 
        fiscal_year_sorting=args.fiscal_year,
        fiscal_start_month=args.fiscal_start_month,
        fiscal_start_day=args.fiscal_start_day
    )
    
    # Clean up temporary directory if it was created
    if args.files and os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    
    if success and args.analyze:
        # Analyze the combined CSV
        analyze_combined_csv(args.output)
