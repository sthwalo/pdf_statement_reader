#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CSV Export Module

Responsible for saving transaction data to CSV files
and combining multiple CSV files into one.
"""

import os
import logging
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def save_to_csv(transactions, output_path, debug=False):
    """
    Save transactions to CSV
    
    Args:
        transactions (list): List of transaction dictionaries
        output_path (str): Path to save CSV file
        debug (bool): Enable debug mode
        
    Returns:
        bool: Success status
        dict: Debug info if debug=True
    """
    debug_info = {
        'output_path': output_path,
        'transaction_count': len(transactions) if transactions else 0,
        'columns': [],
        'error': None
    }
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame(transactions)
        
        # Check if this is camelot format (has Date as first column)
        is_camelot_format = 'Date' in df.columns and set(['Details', 'Debits', 'Credits', 'Balance']).issubset(df.columns)
        
        # Ensure all required columns are present
        for col in ['Details', 'ServiceFee', 'Debits', 'Credits', 'Date', 'Balance']:
            if col not in df.columns:
                df[col] = ''
        
        debug_info['columns'] = list(df.columns)
        
        # Reorder columns based on format
        if is_camelot_format or 'camelot' in output_path:
            # Use camelot column order
            df = df[['Date', 'Details', 'Debits', 'Credits', 'Balance', 'ServiceFee']]
            logger.info("Using camelot column order")
        else:
            # Use traditional column order
            df = df[['Details', 'ServiceFee', 'Debits', 'Credits', 'Date', 'Balance']]
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df)} transactions to {output_path}")
        
        if debug:
            return True, debug_info
        return True
    
    except Exception as e:
        error_msg = f"Error saving to CSV: {e}"
        logger.error(error_msg)
        debug_info['error'] = error_msg
        
        if debug:
            return False, debug_info
        return False

def combine_csv_files(csv_files, output_path, debug=False):
    """
    Combine multiple CSV files into one
    
    Args:
        csv_files (list): List of CSV file paths
        output_path (str): Path to save combined CSV
        debug (bool): Enable debug mode
        
    Returns:
        bool: Success status
        dict: Debug info if debug=True
    """
    debug_info = {
        'input_files': csv_files,
        'output_path': output_path,
        'file_counts': {},
        'total_rows': 0,
        'error': None
    }
    
    try:
        if not csv_files:
            logger.warning("No CSV files to combine")
            debug_info['error'] = "No CSV files to combine"
            
            if debug:
                return False, debug_info
            return False
        
        # Read and combine all CSVs
        dfs = []
        is_camelot_format = False
        
        for csv_file in csv_files:
            try:
                # Check if this is a camelot output file
                if 'camelot' in csv_file or '/camelot/' in csv_file:
                    is_camelot_format = True
                
                df = pd.read_csv(csv_file)
                
                # Add source filename as a column
                filename = os.path.basename(csv_file)
                df['Source'] = filename
                
                # For camelot files, ensure column order is preserved
                if is_camelot_format and set(['Date', 'Details', 'Debits', 'Credits', 'Balance', 'ServiceFee']).issubset(df.columns):
                    # This is a camelot format file, preserve the column order
                    logger.info(f"Detected camelot format for {csv_file}")
                
                debug_info['file_counts'][csv_file] = len(df)
                dfs.append(df)
                logger.info(f"Read {len(df)} rows from {csv_file}")
            except Exception as e:
                error_msg = f"Error reading {csv_file}: {e}"
                logger.error(error_msg)
                debug_info['file_counts'][csv_file] = f"Error: {e}"
        
        if not dfs:
            logger.warning("No valid CSV data to combine")
            debug_info['error'] = "No valid CSV data to combine"
            
            if debug:
                return False, debug_info
            return False
        
        # Combine all dataframes
        combined_df = pd.concat(dfs, ignore_index=True)
        debug_info['total_rows'] = len(combined_df)
        
        # Ensure all required columns are present
        for col in ['Details', 'ServiceFee', 'Debits', 'Credits', 'Date', 'Balance', 'Source']:
            if col not in combined_df.columns:
                combined_df[col] = ''
        
        # Sort by date
        if 'Date' in combined_df.columns:
            # Handle partial dates (DD/MM) by adding current year
            # First preserve original dates
            combined_df['OriginalDate'] = combined_df['Date']
            
            # Add current year to dates for sorting purposes
            current_year = pd.Timestamp.now().year
            
            # Convert DD/MM to DD/MM/YYYY format
            def add_year_to_date(date_str):
                if pd.isna(date_str) or not isinstance(date_str, str):
                    return date_str
                
                # Check if it's in DD/MM format
                if '/' in date_str and len(date_str.split('/')) == 2:
                    return f"{date_str}/{current_year}"
                return date_str
            
            # Apply the conversion
            combined_df['DateWithYear'] = combined_df['Date'].apply(add_year_to_date)
            
            # Parse dates with year for sorting
            combined_df['DateWithYear'] = pd.to_datetime(combined_df['DateWithYear'], 
                                                       format='%d/%m/%Y', 
                                                       errors='coerce')
            
            # Sort by the date with year
            combined_df = combined_df.sort_values('DateWithYear')
            
            # Restore original date format
            combined_df['Date'] = combined_df['OriginalDate']
            
            # Drop temporary columns
            combined_df = combined_df.drop(['OriginalDate', 'DateWithYear'], axis=1)
            
        # Use camelot column order if detected
        if is_camelot_format:
            # Use camelot column order with Source at the end
            column_order = ['Date', 'Details', 'Debits', 'Credits', 'Balance', 'ServiceFee', 'Source']
            # Only include columns that exist in the dataframe
            column_order = [col for col in column_order if col in combined_df.columns]
            # Add any remaining columns
            column_order += [col for col in combined_df.columns if col not in column_order]
            combined_df = combined_df[column_order]
        else:
            # Use traditional column order
            combined_df = combined_df[['Details', 'ServiceFee', 'Debits', 'Credits', 'Date', 'Balance', 'Source']]
        
        # Save combined CSV
        combined_df.to_csv(output_path, index=False)
        logger.info(f"Combined {len(dfs)} CSV files with {len(combined_df)} total transactions to {output_path}")
        
        if debug:
            return True, debug_info
        return True
    
    except Exception as e:
        error_msg = f"Error combining CSV files: {e}"
        logger.error(error_msg)
        debug_info['error'] = error_msg
        
        if debug:
            return False, debug_info
        return False

def main():
    """Test function for direct module execution"""
    import argparse
    import json
    import sys
    
    parser = argparse.ArgumentParser(description='Export transactions to CSV')
    parser.add_argument('--input', help='Path to JSON file with transactions')
    parser.add_argument('--output', required=True, help='Path to save CSV output')
    parser.add_argument('--combine', action='store_true', help='Combine multiple CSV files')
    parser.add_argument('--csv-files', nargs='+', help='CSV files to combine (with --combine)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--debug-output', help='Path to save debug info (JSON)')
    
    args = parser.parse_args()
    
    if args.combine:
        if not args.csv_files:
            print("Error: --csv-files must be specified with --combine")
            return
        
        if args.debug:
            success, debug_info = combine_csv_files(args.csv_files, args.output, debug=True)
        else:
            success = combine_csv_files(args.csv_files, args.output)
        
        if success:
            print(f"✅ Successfully combined CSV files to {args.output}")
        else:
            print(f"❌ Failed to combine CSV files")
    else:
        if not args.input:
            print("Error: --input must be specified")
            return
        
        # Read transactions from JSON
        with open(args.input, 'r') as f:
            transactions = json.load(f)
        
        if args.debug:
            success, debug_info = save_to_csv(transactions, args.output, debug=True)
        else:
            success = save_to_csv(transactions, args.output)
        
        if success:
            print(f"✅ Successfully saved {len(transactions)} transactions to {args.output}")
        else:
            print(f"❌ Failed to save transactions to CSV")
    
    if args.debug and args.debug_output:
        with open(args.debug_output, 'w') as f:
            json.dump(debug_info, f, indent=2)
        print(f"Debug info saved to {args.debug_output}")

if __name__ == '__main__':
    main()
