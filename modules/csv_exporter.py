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
        
        # Ensure all required columns are present
        for col in ['Details', 'ServiceFee', 'Debits', 'Credits', 'Date', 'Balance']:
            if col not in df.columns:
                df[col] = ''
        
        debug_info['columns'] = list(df.columns)
        
        # Reorder columns
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
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                
                # Add source filename as a column
                filename = os.path.basename(csv_file)
                df['Source'] = filename
                
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
            combined_df['Date'] = pd.to_datetime(combined_df['Date'], errors='coerce')
            combined_df = combined_df.sort_values('Date')
            combined_df['Date'] = combined_df['Date'].dt.strftime('%Y-%m-%d')
        
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
