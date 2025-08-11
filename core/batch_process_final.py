#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch processing script for extracting transactions from multiple PDF files
using the modules directly for better integration and efficiency
"""

import os
import sys
import time
import json
import logging
import argparse
import pandas as pd
import concurrent.futures
from datetime import datetime

# Add parent directory to path to find modules
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  

# Import modules
from modules.table_extractor import extract_tables_from_pdf
from modules.transaction_extractor import extract_transactions_from_pdf
from modules.data_cleaner import propagate_dates, clean_transactions, deduplicate_transactions
from modules.csv_exporter import save_to_csv
from modules.csv_analyzer import analyze_csv_transactions, analyze_combined_csv

# Define combine_csv_files function since it's not in the csv_analyzer module
def combine_csv_files(csv_files, output_path):
    """
    Combine multiple CSV files into a single CSV file with a Source column
    
    Args:
        csv_files (list): List of CSV file paths
        output_path (str): Path to save the combined CSV
        
    Returns:
        bool: Success status
    """
    try:
        if not csv_files:
            logger.error("No CSV files provided for combining")
            return False
            
        combined_df = pd.DataFrame()
        
        for csv_file in csv_files:
            try:
                # Read CSV
                df = pd.read_csv(csv_file)
                
                # Add source column with filename
                source_name = os.path.basename(csv_file).replace('_transactions.csv', '').replace('_final.csv', '')
                df['Source'] = source_name
                
                # Append to combined dataframe
                combined_df = pd.concat([combined_df, df], ignore_index=True)
                
                logger.info(f"Added {len(df)} rows from {csv_file}")
            except Exception as e:
                logger.error(f"Error processing {csv_file}: {e}")
        
        # Sort by date if possible
        if 'Date' in combined_df.columns:
            combined_df['Date'] = pd.to_datetime(combined_df['Date'], errors='coerce')
            combined_df = combined_df.sort_values('Date').reset_index(drop=True)
        
        # Save combined CSV
        combined_df.to_csv(output_path, index=False)
        logger.info(f"Combined {len(csv_files)} CSV files with {len(combined_df)} total rows")
        
        return True
    except Exception as e:
        logger.error(f"Error combining CSV files: {e}")
        return False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename=f'batch_process_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
)
logger = logging.getLogger(__name__)

def process_pdf(pdf_path, output_dir, password=None, debug=False):
    """Process a single PDF file using modules directly"""
    try:
        # Create output filename and debug directory
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_transactions.csv")
        debug_dir = os.path.join(output_dir, "debug")
        if debug:
            os.makedirs(debug_dir, exist_ok=True)
        
        logger.info(f"Processing {pdf_path} -> {output_path}")
        print(f"Processing {pdf_path}...")
        
        # Start timing
        start_time = time.time()
        
        # 1. Extract tables
        logger.info(f"Extracting tables from {pdf_path}")
        if debug:
            tables, tables_debug = extract_tables_from_pdf(pdf_path, password, debug=True)
            debug_file = os.path.join(debug_dir, f"{base_name}_tables_debug.json")
            with open(debug_file, 'w') as f:
                json.dump(tables_debug, f, indent=2, default=str)
        else:
            tables = extract_tables_from_pdf(pdf_path, password)
        
        if not tables:
            logger.error(f"No tables extracted from {pdf_path}")
            print(f"❌ Failed to extract tables from {pdf_path}")
            return False, pdf_path, None
        
        logger.info(f"Extracted {len(tables)} tables from {pdf_path}")
        
        # 2. Extract transactions
        logger.info(f"Extracting transactions from tables")
        if debug:
            transactions, tx_debug = extract_transactions_from_pdf(tables, debug=True)
            debug_file = os.path.join(debug_dir, f"{base_name}_transactions_debug.json")
            with open(debug_file, 'w') as f:
                json.dump(tx_debug, f, indent=2, default=str)
        else:
            transactions = extract_transactions_from_pdf(tables)
        
        if not transactions:
            logger.error(f"No transactions extracted from {pdf_path}")
            print(f"❌ Failed to extract transactions from {pdf_path}")
            return False, pdf_path, None
        
        logger.info(f"Extracted {len(transactions)} raw transactions")
        
        # 3. Clean and process transactions
        logger.info(f"Cleaning and processing transactions")
        
        # 3a. Propagate dates
        transactions = propagate_dates(transactions)
        
        # 3b. Clean transactions
        transactions = clean_transactions(transactions)
        
        # 3c. Deduplicate transactions
        transactions = deduplicate_transactions(transactions)
        
        logger.info(f"Cleaned and processed transactions: {len(transactions)} final transactions")
        
        # 4. Save to CSV
        logger.info(f"Saving transactions to {output_path}")
        save_to_csv(transactions, output_path)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        logger.info(f"Successfully processed {pdf_path} in {elapsed_time:.2f} seconds")
        print(f"✅ {pdf_path} -> {output_path} ({elapsed_time:.2f}s)")
        return True, pdf_path, output_path
    
    except Exception as e:
        logger.error(f"Error processing {pdf_path}: {e}")
        print(f"❌ Error processing {pdf_path}: {e}")
        return False, pdf_path, None

def batch_process(input_dir, output_dir, max_workers=4, password=None, file_pattern=None, combine=False, analyze=False, debug=False):
    """Process all PDF files in the input directory"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all PDF files in the input directory
    pdf_files = []
    for file in os.listdir(input_dir):
        if file.lower().endswith('.pdf'):
            if file_pattern and file_pattern not in file:
                continue
            pdf_files.append(os.path.join(input_dir, file))
    
    if not pdf_files:
        logger.warning(f"No PDF files found in {input_dir}")
        print(f"No PDF files found in {input_dir}")
        return []
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    print(f"Found {len(pdf_files)} PDF files to process")
    
    # Process files in parallel
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_pdf, pdf, output_dir, password, debug): pdf for pdf in pdf_files}
        
        for future in concurrent.futures.as_completed(futures):
            pdf = futures[future]
            try:
                success, pdf_path, output_path = future.result()
                results.append((success, pdf_path, output_path))
            except Exception as e:
                logger.error(f"Error processing {pdf}: {e}")
                print(f"❌ Error processing {pdf}: {e}")
                results.append((False, pdf, None))
    
    # Get successful outputs
    successful = [r for r in results if r[0]]
    failed = [r for r in results if not r[0]]
    successful_outputs = [output_path for _, _, output_path in successful if output_path]
    
    # Combine CSVs if requested
    combined_path = None
    if combine and successful_outputs:
        logger.info("Combining CSV files...")
        print("\nCombining CSV files...")
        
        combined_path = os.path.join(output_dir, "combined_transactions.csv")
        combine_success = combine_csv_files(successful_outputs, combined_path)
        
        if combine_success:
            logger.info(f"✅ Successfully combined all transactions to {combined_path}")
            print(f"✅ Combined transactions saved to {combined_path}")
            
            # Analyze combined CSV if requested
            if analyze:
                logger.info(f"Analyzing combined CSV file...")
                print(f"\nAnalyzing combined CSV file...")
                
                analysis_dir = os.path.join(output_dir, "analysis")
                os.makedirs(analysis_dir, exist_ok=True)
                
                analysis_result = analyze_combined_csv(combined_path, analysis_dir)
                
                if analysis_result:
                    logger.info(f"✅ Analysis complete. Results saved to {analysis_dir}")
                    print(f"✅ Analysis complete. Results saved to {analysis_dir}")
                else:
                    logger.error(f"❌ Failed to analyze combined CSV")
                    print(f"❌ Failed to analyze combined CSV")
        else:
            logger.error(f"❌ Failed to combine transactions")
            print(f"❌ Failed to combine transactions")
    
    # Summarize results
    logger.info(f"Batch processing complete: {len(successful)} successful, {len(failed)} failed")
    print(f"\nBatch processing complete: {len(successful)} successful, {len(failed)} failed")
    
    if failed:
        print("\nFailed files:")
        for _, pdf_path, _ in failed:
            print(f"  - {os.path.basename(pdf_path)}")
    
    return successful_outputs, combined_path

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Batch process PDF bank statements')
    parser.add_argument('input_dir', help='Directory containing PDF files')
    parser.add_argument('--output-dir', help='Directory to save CSV files', default='output')
    parser.add_argument('--workers', type=int, help='Maximum number of parallel workers', default=4)
    parser.add_argument('--password', help='Password for encrypted PDFs')
    parser.add_argument('--pattern', help='Only process files containing this pattern')
    parser.add_argument('--combine', action='store_true', help='Combine all CSVs into one file')
    parser.add_argument('--analyze', action='store_true', help='Analyze CSV files after processing')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with detailed logging')
    
    args = parser.parse_args()
    
    # Process all PDF files
    output_files, combined_path = batch_process(
        args.input_dir, 
        args.output_dir, 
        args.workers, 
        args.password,
        args.pattern,
        args.combine,
        args.analyze,
        args.debug
    )
    
    if output_files:
        print(f"\nAll CSV files saved to {args.output_dir}/")
        
        if not args.analyze:
            print(f"\nTo analyze CSV files, run:")
            print(f"  python analyze_csv_output.py {args.output_dir}/*.csv --output-dir {args.output_dir}/analysis")
            
            if args.combine and combined_path:
                print(f"  python analyze_csv_output.py {combined_path} --output-dir {args.output_dir}/analysis --combined")
    
    # Provide additional options for further processing
    if combined_path:
        print(f"\nTo fix balance issues in the combined CSV, run:")
        print(f"  python fix_balance_issues.py {combined_path} --debug")
        
    print(f"\nTo run the full interactive processor with all options, run:")
    print(f"  python pdf_statement_processor.py")
    
    logger.info("Batch processing finished")

if __name__ == '__main__':
    main()
