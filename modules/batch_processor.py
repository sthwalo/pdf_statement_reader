#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch Processing Module

Coordinates the entire extraction process for multiple PDF files,
using the modular components for table extraction, transaction identification,
data cleaning, and CSV export.
"""

import os
import sys
import json
import logging
import argparse
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

# Import transaction extraction functions
from modules.transaction_extractor import extract_transactions_from_pdf
from modules.csv_exporter import save_to_csv, combine_csv_files

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def process_single_pdf(pdf_path, output_dir, password=None, debug=False, extraction_method='regular'):
    """
    Process a single PDF file through the entire extraction pipeline
    
    Args:
        pdf_path (str): Path to the PDF file
        output_dir (str): Directory to save output files
        password (str, optional): Password for encrypted PDF
        debug (bool, optional): Enable debug mode
        extraction_method (str, optional): Extraction method to use ('regular', 'lattice', 'strict_lattice', 'camelot')
        
    Returns:
        dict: Result information
        dict: Debug info if debug=True
    """
    from modules.table_extractor import extract_tables_from_pdf
    from modules.transaction_extractor import extract_transactions_from_pdf
    from modules.data_cleaner import propagate_dates, clean_transactions, deduplicate_transactions
    from modules.csv_exporter import save_to_csv
    
    # Import camelot parser if needed
    if extraction_method == 'camelot':
        from camelot_parser import CamelotBankStatementParser
    
    result = {
        'pdf_path': pdf_path,
        'success': False,
        'output_path': None,
        'transaction_count': 0,
        'error': None
    }
    
    debug_info = {
        'pdf_path': pdf_path,
        'timestamp': datetime.now().isoformat(),
        'stages': {}
    }
    
    try:
        # Create output filename
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        
        # Set up consistent output paths
        if extraction_method == 'camelot':
            # Always use output/camelot directory for camelot extraction
            camelot_output_dir = os.path.join(os.path.dirname(os.path.dirname(pdf_path)), 'output', 'camelot')
            os.makedirs(camelot_output_dir, exist_ok=True)
            output_path = os.path.join(camelot_output_dir, f"{base_name}_transactions.csv")
        else:
            output_path = os.path.join(output_dir, f"{base_name}_{extraction_method}_transactions.csv")
            
        result['output_path'] = output_path
        
        # Use camelot parser if selected
        if extraction_method == 'camelot':
            logger.info(f"Using camelot parser for {pdf_path}")
            parser = CamelotBankStatementParser(debug=debug)
            if debug:
                transactions = parser.extract_transactions_from_pdf(pdf_path, password=password)
                debug_info['stages']['camelot_extraction'] = parser.debug_info
            else:
                transactions = parser.extract_transactions_from_pdf(pdf_path, password=password)
                
            if not transactions:
                error_msg = f"No transactions extracted from {pdf_path} using camelot parser"
                logger.error(error_msg)
                result['error'] = error_msg
                
                if debug:
                    return result, debug_info
                return result
                
            logger.info(f"Extracted {len(transactions)} transactions using camelot parser")
            
            # For camelot extraction, save directly to CSV to preserve format
            logger.info(f"Saving camelot transactions directly to {output_path}")
            df = pd.DataFrame(transactions)
            df.to_csv(output_path, index=False)
            
            # Also save debug info to the camelot directory if debug is enabled
            if debug:
                camelot_debug_dir = os.path.dirname(output_path)
                debug_file = os.path.join(camelot_debug_dir, f"{base_name}_debug.json")
                with open(debug_file, 'w') as f:
                    json.dump(parser.debug_info, f, indent=2, default=str)
                logger.info(f"Saved debug info to {debug_file}")
            
            # Success
            result['success'] = True
            result['transaction_count'] = len(transactions)
            logger.info(f"Successfully processed {pdf_path} with {len(transactions)} transactions")
            
            if debug:
                return result, debug_info
            return result
        else:
            # Regular extraction pipeline
            # 1. Extract tables based on extraction method
            logger.info(f"Extracting tables from {pdf_path} using {extraction_method} method")
            
            if extraction_method == 'lattice':
                from modules.lattice_extractor import LatticeExtractor
                extractor = LatticeExtractor(debug=debug)
                tables = extractor.extract_tables_from_pdf(pdf_path, password)
                if debug:
                    tables_debug = extractor.debug_info
                    debug_info['stages']['table_extraction'] = tables_debug
            elif extraction_method == 'strict_lattice':
                from lattice_strict import LatticeStrictExtractor
                extractor = LatticeStrictExtractor(debug=debug)
                tables = extractor.extract_tables_from_pdf(pdf_path, password)
                if debug:
                    tables_debug = extractor.debug_info
                    debug_info['stages']['table_extraction'] = tables_debug
            else:  # regular
                if debug:
                    from modules.table_extractor import extract_tables_from_pdf
                    tables, tables_debug = extract_tables_from_pdf(pdf_path, password, debug=True)
                    debug_info['stages']['table_extraction'] = tables_debug
                else:
                    from modules.table_extractor import extract_tables_from_pdf
                    tables = extract_tables_from_pdf(pdf_path, password)
            
            if not tables:
                error_msg = f"No tables extracted from {pdf_path}"
                logger.error(error_msg)
                result['error'] = error_msg
                
                if debug:
                    return result, debug_info
                return result
            
            logger.info(f"Extracted {len(tables)} tables from {pdf_path}")
            
            # 2. Extract transactions
            logger.info(f"Extracting transactions from tables")
            if debug:
                transactions, tx_debug = extract_transactions_from_pdf(tables, debug=True)
                debug_info['stages']['transaction_extraction'] = tx_debug
            else:
                transactions = extract_transactions_from_pdf(tables)
        
        if not transactions:
            error_msg = f"No transactions extracted from {pdf_path}"
            logger.error(error_msg)
            result['error'] = error_msg
            
            if debug:
                return result, debug_info
            return result
        
        logger.info(f"Extracted {len(transactions)} raw transactions")
        
        # 3. Clean and process transactions
        logger.info(f"Cleaning and processing transactions")
        
        # 3a. Propagate dates
        if debug:
            transactions, prop_debug = propagate_dates(transactions, debug=True)
            debug_info['stages']['date_propagation'] = prop_debug
        else:
            transactions = propagate_dates(transactions)
        
        # 3b. Clean transactions
        if debug:
            transactions, clean_debug = clean_transactions(transactions, debug=True)
            debug_info['stages']['transaction_cleaning'] = clean_debug
        else:
            transactions = clean_transactions(transactions)
        
        # 3c. Deduplicate transactions
        if debug:
            transactions, dedup_debug = deduplicate_transactions(transactions, debug=True)
            debug_info['stages']['deduplication'] = dedup_debug
        else:
            transactions = deduplicate_transactions(transactions)
        
        if not transactions:
            error_msg = f"No valid transactions after cleaning from {pdf_path}"
            logger.error(error_msg)
            result['error'] = error_msg
            
            if debug:
                return result, debug_info
            return result
        
        logger.info(f"Processed {len(transactions)} clean transactions")
        result['transaction_count'] = len(transactions)
        
        # 4. Save to CSV
        logger.info(f"Saving transactions to {output_path}")
        if debug:
            save_success, save_debug = save_to_csv(transactions, output_path, debug=True)
            debug_info['stages']['csv_export'] = save_debug
        else:
            save_success = save_to_csv(transactions, output_path)
        
        if not save_success:
            error_msg = f"Failed to save transactions to {output_path}"
            logger.error(error_msg)
            result['error'] = error_msg
            
            if debug:
                return result, debug_info
            return result
        
        # Success
        result['success'] = True
        logger.info(f"Successfully processed {pdf_path} with {len(transactions)} transactions")
        
        if debug:
            return result, debug_info
        return result
    
    except Exception as e:
        error_msg = f"Error processing {pdf_path}: {e}"
        logger.error(error_msg)
        result['error'] = error_msg
        debug_info['error'] = error_msg
        
        if debug:
            return result, debug_info
        return result

def process_directory(input_dir, output_dir, password=None, max_workers=None, combine=False, debug=False, extraction_method='regular'):
    """
    Process all PDF files in a directory
    
    Args:
        input_dir (str): Directory containing PDF files
        output_dir (str): Directory to save output files
        password (str, optional): Password for encrypted PDFs
        max_workers (int, optional): Maximum number of worker processes
        combine (bool, optional): Combine all CSVs into one file
        debug (bool, optional): Enable debug mode
        extraction_method (str, optional): Extraction method to use ('regular', 'lattice', 'strict_lattice', 'camelot')
        
    Returns:
        dict: Processing results
    """
    from modules.csv_exporter import combine_csv_files
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create debug directory if needed
    debug_dir = os.path.join(output_dir, "debug") if debug else None
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
    
    # Find all PDF files
    pdf_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    
    if not pdf_files:
        logger.warning(f"No PDF files found in {input_dir}")
        return {
            'success': False,
            'error': f"No PDF files found in {input_dir}",
            'pdf_count': 0,
            'success_count': 0,
            'failed_count': 0,
            'results': []
        }
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    # Process PDFs in parallel
    results = []
    successful_outputs = []
    debug_outputs = {} if debug else None
    # Process each PDF file in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks
        future_to_pdf = {}
        for pdf_file in pdf_files:
            # pdf_file already contains the full path, don't join with input_dir again
            pdf_path = pdf_file
            future = executor.submit(process_single_pdf, pdf_path, output_dir, password, debug, extraction_method)
            future_to_pdf[future] = pdf_path
        
        # Process results as they complete
        for future in as_completed(future_to_pdf):
            pdf_path = future_to_pdf[future]
            try:
                if debug:
                    result, debug_info = future.result()
                    
                    # Save debug info
                    debug_file = os.path.join(debug_dir, f"{os.path.splitext(os.path.basename(pdf_path))[0]}_debug.json")
                    with open(debug_file, 'w') as f:
                        json.dump(debug_info, f, indent=2)
                    
                    debug_outputs[pdf_path] = debug_file
                else:
                    result = future.result()
                
                results.append(result)
                
                if result['success']:
                    logger.info(f"✅ Successfully processed {pdf_path}")
                    successful_outputs.append(result['output_path'])
                else:
                    logger.error(f"❌ Failed to process {pdf_path}: {result['error']}")
            except Exception as e:
                logger.error(f"Error processing {pdf_path}: {e}")
                results.append({
                    'pdf_path': pdf_path,
                    'output_path': None,
                    'success': False,
                    'error': str(e)
                })
    
    # Combine CSVs if requested
    combined_path = None
    if combine and successful_outputs:
        # Check if we're using camelot extraction
        is_camelot_extraction = extraction_method == 'camelot'
        
        # Set appropriate output directory for combined file
        if is_camelot_extraction:
            # For camelot, use the output/camelot directory
            camelot_output_dir = os.path.join(os.path.dirname(output_dir), 'output', 'camelot')
            os.makedirs(camelot_output_dir, exist_ok=True)
            combined_path = os.path.join(camelot_output_dir, "combined_transactions.csv")
        else:
            # For other methods, use the specified output directory
            combined_path = os.path.join(output_dir, "combined_transactions.csv")
        
        logger.info(f"Combining CSV files to {combined_path}")
        
        if debug:
            combine_success, combine_debug = combine_csv_files(successful_outputs, combined_path, debug=True)
            
            # Save combine debug info
            combine_debug_file = os.path.join(debug_dir, "combine_debug.json")
            with open(combine_debug_file, 'w') as f:
                json.dump(combine_debug, f, indent=2)
        else:
            combine_success = combine_csv_files(successful_outputs, combined_path)
        
        if combine_success:
            logger.info(f"✅ Successfully combined all transactions to {combined_path}")
        else:
            logger.error(f"❌ Failed to combine transactions")
    
    # Count successes and failures
    success_count = sum(1 for r in results if r['success'])
    failed_count = len(results) - success_count
    
    # Log summary
    logger.info(f"Batch processing complete: {success_count}/{len(pdf_files)} PDFs processed successfully")
    
    return {
        'success': success_count > 0,
        'pdf_count': len(pdf_files),
        'success_count': success_count,
        'failed_count': failed_count,
        'results': results,
        'combined_path': combined_path,
        'debug_outputs': debug_outputs if debug else None
    }

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Batch process bank statement PDFs')
    parser.add_argument('--input', '-i', required=True, help='Input directory containing PDF files')
    parser.add_argument('--output', '-o', required=True, help='Output directory for CSV files')
    parser.add_argument('--password', '-p', help='Password for encrypted PDFs')
    parser.add_argument('--workers', '-w', type=int, help='Maximum number of worker processes')
    parser.add_argument('--combine', '-c', action='store_true', help='Combine all CSVs into one file')
    parser.add_argument('--debug', '-d', action='store_true', help='Enable debug mode with detailed logging')
    parser.add_argument('--single', '-s', help='Process a single PDF file instead of a directory')
    parser.add_argument('--method', '-m', choices=['regular', 'lattice', 'strict_lattice', 'camelot'], 
                        default='regular', help='Extraction method to use')
    
    args = parser.parse_args()
    
    # Setup logging to file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f"batch_process_{timestamp}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info(f"Starting batch processing with arguments: {args}")
    
    if args.single:
        # Process single PDF
        logger.info(f"Processing single PDF: {args.single} using {args.method} method")
        
        if args.debug:
            result, debug_info = process_single_pdf(args.single, args.output, args.password, debug=True, extraction_method=args.method)
            
            # Save debug info
            debug_dir = os.path.join(args.output, "debug")
            os.makedirs(debug_dir, exist_ok=True)
            debug_file = os.path.join(debug_dir, f"{os.path.splitext(os.path.basename(args.single))[0]}_{args.method}_debug.json")
            with open(debug_file, 'w') as f:
                json.dump(debug_info, f, indent=2)
            
            print(f"Debug info saved to {debug_file}")
        else:
            result = process_single_pdf(args.single, args.output, args.password, extraction_method=args.method)
        
        if result['success']:
            print(f"✅ Successfully processed {args.single}")
            print(f"Transactions saved to: {result['output_path']}")
        else:
            print(f"❌ Failed to process {args.single}: {result['error']}")
    else:
        # Process directory
        logger.info(f"Processing directory: {args.input} -> {args.output} using {args.method} method")
        
        result = process_directory(
            args.input,
            args.output,
            args.password,
            args.workers,
            args.combine,
            args.debug,
            args.method
        )
        
        # Print summary
        print(f"\nBatch processing complete: {result['success_count']}/{result['pdf_count']} PDFs processed successfully")
        
        if args.combine and result['combined_path']:
            print(f"Combined transactions saved to: {result['combined_path']}")
        
        if args.debug:
            print(f"Debug logs saved to: {os.path.join(args.output, 'debug')}")
    
    logger.info("Batch processing finished")
    print(f"Log file: {log_file}")

if __name__ == '__main__':
    main()
