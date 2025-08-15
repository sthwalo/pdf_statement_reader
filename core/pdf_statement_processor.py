#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PDF Statement Processor - Integrated Menu System

This script provides a comprehensive menu-driven interface to process bank statement PDFs
through all stages of extraction, analysis, and reporting.
"""

import os
import sys
import logging
import argparse
import json
import pandas as pd
from datetime import datetime
import concurrent.futures
import time
import subprocess
import shutil

# Add parent directory to path to find modules
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  

# Import modules
from modules.table_extractor import extract_tables_from_pdf
from modules.transaction_identifier import identify_columns
from modules.transaction_extractor import extract_transactions_from_pdf
from modules.data_cleaner import process_transactions
from modules.csv_exporter import save_to_csv, combine_csv_files
from modules.csv_analyzer import analyze_csv_transactions, analyze_combined_csv
from modules.analyze_pdf_structure import analyze_pdf_structure
from modules.page_analyzer_menu import menu_analyze_pdf_pages
from modules.camelot_parser import CamelotBankStatementParser

# Configure logging
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)

# File handler will be created in main()

def setup_logging(output_dir=None):
    """Set up logging to file"""
    global logger
    
    # Remove any existing file handlers
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
    
    if output_dir:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(output_dir, f"pdf_processor_{timestamp}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(log_formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging to {log_file}")
    
    return logger

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Print application header"""
    clear_screen()
    print("=" * 80)
    print("PDF STATEMENT PROCESSOR".center(80))
    print("=" * 80)
    print()

def print_section_header(title):
    """Print section header"""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "-"))
    print("=" * 80 + "\n")

def get_input_path(prompt, file_type=None):
    """Get and validate input path from user"""
    while True:
        path = input(prompt).strip()
        
        if not path:
            print("Path cannot be empty. Please try again.")
            continue
            
        # Handle relative paths
        if not os.path.isabs(path):
            path = os.path.abspath(path)
            
        if not os.path.exists(path):
            print(f"Path does not exist: {path}")
            create = input("Would you like to create this directory? (y/n): ").lower()
            if create == 'y':
                try:
                    os.makedirs(path, exist_ok=True)
                    print(f"Created directory: {path}")
                    return path
                except Exception as e:
                    print(f"Error creating directory: {e}")
            continue
            
        if file_type == 'file' and not os.path.isfile(path):
            print(f"Not a file: {path}")
            continue
            
        if file_type == 'dir' and not os.path.isdir(path):
            print(f"Not a directory: {path}")
            continue
            
        return path

def process_single_pdf(pdf_path, output_dir, debug=False, extraction_method='regular'):
    """
    Process a single PDF file through all stages
    
    Args:
        pdf_path (str): Path to the PDF file
        output_dir (str): Output directory for results
        debug (bool): Enable debug mode
        extraction_method (str): Extraction method to use ('regular', 'lattice', 'strict_lattice', 'camelot')
        
    Returns:
        bool: Success status
        str: Path to output CSV file
    """
    try:
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            return False, None
            
        # Determine output paths
        base_name = os.path.basename(pdf_path).replace('.pdf', '')
        output_csv = os.path.join(output_dir, f"{base_name}_transactions.csv")
        debug_dir = os.path.join(output_dir, "debug")
            
        # Ensure output directories exist
        os.makedirs(output_dir, exist_ok=True)
        if debug:
            os.makedirs(debug_dir, exist_ok=True)
        
        logger.info(f"Processing {pdf_path}")
        
        # Create output filename with extraction method
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        
        # Use dedicated output directory for camelot extraction
        if extraction_method == 'camelot':
            camelot_output_dir = os.path.join(output_dir, 'camelot')
            os.makedirs(camelot_output_dir, exist_ok=True)
            output_csv = os.path.join(camelot_output_dir, f"{base_name}_transactions.csv")
        else:
            output_csv = os.path.join(output_dir, f"{base_name}_{extraction_method}_transactions.csv")
        
        # Use camelot parser if selected
        if extraction_method == 'camelot':
            logger.info(f"Using camelot parser for {pdf_path}")
            parser = CamelotBankStatementParser(debug=debug)
            
            if debug:
                transactions = parser.extract_transactions_from_pdf(pdf_path, password=None)
                with open(os.path.join(debug_dir, f"{base_name}_camelot_extraction.json"), 'w') as f:
                    json.dump(parser.debug_info, f, indent=2, default=str)
            else:
                transactions = parser.extract_transactions_from_pdf(pdf_path, password=None)
                
            if not transactions:
                logger.error(f"No transactions extracted from {pdf_path} using camelot parser")
                return False, None
                
            logger.info(f"Extracted {len(transactions)} transactions using camelot parser")
            
            # For camelot extraction, save directly to CSV to preserve format
            logger.info(f"Saving camelot transactions directly to {output_csv}")
            df = pd.DataFrame(transactions)
            df.to_csv(output_csv, index=False)
            
            # Also save debug info to the camelot directory
            if debug:
                camelot_debug_dir = os.path.dirname(output_csv)
                debug_file = os.path.join(camelot_debug_dir, f"{base_name}_debug.json")
                with open(debug_file, 'w') as f:
                    json.dump(parser.debug_info, f, indent=2, default=str)
                logger.info(f"Saved debug info to {debug_file}")
            
            logger.info(f"Saved {len(transactions)} transactions to {output_csv}")
            return True, output_csv
            
        else:
            # Regular extraction pipeline
            # Step 1: Extract tables from PDF
            logger.info(f"Extracting tables from {pdf_path} using {extraction_method} method")
            
            if debug:
                tables, extract_debug = extract_tables_from_pdf(pdf_path, None, method=extraction_method, debug=True)
                with open(os.path.join(debug_dir, f"{base_name}_table_extraction.json"), 'w') as f:
                    json.dump(extract_debug, f, indent=2, default=str)
            else:
                tables = extract_tables_from_pdf(pdf_path, None, method=extraction_method)
            
            if not tables:
                logger.error(f"No tables found in {pdf_path}")
                return False, None
                
            logger.info(f"Extracted {len(tables)} tables from PDF")
            
            # Step 2: Extract transactions from tables
            logger.info("Extracting transactions from tables...")
            
            if debug:
                transactions, tx_debug = extract_transactions_from_pdf(tables, debug=True)
                with open(os.path.join(debug_dir, f"{base_name}_transaction_extraction.json"), 'w') as f:
                    json.dump(tx_debug, f, indent=2, default=str)
            else:
                transactions = extract_transactions_from_pdf(tables)
                
            if not transactions:
                logger.error(f"No transactions extracted from {pdf_path}")
                return False, None
                
            logger.info(f"Extracted {len(transactions)} transactions from tables")
            
            # Step 3: Process transactions
            logger.info("Processing transactions...")
            
            if debug:
                processed_transactions, proc_debug = process_transactions(transactions, debug=True)
                with open(os.path.join(debug_dir, f"{base_name}_transaction_processing.json"), 'w') as f:
                    json.dump(proc_debug, f, indent=2, default=str)
                logger.info(f"Transactions with balance values: {proc_debug.get('balance_columns_found', 0)}")
                logger.info(f"Balance columns found: {proc_debug.get('balance_columns_found', 0)}")
                logger.info(f"Balance columns fixed: {proc_debug.get('balance_columns_fixed', 0)}")
            else:
                processed_transactions = process_transactions(transactions)
            
            logger.info(f"Processed {len(processed_transactions)} transactions")
        logger.info(f"Processed {len(processed_transactions)} transactions")
        
        # Step 3: Export to CSV
        logger.info(f"Exporting to CSV: {output_csv}")
        save_to_csv(processed_transactions, output_csv)
        logger.info(f"Saved {len(processed_transactions)} transactions to {output_csv}")
        
        logger.info(f"Successfully processed {pdf_path} -> {output_csv}")
        
        return True, output_csv
    except Exception as e:
        logger.error(f"Error processing {pdf_path}: {e}", exc_info=True)
        return False, None

def batch_process_pdfs(input_dir, output_dir, password=None, parser_type='camelot', 
                   combine=False, analyze=False, debug=False, parallel=False, max_workers=None,
                   fiscal_year_sorting=False, fiscal_start_month=3, fiscal_start_day=1):
    """
    Process all PDF files in the specified directory
    
    Args:
        input_dir (str): Directory containing PDF files to process
        output_dir (str): Output directory for results
        max_workers (int): Maximum number of parallel workers
        debug (bool): Enable debug mode
        combine (bool): Combine all CSV files into one
        analyze (bool): Analyze CSV output
        extraction_method (str): Extraction method to use ('regular', 'lattice', 'strict_lattice', 'camelot')
        
    Returns:
        bool: Success status
        list: List of output CSV files
    """
    # Find all PDF files in the directory
    pdf_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    
    if not pdf_files:
        logger.error(f"No PDF files found in {input_dir}")
        return False, []
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    # Process files
    successful = 0
    failed = 0
    output_files = []
    
    if debug:
        # Process sequentially for debugging
        for pdf in pdf_files:
            logger.info(f"Processing {pdf} (debug mode)")
            success, output_file = process_single_pdf(pdf, output_dir, debug, extraction_method)
            if success:
                successful += 1
                if output_file:
                    output_files.append(output_file)
            else:
                failed += 1
    else:
        # Process in parallel for production
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_pdf = {executor.submit(process_single_pdf, pdf, output_dir, debug, extraction_method): pdf for pdf in pdf_files}
            for future in concurrent.futures.as_completed(future_to_pdf):
                pdf = future_to_pdf[future]
                try:
                    success, output_file = future.result()
                    if success:
                        successful += 1
                        if output_file:
                            output_files.append(output_file)
                    else:
                        failed += 1
                except Exception as e:
                    logger.error(f"Error processing {pdf}: {e}")
                    failed += 1
    
    logger.info(f"Batch processing complete. Successful: {successful}, Failed: {failed}")
    
    # Combine CSV files if requested
    combined_output = None
    if combine and output_files:
        combined_csv = os.path.join(output_dir, "combined_transactions.csv")
        logger.info(f"Combining {len(output_files)} CSV files into {combined_csv}")
        
        if debug:
            success, combine_debug = combine_csv_files(
                output_files, 
                combined_csv, 
                fiscal_year_sorting=fiscal_year_sorting,
                fiscal_start_month=fiscal_start_month,
                fiscal_start_day=fiscal_start_day,
                debug=True
            )
            debug_dir = os.path.join(output_dir, "debug")
            os.makedirs(debug_dir, exist_ok=True)
            with open(os.path.join(debug_dir, "combine_csv_debug.json"), 'w') as f:
                json.dump(combine_debug, f, indent=2, default=str)
        else:
            success = combine_csv_files(
                output_files, 
                combined_csv,
                fiscal_year_sorting=fiscal_year_sorting,
                fiscal_start_month=fiscal_start_month,
                fiscal_start_day=fiscal_start_day
            )
            
        if success:
            logger.info(f"Successfully combined CSV files into {combined_csv}")
            combined_output = combined_csv
            output_files.append(combined_csv)
        else:
            logger.error(f"Failed to combine CSV files")
    
    # Analyze CSV output if requested
    if analyze and output_files:
        logger.info("Analyzing CSV output...")
        
        # Create analysis directory
        analysis_dir = os.path.join(output_dir, "analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Find combined CSV if it exists
        if combined_output:
            # Analyze combined CSV
            logger.info(f"Analyzing combined CSV: {combined_output}")
            analyze_combined_csv(combined_output, debug, analysis_dir)
        else:
            # Analyze individual CSV files
            for csv_file in output_files:
                logger.info(f"Analyzing CSV: {csv_file}")
                analyze_csv_transactions(csv_file, debug, analysis_dir)
        
        logger.info(f"Analysis results saved to {analysis_dir}")
    
    return successful > 0, output_files

def analyze_extraction_issues(csv_analysis_dir, structure_analysis_dir, output_dir):
    """
    Analyze extraction issues by correlating PDF structure and CSV quality
    
    Args:
        csv_analysis_dir (str): Directory containing CSV analysis results
        structure_analysis_dir (str): Directory containing PDF structure analysis results
        output_dir (str): Directory to save analysis results
    """
    try:
        # Run the analyze_extraction_issues.py script
        cmd = [
            sys.executable, 
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "analyze_extraction_issues.py"),
            "--csv-analysis", csv_analysis_dir,
            "--structure-analysis", structure_analysis_dir,
            "--output-dir", output_dir
        ]
        
        subprocess.run(cmd, check=True)
        logger.info(f"Extraction issues analysis complete. Results saved to {output_dir}")
        return True
    except Exception as e:
        logger.error(f"Error analyzing extraction issues: {e}")
        return False

def menu_extract_single_pdf():
    """Menu for extracting transactions from a single PDF"""
    print_section_header("EXTRACT TRANSACTIONS FROM SINGLE PDF")
    
    # Get PDF path
    pdf_path = get_input_path("Enter path to PDF file: ", 'file')
    
    # Get output directory
    output_dir = get_input_path("Enter output directory for results: ", 'dir')
    
    # Get extraction method
    print("\nSelect extraction method:")
    print("1. Regular (Default)")
    print("2. Lattice")
    print("3. Strict Lattice")
    print("4. Camelot (Recommended for single-column tables)")
    
    method_choice = input("Enter your choice (1-4, default: 1): ")
    extraction_methods = {
        '1': 'regular',
        '2': 'lattice',
        '3': 'strict_lattice',
        '4': 'camelot'
    }
    extraction_method = extraction_methods.get(method_choice, 'regular')
    
    # Get debug option
    debug = input("Enable debug mode? (y/n, default: n): ").lower() == 'y'
    analyze = input("Analyze CSV output? (y/n): ").lower() == 'y'
    
    # Process PDF
    print(f"\nProcessing PDF file using {extraction_method} extraction method...")
    success, output_file = process_single_pdf(pdf_path, output_dir, debug, extraction_method)
    
    if success:
        print(f"\n✅ Successfully processed PDF: {pdf_path}")
        print(f"Output CSV: {output_file}")
        
        if analyze and output_file:
            print("\nAnalyzing CSV output...")
            analysis_dir = os.path.join(output_dir, "analysis")
            os.makedirs(analysis_dir, exist_ok=True)
            results = analyze_csv_transactions(output_file, debug, analysis_dir)
            
            print(f"\n📊 Analysis Summary:")
            print(f"- Found {len(results.get('issues_found', []))} issues and {len(results.get('warnings', []))} warnings")
            
            if results.get('issues_found'):
                print("\nIssues:")
                for issue in results['issues_found']:
                    print(f"  - {issue}")
            
            print(f"\nAnalysis results saved to {analysis_dir}")
    else:
        print(f"\n❌ Failed to process PDF: {pdf_path}")
    
    input("\nPress Enter to return to main menu...")

def menu_batch_process():
    """Menu for batch processing multiple PDFs"""
    print_section_header("BATCH PROCESS MULTIPLE PDFs")
    
    # Get input directory
    input_dir = get_input_path("Enter directory containing PDF files: ", 'dir')
    
    # Get output directory
    output_dir = get_input_path("Enter output directory for results: ", 'dir')
    
    # Get extraction method
    print("\nSelect extraction method:")
    print("1. Regular (Default)")
    print("2. Lattice")
    print("3. Strict Lattice")
    print("4. Camelot (Recommended for single-column tables)")
    
    method_choice = input("Enter your choice (1-4, default: 1): ")
    extraction_methods = {
        '1': 'regular',
        '2': 'lattice',
        '3': 'strict_lattice',
        '4': 'camelot'
    }
    extraction_method = extraction_methods.get(method_choice, 'regular')
    
    # Get processing options
    max_workers = input("Enter maximum number of worker processes (default: 4): ")
    max_workers = int(max_workers) if max_workers.isdigit() else 4
    
    debug = input("Enable debug mode? (y/n, default: n): ").lower() == 'y'
    combine = input("Combine all CSVs into one file? (y/n, default: n): ").lower() == 'y'
    analyze = input("Analyze CSV output after processing? (y/n, default: n): ").lower() == 'y'
    
    # Add cashbook option (only available for camelot extraction)
    cashbook = False
    if extraction_method == 'camelot':
        cashbook = input("Generate cashbook after processing? (y/n, default: n): ").lower() == 'y'
        
    pattern = input("Filter PDFs by filename pattern (leave empty for all PDFs): ").strip()
    
    # Get fiscal year dates if cashbook is enabled
    fiscal_start = '2024-03-01'
    fiscal_end = '2025-02-28'
    if cashbook:
        fiscal_start = input(f"Enter start date of fiscal year (YYYY-MM-DD, default: {fiscal_start}): ").strip() or fiscal_start
        fiscal_end = input(f"Enter end date of fiscal year (YYYY-MM-DD, default: {fiscal_end}): ").strip() or fiscal_end
    
    # Build command arguments
    batch_script_path = os.path.join(os.path.dirname(__file__), '../modules/batch_processor.py')
    cmd = [sys.executable, batch_script_path, '--input', input_dir, '--output', output_dir]
    
    # Add extraction method
    cmd.extend(['--method', extraction_method])
    
    if debug:
        cmd.append('--debug')
    if combine:
        cmd.append('--combine')
    if analyze:
        cmd.append('--analyze')
    if cashbook:
        cmd.append('--cashbook')
        cmd.extend(['--fiscal-start', fiscal_start])
        cmd.extend(['--fiscal-end', fiscal_end])
    if pattern:
        cmd.extend(['--pattern', pattern])
    
    # Set number of workers
    cmd.extend(['--workers', str(max_workers)])
    
    print("\nProcessing PDF files...")
    logger.info(f"Running batch processing with command: {' '.join(cmd)}")
    
    try:
        # Run the batch processing script
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        if process.returncode == 0:
            # Print output
            print(process.stdout)
            logger.info("Batch processing completed successfully")
            
            # Check if no PDFs were found
            if "No PDF files found" in process.stdout:
                print("\nNo PDF files found in the specified directory. Please check the path and try again.")
                return
                
            # Check for combined CSV - use camelot directory if camelot extraction was used
            if extraction_method == 'camelot':
                combined_csv = os.path.join(output_dir, "camelot", "combined_transactions.csv")
            else:
                combined_csv = os.path.join(output_dir, "combined_transactions.csv")
                
            if combine and os.path.exists(combined_csv):
                print(f"\n✅ Combined CSV saved to: {combined_csv}")
                
                # Ask if user wants to fix balance issues
                fix_balance = input("\nDo you want to fix balance issues in the combined CSV? (y/n): ").lower() == 'y'
                if fix_balance:
                    print("\nFixing balance issues...")
                    fix_script_path = os.path.join(os.path.dirname(__file__), 'fix_balance_issues.py')
                    fix_cmd = [sys.executable, fix_script_path, combined_csv]
                    if debug:
                        fix_cmd.append('--debug')
                    
                    fix_process = subprocess.run(fix_cmd, capture_output=True, text=True)
                    if fix_process.returncode == 0:
                        print(fix_process.stdout)
                        print(f"\n✅ Fixed CSV saved to: {os.path.dirname(combined_csv)}/combined_transactions_fixed.csv")
                    else:
                        print(f"\n❌ Failed to fix balance issues: {fix_process.stderr}")
            
            # Check for analysis results
            analysis_dir = os.path.join(output_dir, "analysis")
            if analyze and os.path.exists(analysis_dir):
                print(f"\nAnalysis results saved to {analysis_dir}")
        else:
            print(f"\n❌ Failed to process PDF files:\n{process.stderr}")
            logger.error(f"Batch processing failed with error: {process.stderr}")
    except Exception as e:
        print(f"\n❌ Error running batch process: {e}")
        logger.error(f"Error running batch process: {e}", exc_info=True)
    
    input("\nPress Enter to return to main menu...")

def menu_analyze_csv():
    """Menu for analyzing CSV files"""
    print_section_header("ANALYZE CSV FILES")
    
    input_path = get_input_path("Enter CSV file or directory containing CSV files: ")
    output_dir = get_input_path("Enter output directory for analysis results (or press Enter for auto-generated): ") or None
    
    is_combined = input("Is this a combined CSV with Source column? (y/n): ").lower() == 'y'
    debug = input("Enable debug mode? (y/n): ").lower() == 'y'
    visualize = input("Generate visualizations? (y/n): ").lower() == 'y'
    
    if not output_dir:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f"csv_analysis_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
    
    print("\nAnalyzing CSV files...")
    
    try:
        if os.path.isdir(input_path):
            # Find all CSV files in directory
            csv_files = []
            for root, _, files in os.walk(input_path):
                for file in files:
                    if file.lower().endswith('.csv'):
                        csv_files.append(os.path.join(root, file))
            
            if not csv_files:
                print(f"No CSV files found in {input_path}")
                input("\nPress Enter to return to main menu...")
                return
            
            print(f"Found {len(csv_files)} CSV files to analyze")
            
            for csv_file in csv_files:
                print(f"Analyzing {csv_file}...")
                if is_combined and os.path.basename(csv_file) == "combined_transactions.csv":
                    analyze_combined_csv(csv_file, debug, output_dir)
                else:
                    analyze_csv_transactions(csv_file, debug, output_dir)
        else:
            # Analyze single CSV file
            if is_combined:
                analyze_combined_csv(input_path, debug, output_dir)
            else:
                analyze_csv_transactions(input_path, debug, output_dir)
        
        print(f"\n✅ Analysis complete. Results saved to {output_dir}")
    except Exception as e:
        print(f"\n❌ Error analyzing CSV: {e}")
    
    input("\nPress Enter to return to main menu...")

def menu_analyze_pdf_structure():
    """Menu for analyzing PDF structure"""
    print_section_header("ANALYZE PDF STRUCTURE")
    
    pdf_path = get_input_path("Enter path to PDF file: ", 'file')
    output_dir = get_input_path("Enter output directory (or press Enter for current directory): ") or os.getcwd()
    
    print("\nAnalyzing PDF structure...")
    
    try:
        structure_data = analyze_pdf_structure(pdf_path)
        
        # Save structure analysis
        base_name = os.path.basename(pdf_path).replace('.pdf', '')
        output_file = os.path.join(output_dir, f"{base_name}_structure_analysis.json")
        
        with open(output_file, 'w') as f:
            json.dump(structure_data, f, indent=2, default=str)
        
        print(f"\n✅ Structure analysis complete. Results saved to {output_file}")
        
        # Print summary
        print("\nSummary:")
        print(f"- Pages: {len(structure_data.get('pages', []))}")
        print(f"- Tables: {sum(len(page.get('tables', [])) for page in structure_data.get('pages', []))}")
        
        # Print table details
        print("\nTable Details:")
        for i, page in enumerate(structure_data.get('pages', [])):
            for j, table in enumerate(page.get('tables', [])):
                print(f"  Page {i+1}, Table {j+1}:")
                if 'column_mapping' in table:
                    print(f"    Column Mapping: {table['column_mapping']}")
                if 'rows' in table:
                    print(f"    Rows: {len(table['rows'])}")
    except Exception as e:
        print(f"\n❌ Error analyzing PDF structure: {e}")
    
    input("\nPress Enter to return to main menu...")

def menu_correlate_issues():
    """Menu for correlating PDF structure and CSV quality issues"""
    print_section_header("CORRELATE EXTRACTION ISSUES")
    
    csv_analysis_dir = get_input_path("Enter directory containing CSV analysis results: ", 'dir')
    structure_analysis_dir = get_input_path("Enter directory containing PDF structure analysis results: ", 'dir')
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = get_input_path(f"Enter output directory (or press Enter for auto-generated): ") or f"extraction_analysis_{timestamp}"
    
    print("\nAnalyzing extraction issues...")
    
    success = analyze_extraction_issues(csv_analysis_dir, structure_analysis_dir, output_dir)
    
    if success:
        print(f"\n✅ Extraction issues analysis complete. Results saved to {output_dir}")
    else:
        print(f"\n❌ Failed to analyze extraction issues")
    
    input("\nPress Enter to return to main menu...")

def menu_page_analyzer():
    """Launch the Page Analyzer menu"""
    menu_analyze_pdf_pages()


def menu_settings():
    """Menu for configuring settings"""
    print_section_header("SETTINGS")
    
    print("1. Configure logging")
    print("2. Configure default directories")
    print("3. Back to main menu")
    
    choice = input("\nEnter your choice (1-3): ")
    
    if choice == '1':
        log_level = input("Enter log level (INFO, DEBUG, WARNING, ERROR): ").upper()
        if log_level in ('INFO', 'DEBUG', 'WARNING', 'ERROR'):
            logger.setLevel(getattr(logging, log_level))
            print(f"Log level set to {log_level}")
        else:
            print("Invalid log level")
    elif choice == '2':
        # This would typically save settings to a config file
        print("Feature not yet implemented")
    
    input("\nPress Enter to return to main menu...")

def display_menu():
    """Display the main menu"""
    print("\n" + "=" * 80)
    print(" " * 30 + "PDF STATEMENT PROCESSOR")
    print("=" * 80)
    print("\n1. Extract Transactions from Single PDF")
    print("2. Batch Process Multiple PDFs")
    print("3. Analyze CSV Files")
    print("4. Generate Cashbook & Trial Balance")
    print("5. Analyze PDF Structure")
    print("6. Correlate Extraction Issues")
    print("7. Page Analyzer (Compare Pages)")
    print("8. Settings")
    print("0. Exit")

def menu_process_cashbook():
    """Menu for generating cashbook and trial balance"""
    print_section_header("Generate Cashbook and Trial Balance")
    
    # Get input directory
    input_dir = get_input_path("Enter directory containing CSV files: ", file_type='dir')
    
    # Get output path
    default_output = os.path.join(os.path.dirname(input_dir), "cashbook.xlsx")
    output_path = input(f"Enter output Excel file path (default: {default_output}): ").strip() or default_output
    
    # Get fiscal year date range
    default_start = '2024-03-01'
    default_end = '2025-02-28'
    
    start_date = input(f"Enter start date of fiscal year (YYYY-MM-DD, default: {default_start}): ").strip() or default_start
    end_date = input(f"Enter end date of fiscal year (YYYY-MM-DD, default: {default_end}): ").strip() or default_end
    
    # Balance verification options
    print("\nBalance Verification Options:")
    print("1. Generate cashbook without balance verification")
    print("2. Verify balances against expected values (without forcing adjustments)")
    
    balance_option = input("\nSelect option (1-2, default: 1): ").strip() or "1"
    
    expected_opening_balance = None
    expected_closing_balance = None
    
    if balance_option == "2":
        print("\nEnter expected balance values from bank statements for verification:")
        try:
            expected_opening_balance = float(input("Expected opening balance at start of fiscal year: ").strip().replace(',', ''))
            expected_closing_balance = float(input("Expected closing balance at end of fiscal year: ").strip().replace(',', ''))
            print(f"\nWill verify balances against: Opening={expected_opening_balance}, Closing={expected_closing_balance}")
            print("Note: Transactions will be properly sorted and balances calculated naturally.")
            print("      Any differences between calculated and expected balances will be reported.")
        except ValueError:
            print("\nInvalid balance values entered. Will proceed without balance verification.")
            expected_opening_balance = None
            expected_closing_balance = None
    
    # Debug mode
    debug = input("\nEnable debug mode? (y/n, default: n): ").lower() == 'y'
    
    print("\nProcessing cashbook...")
    
    try:
        # Import the simple_cashbook module
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        from modules.simple_cashbook import process_simple_cashbook
        
        # Check if we have a fixed CSV file first
        fixed_csv_path = os.path.join(input_dir, 'combined_transactions_fixed.csv')
        regular_csv_path = os.path.join(input_dir, 'combined_transactions.csv')
        
        # Use the fixed CSV if it exists, otherwise use the regular combined CSV
        if os.path.exists(fixed_csv_path):
            input_csv_path = fixed_csv_path
            print(f"Using fixed CSV file: {fixed_csv_path}")
        elif os.path.exists(regular_csv_path):
            input_csv_path = regular_csv_path
            print(f"Using combined CSV file: {regular_csv_path}")
        else:
            print(f"Error: No combined CSV file found in {input_dir}")
            return False
        
        # Process the cashbook
        result = process_simple_cashbook(
            input_csv=input_csv_path, 
            output_file=output_path, 
            fiscal_start_month=fiscal_start_month,
            fiscal_start_day=fiscal_start_day
        )
        
        if result['success']:
            print(f"\n✅ Cashbook generated successfully with {result.get('transaction_count', 0)} transactions")
            print(f"Output file: {result['output_path']}")
            
            if expected_opening_balance is not None and expected_closing_balance is not None:
                print("\nBalance Verification Results:")
                opening_balance = result.get('opening_balance')
                closing_balance = result.get('closing_balance')
                opening_diff = result.get('opening_balance_difference')
                closing_diff = result.get('closing_balance_difference')
                
                print(f"Actual opening balance: {opening_balance:.2f}")
                print(f"Expected opening balance: {expected_opening_balance:.2f}")
                if opening_diff is not None:
                    print(f"Opening balance difference: {opening_diff:.2f}")
                    if abs(opening_diff) < 0.01:
                        print("✓ Opening balance matches expected value")
                    else:
                        print("⚠ Opening balance differs from expected value")
                
                print(f"\nActual closing balance: {closing_balance:.2f}")
                print(f"Expected closing balance: {expected_closing_balance:.2f}")
                if closing_diff is not None:
                    print(f"Closing balance difference: {closing_diff:.2f}")
                    if abs(closing_diff) < 0.01:
                        print("✓ Closing balance matches expected value")
                    else:
                        print("⚠ Closing balance differs from expected value")
                        print("Note: No adjustment was made as requested. Transactions were properly sorted and balances calculated naturally.")
        else:
            print(f"\n❌ Failed to generate cashbook: {result.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"\n❌ Error processing cashbook: {e}")
        if debug:
            import traceback
            traceback.print_exc()
    
    input("\nPress Enter to return to main menu...")

def main_menu():
    """Display main menu and handle user choices"""
    while True:
        display_menu()
        
        choice = input("\nEnter your choice (0-8): ")
        
        if choice == '1':
            menu_extract_single_pdf()
        elif choice == '2':
            menu_batch_process()
        elif choice == '3':
            menu_analyze_csv()
        elif choice == '4':
            menu_process_cashbook()
        elif choice == '5':
            menu_analyze_pdf_structure()
        elif choice == '6':
            menu_correlate_issues()
        elif choice == '7':
            menu_page_analyzer()
        elif choice == '8':
            menu_settings()
        elif choice == '0':
            print("\nExiting PDF Statement Processor. Goodbye!")
            break
        else:
            print("\nInvalid choice. Please try again.")
            input("Press Enter to continue...")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='PDF Statement Processor - Integrated Menu System')
    parser.add_argument('--output-dir', '-o', help='Output directory for results')
    parser.add_argument('--batch', '-b', help='Batch process PDFs in specified directory')
    parser.add_argument('--analyze', '-a', help='Analyze CSV file or directory')
    parser.add_argument('--combine', action='store_true', help='Combine all CSV files into one')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--parallel', action='store_true', help='Process PDFs in parallel')
    parser.add_argument('--max-workers', type=int, default=None, help='Maximum number of parallel workers')
    parser.add_argument('--fiscal-year', action='store_true', help='Sort by fiscal year instead of calendar year')
    parser.add_argument('--fiscal-start-month', type=int, default=3, help='Month when fiscal year starts (1-12)')
    parser.add_argument('--fiscal-start-day', type=int, default=1, help='Day when fiscal year starts (1-31)')
    
    args = parser.parse_args()
    
    # Set up logging
    output_dir = args.output_dir or os.getcwd()
    os.makedirs(output_dir, exist_ok=True)
    setup_logging(output_dir)
    
    # Handle command line arguments for non-interactive mode
    if args.batch:
        batch_process_pdfs(args.batch, output_dir, combine=args.combine, analyze=args.analyze, 
                          debug=args.debug, parallel=args.parallel, max_workers=args.max_workers,
                          fiscal_year_sorting=args.fiscal_year, fiscal_start_month=args.fiscal_start_month,
                          fiscal_start_day=args.fiscal_start_day)
    elif args.analyze:
        if os.path.isdir(args.analyze):
            for root, _, files in os.walk(args.analyze):
                for file in files:
                    if file.lower().endswith('.csv'):
                        csv_file = os.path.join(root, file)
                        if file == "combined_transactions.csv":
                            analyze_combined_csv(csv_file, False, output_dir)
                        else:
                            analyze_csv_transactions(csv_file, False, output_dir)
        else:
            analyze_csv_transactions(args.analyze, False, output_dir)
        return
    
    # Start interactive menu
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted. Exiting...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"\nAn unexpected error occurred: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()
