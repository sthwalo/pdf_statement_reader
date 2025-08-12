#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Lattice Extraction CLI

Command-line interface for the specialized lattice extraction module.
This tool focuses specifically on extracting transaction data using tabula's lattice mode.
"""

import os
import sys
import json
import logging
import argparse

# Add the project root to the path to ensure modules can be found
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import after path setup
from modules.lattice_extractor import LatticeExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Import pandas after path setup
import pandas as pd

def main():
    """Main function for the lattice extraction CLI"""
    parser = argparse.ArgumentParser(
        description='Extract transactions from PDF statements using lattice mode',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract transactions from a single PDF
  python lattice_extract.py /path/to/statement.pdf --output output/transactions.csv
  
  # Extract with debug info
  python lattice_extract.py /path/to/statement.pdf --debug --debug-info output/debug.json
  
  # Compare with regular extraction
  python lattice_extract.py /path/to/statement.pdf --compare
        """
    )
    
    parser.add_argument('pdf_path', help='Path to the PDF statement file')
    parser.add_argument('--output', help='Path to save extracted transactions CSV')
    parser.add_argument('--password', help='Password for encrypted PDF')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with detailed logging')
    parser.add_argument('--debug-info', help='Path to save debug information as JSON')
    parser.add_argument('--compare', action='store_true', 
                        help='Compare with regular extraction (uses both modules)')
    
    args = parser.parse_args()
    
    # Validate PDF path
    if not os.path.isfile(args.pdf_path):
        logger.error(f"PDF file not found: {args.pdf_path}")
        return 1
    
    # Set default output path if not provided
    if not args.output:
        pdf_name = os.path.splitext(os.path.basename(args.pdf_path))[0]
        args.output = f"data/output/{pdf_name}_lattice_transactions.csv"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Create lattice extractor
    extractor = LatticeExtractor(debug=args.debug)
    
    # Process PDF
    logger.info(f"Processing PDF: {args.pdf_path}")
    transactions = extractor.process_pdf(args.pdf_path, args.password)
    
    # Export to CSV
    if transactions:
        extractor.export_to_csv(transactions, args.output)
        logger.info(f"✅ Extracted {len(transactions)} transactions using lattice mode")
        logger.info(f"✅ Saved to: {args.output}")
    else:
        logger.error("❌ No transactions extracted")
    
    # Save debug info if requested
    if args.debug and args.debug_info:
        debug_info = extractor.get_debug_info()
        os.makedirs(os.path.dirname(args.debug_info), exist_ok=True)
        with open(args.debug_info, 'w') as f:
            json.dump(debug_info, f, indent=2)
        logger.info(f"✅ Debug info saved to: {args.debug_info}")
    
    # Compare with regular extraction if requested
    if args.compare:
        logger.info("Comparing with regular extraction...")
        
        # Import regular extraction modules
        from modules.table_extractor import extract_tables_from_pdf
        from modules.transaction_extractor import extract_transactions_from_pdf
        
        # Extract using regular method
        tables = extract_tables_from_pdf(args.pdf_path, args.password)
        regular_transactions = extract_transactions_from_pdf(tables)
        
        # Save regular transactions
        regular_output = args.output.replace('_lattice_transactions.csv', '_regular_transactions.csv')
        df = pd.DataFrame(regular_transactions)
        df.to_csv(regular_output, index=False)
        
        # Compare results
        logger.info(f"Regular extraction: {len(regular_transactions)} transactions")
        logger.info(f"Lattice extraction: {len(transactions)} transactions")
        logger.info(f"Difference: {len(regular_transactions) - len(transactions)} transactions")
        
        # Save comparison report
        comparison_output = args.output.replace('_lattice_transactions.csv', '_comparison.txt')
        with open(comparison_output, 'w') as f:
            f.write(f"PDF: {args.pdf_path}\n")
            f.write(f"Regular extraction: {len(regular_transactions)} transactions\n")
            f.write(f"Lattice extraction: {len(transactions)} transactions\n")
            f.write(f"Difference: {len(regular_transactions) - len(transactions)} transactions\n\n")
            
            # Compare fields
            if regular_transactions and transactions:
                regular_fields = set()
                lattice_fields = set()
                
                for tx in regular_transactions:
                    for field, value in tx.items():
                        if value:
                            regular_fields.add(field)
                
                for tx in transactions:
                    for field, value in tx.items():
                        if value:
                            lattice_fields.add(field)
                
                f.write("Fields in regular extraction: " + ", ".join(sorted(regular_fields)) + "\n")
                f.write("Fields in lattice extraction: " + ", ".join(sorted(lattice_fields)) + "\n")
                f.write("Fields only in regular: " + ", ".join(sorted(regular_fields - lattice_fields)) + "\n")
                f.write("Fields only in lattice: " + ", ".join(sorted(lattice_fields - regular_fields)) + "\n")
        
        logger.info(f"✅ Comparison saved to: {comparison_output}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
