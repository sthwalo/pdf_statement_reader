#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PDF Statement Reader - Entry Point

This script serves as the entry point for the PDF Statement Reader application.
It runs the main processor from the core directory or provides direct access to CSV combining functionality.
"""

import os
import sys
import argparse

def main():
    """Main entry point function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='PDF Statement Reader')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Add combine command
    combine_parser = subparsers.add_parser('combine', help='Combine CSV files')
    combine_parser.add_argument('--files', nargs='+', help='List of CSV files to combine')
    combine_parser.add_argument('--input-dir', help='Directory containing CSV files to combine')
    combine_parser.add_argument('--output', required=True, help='Output CSV file')
    combine_parser.add_argument('--fiscal-year', action='store_true', help='Sort by fiscal year')
    combine_parser.add_argument('--fiscal-start-month', type=int, default=3, help='Month when fiscal year starts (1-12)')
    combine_parser.add_argument('--fiscal-start-day', type=int, default=1, help='Day when fiscal year starts (1-31)')
    combine_parser.add_argument('--analyze', action='store_true', help='Analyze the combined CSV after combining')
    
    args = parser.parse_args()
    
    # Get the absolute path to the core directory
    core_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'core')
    
    # Add the project root to the path to ensure modules can be found
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # Handle combine command
    if args.command == 'combine':
        from modules.combine_csvs import combine_csv_files, analyze_combined_csv
        
        # Check if we have files or input directory
        if not args.files and not args.input_dir:
            print("Error: Either --files or --input-dir must be specified")
            sys.exit(1)
        
        # If files are specified, use them directly
        if args.files:
            # Create a temporary directory to store the files
            import tempfile
            import shutil
            temp_dir = tempfile.mkdtemp()
            
            try:
                # Copy files to temp directory
                for file in args.files:
                    if os.path.exists(file):
                        shutil.copy(file, os.path.join(temp_dir, os.path.basename(file)))
                    else:
                        print(f"Warning: File not found: {file}")
                
                # Combine files
                success = combine_csv_files(
                    temp_dir, 
                    args.output, 
                    fiscal_year_sorting=args.fiscal_year,
                    fiscal_start_month=args.fiscal_start_month,
                    fiscal_start_day=args.fiscal_start_day
                )
            finally:
                # Clean up temp directory
                shutil.rmtree(temp_dir)
        else:
            # Use input directory
            success = combine_csv_files(
                args.input_dir, 
                args.output, 
                fiscal_year_sorting=args.fiscal_year,
                fiscal_start_month=args.fiscal_start_month,
                fiscal_start_day=args.fiscal_start_day
            )
        
        # Analyze if requested
        if success and args.analyze:
            analyze_combined_csv(args.output)
            
        sys.exit(0 if success else 1)
    else:
        # Import the main processor
        sys.path.append(core_dir)
        from pdf_statement_processor import main as processor_main
        
        # Run the main processor
        processor_main()

if __name__ == "__main__":
    # If no arguments were passed, run the main function
    if len(sys.argv) == 1:
        # Get the path to the main processor
        processor_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            'core', 
            'pdf_statement_processor.py'
        )
        
        # Execute the main processor with no arguments
        os.execv(sys.executable, [sys.executable, processor_path])
    else:
        main()
