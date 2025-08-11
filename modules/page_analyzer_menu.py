#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Page Analyzer Menu Integration

Provides menu functions for integrating the Page Analyzer module into the main application.
"""

import os
import sys
import json
import logging
from typing import List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Import page analyzer module
from modules.page_analyzer import PageAnalyzer, analyze_pdf_pages


def menu_analyze_pdf_pages():
    """
    Interactive menu for analyzing PDF pages.
    """
    print("\n" + "=" * 80)
    print(" " * 30 + "PAGE ANALYZER")
    print("=" * 80)
    
    print("\n1. Analyze single PDF")
    print("2. Compare specific pages (e.g., Page 16 vs. 17)")
    print("3. Batch analyze folder")
    print("0. Return to main menu")
    
    choice = input("\nEnter your choice (0-3): ")
    
    if choice == "1":
        analyze_single_pdf()
    elif choice == "2":
        compare_specific_pages()
    elif choice == "3":
        batch_analyze_folder()
    elif choice == "0":
        return
    else:
        print("Invalid choice. Please try again.")
        menu_analyze_pdf_pages()


def analyze_single_pdf():
    """
    Analyze all pages in a single PDF.
    """
    pdf_path = input("\nEnter the path to the PDF file: ")
    
    if not os.path.exists(pdf_path):
        print(f"Error: File not found: {pdf_path}")
        return
    
    # Ask if user wants to analyze specific pages
    specific_pages = input("\nAnalyze specific pages? (y/n): ").lower()
    
    pages = None
    if specific_pages == 'y':
        pages_input = input("Enter page numbers separated by commas (e.g., 1,5,10): ")
        try:
            pages = [int(p) for p in pages_input.split(',')]
        except ValueError:
            print("Invalid page numbers. Using all pages.")
    
    # Ask for output file
    output_file = input("\nSave analysis to file? (Enter path or leave empty for console output): ")
    
    print("\nAnalyzing PDF pages. This may take a moment...")
    
    try:
        # Analyze PDF
        report = analyze_pdf_pages(pdf_path, pages)
        
        # Save or display results
        if output_file:
            # Check if output_file is a directory
            if os.path.isdir(output_file):
                # Create a filename based on the PDF name
                pdf_name = os.path.basename(pdf_path).split('.')[0]
                output_file = os.path.join(output_file, f"{pdf_name}_page_analysis.json")
            
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\nAnalysis report saved to {output_file}")
        else:
            display_analysis_report(report)
    
    except Exception as e:
        logger.error(f"Error analyzing PDF: {e}")
        print(f"\nError analyzing PDF: {e}")


def compare_specific_pages():
    """
    Compare specific pages in a PDF.
    """
    pdf_path = input("\nEnter the path to the PDF file: ")
    
    if not os.path.exists(pdf_path):
        print(f"Error: File not found: {pdf_path}")
        return
    
    # Get page pairs to compare
    pairs_input = input("\nEnter page pairs to compare (e.g., 16-17,20-21): ")
    
    try:
        compare_pages = []
        for pair in pairs_input.split(','):
            a, b = map(int, pair.split('-'))
            compare_pages.append((a, b))
    except (ValueError, IndexError):
        print("Invalid page pairs. Format should be like '16-17,20-21'.")
        return
    
    # Ask for output file
    output_file = input("\nSave analysis to file? (Enter path or leave empty for console output): ")
    
    print("\nComparing PDF pages. This may take a moment...")
    
    try:
        # Get pages to analyze (all pages in the pairs)
        pages = list(set([p for pair in compare_pages for p in pair]))
        
        # Analyze PDF
        report = analyze_pdf_pages(pdf_path, pages, compare_pages)
        
        # Save or display results
        if output_file:
            # Check if output_file is a directory
            if os.path.isdir(output_file):
                # Create a filename based on the PDF name
                pdf_name = os.path.basename(pdf_path).split('.')[0]
                output_file = os.path.join(output_file, f"{pdf_name}_page_comparison.json")
            
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\nAnalysis report saved to {output_file}")
        else:
            display_comparison_report(report)
    
    except Exception as e:
        logger.error(f"Error comparing PDF pages: {e}")
        print(f"\nError comparing PDF pages: {e}")


def batch_analyze_folder():
    """
    Analyze all PDFs in a folder.
    """
    folder_path = input("\nEnter the path to the folder containing PDFs: ")
    
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        print(f"Error: Folder not found: {folder_path}")
        return
    
    # Get PDF files in the folder
    pdf_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith('.pdf') and os.path.isfile(os.path.join(folder_path, f))
    ]
    
    if not pdf_files:
        print(f"No PDF files found in {folder_path}")
        return
    
    print(f"\nFound {len(pdf_files)} PDF files.")
    
    # Ask for output directory
    output_dir = input("\nSave analyses to directory? (Enter path or leave empty for console output): ")
    
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except Exception as e:
            print(f"Error creating output directory: {e}")
            return
    
    # Process each PDF
    for i, pdf_path in enumerate(pdf_files):
        print(f"\nProcessing {i+1}/{len(pdf_files)}: {os.path.basename(pdf_path)}")
        
        try:
            # Analyze PDF
            report = analyze_pdf_pages(pdf_path)
            
            # Save or display results
            if output_dir:
                output_file = os.path.join(output_dir, f"{os.path.basename(pdf_path).split('.')[0]}_analysis.json")
                with open(output_file, 'w') as f:
                    json.dump(report, f, indent=2)
                print(f"  Analysis saved to {output_file}")
            else:
                print(f"\nAnalysis for {os.path.basename(pdf_path)}:")
                print(f"  Total pages: {report['total_pages']}")
                print(f"  Successful extraction pages: {len(report['successful_extraction_pages'])}")
                print(f"  Failed extraction pages: {len(report['failed_extraction_pages'])}")
                
                if report['failed_extraction_pages']:
                    print(f"  Pages with extraction issues: {report['failed_extraction_pages']}")
        
        except Exception as e:
            logger.error(f"Error analyzing {pdf_path}: {e}")
            print(f"  Error analyzing PDF: {e}")
    
    print("\nBatch analysis complete.")


def display_analysis_report(report):
    """
    Display an analysis report in the console.
    
    Args:
        report (dict): Analysis report from analyze_pdf_pages
    """
    print("\n" + "=" * 80)
    print(f" PDF Analysis Summary: {os.path.basename(report['pdf_path'])}")
    print("=" * 80)
    
    print(f"\nTotal pages: {report['total_pages']}")
    print(f"Analyzed pages: {report['analyzed_pages']}")
    print(f"Successful extraction pages: {report['successful_extraction_pages']}")
    print(f"Failed extraction pages: {report['failed_extraction_pages']}")
    
    print("\nPage Details:")
    for page_data in report['page_details']:
        page_num = page_data['page']
        success = page_data['extraction_success']
        
        print(f"\n  Page {page_num}: {'Success' if success else 'Failed'}")
        
        # Show table extraction results
        stream_tables = len(page_data['tables']['tabula_stream'])
        lattice_tables = len(page_data['tables']['tabula_lattice'])
        
        print(f"    Tables (Stream mode): {stream_tables}")
        print(f"    Tables (Lattice mode): {lattice_tables}")
        
        # Show text extraction results if available
        if 'text_blocks' in page_data:
            print(f"    Text blocks: {len(page_data['text_blocks'])}")
        
        # Show layout fingerprint if available
        if page_data.get('layout_fingerprint'):
            fp = page_data['layout_fingerprint']
            print(f"    Text density: {fp.get('text_density', 0):.4f}")
            print(f"    Text columns: {fp.get('text_columns', 0)}")
            print(f"    Has tables: {fp.get('has_tables', False)}")
            print(f"    Has lines: {fp.get('has_lines', False)}")
    
    print("\nGlobal Recommendations:")
    for rec in report['global_recommendations']:
        print(f"  - {rec}")


def display_comparison_report(report):
    """
    Display a comparison report in the console.
    
    Args:
        report (dict): Analysis report from analyze_pdf_pages with comparisons
    """
    print("\n" + "=" * 80)
    print(f" PDF Comparison Summary: {os.path.basename(report['pdf_path'])}")
    print("=" * 80)
    
    if not report['comparisons']:
        print("\nNo page comparisons available.")
        return
    
    for key, comparison in report['comparisons'].items():
        page_a = comparison['page_a']
        page_b = comparison['page_b']
        
        print(f"\nComparison: Page {page_a} vs Page {page_b}")
        print("-" * 50)
        
        if comparison['differences']:
            print("\nDifferences:")
            for diff in comparison['differences']:
                print(f"  - {diff}")
        else:
            print("\nNo significant structural differences detected.")
        
        print("\nRecommendations:")
        if comparison['recommendations']:
            for rec in comparison['recommendations']:
                print(f"  - {rec}")
        else:
            print("  - No specific recommendations.")
    
    print("\nGlobal Recommendations:")
    for rec in report['global_recommendations']:
        print(f"  - {rec}")


if __name__ == "__main__":
    menu_analyze_pdf_pages()
