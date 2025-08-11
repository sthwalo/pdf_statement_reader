#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CSV Analyzer Module

This module provides tools for analyzing CSV transaction data to identify
potential issues, inconsistencies, and anomalies in the extracted data.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import re
import matplotlib.pyplot as plt
from collections import Counter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def analyze_csv_transactions(csv_path, debug=False, output_dir=None):
    """
    Analyze CSV transaction data to identify potential issues and anomalies
    
    Args:
        csv_path (str): Path to the CSV file
        debug (bool): Enable debug mode
        output_dir (str): Directory to save analysis results
        
    Returns:
        dict: Analysis results
        dict: Debug info if debug=True
    """
    debug_info = {
        'csv_path': csv_path,
        'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'issues_found': [],
        'warnings': [],
        'statistics': {}
    }
    
    try:
        # Read CSV file
        df = pd.read_csv(csv_path)
        
        # Basic statistics
        row_count = len(df)
        debug_info['statistics']['row_count'] = row_count
        
        # Check for expected columns
        expected_columns = ['Date', 'Details', 'Debits', 'Credits', 'Balance']
        missing_columns = [col for col in expected_columns if col not in df.columns]
        if missing_columns:
            issue = f"Missing expected columns: {', '.join(missing_columns)}"
            debug_info['issues_found'].append(issue)
            logger.warning(issue)
        
        # Column presence statistics
        column_presence = {col: df[col].notna().sum() for col in df.columns}
        column_presence_pct = {col: df[col].notna().mean() * 100 for col in df.columns}
        debug_info['statistics']['column_presence'] = column_presence
        debug_info['statistics']['column_presence_pct'] = column_presence_pct
        
        # Check for empty values in critical columns
        for col in ['Date', 'Balance']:
            if col in df.columns:
                empty_count = df[col].isna().sum()
                if empty_count > 0:
                    issue = f"Found {empty_count} empty values in {col} column ({empty_count/row_count*100:.2f}%)"
                    debug_info['issues_found'].append(issue)
                    logger.warning(issue)
        
        # Analyze date column
        if 'Date' in df.columns:
            # Convert to datetime
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            invalid_dates = df['Date'].isna().sum()
            if invalid_dates > 0:
                issue = f"Found {invalid_dates} invalid dates ({invalid_dates/row_count*100:.2f}%)"
                debug_info['issues_found'].append(issue)
                logger.warning(issue)
                
            # Check for date gaps
            if not df['Date'].isna().all():
                df_sorted = df.sort_values('Date')
                df_sorted['date_diff'] = df_sorted['Date'].diff().dt.days
                large_gaps = df_sorted[df_sorted['date_diff'] > 31]
                if not large_gaps.empty:
                    for _, row in large_gaps.iterrows():
                        warning = f"Large gap of {row['date_diff']} days before {row['Date'].strftime('%Y-%m-%d')}"
                        debug_info['warnings'].append(warning)
                        logger.warning(warning)
                
                # Date range
                date_range = (df['Date'].max() - df['Date'].min()).days
                debug_info['statistics']['date_range_days'] = date_range
                debug_info['statistics']['start_date'] = df['Date'].min().strftime('%Y-%m-%d')
                debug_info['statistics']['end_date'] = df['Date'].max().strftime('%Y-%m-%d')
        
        # Analyze balance column
        if 'Balance' in df.columns:
            # Convert to numeric
            df['Balance'] = pd.to_numeric(df['Balance'].astype(str).str.replace(',', ''), errors='coerce')
            invalid_balances = df['Balance'].isna().sum()
            if invalid_balances > 0:
                issue = f"Found {invalid_balances} invalid balance values ({invalid_balances/row_count*100:.2f}%)"
                debug_info['issues_found'].append(issue)
                logger.warning(issue)
                
            # Check for balance continuity
            if 'Date' in df.columns and not df['Date'].isna().all() and not df['Balance'].isna().all():
                df_sorted = df.sort_values('Date')
                df_sorted['balance_diff'] = df_sorted['Balance'].diff()
                
                # Check if debits and credits match balance changes
                if 'Debits' in df.columns and 'Credits' in df.columns:
                    df_sorted['Debits'] = pd.to_numeric(df_sorted['Debits'].astype(str).str.replace(',', ''), errors='coerce')
                    df_sorted['Credits'] = pd.to_numeric(df_sorted['Credits'].astype(str).str.replace(',', ''), errors='coerce')
                    
                    # Calculate expected balance change
                    df_sorted['expected_change'] = df_sorted['Credits'].fillna(0) - df_sorted['Debits'].fillna(0)
                    
                    # Compare with actual change
                    df_sorted['balance_mismatch'] = (df_sorted['balance_diff'] - df_sorted['expected_change']).abs()
                    mismatches = df_sorted[df_sorted['balance_mismatch'] > 0.01].dropna(subset=['balance_mismatch'])
                    
                    if not mismatches.empty:
                        issue = f"Found {len(mismatches)} balance mismatches where debits/credits don't match balance changes"
                        debug_info['issues_found'].append(issue)
                        logger.warning(issue)
                        
                        # Sample of mismatches
                        debug_info['balance_mismatches'] = mismatches[['Date', 'Details', 'Debits', 'Credits', 'Balance', 'balance_diff', 'expected_change', 'balance_mismatch']].head(10).to_dict('records')
        
        # Analyze transaction details
        if 'Details' in df.columns:
            # Check for truncated details
            truncated_pattern = re.compile(r'\.{3}$|…$')
            truncated_details = df['Details'].str.contains(truncated_pattern, na=False).sum()
            if truncated_details > 0:
                warning = f"Found {truncated_details} potentially truncated transaction details"
                debug_info['warnings'].append(warning)
                logger.warning(warning)
            
            # Analyze most common transaction types
            common_words = Counter()
            for details in df['Details'].dropna():
                words = details.split()
                if words:
                    common_words[words[0]] += 1
            
            debug_info['statistics']['common_transaction_types'] = dict(common_words.most_common(10))
        
        # Generate visualizations if output directory is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Transaction count by date
            if 'Date' in df.columns and not df['Date'].isna().all():
                plt.figure(figsize=(12, 6))
                df['Date'].dt.to_period('M').value_counts().sort_index().plot(kind='bar')
                plt.title('Transaction Count by Month')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'transaction_count_by_month.png'))
                plt.close()
                
                # Balance over time
                if 'Balance' in df.columns and not df['Balance'].isna().all():
                    plt.figure(figsize=(12, 6))
                    df_sorted = df.sort_values('Date')
                    plt.plot(df_sorted['Date'], df_sorted['Balance'])
                    plt.title('Balance Over Time')
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, 'balance_over_time.png'))
                    plt.close()
        
        # Save analysis results if output directory is provided
        if output_dir:
            base_name = os.path.basename(csv_path).replace('.csv', '')
            with open(os.path.join(output_dir, f"{base_name}_analysis.json"), 'w') as f:
                json.dump(debug_info, f, indent=2, default=str)
        
        logger.info(f"CSV analysis complete for {csv_path}")
        logger.info(f"Found {len(debug_info['issues_found'])} issues and {len(debug_info['warnings'])} warnings")
        
        return debug_info
        
    except Exception as e:
        error_msg = f"Error analyzing CSV: {e}"
        logger.error(error_msg, exc_info=True)
        debug_info['error'] = error_msg
        return debug_info

def analyze_multiple_csvs(csv_files, output_dir=None, debug=False):
    """
    Analyze multiple CSV files and compare results
    
    Args:
        csv_files (list): List of CSV file paths
        output_dir (str): Directory to save analysis results
        debug (bool): Enable debug mode
        
    Returns:
        dict: Comparative analysis results
    """
    if not csv_files:
        logger.warning("No CSV files provided for analysis")
        return {"error": "No CSV files provided"}
    
    results = {}
    combined_issues = []
    combined_warnings = []
    
    # Analyze each CSV file
    for csv_file in csv_files:
        logger.info(f"Analyzing {csv_file}")
        result = analyze_csv_transactions(csv_file, debug, output_dir)
        results[csv_file] = result
        
        combined_issues.extend([f"{csv_file}: {issue}" for issue in result.get('issues_found', [])])
        combined_warnings.extend([f"{csv_file}: {warning}" for warning in result.get('warnings', [])])
    
    # Comparative analysis
    comparative_analysis = {
        'files_analyzed': len(csv_files),
        'combined_issues': combined_issues,
        'combined_warnings': combined_warnings,
        'comparative_statistics': {}
    }
    
    # Compare transaction counts, date ranges, etc.
    if len(csv_files) > 1:
        row_counts = {csv_file: results[csv_file].get('statistics', {}).get('row_count', 0) for csv_file in csv_files}
        comparative_analysis['comparative_statistics']['row_counts'] = row_counts
        
        # Check for date overlaps
        date_ranges = {}
        for csv_file in csv_files:
            stats = results[csv_file].get('statistics', {})
            if 'start_date' in stats and 'end_date' in stats:
                date_ranges[csv_file] = {
                    'start_date': stats['start_date'],
                    'end_date': stats['end_date']
                }
        
        comparative_analysis['comparative_statistics']['date_ranges'] = date_ranges
        
        # Check for duplicate transactions across files
        # This would require more complex logic to compare transaction details
    
    # Save comparative analysis if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "comparative_analysis.json"), 'w') as f:
            json.dump(comparative_analysis, f, indent=2, default=str)
    
    logger.info(f"Comparative analysis complete for {len(csv_files)} CSV files")
    logger.info(f"Found {len(combined_issues)} combined issues and {len(combined_warnings)} combined warnings")
    
    return comparative_analysis

def analyze_combined_csv(csv_path, debug=False, output_dir=None):
    """
    Specialized analysis for combined CSV files with a Source column
    
    Args:
        csv_path (str): Path to the combined CSV file
        debug (bool): Enable debug mode
        output_dir (str): Directory to save analysis results
        
    Returns:
        dict: Analysis results
    """
    try:
        # Read CSV file
        df = pd.read_csv(csv_path)
        
        # Check if this is a combined CSV with a Source column
        if 'Source' not in df.columns:
            logger.warning(f"{csv_path} does not appear to be a combined CSV (no Source column)")
            return analyze_csv_transactions(csv_path, debug, output_dir)
        
        # Basic analysis first
        basic_analysis = analyze_csv_transactions(csv_path, debug, output_dir)
        
        # Additional analysis for combined CSV
        source_counts = df['Source'].value_counts().to_dict()
        basic_analysis['statistics']['source_counts'] = source_counts
        
        # Check for date continuity between sources
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            
            # Group by source and check date ranges
            source_date_ranges = {}
            for source in df['Source'].unique():
                source_df = df[df['Source'] == source]
                if not source_df['Date'].isna().all():
                    source_date_ranges[source] = {
                        'start_date': source_df['Date'].min().strftime('%Y-%m-%d'),
                        'end_date': source_df['Date'].max().strftime('%Y-%m-%d'),
                        'days': (source_df['Date'].max() - source_df['Date'].min()).days
                    }
            
            basic_analysis['statistics']['source_date_ranges'] = source_date_ranges
            
            # Check for gaps between sources
            sources_by_end_date = sorted(
                [(source, pd.to_datetime(info['end_date'])) for source, info in source_date_ranges.items()],
                key=lambda x: x[1]
            )
            
            sources_by_start_date = sorted(
                [(source, pd.to_datetime(info['start_date'])) for source, info in source_date_ranges.items()],
                key=lambda x: x[1]
            )
            
            gaps = []
            for i in range(len(sources_by_end_date) - 1):
                current_source, current_end = sources_by_end_date[i]
                next_source, next_start = sources_by_start_date[i + 1]
                
                gap_days = (next_start - current_end).days
                if gap_days > 1:  # More than 1 day gap
                    gaps.append({
                        'from_source': current_source,
                        'to_source': next_source,
                        'gap_days': gap_days,
                        'gap_end': current_end.strftime('%Y-%m-%d'),
                        'gap_start': next_start.strftime('%Y-%m-%d')
                    })
            
            if gaps:
                basic_analysis['date_gaps_between_sources'] = gaps
                for gap in gaps:
                    warning = f"Date gap of {gap['gap_days']} days between {gap['from_source']} and {gap['to_source']}"
                    basic_analysis['warnings'].append(warning)
                    logger.warning(warning)
        
        # Generate visualizations for combined CSV
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Transactions by source
            plt.figure(figsize=(12, 6))
            df['Source'].value_counts().plot(kind='bar')
            plt.title('Transaction Count by Source')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'transaction_count_by_source.png'))
            plt.close()
            
            # Balance by source if balance column exists
            if 'Balance' in df.columns and 'Date' in df.columns:
                plt.figure(figsize=(12, 6))
                df['Balance'] = pd.to_numeric(df['Balance'].astype(str).str.replace(',', ''), errors='coerce')
                
                # Plot balance over time with source colors
                df_sorted = df.sort_values('Date')
                sources = df['Source'].unique()
                
                for i, source in enumerate(sources):
                    source_df = df_sorted[df_sorted['Source'] == source]
                    plt.plot(source_df['Date'], source_df['Balance'], label=source)
                
                plt.legend()
                plt.title('Balance Over Time by Source')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'balance_by_source.png'))
                plt.close()
        
        # Save analysis results if output directory is provided
        if output_dir:
            base_name = os.path.basename(csv_path).replace('.csv', '')
            with open(os.path.join(output_dir, f"{base_name}_combined_analysis.json"), 'w') as f:
                json.dump(basic_analysis, f, indent=2, default=str)
        
        return basic_analysis
        
    except Exception as e:
        error_msg = f"Error analyzing combined CSV: {e}"
        logger.error(error_msg, exc_info=True)
        return {"error": error_msg}

def main():
    """Command line interface for CSV analyzer"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze CSV transaction data')
    parser.add_argument('csv_path', help='Path to CSV file or directory containing CSV files')
    parser.add_argument('--output-dir', '-o', help='Directory to save analysis results')
    parser.add_argument('--combined', '-c', action='store_true', help='Analyze as a combined CSV with Source column')
    parser.add_argument('--debug', '-d', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Check if path is a directory or file
    if os.path.isdir(args.csv_path):
        # Find all CSV files in directory
        csv_files = []
        for root, _, files in os.walk(args.csv_path):
            for file in files:
                if file.lower().endswith('.csv'):
                    csv_files.append(os.path.join(root, file))
        
        if not csv_files:
            logger.error(f"No CSV files found in {args.csv_path}")
            return
        
        logger.info(f"Found {len(csv_files)} CSV files to analyze")
        
        # Analyze multiple CSV files
        results = analyze_multiple_csvs(csv_files, args.output_dir, args.debug)
        
        print(f"✅ Analyzed {len(csv_files)} CSV files")
        print(f"Found {len(results['combined_issues'])} issues and {len(results['combined_warnings'])} warnings")
        
        if args.output_dir:
            print(f"Analysis results saved to {args.output_dir}")
    else:
        # Analyze single CSV file
        if args.combined:
            results = analyze_combined_csv(args.csv_path, args.debug, args.output_dir)
        else:
            results = analyze_csv_transactions(args.csv_path, args.debug, args.output_dir)
        
        print(f"✅ Analyzed {args.csv_path}")
        print(f"Found {len(results.get('issues_found', []))} issues and {len(results.get('warnings', []))} warnings")
        
        if args.output_dir:
            print(f"Analysis results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
