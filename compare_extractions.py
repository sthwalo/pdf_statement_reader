#!/usr/bin/env python3
"""
Compare extraction results from different methods.
"""
import os
import sys
import argparse
import pandas as pd
import logging
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from tabulate import tabulate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def setup_argparse() -> argparse.Namespace:
    """Set up command line arguments."""
    parser = argparse.ArgumentParser(description='Compare extraction results from different methods.')
    parser.add_argument('pdf_path', help='Path to the PDF file')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    return parser.parse_args()

def get_extraction_files(pdf_path: str) -> Dict[str, str]:
    """Get paths to extraction result files."""
    base_name = os.path.basename(pdf_path)
    name_without_ext = os.path.splitext(base_name)[0]
    output_dir = os.path.join(os.path.dirname(os.path.dirname(pdf_path)), 'output')
    
    extraction_files = {
        'regular': os.path.join(output_dir, f"{name_without_ext}_regular_transactions.csv"),
        'lattice': os.path.join(output_dir, f"{name_without_ext}_lattice_transactions.csv"),
        'strict_lattice': os.path.join(output_dir, f"{name_without_ext}_strict_lattice_transactions.csv"),
        'camelot': os.path.join(output_dir, f"{name_without_ext}_camelot_transactions.csv")
    }
    
    # Filter out files that don't exist
    return {k: v for k, v in extraction_files.items() if os.path.exists(v)}

def load_transactions(file_path: str) -> pd.DataFrame:
    """Load transactions from CSV file."""
    try:
        df = pd.read_csv(file_path)
        # Standardize column names
        df.columns = [col.strip() for col in df.columns]
        
        # Check and normalize column names
        column_mapping = {
            'date': 'Date',
            'details': 'Details',
            'debits': 'Debits',
            'credits': 'Credits',
            'balance': 'Balance',
            'servicefee': 'ServiceFee'
        }
        
        # Rename columns to standard format (case-insensitive)
        renamed_columns = {}
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in column_mapping:
                renamed_columns[col] = column_mapping[col_lower]
        
        if renamed_columns:
            df = df.rename(columns=renamed_columns)
        
        # Ensure all required columns exist
        for col in ['Date', 'Details', 'Debits', 'Credits', 'Balance', 'ServiceFee']:
            if col not in df.columns:
                df[col] = ''
                
        return df
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return pd.DataFrame()

def compare_transactions(transactions: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
    """Compare transactions from different methods."""
    results = {}
    
    for method, df in transactions.items():
        if df.empty:
            results[method] = {
                'count': 0,
                'has_date': 0,
                'has_details': 0,
                'has_debits': 0,
                'has_credits': 0,
                'has_balance': 0,
                'has_service_fee': 0
            }
            continue
            
        # Count non-empty values in each column
        has_date = df['Date'].astype(str).str.strip().str.len() > 0
        has_details = df['Details'].astype(str).str.strip().str.len() > 0
        has_debits = df['Debits'].astype(str).str.strip().str.len() > 0
        has_credits = df['Credits'].astype(str).str.strip().str.len() > 0
        has_balance = df['Balance'].astype(str).str.strip().str.len() > 0
        has_service_fee = df['ServiceFee'].astype(str).str.strip().str.len() > 0
        
        results[method] = {
            'count': len(df),
            'has_date': has_date.sum(),
            'has_details': has_details.sum(),
            'has_debits': has_debits.sum(),
            'has_credits': has_credits.sum(),
            'has_balance': has_balance.sum(),
            'has_service_fee': has_service_fee.sum()
        }
    
    return results

def print_comparison_table(results: Dict[str, Dict]) -> None:
    """Print comparison table."""
    table_data = []
    headers = ['Method', 'Count', 'Date', 'Details', 'Debits', 'Credits', 'Balance', 'ServiceFee']
    
    for method, stats in results.items():
        row = [
            method,
            stats['count'],
            f"{stats['has_date']} ({stats['has_date']/stats['count']*100:.1f}%)" if stats['count'] > 0 else "0 (0.0%)",
            f"{stats['has_details']} ({stats['has_details']/stats['count']*100:.1f}%)" if stats['count'] > 0 else "0 (0.0%)",
            f"{stats['has_debits']} ({stats['has_debits']/stats['count']*100:.1f}%)" if stats['count'] > 0 else "0 (0.0%)",
            f"{stats['has_credits']} ({stats['has_credits']/stats['count']*100:.1f}%)" if stats['count'] > 0 else "0 (0.0%)",
            f"{stats['has_balance']} ({stats['has_balance']/stats['count']*100:.1f}%)" if stats['count'] > 0 else "0 (0.0%)",
            f"{stats['has_service_fee']} ({stats['has_service_fee']/stats['count']*100:.1f}%)" if stats['count'] > 0 else "0 (0.0%)"
        ]
        table_data.append(row)
    
    print(tabulate(table_data, headers=headers, tablefmt='grid'))

def find_unique_transactions(transactions: Dict[str, pd.DataFrame]) -> Dict[str, List[str]]:
    """Find transactions unique to each method."""
    unique_transactions = {}
    
    # Create a copy of each DataFrame with fingerprints
    fingerprinted_dfs = {}
    for method, df in transactions.items():
        if df.empty:
            unique_transactions[method] = []
            fingerprinted_dfs[method] = df.copy()
            continue
            
        # Create a copy to avoid modifying the original
        df_copy = df.copy()
        df_copy['Fingerprint'] = df_copy.apply(
            lambda row: f"{row['Date']}|{row['Details'][:50]}|{row['Debits']}|{row['Credits']}",
            axis=1
        )
        fingerprinted_dfs[method] = df_copy
    
    # Find unique transactions for each method
    for method, df in fingerprinted_dfs.items():
        if df.empty:
            unique_transactions[method] = []
            continue
            
        # Find fingerprints unique to this method
        other_fingerprints = set()
        for other_method, other_df in fingerprinted_dfs.items():
            if other_method != method and not other_df.empty:
                other_fingerprints.update(other_df['Fingerprint'].tolist())
        
        unique_fingerprints = set(df['Fingerprint'].tolist()) - other_fingerprints
        unique_transactions[method] = df[df['Fingerprint'].isin(unique_fingerprints)].to_dict('records')
    
    return unique_transactions

def plot_comparison(results: Dict[str, Dict]) -> None:
    """Plot comparison of extraction methods."""
    methods = list(results.keys())
    counts = [results[method]['count'] for method in methods]
    
    plt.figure(figsize=(10, 6))
    plt.bar(methods, counts)
    plt.title('Transaction Count by Extraction Method')
    plt.xlabel('Method')
    plt.ylabel('Transaction Count')
    plt.tight_layout()
    
    # Save the plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(args.pdf_path)), 'output')
    plot_path = os.path.join(output_dir, 'extraction_comparison.png')
    plt.savefig(plot_path)
    logger.info(f"Saved comparison plot to {plot_path}")

def main():
    """Main function."""
    global args
    args = setup_argparse()
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    logger.info(f"Comparing extraction results for {args.pdf_path}")
    
    # Get extraction files
    extraction_files = get_extraction_files(args.pdf_path)
    logger.info(f"Found {len(extraction_files)} extraction files: {', '.join(extraction_files.keys())}")
    
    if not extraction_files:
        logger.error("No extraction files found. Run extraction methods first.")
        sys.exit(1)
    
    # Load transactions
    transactions = {method: load_transactions(file_path) for method, file_path in extraction_files.items()}
    
    # Compare transactions
    results = compare_transactions(transactions)
    
    # Print comparison table
    print_comparison_table(results)
    
    # Find unique transactions
    unique_transactions = find_unique_transactions(transactions)
    
    # Print unique transactions
    for method, unique_txs in unique_transactions.items():
        if unique_txs:
            logger.info(f"{method} has {len(unique_txs)} unique transactions")
            if args.debug and unique_txs:
                logger.debug(f"Sample unique transaction from {method}:")
                sample = unique_txs[0]
                for key, value in sample.items():
                    if key != 'Fingerprint':
                        logger.debug(f"  {key}: {value}")
    
    # Plot comparison
    try:
        plot_comparison(results)
    except Exception as e:
        logger.error(f"Error creating comparison plot: {e}")
    
    logger.info("Comparison complete")

if __name__ == "__main__":
    main()
