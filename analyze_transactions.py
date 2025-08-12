#!/usr/bin/env python3
"""
Transaction Analysis Script for Bank Statements

This script provides analysis tools for both individual bank statement CSV files
and the combined transactions file.
"""

import os
import sys
import pandas as pd
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

def clean_numeric(value):
    """Convert string numeric values with commas to float."""
    if pd.isna(value) or value == '':
        return 0.0
    
    # Remove commas and convert to float
    if isinstance(value, str):
        return float(value.replace(',', ''))
    return float(value)

def load_csv(file_path):
    """Load a CSV file into a pandas DataFrame."""
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} transactions from {os.path.basename(file_path)}")
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def analyze_individual_file(file_path):
    """Analyze an individual transaction file."""
    df = load_csv(file_path)
    if df is None:
        return
    
    # Clean numeric columns
    for col in ['Debits', 'Credits', 'Balance']:
        if col in df.columns:
            df[col] = df[col].apply(clean_numeric)
    
    # Basic statistics
    total_debits = df['Debits'].sum()
    total_credits = df['Credits'].sum()
    net_flow = total_credits - total_debits
    
    print(f"\nAnalysis for {os.path.basename(file_path)}:")
    print(f"Total transactions: {len(df)}")
    print(f"Total debits: R {total_debits:,.2f}")
    print(f"Total credits: R {total_credits:,.2f}")
    print(f"Net cash flow: R {net_flow:,.2f}")
    
    # Transaction counts by type
    service_fee_count = df[df['ServiceFee'] == 'Y'].shape[0]
    print(f"Transactions with service fees: {service_fee_count} ({service_fee_count/len(df)*100:.1f}%)")
    
    # Show first few transactions
    print("\nSample transactions:")
    print(df.head(5).to_string())
    
    return df

def analyze_combined_file(file_path):
    """Analyze the combined transactions file."""
    df = load_csv(file_path)
    if df is None:
        return
    
    # Clean numeric columns
    for col in ['Debits', 'Credits', 'Balance']:
        if col in df.columns:
            df[col] = df[col].apply(clean_numeric)
    
    # Basic statistics
    total_debits = df['Debits'].sum()
    total_credits = df['Credits'].sum()
    net_flow = total_credits - total_debits
    
    print(f"\nAnalysis for Combined Transactions:")
    print(f"Total transactions: {len(df)}")
    print(f"Total debits: R {total_debits:,.2f}")
    print(f"Total credits: R {total_credits:,.2f}")
    print(f"Net cash flow: R {net_flow:,.2f}")
    
    # Transactions by source file
    if 'Source' in df.columns:
        source_counts = df['Source'].value_counts()
        print("\nTransactions by source file:")
        for source, count in source_counts.items():
            print(f"  {source}: {count} transactions")
    
    # Show first few transactions
    print("\nSample transactions:")
    print(df.head(5).to_string())
    
    return df

def compare_with_combined(individual_files, combined_file):
    """Compare individual files with the combined file."""
    combined_df = load_csv(combined_file)
    if combined_df is None:
        return
    
    # Clean numeric columns in combined file
    for col in ['Debits', 'Credits', 'Balance']:
        if col in combined_df.columns:
            combined_df[col] = combined_df[col].apply(clean_numeric)
    
    # Calculate total transactions in individual files
    total_individual_transactions = 0
    total_individual_debits = 0
    total_individual_credits = 0
    
    for file_path in individual_files:
        df = load_csv(file_path)
        if df is None:
            continue
        
        # Clean numeric columns
        for col in ['Debits', 'Credits', 'Balance']:
            if col in df.columns:
                df[col] = df[col].apply(clean_numeric)
        
        total_individual_transactions += len(df)
        total_individual_debits += df['Debits'].sum()
        total_individual_credits += df['Credits'].sum()
    
    # Compare totals
    print("\nComparison between individual files and combined file:")
    print(f"Total transactions in individual files: {total_individual_transactions}")
    print(f"Total transactions in combined file: {len(combined_df)}")
    print(f"Difference: {total_individual_transactions - len(combined_df)}")
    
    print(f"\nTotal debits in individual files: R {total_individual_debits:,.2f}")
    print(f"Total debits in combined file: R {combined_df['Debits'].sum():,.2f}")
    print(f"Difference: R {total_individual_debits - combined_df['Debits'].sum():,.2f}")
    
    print(f"\nTotal credits in individual files: R {total_individual_credits:,.2f}")
    print(f"Total credits in combined file: R {combined_df['Credits'].sum():,.2f}")
    print(f"Difference: R {total_individual_credits - combined_df['Credits'].sum():,.2f}")

def visualize_transactions(file_path, output_dir=None):
    """Create visualizations for transaction data."""
    df = load_csv(file_path)
    if df is None:
        return
    
    # Ensure output directory exists
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(file_path), 'analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    # Clean numeric columns
    for col in ['Debits', 'Credits', 'Balance']:
        if col in df.columns:
            df[col] = df[col].apply(clean_numeric)
    
    # File basename for output files
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # 1. Debits vs Credits bar chart
    plt.figure(figsize=(10, 6))
    total_debits = df['Debits'].sum()
    total_credits = df['Credits'].sum()
    plt.bar(['Debits', 'Credits'], [total_debits, total_credits])
    plt.title('Total Debits vs Credits')
    plt.ylabel('Amount (R)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for i, v in enumerate([total_debits, total_credits]):
        plt.text(i, v + 0.1, f'R {v:,.2f}', ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{base_name}_debits_vs_credits.png'))
    plt.close()
    
    # 2. Transaction count by source (if combined file)
    if 'Source' in df.columns:
        plt.figure(figsize=(12, 6))
        source_counts = df['Source'].value_counts()
        source_counts.plot(kind='bar')
        plt.title('Transaction Count by Source')
        plt.xlabel('Source File')
        plt.ylabel('Number of Transactions')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{base_name}_transactions_by_source.png'))
        plt.close()
    
    print(f"\nVisualizations saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Analyze bank statement transaction files')
    parser.add_argument('--file', help='Path to a single CSV file to analyze')
    parser.add_argument('--combined', help='Path to the combined transactions CSV file')
    parser.add_argument('--dir', help='Directory containing individual transaction files')
    parser.add_argument('--compare', action='store_true', help='Compare individual files with combined file')
    parser.add_argument('--visualize', action='store_true', help='Create visualizations')
    parser.add_argument('--output', help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    if args.file:
        df = analyze_individual_file(args.file)
        if args.visualize:
            visualize_transactions(args.file, args.output)
    
    elif args.combined:
        df = analyze_combined_file(args.combined)
        if args.visualize:
            visualize_transactions(args.combined, args.output)
    
    elif args.dir and args.compare and args.combined:
        # Get all CSV files in the directory
        csv_files = [os.path.join(args.dir, f) for f in os.listdir(args.dir) 
                    if f.endswith('.csv') and f != os.path.basename(args.combined)]
        compare_with_combined(csv_files, args.combined)
    
    elif args.dir:
        # Analyze all CSV files in the directory
        csv_files = [os.path.join(args.dir, f) for f in os.listdir(args.dir) if f.endswith('.csv')]
        for file_path in csv_files:
            analyze_individual_file(file_path)
            if args.visualize:
                visualize_transactions(file_path, args.output)
    
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
