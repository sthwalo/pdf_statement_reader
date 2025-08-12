#!/usr/bin/env python3
"""
Lattice Strict Extractor - Enhanced PDF statement extraction using strict regex parsing
"""

import os
import re
import sys
import json
import logging
import argparse
import pandas as pd
from typing import List, Dict, Optional, Tuple, Set, Any
from datetime import datetime

# Import from local modules
from modules.lattice_extractor import LatticeExtractor
from modules import transaction_extractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class LatticeStrictExtractor(LatticeExtractor):
    """
    Enhanced LatticeExtractor with strict regex-based parsing for better accuracy
    """

    def __init__(self, debug: bool = False):
        """Initialize the LatticeStrictExtractor"""
        super().__init__(debug=debug)
        self.debug_info = {"errors": [], "warnings": [], "tables_analyzed": 0}

    def parse_bank_statement_table(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Parse a bank statement table using strict regex patterns
        
        Args:
            df (DataFrame): DataFrame containing the table data
            
        Returns:
            list: List of transaction dictionaries
        """
        transactions = []
        
        # Skip if DataFrame is empty or too small
        if df is None or df.empty or len(df) < 2:
            logger.warning("Empty or too small table, skipping")
            return transactions
            
        # Try to identify the header row
        header_row = self._find_header_row(df)
        if header_row is None:
            logger.warning("Could not find header row, using first row")
            header_row = 0
            
        # Skip the header row
        data_rows = df.iloc[header_row + 1:]
        
        # Convert to string for easier processing
        data_rows_str = data_rows.astype(str)
        
        # Track rows that have been processed as part of multi-line transactions
        processed_rows = set()
        
        # Process each row
        for i in range(len(data_rows_str)):
            if i in processed_rows:
                continue
                
            # Get the current row
            row = data_rows_str.iloc[i]
            
            # Check if this is the start of a transaction
            if self._is_transaction_start(row):
                # Extract the transaction data
                transaction, additional_rows = self._extract_transaction(data_rows_str, i)
                
                # Mark additional rows as processed
                processed_rows.update(additional_rows)
                
                # Add the transaction if valid
                if self._is_valid_transaction(transaction):
                    transactions.append(transaction)
        
        logger.info(f"Extracted {len(transactions)} transactions from table")
        return transactions
        
    def _find_header_row(self, df: pd.DataFrame) -> Optional[int]:
        """
        Find the header row in the table
        
        Args:
            df (DataFrame): DataFrame containing the table data
            
        Returns:
            int or None: Index of the header row, or None if not found
        """
        # Common header terms
        header_terms = [
            'date', 'details', 'description', 'debit', 'credit', 
            'balance', 'amount', 'reference', 'transaction', 'fee'
        ]
        
        # Convert DataFrame to string for easier searching
        df_str = df.astype(str).apply(lambda x: x.str.lower())
        
        # Check each row for header terms
        for i in range(min(5, len(df))):  # Check only first 5 rows
            row = df_str.iloc[i]
            matches = sum(row.str.contains('|'.join(header_terms), regex=True, na=False))
            
            # If we find at least 3 header terms, consider it a header row
            if matches >= 3:
                logger.info(f"Found header row at index {i} with {matches} header terms")
                return i
                
        return None
        
    def _is_transaction_start(self, row: pd.Series) -> bool:
        """
        Check if a row is the start of a transaction
        
        Args:
            row (Series): Row to check
            
        Returns:
            bool: True if the row is the start of a transaction
        """
        # Check for date pattern
        date_pattern = r'\d{1,2}[/\-\.]\d{1,2}[/\-\.]?(\d{2}|\d{4})?|\d{4}-\d{2}-\d{2}|\d{2}\s+\d{2}'
        has_date = any(bool(re.search(date_pattern, str(cell))) for cell in row)
        
        # Check for amount pattern
        amount_pattern = r'\d+[\.,]\d{2}-?|\d+[\.,]\d{3}[\.,]\d{2}-?'
        has_amount = any(bool(re.search(amount_pattern, str(cell))) for cell in row)
        
        # Check for transaction details
        details_indicators = ['balance', 'payment', 'transfer', 'withdrawal', 'deposit', 'fee']
        has_details = any(any(indicator in str(cell).lower() for indicator in details_indicators) for cell in row)
        
        # A transaction start should have at least a date or details, and an amount
        return (has_date or has_details) and has_amount
        
    def _extract_transaction(self, df: pd.DataFrame, start_idx: int) -> Tuple[Dict[str, Any], Set[int]]:
        """
        Extract a complete transaction, which may span multiple rows
        
        Args:
            df (DataFrame): DataFrame containing the table data
            start_idx (int): Starting index of the transaction
            
        Returns:
            tuple: (transaction dict, set of additional row indices used)
        """
        transaction = {
            'Date': '',
            'Details': '',
            'Debits': '',
            'Credits': '',
            'Balance': '',
            'ServiceFee': ''
        }
        
        additional_rows = set()
        
        # Get the starting row
        start_row = df.iloc[start_idx]
        
        # Extract data from the starting row
        self._extract_row_data(start_row, transaction)
        
        # Check for continuation rows (rows that add to the details of this transaction)
        i = start_idx + 1
        while i < len(df):
            row = df.iloc[i]
            
            # A continuation row typically has no date and no amounts
            # but contains additional details text
            if self._is_continuation_row(row):
                # Extract details from continuation row
                details = self._extract_details(row)
                if details:
                    transaction['Details'] += ' ' + details
                    additional_rows.add(i)
                    i += 1
                else:
                    break
            else:
                break
                
        # Clean up the transaction data
        self._clean_transaction(transaction)
        
        return transaction, additional_rows
        
    def _is_continuation_row(self, row: pd.Series) -> bool:
        """
        Check if a row is a continuation of a previous transaction
        
        Args:
            row (Series): Row to check
            
        Returns:
            bool: True if the row is a continuation
        """
        # Continuation rows typically have no date and no amounts
        date_pattern = r'\d{1,2}[/\-\.]\d{1,2}[/\-\.]?(\d{2}|\d{4})?|\d{4}-\d{2}-\d{2}|\d{2}\s+\d{2}'
        amount_pattern = r'\d+[\.,]\d{2}-?|\d+[\.,]\d{3}[\.,]\d{2}-?'
        
        has_date = any(bool(re.search(date_pattern, str(cell))) for cell in row)
        has_amount = any(bool(re.search(amount_pattern, str(cell))) for cell in row)
        
        # Check if the row has any non-empty text
        has_text = any(str(cell).strip() and str(cell).strip().lower() != 'nan' for cell in row)
        
        # A continuation row should have text but no date or amounts
        return has_text and not has_date and not has_amount
        
    def _extract_row_data(self, row: pd.Series, transaction: Dict[str, Any]) -> None:
        """
        Extract data from a row and update the transaction dict
        
        Args:
            row (Series): Row to extract data from
            transaction (dict): Transaction dict to update
        """
        # Extract date
        date_pattern = r'\d{1,2}[/\-\.]\d{1,2}[/\-\.]?(\d{2}|\d{4})?|\d{4}-\d{2}-\d{2}|\d{2}\s+\d{2}'
        for cell in row:
            cell_str = str(cell)
            date_match = re.search(date_pattern, cell_str)
            if date_match and not transaction['Date']:
                transaction['Date'] = date_match.group(0)
                
        # Extract amounts (debits, credits, balance)
        amount_pattern = r'(\d+[\.,]\d{2}-?)|\d+[\.,]\d{3}[\.,]\d{2}-?'
        for cell in row:
            cell_str = str(cell)
            amount_match = re.search(amount_pattern, cell_str)
            if amount_match:
                amount = amount_match.group(0)
                
                # Determine if it's a debit, credit, or balance based on context
                if cell_str.lower().endswith('-') or any(term in cell_str.lower() for term in ['debit', 'payment', 'withdrawal']):
                    if not transaction['Debits']:
                        transaction['Debits'] = amount
                elif any(term in cell_str.lower() for term in ['credit', 'deposit']):
                    if not transaction['Credits']:
                        transaction['Credits'] = amount
                elif any(term in cell_str.lower() for term in ['balance', 'closing']):
                    if not transaction['Balance']:
                        transaction['Balance'] = amount
                        
        # Extract details
        details = self._extract_details(row)
        if details:
            transaction['Details'] = details
            
        # Check for service fee
        if any('##' in str(cell) for cell in row) or any('fee' in str(cell).lower() for cell in row):
            transaction['ServiceFee'] = '##'
            
    def _extract_details(self, row: pd.Series) -> str:
        """
        Extract transaction details from a row
        
        Args:
            row (Series): Row to extract details from
            
        Returns:
            str: Extracted details
        """
        # Combine all non-empty cells that don't match date or amount patterns
        date_pattern = r'\d{1,2}[/\-\.]\d{1,2}[/\-\.]?(\d{2}|\d{4})?|\d{4}-\d{2}-\d{2}|\d{2}\s+\d{2}'
        amount_pattern = r'\d+[\.,]\d{2}-?|\d+[\.,]\d{3}[\.,]\d{2}-?'
        
        details_parts = []
        
        for cell in row:
            cell_str = str(cell).strip()
            if cell_str and cell_str.lower() != 'nan':
                # Skip cells that are just dates or amounts
                if (re.fullmatch(date_pattern, cell_str) or 
                    re.fullmatch(amount_pattern, cell_str)):
                    continue
                    
                # Add to details if it contains text
                details_parts.append(cell_str)
                
        return ' '.join(details_parts)
        
    def _clean_transaction(self, transaction: Dict[str, Any]) -> None:
        """
        Clean and normalize transaction data
        
        Args:
            transaction (dict): Transaction dict to clean
        """
        # Clean details
        if transaction['Details']:
            # Remove excess whitespace
            transaction['Details'] = re.sub(r'\s+', ' ', transaction['Details']).strip()
            
            # Truncate long details
            if len(transaction['Details']) > 200:
                transaction['Details'] = transaction['Details'][:197] + '...'
                
        # Normalize date format if possible
        if transaction['Date']:
            # Try to convert to a standard format
            date_parts = re.findall(r'\d+', transaction['Date'])
            if len(date_parts) >= 2:
                # Assume MM DD format
                transaction['Date'] = f"{date_parts[0]} {date_parts[1]}"
                
        # Clean amount fields
        for field in ['Debits', 'Credits', 'Balance']:
            if transaction[field]:
                # Remove any non-numeric characters except . and ,
                amount = re.sub(r'[^\d\.,\-]', '', transaction[field])
                transaction[field] = amount
                
    def _is_valid_transaction(self, transaction: Dict[str, Any]) -> bool:
        """
        Check if a transaction is valid
        
        Args:
            transaction (dict): Transaction to check
            
        Returns:
            bool: True if the transaction is valid
        """
        # A valid transaction should have either a date or details
        has_date = bool(transaction['Date'])
        has_details = bool(transaction['Details'])
        
        # And at least one amount field
        has_amount = bool(transaction['Debits']) or bool(transaction['Credits']) or bool(transaction['Balance'])
        
        return (has_date or has_details) and has_amount
        
    def extract_transactions_from_pdf(self, pdf_path: str, password: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Extract transactions from a PDF using strict regex parsing
        
        Args:
            pdf_path (str): Path to the PDF file
            password (str, optional): Password for the PDF
            
        Returns:
            list: List of transaction dictionaries
        """
        # Extract tables from PDF using parent class method
        tables = self.extract_tables_from_pdf(pdf_path, password)
        
        all_transactions = []
        
        # Process each table
        for i, table in enumerate(tables):
            logger.info(f"Processing table {i+1}/{len(tables)}")
            
            # Check if the table looks like a transaction table
            if self._is_transaction_table(table):
                # Parse the table
                transactions = self.parse_bank_statement_table(table)
                all_transactions.extend(transactions)
            else:
                logger.info(f"Table {i+1} is not a transaction table, skipping")
                
        # Post-process transactions
        if all_transactions:
            all_transactions = self.post_process_transactions(all_transactions)
            
        logger.info(f"Extracted {len(all_transactions)} total transactions from PDF using strict lattice mode")
        return all_transactions
        
    def _is_transaction_table(self, df: pd.DataFrame) -> bool:
        """
        Check if a table contains transaction data
        
        Args:
            df (DataFrame): DataFrame to check
            
        Returns:
            bool: True if the table contains transaction data
        """
        if df is None or df.empty or len(df) < 2:
            return False
            
        # Convert to string for easier processing
        df_str = df.astype(str)
        
        # Check for date pattern
        date_pattern = r'\d{1,2}[/\-\.]\d{1,2}[/\-\.]?(\d{2}|\d{4})?|\d{4}-\d{2}-\d{2}|\d{2}\s+\d{2}'
        has_date = df_str.apply(lambda x: x.str.contains(date_pattern, regex=True, na=False)).any().any()
        
        # Check for amount pattern
        amount_pattern = r'\d+[\.,]\d{2}-?|\d+[\.,]\d{3}[\.,]\d{2}-?'
        has_amount = df_str.apply(lambda x: x.str.contains(amount_pattern, regex=True, na=False)).any().any()
        
        # Check for transaction details
        details_indicators = ['balance', 'payment', 'transfer', 'withdrawal', 'deposit', 'fee']
        has_details = any(df_str.apply(lambda x: x.str.contains(indicator, case=False, regex=False, na=False)).any().any() 
                          for indicator in details_indicators)
        
        # Count potential transaction rows
        potential_rows = 0
        for i in range(len(df_str)):
            row = df_str.iloc[i]
            if self._is_transaction_start(row):
                potential_rows += 1
                
        is_transaction_table = has_date and has_amount and has_details and potential_rows > 0
        
        logger.info(f"Lattice strict table analysis - has_date: {has_date}, has_amount: {has_amount}, "
                   f"has_details: {has_details}, potential_rows: {potential_rows}, "
                   f"is_transaction_table: {is_transaction_table}")
                   
        return is_transaction_table


def main():
    """Main function to run the lattice strict extractor"""
    parser = argparse.ArgumentParser(description="Extract transactions from PDF bank statements using strict lattice mode")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--output", help="Output path for the CSV file")
    parser.add_argument("--password", help="Password for the PDF file")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--debug-info", help="Path to save debug info JSON")
    parser.add_argument("--compare", action="store_true", help="Compare with regular extraction")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    else:
        # Default output path
        output_dir = os.path.join(os.path.dirname(os.path.abspath(args.pdf_path)), "../output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output filename
        pdf_basename = os.path.basename(args.pdf_path)
        pdf_name = os.path.splitext(pdf_basename)[0]
        args.output = os.path.join(output_dir, f"{pdf_name}_strict_lattice_transactions.csv")
    
    # Extract transactions using strict lattice mode
    extractor = LatticeStrictExtractor(debug=args.debug)
    strict_lattice_transactions = extractor.extract_transactions_from_pdf(args.pdf_path, args.password)
    
    # Save transactions to CSV
    if strict_lattice_transactions:
        df = pd.DataFrame(strict_lattice_transactions)
        df.to_csv(args.output, index=False)
        logger.info(f"✅ Saved {len(strict_lattice_transactions)} transactions to {args.output}")
    else:
        logger.warning("❌ No transactions extracted")
        
    # Save debug info if requested
    if args.debug_info:
        with open(args.debug_info, 'w') as f:
            json.dump(extractor.debug_info, f, indent=2)
            
    # Compare with regular extraction if requested
    if args.compare:
        # Extract transactions using regular mode
        # First extract tables from PDF
        tables = extractor.extract_tables_from_pdf(args.pdf_path, args.password)
        # Then extract transactions using transaction_extractor module
        regular_transactions = transaction_extractor.extract_transactions_from_pdf(tables, debug=args.debug)
        
        # Extract transactions using original lattice mode
        lattice_extractor = LatticeExtractor(debug=args.debug)
        # First extract tables
        lattice_tables = lattice_extractor.extract_tables_from_pdf(args.pdf_path, args.password)
        # Then process each table
        lattice_transactions = []
        for table in lattice_tables:
            # Identify columns
            column_mapping, header_row = lattice_extractor.identify_columns(table)
            # Extract transactions
            transactions = lattice_extractor.extract_transactions_from_table(table, column_mapping, header_row)
            lattice_transactions.extend(transactions)
        # Post-process transactions
        if lattice_transactions:
            lattice_transactions = lattice_extractor.post_process_transactions(lattice_transactions)
        
        # Save regular transactions to CSV
        regular_output = os.path.join(os.path.dirname(args.output), f"{pdf_name}_regular_transactions.csv")
        pd.DataFrame(regular_transactions).to_csv(regular_output, index=False)
        
        # Save original lattice transactions to CSV
        lattice_output = os.path.join(os.path.dirname(args.output), f"{pdf_name}_lattice_transactions.csv")
        pd.DataFrame(lattice_transactions).to_csv(lattice_output, index=False)
        
        # Generate comparison report
        comparison_output = os.path.join(os.path.dirname(args.output), f"{pdf_name}_comparison.txt")
        with open(comparison_output, 'w') as f:
            f.write(f"PDF: {args.pdf_path}\n")
            f.write(f"Regular extraction: {len(regular_transactions)} transactions\n")
            f.write(f"Lattice extraction: {len(lattice_transactions)} transactions\n")
            f.write(f"Strict lattice extraction: {len(strict_lattice_transactions)} transactions\n\n")
            
            # Compare fields
            regular_fields = set()
            lattice_fields = set()
            strict_fields = set()
            
            for tx in regular_transactions:
                regular_fields.update(tx.keys())
            
            for tx in lattice_transactions:
                lattice_fields.update(tx.keys())
                
            for tx in strict_lattice_transactions:
                strict_fields.update(tx.keys())
                
            f.write(f"Fields in regular extraction: {', '.join(sorted(regular_fields))}\n")
            f.write(f"Fields in lattice extraction: {', '.join(sorted(lattice_fields))}\n")
            f.write(f"Fields in strict lattice extraction: {', '.join(sorted(strict_fields))}\n\n")
            
            f.write(f"Fields only in regular: {', '.join(sorted(regular_fields - lattice_fields - strict_fields))}\n")
            f.write(f"Fields only in lattice: {', '.join(sorted(lattice_fields - regular_fields - strict_fields))}\n")
            f.write(f"Fields only in strict lattice: {', '.join(sorted(strict_fields - regular_fields - lattice_fields))}\n")
            
        logger.info(f"✅ Comparison saved to: {comparison_output}")
        
    # Log success
    if strict_lattice_transactions:
        logger.info(f"✅ Extracted {len(strict_lattice_transactions)} transactions using strict lattice mode")
        logger.info(f"✅ Saved to: {args.output}")
    else:
        logger.error("❌ Failed to extract any transactions")
        

if __name__ == "__main__":
    main()
