#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Camelot-based Bank Statement Parser

This module provides functionality to extract bank statement transactions from PDF files
using the camelot-py library, which is particularly effective for extracting tables from PDFs.
"""

import os
import re
import sys
import json
import logging
import argparse
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional, Union
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Try to import camelot
try:
    import camelot
    CAMELOT_AVAILABLE = True
except ImportError:
    logger.warning("camelot-py not installed. Install with: pip install camelot-py[cv]")
    CAMELOT_AVAILABLE = False


class CamelotBankStatementParser:
    """
    Parser for bank statements using camelot-py for table extraction.
    """
    
    def __init__(self, debug=False):
        """
        Initialize the parser.
        
        Args:
            debug (bool): Enable debug mode
        """
        self.debug = debug
        self.debug_info = {
            'timestamp': datetime.now().isoformat(),
            'tables_extracted': 0,
            'tables': [],
            'transactions': [],
            'errors': []
        }
        
        # Check if camelot is available
        if not CAMELOT_AVAILABLE:
            raise ImportError("camelot-py is required for this parser. Install with: pip install camelot-py[cv]")
    
    def extract_tables_from_pdf(self, pdf_path: str, password: Optional[str] = None) -> List[pd.DataFrame]:
        """
        Extract tables from a PDF file using camelot.
        
        Args:
            pdf_path (str): Path to the PDF file
            password (str, optional): Password for encrypted PDF
            
        Returns:
            List[pd.DataFrame]: List of extracted tables as pandas DataFrames
        """
        logger.info(f"Extracting tables from {pdf_path} using camelot")
        
        tables = []
        
        try:
            # Extract tables using camelot's lattice mode
            camelot_tables = camelot.read_pdf(
                pdf_path,
                pages='all',
                flavor='lattice',
                password=password
            )
            
            logger.info(f"Extracted {len(camelot_tables)} tables using camelot lattice mode")
            
            # Convert to pandas DataFrames and add metadata
            for i, table in enumerate(camelot_tables):
                df = table.df
                
                # Add metadata
                df._page_number = table.page
                df._extraction_method = 'camelot_lattice'
                df._table_index = i
                
                tables.append(df)
                
                # Add to debug info
                if self.debug:
                    self.debug_info['tables'].append({
                        'page': table.page,
                        'index': i,
                        'rows': len(df),
                        'columns': len(df.columns),
                        'accuracy': table.accuracy,
                        'whitespace': table.whitespace,
                        'extraction_method': 'camelot_lattice'
                    })
            
            # If no tables found with lattice, try stream mode
            if len(tables) == 0:
                logger.info("No tables found with lattice mode, trying stream mode")
                
                camelot_tables = camelot.read_pdf(
                    pdf_path,
                    pages='all',
                    flavor='stream',
                    password=password
                )
                
                logger.info(f"Extracted {len(camelot_tables)} tables using camelot stream mode")
                
                # Convert to pandas DataFrames and add metadata
                for i, table in enumerate(camelot_tables):
                    df = table.df
                    
                    # Add metadata
                    df._page_number = table.page
                    df._extraction_method = 'camelot_stream'
                    df._table_index = i
                    
                    tables.append(df)
                    
                    # Add to debug info
                    if self.debug:
                        self.debug_info['tables'].append({
                            'page': table.page,
                            'index': i,
                            'rows': len(df),
                            'columns': len(df.columns),
                            'accuracy': table.accuracy,
                            'whitespace': table.whitespace,
                            'extraction_method': 'camelot_stream'
                        })
            
            # Update debug info
            if self.debug:
                self.debug_info['tables_extracted'] = len(tables)
            
        except Exception as e:
            error_msg = f"Error extracting tables from {pdf_path}: {str(e)}"
            logger.error(error_msg)
            
            if self.debug:
                self.debug_info['errors'].append(error_msg)
        
        return tables
    
    def extract_transactions_from_pdf(self, pdf_path: str, password: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Extract transactions from a PDF file.
        
        Args:
            pdf_path (str): Path to the PDF file
            password (str, optional): Password for encrypted PDF
            
        Returns:
            List[Dict[str, str]]: List of extracted transactions
        """
        logger.info(f"Extracting transactions from {pdf_path}")
        
        # Extract tables
        tables = self.extract_tables_from_pdf(pdf_path, password)
        
        if not tables:
            logger.error(f"No tables extracted from {pdf_path}")
            return []
        
        # Extract transactions from tables
        transactions = []
        
        for table in tables:
            # Try to extract transactions from the table
            table_transactions = self.extract_transactions_from_table(table)
            
            if table_transactions:
                transactions.extend(table_transactions)
                logger.info(f"Extracted {len(table_transactions)} transactions from table on page {table._page_number}")
            else:
                # If no transactions found, try single-column extraction
                single_col_transactions = self.extract_transactions_from_single_column_table(table)
                
                if single_col_transactions:
                    transactions.extend(single_col_transactions)
                    logger.info(f"Extracted {len(single_col_transactions)} transactions from single-column table on page {table._page_number}")
        
        # Post-process transactions
        if transactions:
            transactions = self.post_process_transactions(transactions)
        
        # Update debug info
        if self.debug:
            self.debug_info['transactions'] = transactions
        
        return transactions
    
    def extract_transactions_from_table(self, table: pd.DataFrame) -> List[Dict[str, str]]:
        """
        Extract transactions from a table.
        
        Args:
            table (pd.DataFrame): Table as pandas DataFrame
            
        Returns:
            List[Dict[str, str]]: List of extracted transactions
        """
        # Skip tables with less than 2 rows (header + at least one transaction)
        if len(table) < 2:
            return []
        
        # Try to identify columns
        column_mapping = self.identify_columns(table)
        
        if not column_mapping:
            return []
        
        # Extract transactions
        transactions = []
        
        # Determine header row
        header_row = 0
        
        # Process each row
        for i in range(1, len(table)):  # Skip header row
            row_data = {}
            
            # Extract data from each mapped column
            for field, col_idx in column_mapping.items():
                if col_idx is not None and col_idx < len(table.columns):
                    value = table.iloc[i, col_idx]
                    
                    # Clean up value
                    if isinstance(value, str):
                        value = value.strip()
                    
                    row_data[field] = value
            
            # Check if row has meaningful data
            if self.has_meaningful_data(row_data):
                # Add metadata
                row_data['_page_number'] = getattr(table, '_page_number', None)
                row_data['_extraction_method'] = getattr(table, '_extraction_method', 'camelot')
                row_data['_row_index'] = i
                
                transactions.append(row_data)
        
        return transactions
    
    def extract_transactions_from_single_column_table(self, table: pd.DataFrame) -> List[Dict[str, str]]:
        """
        Extract transactions from a single-column table.
        
        Args:
            table (pd.DataFrame): Table as pandas DataFrame
            
        Returns:
            List[Dict[str, str]]: List of extracted transactions
        """
        # Skip tables with less than 2 rows
        if len(table) < 2:
            return []
        
        # Check if this is a single-column table
        if len(table.columns) != 1:
            return []
        
        # Extract text from the single column
        text = '\n'.join(table[0].astype(str).tolist())
        
        # Parse text using regex patterns
        return self.parse_bank_statement(text)
    
    def identify_columns(self, table: pd.DataFrame) -> Dict[str, Optional[int]]:
        """
        Identify columns in a table.
        
        Args:
            table (pd.DataFrame): Table as pandas DataFrame
            
        Returns:
            Dict[str, Optional[int]]: Mapping of fields to column indices
        """
        # Initialize column mapping
        column_mapping = {
            'Date': None,
            'Details': None,
            'Debits': None,
            'Credits': None,
            'Balance': None
        }
        
        # Check if table has at least one row
        if len(table) == 0:
            return column_mapping
        
        # Check header row (first row)
        header_row = table.iloc[0]
        
        # Map columns based on header text
        for i, header in enumerate(header_row):
            if not isinstance(header, str):
                continue
                
            header = header.strip().lower()
            
            # Date column
            if any(date_pattern in header for date_pattern in ['date', 'day', 'time']):
                column_mapping['Date'] = i
            
            # Details column
            elif any(details_pattern in header for details_pattern in ['details', 'description', 'narrative', 'transaction', 'particulars']):
                column_mapping['Details'] = i
            
            # Debits column
            elif any(debit_pattern in header for debit_pattern in ['debit', 'withdrawal', 'payments', 'out']):
                column_mapping['Debits'] = i
            
            # Credits column
            elif any(credit_pattern in header for credit_pattern in ['credit', 'deposit', 'receipts', 'in']):
                column_mapping['Credits'] = i
            
            # Balance column
            elif any(balance_pattern in header for balance_pattern in ['balance', 'total']):
                column_mapping['Balance'] = i
        
        # If we couldn't identify columns by header, try to infer from content
        if all(v is None for v in column_mapping.values()):
            # Check first few rows for patterns
            for i in range(min(5, len(table.columns))):
                # Sample values from this column
                sample_values = [str(table.iloc[j, i]).strip() for j in range(1, min(10, len(table))) if pd.notna(table.iloc[j, i])]
                
                if not sample_values:
                    continue
                
                # Check for date patterns
                date_pattern = re.compile(r'\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?')
                date_matches = [bool(date_pattern.match(val)) for val in sample_values]
                
                if sum(date_matches) / len(sample_values) > 0.5:
                    column_mapping['Date'] = i
                    continue
                
                # Check for amount patterns (debits/credits)
                amount_pattern = re.compile(r'[-+]?\d+\.\d{2}|\d+,\d{3}\.\d{2}')
                amount_matches = [bool(amount_pattern.search(val)) for val in sample_values]
                
                if sum(amount_matches) / len(sample_values) > 0.5:
                    # If we already found a debit column, this is probably credits
                    if column_mapping['Debits'] is not None:
                        column_mapping['Credits'] = i
                    else:
                        column_mapping['Debits'] = i
                    continue
                
                # If this column doesn't match any patterns, it's probably details
                if column_mapping['Details'] is None:
                    column_mapping['Details'] = i
        
        return column_mapping
    
    def has_meaningful_data(self, row_data: Dict[str, str]) -> bool:
        """
        Check if a row has meaningful transaction data.
        
        Args:
            row_data (Dict[str, str]): Row data
            
        Returns:
            bool: True if row has meaningful data, False otherwise
        """
        # Check if row has date and at least one amount
        has_date = False
        has_amount = False
        
        if 'Date' in row_data and row_data['Date']:
            # Check if date is in a valid format
            date_str = str(row_data['Date']).strip()
            date_pattern = re.compile(r'\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?')
            
            if date_pattern.match(date_str):
                has_date = True
        
        # Check for amounts
        for field in ['Debits', 'Credits']:
            if field in row_data and row_data[field]:
                amount_str = str(row_data[field]).strip()
                
                # Remove currency symbols and commas
                amount_str = re.sub(r'[£$€,]', '', amount_str)
                
                # Check if it's a valid amount
                amount_pattern = re.compile(r'[-+]?\d+\.\d{2}|\d+')
                
                if amount_pattern.match(amount_str):
                    has_amount = True
                    break
        
        # Check if row has details
        has_details = 'Details' in row_data and row_data['Details'] and str(row_data['Details']).strip()
        
        # Row must have either date or details, and at least one amount
        return (has_date or has_details) and has_amount
    
    def parse_bank_statement(self, text: str) -> List[Dict[str, str]]:
        """
        Parse bank statement text using regex patterns.
        
        Args:
            text (str): Bank statement text
            
        Returns:
            List[Dict[str, str]]: List of extracted transactions
        """
        transactions = []
        
        # Define regex patterns for different transaction formats
        patterns = [
            # Pattern 1: Date, Details, Amount (Debit/Credit), Balance
            r'(\d{1,2}/\d{1,2}(?:/\d{2,4})?)\s+([A-Za-z0-9\s.,&\-\']+?)\s+([-+]?\d+\.\d{2})\s+([-+]?\d+\.\d{2})',
            
            # Pattern 2: Date, Details, Debit, Credit, Balance
            r'(\d{1,2}/\d{1,2}(?:/\d{2,4})?)\s+([A-Za-z0-9\s.,&\-\']+?)\s+(\d+\.\d{2})?\s+(\d+\.\d{2})?\s+([-+]?\d+\.\d{2})',
        ]
        
        # Try each pattern
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            
            for match in matches:
                if len(match.groups()) == 4:
                    # Pattern 1: Date, Details, Amount, Balance
                    date, details, amount, balance = match.groups()
                    
                    # Determine if amount is debit or credit
                    if amount.startswith('-'):
                        debit = amount.lstrip('-')
                        credit = ''
                    else:
                        debit = ''
                        credit = amount
                    
                    transaction = {
                        'Date': date,
                        'Details': details.strip(),
                        'Debits': debit,
                        'Credits': credit,
                        'Balance': balance
                    }
                    
                    transactions.append(transaction)
                    
                elif len(match.groups()) == 5:
                    # Pattern 2: Date, Details, Debit, Credit, Balance
                    date, details, debit, credit, balance = match.groups()
                    
                    transaction = {
                        'Date': date,
                        'Details': details.strip(),
                        'Debits': debit or '',
                        'Credits': credit or '',
                        'Balance': balance
                    }
                    
                    transactions.append(transaction)
        
        return transactions
    
    def post_process_transactions(self, transactions: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Post-process extracted transactions.
        
        Args:
            transactions (List[Dict[str, str]]): List of extracted transactions
            
        Returns:
            List[Dict[str, str]]: List of post-processed transactions
        """
        processed_transactions = []
        
        for tx in transactions:
            # Create a copy of the transaction
            processed_tx = tx.copy()
            
            # Normalize date format
            if 'Date' in processed_tx and processed_tx['Date']:
                date_str = str(processed_tx['Date']).strip()
                
                # Try to parse date
                date_match = re.match(r'(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?', date_str)
                
                if date_match:
                    day, month, year = date_match.groups()
                    
                    # If year is missing, use current year
                    if not year:
                        year = str(datetime.now().year)
                    # If year is 2-digit, convert to 4-digit
                    elif len(year) == 2:
                        year = '20' + year
                    
                    # Format date as YYYY-MM-DD
                    processed_tx['Date'] = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
            
            # Clean up amount fields
            for field in ['Debits', 'Credits', 'Balance']:
                if field in processed_tx and processed_tx[field]:
                    amount_str = str(processed_tx[field]).strip()
                    
                    # Remove currency symbols and commas
                    amount_str = re.sub(r'[£$€,]', '', amount_str)
                    
                    # Convert to float and format with 2 decimal places
                    try:
                        amount = float(amount_str)
                        processed_tx[field] = f"{amount:.2f}"
                    except ValueError:
                        # Keep original if conversion fails
                        pass
            
            # Ensure debits are negative
            if 'Debits' in processed_tx and processed_tx['Debits'] and not processed_tx['Debits'].startswith('-'):
                try:
                    amount = float(processed_tx['Debits'])
                    if amount > 0:
                        processed_tx['Debits'] = f"-{processed_tx['Debits']}"
                except ValueError:
                    pass
            
            processed_transactions.append(processed_tx)
        
        return processed_transactions
    
    def save_transactions_to_csv(self, transactions: List[Dict[str, str]], output_path: str) -> None:
        """
        Save transactions to a CSV file.
        
        Args:
            transactions (List[Dict[str, str]]): List of transactions
            output_path (str): Path to save CSV file
        """
        if not transactions:
            logger.warning("No transactions to save")
            return
        
        # Create DataFrame
        df = pd.DataFrame(transactions)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        
        logger.info(f"Saved {len(transactions)} transactions to {output_path}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Extract bank statement transactions from PDF using camelot')
    parser.add_argument('pdf_path', help='Path to the PDF file')
    parser.add_argument('--output', '-o', required=True, help='Output CSV file path')
    parser.add_argument('--password', '-p', help='Password for encrypted PDF')
    parser.add_argument('--debug', '-d', action='store_true', help='Enable debug mode')
    parser.add_argument('--debug-info', help='Path to save debug info JSON')
    
    args = parser.parse_args()
    
    # Check if camelot is available
    if not CAMELOT_AVAILABLE:
        logger.error("camelot-py is required for this parser. Install with: pip install camelot-py[cv]")
        sys.exit(1)
    
    # Create parser
    extractor = CamelotBankStatementParser(debug=args.debug)
    
    # Extract transactions
    camelot_transactions = extractor.extract_transactions_from_pdf(args.pdf_path, args.password)
    
    # Save to CSV
    if camelot_transactions:
        extractor.save_transactions_to_csv(camelot_transactions, args.output)
    
    # Save debug info if requested
    if args.debug and args.debug_info:
        with open(args.debug_info, 'w') as f:
            json.dump(extractor.debug_info, f, indent=2)
    
    # Legacy comparison feature removed as camelot is now the default extraction method
    # If you need to compare extraction methods, please use the individual extraction modules directly
    
    # Log success
    if camelot_transactions:
        logger.info(f"✅ Extracted {len(camelot_transactions)} transactions using camelot")
        logger.info(f"✅ Saved to: {args.output}")
    else:
        logger.error("❌ Failed to extract any transactions")


if __name__ == "__main__":
    main()
