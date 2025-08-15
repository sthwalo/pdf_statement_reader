#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Camelot-based Bank Statement Parser

Uses camelot-py to extract tables from PDFs and applies strict regex-based parsing
to accurately extract transaction data with proper alignment.
"""

import re
import os
import json
import argparse
import logging
import pandas as pd
import camelot
from typing import List, Dict, Optional, Tuple, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class CamelotBankStatementParser:
    """
    Parser for bank statements using camelot-py for table extraction
    and strict regex-based parsing for transaction data.
    """
    
    def __init__(self, debug: bool = False):
        """Initialize the parser with optional debug mode"""
        self.debug = debug
        self.debug_info = {
            'tables_processed': 0,
            'transaction_tables': 0,
            'total_transactions': 0,
            'table_results': []
        }
    
    def extract_tables_from_pdf(self, pdf_path: str, password: Optional[str] = None) -> List[pd.DataFrame]:
        """Extract tables from PDF using camelot"""
        logger.info(f"Extracting tables from {pdf_path} using camelot")
        
        try:
            # Extract tables using camelot's lattice mode
            tables = camelot.read_pdf(
                pdf_path,
                pages='all',
                flavor='lattice',
                password=password
            )
            
            logger.info(f"Extracted {len(tables)} tables from PDF")
            
            # Convert to pandas DataFrames
            dataframes = []
            for i, table in enumerate(tables):
                df = table.df
                # Store page number as metadata
                df._page_number = table.page
                
                # Debug info
                if self.debug and i < 10:  # Only show first 10 tables to avoid log spam
                    logger.debug(f"Table {i} on page {table.page} has shape {df.shape}")
                    if not df.empty:
                        logger.debug(f"Table {i} sample:\n{df.iloc[:3, :].to_string()}")
                        
                dataframes.append(df)
            
            return dataframes
            
        except Exception as e:
            logger.error(f"Error extracting tables: {e}")
            return []
    
    def parse_bank_statement(self, text: str) -> List[Dict]:
        """
        Strict parser for bank statements with this specific format:
        | Details | Service Fee | Debits | Credits | Date | Balance |
        
        Handles:
        - Negative amounts (with trailing '-')
        - Multi-line details
        - Service fee markers ('##')
        - Various date formats
        - Proper column alignment
        """
        
        # Enhanced regex pattern for this specific format
        pattern = re.compile(
            r'^(?P<details>(?:[^\|]+(?:\n\s+[^\|]+)*))\|'  # Multi-line details
            r'(?P<service_fee>\s*##?\s*)\|'  # Service fee marker
            r'(?P<debit>\s*(?:[\d,]+\.\d{2}-?)?\s*)\|'  # Debit (with optional negative)
            r'(?P<credit>\s*(?:[\d,]+\.\d{2})?\s*)\|'  # Credit
            r'(?P<date>\s*\d{2}\s+\d{2}\s*)\|'  # Date (MM DD)
            r'(?P<balance>\s*[\d,]+\.\d{2}\s*)\|?'  # Balance
        )
        
        transactions = []
        for match in pattern.finditer(text):
            trans = {
                'details': self.clean_text(match.group('details')),
                'service_fee': self.parse_service_fee(match.group('service_fee')),
                'debit': self.parse_amount(match.group('debit')),
                'credit': self.parse_amount(match.group('credit')),
                'date': self.parse_date(match.group('date')),
                'balance': self.parse_amount(match.group('balance'), allow_negative=False)
            }
            
            # Validate debit/credit exclusivity
            if trans['debit'] != 0 and trans['credit'] != 0:
                logger.warning(f"Transaction has both debit and credit: {trans}")
                # Choose the non-zero value with higher absolute value
                if abs(trans['debit']) >= abs(trans['credit']):
                    trans['credit'] = 0
                else:
                    trans['debit'] = 0
                
            transactions.append(trans)
        
        return transactions

    def clean_text(self, text: str) -> str:
        """Clean and normalize multi-line text"""
        if not text:
            return ""
        lines = [line.strip() for line in text.split('\n')]
        return ' '.join(filter(None, lines))

    def parse_service_fee(self, text: str) -> str:
        """Convert service fee marker to string representation"""
        return '##' if '##' in text else ''

    def parse_amount(self, text: str, allow_negative: bool = True) -> float:
        """Parse currency amount with optional negative"""
        if not text or pd.isna(text):
            return 0.0
        
        text = str(text).strip()
        if not text:
            return 0.0
        
        is_negative = text.endswith('-')
        clean_text = text.replace(',', '').replace('-', '')
        
        try:
            amount = float(clean_text)
            return -amount if is_negative and allow_negative else amount
        except ValueError:
            return 0.0

    def parse_date(self, text: str) -> str:
        """Normalize date format (MM DD to DD/MM)"""
        if not text or pd.isna(text):
            return ""
            
        text = str(text).strip()
        parts = text.strip().split()
        if len(parts) == 2:
            return f"{parts[1]}/{parts[0]}"  # Convert to DD/MM
        return text.strip()

    def format_currency(self, amount: float, allow_negative: bool = False) -> str:
        """Format amount with thousands separators"""
        if amount == 0:
            return ""
        abs_amount = abs(amount)
        formatted = f"{abs_amount:,.2f}"
        if amount < 0 and allow_negative:
            return f"{formatted}-"
        return formatted
    
    def is_transaction_table(self, df: pd.DataFrame) -> bool:
        """
        Determine if a table contains transaction data based on column headers
        and content patterns
        """
        if df.empty or df.shape[0] < 2:
            if self.debug:
                logger.debug(f"Table rejected: too small - shape {df.shape}")
            return False
            
        # Special case for single-column tables (common in camelot extraction)
        if df.shape[1] == 1:
            # Check if this is a transaction table with all data in one column
            content = '\n'.join(df[0].astype(str).tolist())
            
            # Check for header terms in the content
            header_terms = ['date', 'details', 'debit', 'credit', 'balance', 'service', 'fee']
            header_count = sum(1 for term in header_terms if re.search(term, content, re.IGNORECASE))
            
            # Check for transaction patterns
            has_date = bool(re.search(r'\b\d{2}\s+\d{2}\b', content))  # DD MM format
            has_amount = bool(re.search(r'\b[\d,]+\.\d{2}\b', content))  # Currency format
            has_details = bool(re.search(r'payment|transfer|balance|brought|forward', content, re.IGNORECASE))
            
            # Check for transaction markers
            has_transactions = ('balance brought forward' in content.lower() or 
                              'ib payment' in content.lower())
            
            is_transaction = (header_count >= 3 and has_date and has_amount and 
                             has_details and has_transactions)
            
            if self.debug:
                logger.debug(f"Single-column table analysis - headers: {header_count}/7, "
                          f"has_date: {has_date}, has_amount: {has_amount}, "
                          f"has_details: {has_details}, has_transactions: {has_transactions}, "
                          f"is_transaction_table: {is_transaction}")
            
            return is_transaction
        
        # Standard multi-column table analysis
        if df.shape[1] < 4:
            if self.debug:
                logger.debug(f"Table rejected: too few columns - shape {df.shape}")
            return False
            
        # Convert all cells to string for pattern matching
        df_str = df.astype(str)
        
        # Check for header terms
        header_terms = ['date', 'details', 'debit', 'credit', 'balance']
        header_count = 0
        found_headers = []
        
        for term in header_terms:
            if df_str.apply(lambda x: x.str.contains(term, case=False, na=False)).any().any():
                header_count += 1
                found_headers.append(term)
                
        # Check for date patterns (DD/MM or MM DD)
        date_pattern = r'\b\d{2}[\s/]\d{2}\b'
        has_date = df_str.apply(lambda x: x.str.contains(date_pattern, regex=True, na=False)).any().any()
        
        # Check for currency amount patterns
        amount_pattern = r'\b\d{1,3}(?:,\d{3})*\.\d{2}\b'
        has_amount = df_str.apply(lambda x: x.str.contains(amount_pattern, regex=True, na=False)).any().any()
        
        # Check for transaction details
        details_pattern = r'[A-Za-z]{3,}'
        has_details = df_str.apply(lambda x: x.str.contains(details_pattern, regex=True, na=False)).any().any()
        
        # Count potential transaction rows (rows with date and amount)
        potential_rows = 0
        for i in range(1, df.shape[0]):  # Skip header row
            row_str = ' '.join(df_str.iloc[i].astype(str))
            if re.search(date_pattern, row_str) and re.search(amount_pattern, row_str):
                potential_rows += 1
                
        # Decision criteria
        is_transaction = (header_count >= 3 and has_date and has_amount and has_details and potential_rows > 0)
        
        if self.debug:
            logger.debug(f"Multi-column table analysis - headers: {found_headers} ({header_count}/5), "  
                      f"has_date: {has_date}, has_amount: {has_amount}, has_details: {has_details}, "
                      f"potential_rows: {potential_rows}, is_transaction_table: {is_transaction}")
            
        return is_transaction
    
    def find_header_row(self, df: pd.DataFrame) -> int:
        """Find the header row in a transaction table"""
        header_terms = ['details', 'date', 'debit', 'credit', 'balance', 'service']
        
        for i in range(min(5, len(df))):  # Check first 5 rows
            row_str = ' '.join(df.iloc[i].astype(str).tolist()).lower()
            term_count = sum(1 for term in header_terms if term in row_str)
            
            if term_count >= 3:
                logger.info(f"Found header row at index {i} with {term_count} header terms")
                return i
        
        # If no clear header found, check for date patterns in first row
        date_pattern = r'\b\d{2}\s+\d{2}\b'
        first_row = ' '.join(df.iloc[0].astype(str).tolist())
        if re.search(date_pattern, first_row):
            logger.info("First row contains dates, assuming no header row")
            return -1
        
        logger.warning("Could not find header row, using first row")
        return 0
    
    def convert_table_to_text(self, df: pd.DataFrame, header_row: int) -> str:
        """Convert DataFrame to text format for regex parsing"""
        # Skip header row if found
        start_row = header_row + 1 if header_row >= 0 else 0
        
        # Create text representation
        text_rows = []
        for i in range(start_row, len(df)):
            row = df.iloc[i].astype(str).tolist()
            text_rows.append('|' + '|'.join(row) + '|')
        
        return '\n'.join(text_rows)
    
    def extract_transactions_from_table(self, df: pd.DataFrame) -> List[Dict]:
        """Extract transactions from a single table"""
        # Special case for single-column tables
        if df.shape[1] == 1:
            return self.extract_transactions_from_single_column(df)
            
        # Standard multi-column table processing
        # Find header row
        header_row = self.find_header_row(df)
        if header_row < 0:
            logger.warning("Could not find header row, using first row")
            header_row = 0
            
        # Convert table to text format for regex parsing
        text = self.convert_table_to_text(df, header_row)
        
        # Parse transactions using regex
        transactions = self.parse_bank_statement(text)
        
        logger.info(f"Extracted {len(transactions)} transactions from table")
        
        return transactions
        
    def extract_transactions_from_single_column(self, df: pd.DataFrame) -> List[Dict]:
        """Extract transactions from a single-column table"""
        # Join all content into a single string
        content = '\n'.join(df[0].astype(str).tolist())
        
        if self.debug:
            # Print a sample of the content to understand its structure
            logger.debug(f"Single-column table content sample (first 500 chars):\n{content[:500]}")
        
        # Check for transaction indicators in the content
        has_balance_brought_forward = 'BALANCE BROUGHT FORWARD' in content
        has_ib_payment = 'IB PAYMENT' in content
        has_date_pattern = bool(re.search(r'\b\d{2}\s+\d{2}\b', content))  # DD MM format
        has_amount_pattern = bool(re.search(r'\b[\d,]+\.\d{2}\b', content))  # Currency format
        
        if self.debug:
            logger.debug(f"Content indicators: balance_brought_forward={has_balance_brought_forward}, "
                      f"ib_payment={has_ib_payment}, date_pattern={has_date_pattern}, "
                      f"amount_pattern={has_amount_pattern}")
        
        # Try different patterns to extract transactions
        # First pattern: Look for transaction blocks with specific structure
        transaction_pattern1 = re.compile(
            r'((?:IB PAYMENT|BALANCE)[^\n]+)\s*\n'  # Transaction description line
            r'(\d[\d,]*\.\d{2}(?:-|))\s*\n'  # Amount (possibly with negative sign)
            r'(\d{2}\s+\d{2})\s*\n'  # Date (DD MM)
            r'(\d[\d,]*\.\d{2})\s*'  # Balance
        )
        
        # Second pattern: More flexible pattern for various transaction formats
        transaction_pattern2 = re.compile(
            r'([A-Z][A-Z\s\d-]+)\s*\n'  # Details (uppercase text)
            r'(?:(\d[\d,]*\.\d{2}(?:-|)?))?\s*\n'  # Optional amount
            r'(\d{2}\s+\d{2})\s*\n'  # Date (DD MM)
            r'(\d[\d,]*\.\d{2})\s*'  # Balance
        )
        
        # Try first pattern
        matches1 = list(transaction_pattern1.finditer(content))
        if self.debug:
            logger.debug(f"Pattern 1 found {len(matches1)} matches")
            if matches1:
                sample_match = matches1[0]
                logger.debug(f"Sample match 1: {sample_match.groups()}")
        
        # Try second pattern if first one didn't work
        matches2 = list(transaction_pattern2.finditer(content))
        if self.debug:
            logger.debug(f"Pattern 2 found {len(matches2)} matches")
            if matches2:
                sample_match = matches2[0]
                logger.debug(f"Sample match 2: {sample_match.groups()}")
        
        # Use the pattern that found more matches
        matches = matches1 if len(matches1) >= len(matches2) else matches2
        pattern_used = 1 if len(matches1) >= len(matches2) else 2
        
        if self.debug:
            logger.debug(f"Using pattern {pattern_used} with {len(matches)} matches")
        
        transactions = []
        for match in matches:
            details = match.group(1).strip() if match.group(1) else ""
            
            # Handle different patterns
            if pattern_used == 1:
                # Pattern 1: Details, Amount, Date, Balance
                amount_str = match.group(2) if match.group(2) else "0"
                date = match.group(3) if match.group(3) else ""
                balance = match.group(4) if match.group(4) else "0"
                
                # Determine if amount is debit or credit
                if amount_str.endswith('-'):
                    debit = amount_str.rstrip('-')
                    credit = "0"
                else:
                    # Assume credit if no negative sign
                    debit = "0"
                    credit = amount_str
            else:
                # Pattern 2: Details, Amount (optional), Date, Balance
                amount_str = match.group(2) if match.group(2) else "0"
                date = match.group(3) if match.group(3) else ""
                balance = match.group(4) if match.group(4) else "0"
                
                # Determine if amount is debit or credit
                if amount_str.endswith('-'):
                    debit = amount_str.rstrip('-')
                    credit = "0"
                else:
                    # Assume credit if no negative sign
                    debit = "0"
                    credit = amount_str
            
            # Clean and format values
            debit = self.parse_amount(debit)
            credit = self.parse_amount(credit)
            date = self.parse_date(date)
            balance = self.parse_amount(balance, allow_negative=False)
            
            # Check for service fee in the details
            service_fee = 'Y' if '##' in content or 'FEE-' in details else ''
            
            # Create transaction dictionary
            transaction = {
                'Details': details,
                'ServiceFee': service_fee,
                'Debits': debit,
                'Credits': credit,
                'Date': date,
                'Balance': balance
            }
            
            transactions.append(transaction)
            
        if self.debug:
            logger.debug(f"Extracted {len(transactions)} transactions from single-column table")
            
        return transactions
    
    def _extract_header_info(self, pdf_path: str) -> dict:
        """Extract header information from the first page of the PDF"""
        import PyPDF2
        import re
        
        header_info = {
            'accountnumber': '',
            'statementperiod': ''
        }
        
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                first_page = reader.pages[0]
                text = first_page.extract_text()
                
                # Extract account number
                account_match = re.search(r'Account\s+Number\s*\n?\s*(\d+\s*\d+\s*\d+\s*\d+)', text, re.IGNORECASE)
                if account_match:
                    # Remove spaces from account number
                    header_info['accountnumber'] = account_match.group(1).replace(' ', '')
                
                # Extract statement period
                period_match = re.search(r'Statement\s+from\s+(\d+\s+\w+\s+\d{4})\s+to\s+(\d+\s+\w+\s+\d{4})', text, re.IGNORECASE)
                if period_match:
                    header_info['statementperiod'] = f"{period_match.group(1)} - {period_match.group(2)}"
                    
                if self.debug:
                    logger.debug(f"Extracted header info: {header_info}")
                    
        except Exception as e:
            logger.warning(f"Error extracting header info: {e}")
            
        return header_info
        
    def extract_transactions_from_pdf(self, pdf_path: str, password: Optional[str] = None) -> List[Dict]:
        """Extract transactions from all tables in a PDF"""

        # Extract header information first
        header_info = self._extract_header_info(pdf_path)

        # Extract tables from PDF
        tables = self.extract_tables_from_pdf(pdf_path, password)
        
        all_transactions = []
        self.debug_info['tables_processed'] = len(tables)
        
        # Process each table
        for i, table in enumerate(tables):
            logger.info(f"Processing table {i+1}/{len(tables)}")
            
            table_debug = {
                'table_index': i,
                'rows': len(table),
                'columns': len(table.columns),
                'is_transaction_table': False,
                'transactions_extracted': 0
            }
            
            # Extract transactions from table
            transactions = self.extract_transactions_from_table(table)
            
            if transactions:
                table_debug['is_transaction_table'] = True
                table_debug['transactions_extracted'] = len(transactions)
                self.debug_info['transaction_tables'] += 1
                all_transactions.extend(transactions)
            else:
                logger.info(f"Table {i+1} is not a transaction table, skipping")
            
            if self.debug:
                self.debug_info['table_results'].append(table_debug)
        
        # Post-process transactions
        if all_transactions:
            # Add header info to each transaction before post-processing
            for tx in all_transactions:
                tx.update(header_info)
                
            all_transactions = self.post_process_transactions(all_transactions)
        
        self.debug_info['total_transactions'] = len(all_transactions)
        logger.info(f"Extracted {len(all_transactions)} total transactions from PDF using camelot")
        
        return all_transactions
    
    def post_process_transactions(self, transactions: List[Dict]) -> List[Dict]:
        """Post-process extracted transactions to clean and standardize data"""
        clean_transactions = []
        
        for tx in transactions:
            # Normalize keys to lowercase for consistency
            normalized_tx = {k.lower(): v for k, v in tx.items()}
            
            # Skip empty transactions
            if not normalized_tx.get('date', '') and not normalized_tx.get('details', '') and \
               float(normalized_tx.get('debits', 0)) == 0 and float(normalized_tx.get('credits', 0)) == 0:
                continue
                
            # Clean and standardize transaction data
            clean_tx = {
                'Date': normalized_tx.get('date', ''),
                'Details': normalized_tx.get('details', ''),
                'Debits': normalized_tx.get('debits', 0),
                'Credits': normalized_tx.get('credits', 0),
                'Balance': normalized_tx.get('balance', 0),
                'ServiceFee': normalized_tx.get('servicefee', '')
            }
            
            # Add header info if available
            if normalized_tx.get('accountnumber'):
                clean_tx['AccountNumber'] = normalized_tx.get('accountnumber')
            if normalized_tx.get('statementperiod'):
                clean_tx['StatementPeriod'] = normalized_tx.get('statementperiod')
            
            clean_transactions.append(clean_tx)
        
        # Format currency values
        for tx in clean_transactions:
            tx['Debits'] = self.format_currency(tx['Debits'], allow_negative=True) if float(tx['Debits']) != 0 else ''
            tx['Credits'] = self.format_currency(tx['Credits']) if float(tx['Credits']) != 0 else ''
            tx['Balance'] = self.format_currency(tx['Balance']) if float(tx['Balance']) != 0 else ''
        
        # Deduplicate by creating a fingerprint
        unique_transactions = []
        seen_fingerprints = set()
        
        for tx in clean_transactions:
            # Create a fingerprint based on key fields
            fingerprint = f"{tx['Date']}|{tx['Details'][:50]}|{tx['Debits']}|{tx['Credits']}"
            
            if fingerprint not in seen_fingerprints:
                seen_fingerprints.add(fingerprint)
                unique_transactions.append(tx)
                
        if self.debug:
            logger.debug(f"Post-processed {len(transactions)} transactions to {len(unique_transactions)} unique transactions")
            
        return unique_transactions


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Extract bank statement transactions using camelot')
    parser.add_argument('pdf_path', help='Path to the PDF file')
    parser.add_argument('--password', help='Password for encrypted PDF')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--output', help='Path to save transactions (CSV)')
    parser.add_argument('--debug-info', help='Path to save debug info (JSON)')
    parser.add_argument('--compare', action='store_true', help='Compare with other extraction methods')
    
    args = parser.parse_args()
    
    # Set debug level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Set default output path if not specified
    if not args.output:
        pdf_basename = os.path.basename(args.pdf_path)
        output_dir = os.path.join(os.path.dirname(args.pdf_path), '../output')
        os.makedirs(output_dir, exist_ok=True)
        pdf_name = os.path.splitext(pdf_basename)[0]
        args.output = os.path.join(output_dir, f"{pdf_name}_camelot_transactions.csv")
    
    # Extract transactions using camelot
    extractor = CamelotBankStatementParser(debug=args.debug)
    camelot_transactions = extractor.extract_transactions_from_pdf(args.pdf_path, args.password)
    
    # Save transactions to CSV
    if camelot_transactions:
        df = pd.DataFrame(camelot_transactions)
        df.to_csv(args.output, index=False)
        logger.info(f"✅ Saved {len(camelot_transactions)} transactions to {args.output}")
    else:
        logger.warning("❌ No transactions extracted")
        
    # Save debug info if requested
    if args.debug_info:
        with open(args.debug_info, 'w') as f:
            json.dump(extractor.debug_info, f, indent=2)
            
    # Compare with other extraction methods if requested
    if args.compare:
        # Import other extraction methods
        from modules import transaction_extractor
        from modules.lattice_extractor import LatticeExtractor
        from lattice_strict import LatticeStrictExtractor
        
        # Extract transactions using regular mode
        tables = extractor.extract_tables_from_pdf(args.pdf_path, args.password)
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
            
        # Extract transactions using strict lattice mode
        strict_extractor = LatticeStrictExtractor(debug=args.debug)
        strict_lattice_transactions = strict_extractor.extract_transactions_from_pdf(args.pdf_path, args.password)
        
        # Save regular transactions to CSV
        pdf_name = os.path.splitext(os.path.basename(args.pdf_path))[0]
        output_dir = os.path.dirname(args.output)
        
        regular_output = os.path.join(output_dir, f"{pdf_name}_regular_transactions.csv")
        pd.DataFrame(regular_transactions).to_csv(regular_output, index=False)
        
        # Save original lattice transactions to CSV
        lattice_output = os.path.join(output_dir, f"{pdf_name}_lattice_transactions.csv")
        pd.DataFrame(lattice_transactions).to_csv(lattice_output, index=False)
        
        # Save strict lattice transactions to CSV
        strict_output = os.path.join(output_dir, f"{pdf_name}_strict_lattice_transactions.csv")
        pd.DataFrame(strict_lattice_transactions).to_csv(strict_output, index=False)
        
        # Generate comparison report
        comparison_output = os.path.join(output_dir, f"{pdf_name}_full_comparison.txt")
        with open(comparison_output, 'w') as f:
            f.write(f"PDF: {args.pdf_path}\n")
            f.write(f"Regular extraction: {len(regular_transactions)} transactions\n")
            f.write(f"Lattice extraction: {len(lattice_transactions)} transactions\n")
            f.write(f"Strict lattice extraction: {len(strict_lattice_transactions)} transactions\n")
            f.write(f"Camelot extraction: {len(camelot_transactions)} transactions\n\n")
            
            # Compare fields
            regular_fields = set()
            lattice_fields = set()
            strict_fields = set()
            camelot_fields = set()
            
            for tx in regular_transactions:
                if isinstance(tx, dict):
                    regular_fields.update(tx.keys())
            
            for tx in lattice_transactions:
                if isinstance(tx, dict):
                    lattice_fields.update(tx.keys())
                
            for tx in strict_lattice_transactions:
                if isinstance(tx, dict):
                    strict_fields.update(tx.keys())
                
            for tx in camelot_transactions:
                if isinstance(tx, dict):
                    camelot_fields.update(tx.keys())
                
            f.write(f"Fields in regular extraction: {', '.join(sorted(regular_fields))}\n")
            f.write(f"Fields in lattice extraction: {', '.join(sorted(lattice_fields))}\n")
            f.write(f"Fields in strict lattice extraction: {', '.join(sorted(strict_fields))}\n")
            f.write(f"Fields in camelot extraction: {', '.join(sorted(camelot_fields))}\n\n")
            
            f.write(f"Fields only in regular: {', '.join(sorted(regular_fields - lattice_fields - strict_fields - camelot_fields))}\n")
            f.write(f"Fields only in lattice: {', '.join(sorted(lattice_fields - regular_fields - strict_fields - camelot_fields))}\n")
            f.write(f"Fields only in strict lattice: {', '.join(sorted(strict_fields - regular_fields - lattice_fields - camelot_fields))}\n")
            f.write(f"Fields only in camelot: {', '.join(sorted(camelot_fields - regular_fields - lattice_fields - strict_fields))}\n")
            
        logger.info(f"✅ Full comparison saved to: {comparison_output}")
        
    # Log success
    if camelot_transactions:
        logger.info(f"✅ Extracted {len(camelot_transactions)} transactions using camelot")
        logger.info(f"✅ Saved to: {args.output}")
    else:
        logger.error("❌ Failed to extract any transactions")


def combine_statement_csvs(csv_files: List[str], output_file: str, fiscal_year_sorting: bool = False, fiscal_start_month: int = 3, fiscal_start_day: int = 1) -> None:
    """Combine multiple statement CSV files in chronological order
    
    Args:
        csv_files: List of CSV files to combine
        output_file: Output CSV file
        fiscal_year_sorting: If True, sort by fiscal year instead of calendar year
        fiscal_start_month: Month when fiscal year starts (1-12)
        fiscal_start_day: Day when fiscal year starts (1-31)
    """
    import pandas as pd
    import re
    from datetime import datetime
    
    all_dfs = []
    file_dates = []
    
    # First pass: Read all CSVs and extract their statement periods
    for file_path in csv_files:
        try:
            # Read the CSV
            df = pd.read_csv(file_path)
            
            if df.empty:
                logger.warning(f"Empty CSV file: {file_path}")
                continue
                
            # Check if the CSV has the required columns
            if 'StatementPeriod' not in df.columns:
                logger.warning(f"CSV file missing StatementPeriod column: {file_path}")
                continue
                
            # Extract the start date from the statement period
            # Format example: "16 February 2024 - 16 March 2024"
            statement_period = df['StatementPeriod'].iloc[0]
            
            # Extract start date using regex
            start_date_match = re.search(r'(\d+\s+\w+\s+\d{4})', statement_period)
            if not start_date_match:
                logger.warning(f"Could not extract date from statement period: {statement_period}")
                continue
                
            start_date_str = start_date_match.group(1)
            
            try:
                # Parse the date
                start_date = datetime.strptime(start_date_str, '%d %B %Y')
                
                # Store the dataframe and its start date
                all_dfs.append(df)
                file_dates.append(start_date)
                logger.info(f"Added {len(df)} transactions from {file_path} with period starting {start_date_str}")
                
            except ValueError as e:
                logger.warning(f"Error parsing date '{start_date_str}': {e}")
                continue
                
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
    
    if not all_dfs:
        logger.error("No valid CSV files found to combine")
        return
    
    # Sort dataframes by their statement period start dates
    sorted_dfs = [df for _, df in sorted(zip(file_dates, all_dfs))]
    
    # Combine all dataframes
    combined_df = pd.concat(sorted_dfs, ignore_index=True)
    
    # Sort transactions by date within each statement period
    # First convert Date column to datetime if possible
    try:
        # Try different date formats
        date_formats = ['%d/%m', '%d/%m/%Y', '%Y-%m-%d']
        converted = False
        
        for date_format in date_formats:
            try:
                # Add a dummy year if the format doesn't include year
                if date_format == '%d/%m':
                    # Extract year from StatementPeriod
                    combined_df['TempYear'] = combined_df['StatementPeriod'].str.extract(r'(\d{4})').iloc[:, 0]
                    # Combine with Date
                    combined_df['FullDate'] = combined_df.apply(
                        lambda row: f"{row['Date']}/{row['TempYear']}" if pd.notna(row['Date']) else None, 
                        axis=1
                    )
                    combined_df['DateObj'] = pd.to_datetime(combined_df['FullDate'], format='%d/%m/%Y', errors='coerce')
                    combined_df.drop(['TempYear', 'FullDate'], axis=1, inplace=True)
                else:
                    combined_df['DateObj'] = pd.to_datetime(combined_df['Date'], format=date_format, errors='coerce')
                    
                if not combined_df['DateObj'].isna().all():
                    converted = True
                    break
            except Exception as e:
                logger.debug(f"Failed to parse dates with format {date_format}: {e}")
        
        # Add fiscal period based on transaction date if requested
        if fiscal_year_sorting and converted:
            # Create a function to determine fiscal year for each transaction
            def get_fiscal_period(date_obj):
                if pd.isna(date_obj):
                    return None
                    
                # Get month and day from date
                month = date_obj.month
                day = date_obj.day
                year = date_obj.year
                
                # If date is before fiscal year start, it belongs to previous fiscal year
                if month < fiscal_start_month or (month == fiscal_start_month and day < fiscal_start_day):
                    fiscal_year = year - 1
                else:
                    fiscal_year = year
                    
                fiscal_year_end = fiscal_year + 1
                return f"FY{fiscal_year}-{fiscal_year_end}"
            
            # Apply fiscal period to each transaction
            combined_df['FiscalPeriod'] = combined_df['DateObj'].apply(get_fiscal_period)
            logger.info(f"Added fiscal periods to transactions based on individual transaction dates")
            
            # Sort by fiscal period, then by date
            combined_df.sort_values(['FiscalPeriod', 'DateObj'], inplace=True)
        elif converted:
            # Just sort by date within statement period
            combined_df.sort_values(['StatementPeriod', 'DateObj'], inplace=True)
        else:
            logger.warning("Could not convert dates to datetime for sorting, using original order")
            
        # Clean up temporary columns
        if 'DateObj' in combined_df.columns:
            combined_df.drop('DateObj', axis=1, inplace=True)
            
    except Exception as e:
        logger.warning(f"Error processing dates: {e}")
    
    # Save combined CSV
    combined_df.to_csv(output_file, index=False)
    logger.info(f"✅ Combined {len(all_dfs)} CSV files with {len(combined_df)} total transactions")
    logger.info(f"✅ Saved to: {output_file}")


if __name__ == "__main__":
    # Check if the command is to combine CSVs
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "combine":
        # Parse arguments for combine function
        parser = argparse.ArgumentParser(description='Combine multiple statement CSV files')
        parser.add_argument('--files', nargs='+', required=True, help='List of CSV files to combine')
        parser.add_argument('--output', required=True, help='Output CSV file')
        parser.add_argument('--fiscal-year', action='store_true', help='Sort by fiscal year instead of calendar year')
        parser.add_argument('--fiscal-start-month', type=int, default=3, help='Month when fiscal year starts (1-12)')
        parser.add_argument('--fiscal-start-day', type=int, default=1, help='Day when fiscal year starts (1-31)')
        
        # Parse only the relevant args
        combine_args, _ = parser.parse_known_args(sys.argv[2:])
        
        # Combine CSV files
        combine_statement_csvs(
            combine_args.files, 
            combine_args.output, 
            fiscal_year_sorting=combine_args.fiscal_year,
            fiscal_start_month=combine_args.fiscal_start_month,
            fiscal_start_day=combine_args.fiscal_start_day
        )
    else:
        # Run the normal main function
        main()
