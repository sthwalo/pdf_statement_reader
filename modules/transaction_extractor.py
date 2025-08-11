#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Transaction Extraction Module

Responsible for extracting transaction data from identified tables
based on column mappings.
"""

import re
import json
import logging
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def extract_json_text(value):
    """Extract text from JSON-like string"""
    if not value or pd.isna(value):
        return ""
    
    value_str = str(value).strip()
    
    # Check if it looks like a JSON object
    if value_str.startswith('{') and value_str.endswith('}'):
        try:
            # Try to parse as JSON
            data = json.loads(value_str.replace("'", '"'))
            if isinstance(data, dict) and 'text' in data:
                return str(data['text'])
        except:
            # If not valid JSON, try regex extraction
            text_match = re.search(r"'text':\s*'([^']+)'", value_str)
            if text_match:
                return text_match.group(1)
    
    return value_str

def has_meaningful_data(row_data):
    """Check if a row has meaningful transaction data"""
    # Skip rows that are likely headers
    if row_data.get('Details', '').lower() in ['details', 'description'] and \
       (row_data.get('Debits', '').lower() == 'debits' or row_data.get('Credits', '').lower() == 'credits'):
        return False
    
    # Special case for balance brought forward - always keep these
    if 'balance brought forward' in row_data.get('Details', '').lower():
        return True
    
    # Skip rows with malformed data (likely parsing errors)
    if len(row_data.get('Details', '')) > 100 or \
       (row_data.get('Date', '') and len(row_data.get('Date', '')) > 10) or \
       (row_data.get('Balance', '') and len(row_data.get('Balance', '')) > 30):
        return False
    
    # Skip rows with no financial data
    has_amount = bool(row_data.get('Debits') or row_data.get('Credits') or row_data.get('ServiceFee'))
    has_details = bool(row_data.get('Details'))
    
    return has_amount or (has_details and row_data.get('Date'))

def extract_transactions_from_table(df, column_mapping, header_row=None, debug=False):
    """
    Extract transaction data from a table based on column mapping
    
    Args:
        df (DataFrame): Pandas DataFrame containing the table
        column_mapping (dict): Mapping of column indices to field names
        header_row (int, optional): Index of header row to skip
        debug (bool, optional): Enable debug mode
        
    Returns:
        list: List of transaction dictionaries
        dict: Debug info if debug=True
    """
    transactions = []
    
    debug_info = {
        'total_rows': len(df),
        'header_row': header_row,
        'column_mapping': column_mapping,
        'skipped_rows': [],
        'extracted_rows': [],
        'error': None
    }
    
    try:
        # Skip header row if identified
        start_row = header_row + 1 if header_row is not None else 0
        
        # Process each row
        prev_transaction = None
        for i in range(start_row, len(df)):
            row_data = {
                'Details': '',
                'ServiceFee': '',
                'Debits': '',
                'Credits': '',
                'Date': '',
                'Balance': ''
            }
            
            # Extract data from each column based on mapping
            for field, col_idx in column_mapping.items():
                # Skip metadata fields (those starting with underscore)
                if field.startswith('_'):
                    continue
                    
                if col_idx is not None and col_idx < len(df.columns):
                    value = df.iloc[i, col_idx]
                    # Extract text from JSON-like strings
                    extracted_value = extract_json_text(value)
                    
                    # For lattice mode, check adjacent columns if they exist
                    is_lattice = column_mapping.get('_lattice_mode', False)
                    adjacent_columns = column_mapping.get('_adjacent_columns', {})
                    
                    if is_lattice and field in adjacent_columns:
                        adj_col = adjacent_columns[field]
                        if adj_col < len(df.columns):
                            adj_value = df.iloc[i, adj_col]
                            adj_extracted = extract_json_text(adj_value)
                            
                            # Combine values based on field type
                            if field == 'Details':
                                # For details, concatenate with space if both have content
                                if extracted_value and adj_extracted:
                                    extracted_value = f"{extracted_value} {adj_extracted}"
                                elif not extracted_value and adj_extracted:
                                    extracted_value = adj_extracted
                            elif field in ['Debits', 'Credits', 'Balance']:
                                # For amount fields, use the one that looks like a number
                                if re.search(r'\d+\.\d{2}|\d+,\d{2}|-\d+', adj_extracted) and not re.search(r'\d+\.\d{2}|\d+,\d{2}|-\d+', extracted_value):
                                    extracted_value = adj_extracted
                            elif field == 'Date':
                                # For date, use the one that looks like a date
                                if re.search(r'\d{1,2}[/\-\.]\d{1,2}|\d{4}-\d{2}-\d{2}', adj_extracted) and not re.search(r'\d{1,2}[/\-\.]\d{1,2}|\d{4}-\d{2}-\d{2}', extracted_value):
                                    extracted_value = adj_extracted
                    
                    row_data[field] = extracted_value
            
            # Skip rows that are likely headers or non-transaction data
            if row_data.get('Details', '').lower() in ['details', 'description', 'opening balance', 'closing balance'] or \
               row_data.get('Debits', '').lower() == 'debits' or \
               row_data.get('Credits', '').lower() == 'credits':
                debug_info['skipped_rows'].append({
                    'row_index': i,
                    'reason': 'header_like_content',
                    'content': row_data
                })
                continue
            
            # Check if this is a continuation of previous transaction (multiline details)
            is_continuation = False
            if prev_transaction:
                # Case 1: No date, debits, or credits - likely a continuation of details
                if not row_data.get('Date') and not row_data.get('Debits') and not row_data.get('Credits') and row_data.get('Details'):
                    # This is likely a continuation of the previous transaction's details
                    prev_transaction['Details'] += ' ' + row_data.get('Details', '')
                    is_continuation = True
                    debug_info['skipped_rows'].append({
                        'row_index': i,
                        'reason': 'merged_with_previous',
                        'content': row_data
                    })
                # Case 2: Same date as previous transaction with only details - could be a continuation
                elif row_data.get('Date') == prev_transaction.get('Date') and row_data.get('Details') and not row_data.get('Debits') and not row_data.get('Credits'):
                    # This could be additional details for the same transaction
                    prev_transaction['Details'] += ' ' + row_data.get('Details', '')
                    is_continuation = True
                    debug_info['skipped_rows'].append({
                        'row_index': i,
                        'reason': 'merged_with_previous_same_date',
                        'content': row_data
                    })
            
            # Add transaction if it has meaningful data and is not a continuation
            if not is_continuation and has_meaningful_data(row_data):
                transactions.append(row_data)
                prev_transaction = row_data  # Set as previous for potential continuation
                debug_info['extracted_rows'].append({
                    'row_index': i,
                    'content': row_data
                })
            elif not is_continuation:
                debug_info['skipped_rows'].append({
                    'row_index': i,
                    'reason': 'no_meaningful_data',
                    'content': row_data
                })
        
        logger.info(f"Extracted {len(transactions)} transactions from table")
        
        if debug:
            return transactions, debug_info
        return transactions
    
    except Exception as e:
        error_msg = f"Error extracting transactions: {e}"
        logger.error(error_msg)
        debug_info['error'] = error_msg
        
        if debug:
            return [], debug_info
        return []

def extract_transactions_from_pdf(tables, debug=False):
    """
    Extract transactions from all tables in a PDF
    
    Args:
        tables (list): List of pandas DataFrames containing tables
        debug (bool, optional): Enable debug mode
        
    Returns:
        list: List of transaction dictionaries
        dict: Debug info if debug=True
    """
    import sys
    sys.path.append('..')
    from modules.transaction_identifier import is_transaction_table, identify_columns
    
    all_transactions = []
    
    debug_info = {
        'total_tables': len(tables),
        'transaction_tables': 0,
        'non_transaction_tables': 0,
        'table_results': [],
        'total_transactions': 0,
        'error': None
    }
    
    try:
        # Process each table
        for i, table in enumerate(tables):
            logger.info(f"Processing table {i+1}/{len(tables)}")
            
            table_debug = {
                'table_index': i,
                'is_transaction_table': False,
                'transactions_extracted': 0
            }
            
            # Check if this is a transaction table
            if debug:
                is_tx, tx_debug = is_transaction_table(table, debug=True)
                table_debug['transaction_analysis'] = tx_debug
            else:
                is_tx = is_transaction_table(table)
            
            table_debug['is_transaction_table'] = is_tx
            
            if is_tx:
                debug_info['transaction_tables'] += 1
                
                # Get extraction mode from table metadata if available
                extraction_mode = None
                if hasattr(table, '_extraction_mode'):
                    extraction_mode = table._extraction_mode
                    logger.info(f"Table {i+1} has extraction mode metadata: {extraction_mode}")
                # Fall back to detection based on table structure
                elif len(table.columns) > 10:  # Lattice mode often creates more columns
                    extraction_mode = 'lattice'
                    logger.info(f"Table {i+1} appears to be from lattice mode (columns: {len(table.columns)})")
                
                # Get page number if available
                page_number = getattr(table, '_page_number', None)
                if page_number:
                    logger.info(f"Table {i+1} is from page {page_number}")
                    table_debug['page_number'] = page_number
                
                # Identify columns
                if debug:
                    (column_mapping, header_row), col_debug = identify_columns(table, debug=True, extraction_mode=extraction_mode)
                    table_debug['column_mapping_analysis'] = col_debug
                    table_debug['extraction_mode'] = extraction_mode or col_debug.get('extraction_mode')
                else:
                    column_mapping, header_row = identify_columns(table, extraction_mode=extraction_mode)
                
                table_debug['column_mapping'] = column_mapping
                table_debug['header_row'] = header_row
                
                # Extract transactions
                if debug:
                    transactions, tx_extract_debug = extract_transactions_from_table(
                        table, column_mapping, header_row, debug=True
                    )
                    table_debug['extraction_details'] = tx_extract_debug
                else:
                    transactions = extract_transactions_from_table(table, column_mapping, header_row)
                
                all_transactions.extend(transactions)
                table_debug['transactions_extracted'] = len(transactions)
                debug_info['total_transactions'] += len(transactions)
            else:
                debug_info['non_transaction_tables'] += 1
                logger.info(f"Table {i+1} is not a transaction table, skipping")
            
            debug_info['table_results'].append(table_debug)
        
        logger.info(f"Extracted {len(all_transactions)} total transactions from PDF")
        
        if debug:
            return all_transactions, debug_info
        return all_transactions
    
    except Exception as e:
        error_msg = f"Error processing tables: {e}"
        logger.error(error_msg)
        debug_info['error'] = error_msg
        
        if debug:
            return [], debug_info
        return []

def main():
    """Test function for direct module execution"""
    import argparse
    import json
    import sys
    
    # Add parent directory to path for imports
    sys.path.append('..')
    from modules.table_extractor import extract_tables_from_pdf
    
    parser = argparse.ArgumentParser(description='Extract transactions from PDF')
    parser.add_argument('pdf_path', help='Path to the PDF file')
    parser.add_argument('--password', help='Password for encrypted PDF')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--output', help='Path to save debug info (JSON)')
    parser.add_argument('--transactions', help='Path to save raw transactions (JSON)')
    
    args = parser.parse_args()
    
    # Extract tables
    tables = extract_tables_from_pdf(args.pdf_path, args.password)
    
    # Extract transactions
    if args.debug:
        transactions, debug_info = extract_transactions_from_pdf(tables, debug=True)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(debug_info, f, indent=2)
            print(f"Debug info saved to {args.output}")
    else:
        transactions = extract_transactions_from_pdf(tables)
    
    print(f"Extracted {len(transactions)} transactions")
    
    if args.transactions:
        with open(args.transactions, 'w') as f:
            json.dump(transactions, f, indent=2)
        print(f"Transactions saved to {args.transactions}")

if __name__ == '__main__':
    main()
