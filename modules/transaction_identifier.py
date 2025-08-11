#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Transaction Identification Module

Responsible for identifying which tables contain transaction data
and mapping columns to expected transaction fields.
"""

import re
import logging
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def is_transaction_table(df, debug=False):
    """
    Identify if a dataframe contains transaction data based on content patterns
    
    Args:
        df (DataFrame): Pandas DataFrame to analyze
        debug (bool): Enable debug mode with detailed logging
        
    Returns:
        bool: True if table contains transaction data
        dict: Debug info if debug=True
    """
    debug_info = {
        'has_date': False,
        'has_amount': False,
        'has_details': False,
        'date_columns': [],
        'amount_columns': [],
        'details_columns': [],
        'row_count': len(df) if df is not None else 0,
        'column_count': len(df.columns) if df is not None else 0
    }
    
    if df is None or df.empty or len(df) < 2:  # Need at least header + one row
        if debug:
            return False, debug_info
        return False
    
    try:
        # Convert all values to string for pattern matching
        df_str = df.astype(str)
        
        # Look for column headers or content patterns
        has_date = False
        has_amount = False
        has_details = False
        
        # Check each column for patterns
        for col_idx in range(len(df.columns)):
            col_values = df_str.iloc[:, col_idx].str.lower()
            
            # Check for date patterns (DD/MM or DD-MM)
            date_pattern = r'\d{1,2}[/\-\.]\d{1,2}|\d{4}-\d{2}-\d{2}'
            if col_values.str.contains(date_pattern, regex=True, na=False).any():
                has_date = True
                debug_info['date_columns'].append(col_idx)
                logger.info(f"Found date pattern in column {col_idx}")
                
            # Check for amount patterns (numbers with decimal points)
            amount_pattern = r'\d+\.\d{2}|\d+,\d{2}|-\d+\.\d{2}'
            if col_values.str.contains(amount_pattern, regex=True, na=False).any():
                has_amount = True
                debug_info['amount_columns'].append(col_idx)
                logger.info(f"Found amount pattern in column {col_idx}")
                
            # Check for transaction details
            details_pattern = r'payment|transfer|debit|credit|withdrawal|deposit|fee|charge|interest'
            if col_values.str.contains(details_pattern, case=False, regex=True, na=False).any():
                has_details = True
                debug_info['details_columns'].append(col_idx)
                logger.info(f"Found transaction details in column {col_idx}")
        
        # If we have at least two of the three features, it's likely a transaction table
        is_tx_table = (has_date and has_amount) or (has_date and has_details) or (has_amount and has_details)
        
        debug_info['has_date'] = has_date
        debug_info['has_amount'] = has_amount
        debug_info['has_details'] = has_details
        
        logger.info(f"Table analysis - has_date: {has_date}, has_amount: {has_amount}, has_details: {has_details}, is_transaction_table: {is_tx_table}")
        
        if debug:
            return is_tx_table, debug_info
        return is_tx_table
    
    except Exception as e:
        error_msg = f"Error in is_transaction_table: {e}"
        logger.error(error_msg)
        debug_info['error'] = error_msg
        
        if debug:
            return False, debug_info
        return False

def identify_columns(df, debug=False, extraction_mode=None):
    """
    Identify which columns correspond to our target fields
    
    Args:
        df (DataFrame): Pandas DataFrame to analyze
        debug (bool): Enable debug mode with detailed logging
        extraction_mode (str, optional): 'stream' or 'lattice' to apply mode-specific logic
        
    Returns:
        tuple: (column_mapping, header_row)
        dict: Debug info if debug=True
    """
    column_mapping = {
        'Details': None,
        'ServiceFee': None,
        'Debits': None,
        'Credits': None,
        'Date': None,
        'Balance': None
    }
    
    header_row = None
    
    debug_info = {
        'header_candidates': [],
        'column_mapping_steps': [],
        'header_row': None,
        'final_mapping': {},
        'extraction_mode': extraction_mode,
        'error': None
    }
    
    # Try to detect if this is a lattice-extracted table
    if extraction_mode is None:
        # Lattice tables often have more columns and sometimes empty columns
        empty_cols = sum(df.iloc[:, i].astype(str).str.strip().eq('').all() for i in range(len(df.columns)))
        narrow_cols = sum(df.iloc[:, i].astype(str).str.len().mean() < 3 for i in range(len(df.columns)))
        
        if len(df.columns) > 10 or empty_cols > 2 or narrow_cols > 2:
            extraction_mode = 'lattice'
            logger.info(f"Detected lattice mode table (columns: {len(df.columns)}, empty: {empty_cols}, narrow: {narrow_cols})")
        else:
            extraction_mode = 'stream'
            logger.info(f"Detected stream mode table (columns: {len(df.columns)}, empty: {empty_cols}, narrow: {narrow_cols})")
        
        debug_info['extraction_mode'] = extraction_mode
    
    try:
        # First check for header row
        for i in range(min(5, len(df))):
            row = df.iloc[i].astype(str)
            header_values = [str(val).lower() for val in row]
            
            debug_info['header_candidates'].append({
                'row': i,
                'values': header_values
            })
            
            # Check if this row contains standard column headers
            if ('detail' in ' '.join(header_values) or 'description' in ' '.join(header_values)) and \
               ('debit' in ' '.join(header_values) or 'credit' in ' '.join(header_values)):
                header_row = i
                debug_info['header_row'] = i
                break
        
        # First, check all rows for potential header information
        # This helps with multi-row headers where date might be in one row and debits/credits in another
        potential_headers = {}
        for i in range(min(5, len(df))):
            row = df.iloc[i].astype(str)
            for col in range(len(df.columns)):
                if col >= len(df.columns):
                    continue
                    
                cell_val = str(row[col]).lower()
                if 'detail' in cell_val or 'description' in cell_val:
                    potential_headers.setdefault('Details', []).append((col, i))
                elif 'fee' in cell_val or 'service' in cell_val:
                    potential_headers.setdefault('ServiceFee', []).append((col, i))
                elif 'debit' in cell_val or 'withdrawal' in cell_val:
                    potential_headers.setdefault('Debits', []).append((col, i))
                elif 'credit' in cell_val or 'deposit' in cell_val:
                    potential_headers.setdefault('Credits', []).append((col, i))
                elif 'date' in cell_val:
                    potential_headers.setdefault('Date', []).append((col, i))
                elif 'balance' in cell_val:
                    potential_headers.setdefault('Balance', []).append((col, i))
        
        # If header row found, map columns based on headers
        if header_row is not None:
            mapping_step = {'source': 'header_row', 'mapping': {}}
            
            for col in range(len(df.columns)):
                if col >= len(df.columns):
                    continue
                    
                header_val = str(df.iloc[header_row, col]).lower()
                
                if 'detail' in header_val or 'description' in header_val:
                    column_mapping['Details'] = col
                    mapping_step['mapping']['Details'] = {'col': col, 'header': header_val}
                elif 'fee' in header_val or 'service' in header_val:
                    column_mapping['ServiceFee'] = col
                    mapping_step['mapping']['ServiceFee'] = {'col': col, 'header': header_val}
                elif 'debit' in header_val or 'withdrawal' in header_val:
                    column_mapping['Debits'] = col
                    mapping_step['mapping']['Debits'] = {'col': col, 'header': header_val}
                elif 'credit' in header_val or 'deposit' in header_val:
                    column_mapping['Credits'] = col
                    mapping_step['mapping']['Credits'] = {'col': col, 'header': header_val}
                elif 'date' in header_val:
                    column_mapping['Date'] = col
                    mapping_step['mapping']['Date'] = {'col': col, 'header': header_val}
                elif 'balance' in header_val:
                    column_mapping['Balance'] = col
                    mapping_step['mapping']['Balance'] = {'col': col, 'header': header_val}
                    
            # If we didn't find certain columns in the header row, check the potential headers
            for field, cols in potential_headers.items():
                if column_mapping[field] is None and cols:
                    # Use the most common column for this field
                    from collections import Counter
                    # Extract just the column numbers for counting
                    col_nums = [col_tuple[0] for col_tuple in cols]
                    most_common_col = Counter(col_nums).most_common(1)[0][0]
                    column_mapping[field] = most_common_col
                    mapping_step['mapping'][field] = {
                        'col': most_common_col, 
                        'method': 'multi_row_header',
                        'header_rows': [row for col, row in cols if col == most_common_col]
                    }
            
            debug_info['column_mapping_steps'].append(mapping_step)
        
        # Apply potential headers even if no header row was found
        if header_row is None and potential_headers:
            mapping_step = {'source': 'potential_headers', 'mapping': {}}
            
            for field, cols in potential_headers.items():
                if column_mapping[field] is None and cols:
                    # Use the most common column for this field
                    from collections import Counter
                    # Extract just the column numbers for counting
                    col_nums = [col_tuple[0] for col_tuple in cols]
                    most_common_col = Counter(col_nums).most_common(1)[0][0]
                    column_mapping[field] = most_common_col
                    mapping_step['mapping'][field] = {
                        'col': most_common_col, 
                        'method': 'multi_row_header',
                        'header_rows': [row for col, row in cols if col == most_common_col]
                    }
            
            debug_info['column_mapping_steps'].append(mapping_step)
        
        # Apply lattice-specific adjustments if needed
        if extraction_mode == 'lattice':
            # Lattice mode often creates extra columns - check for merged cells that might be split
            # Look for adjacent columns that might belong together
            for field in ['Details', 'Date', 'Debits', 'Credits', 'Balance']:
                if column_mapping[field] is not None:
                    continue
                    
                # For each potential field, check adjacent columns for related content
                for col in range(len(df.columns) - 1):
                    if col in column_mapping.values():
                        continue
                        
                    # Check this column and next column together
                    combined_values = []
                    for i in range(min(10, len(df))):
                        val1 = str(df.iloc[i, col]) if col < len(df.columns) else ''
                        val2 = str(df.iloc[i, col+1]) if col+1 < len(df.columns) else ''
                        combined_values.append(f"{val1} {val2}".lower())
                    
                    combined_text = ' '.join(combined_values)
                    
                    # Check for field-specific patterns in combined text
                    if field == 'Date' and re.search(r'\d{1,2}[/\-\.]\d{1,2}|\d{4}-\d{2}-\d{2}', combined_text):
                        column_mapping['Date'] = col
                        logger.info(f"Lattice mode: Identified Date column at {col} from combined columns")
                        break
                    elif field == 'Details' and len(combined_text) > 50:
                        column_mapping['Details'] = col
                        logger.info(f"Lattice mode: Identified Details column at {col} from combined columns")
                        break
                    elif field in ['Debits', 'Credits'] and re.search(r'\d+\.\d{2}|\d+,\d{2}', combined_text):
                        # Assign first unassigned amount column
                        if column_mapping['Debits'] is None:
                            column_mapping['Debits'] = col
                            logger.info(f"Lattice mode: Identified Debits column at {col} from combined columns")
                        elif column_mapping['Credits'] is None:
                            column_mapping['Credits'] = col
                            logger.info(f"Lattice mode: Identified Credits column at {col} from combined columns")
                        break
                    elif field == 'Balance' and re.search(r'balance|\d{4,}\.\d{2}|\d{4,},\d{2}', combined_text):
                        column_mapping['Balance'] = col
                        logger.info(f"Lattice mode: Identified Balance column at {col} from combined columns")
                        break
        
        # If columns still missing, identify by content patterns
        if None in column_mapping.values():
            mapping_step = {'source': 'content_patterns', 'mapping': {}}
            
            # Find details column (usually has longest text)
            if column_mapping['Details'] is None:
                text_lengths = []
                for col in range(len(df.columns)):
                    if col in column_mapping.values():
                        text_lengths.append(0)
                        continue
                    
                    col_values = df.iloc[3:, col].astype(str) if len(df) > 3 else df.iloc[:, col].astype(str)
                    avg_len = col_values.str.len().mean()
                    text_lengths.append(avg_len)
                
                if text_lengths and max(text_lengths) > 10:
                    column_mapping['Details'] = text_lengths.index(max(text_lengths))
                    mapping_step['mapping']['Details'] = {
                        'col': text_lengths.index(max(text_lengths)), 
                        'method': 'longest_text',
                        'avg_length': max(text_lengths)
                    }
            
            # Find amount columns first to avoid misidentifying monetary columns as dates
            amount_cols = []
            for col in range(len(df.columns)):
                if col in column_mapping.values():
                    continue
                
                col_values = df.iloc[:, col].astype(str)
                # Look for monetary patterns: decimal numbers, commas, currency symbols, trailing negatives
                if col_values.str.contains(r'\d+\.\d{2}|\d+,\d{2}|-\d+|\d+\.\d{2}-|\d+,\d{2}-|[$€£¥]', regex=True, na=False).any():
                    amount_cols.append(col)
            
            # Assign amount columns based on content
            if amount_cols:
                # Sort by column index
                amount_cols.sort()
                
                # Check for balance column first - look for columns with larger numbers or "balance" in nearby text
                balance_col = None
                
                # First, check for explicit balance indicators in column headers or nearby cells
                balance_keywords = ['balance', 'closing', 'opening', 'bal', 'b/fwd', 'c/fwd']
                for r in range(min(10, len(df))):
                    for col in range(len(df.columns)):
                        cell_val = str(df.iloc[r, col]).lower()
                        
                        # Check if any balance keyword is in this cell
                        if any(keyword in cell_val for keyword in balance_keywords):
                            # Look for a nearby numeric column (current, previous, or next column)
                            for c in range(max(0, col-1), min(len(df.columns), col+2)):
                                if c in amount_cols:
                                    balance_col = c
                                    mapping_step['mapping']['Balance'] = {
                                        'col': c,
                                        'method': 'balance_keyword_proximity',
                                        'keyword': cell_val
                                    }
                                    break
                        
                        if balance_col is not None:
                            break
                    if balance_col is not None:
                        break
                
                # If no explicit balance indicators, check for columns with larger numbers
                if balance_col is None:
                    for col in amount_cols:
                        col_values = df.iloc[:, col].astype(str)
                        numeric_values = []
                        for val in col_values:
                            val = re.sub(r'[^0-9.,\-]', '', val)
                            if val:
                                try:
                                    # Convert to numeric, handling commas as thousands separators
                                    val = float(val.replace(',', ''))
                                    numeric_values.append(val)
                                except ValueError:
                                    pass
                        
                        # If this column has mostly large values (> 1000), it's likely a balance column
                        if numeric_values and sum(abs(v) > 1000 for v in numeric_values) / len(numeric_values) > 0.5:
                            balance_col = col
                            mapping_step['mapping']['Balance'] = {
                                'col': col,
                                'method': 'large_values_heuristic',
                                'avg_value': sum(abs(v) for v in numeric_values) / len(numeric_values)
                            }
                            break
                        
                        # Check if this column has consistently increasing or decreasing values
                        if len(numeric_values) >= 3:
                            differences = [numeric_values[i+1] - numeric_values[i] for i in range(len(numeric_values)-1)]
                            if all(d < 0 for d in differences) or all(d > 0 for d in differences):
                                balance_col = col
                                mapping_step['mapping']['Balance'] = {
                                    'col': col,
                                    'method': 'consistent_trend',
                                    'trend': 'increasing' if all(d > 0 for d in differences) else 'decreasing'
                                }
                                break
                
                # If we found a balance column, assign it
                if balance_col is not None and column_mapping['Balance'] is None:
                    column_mapping['Balance'] = balance_col
                    mapping_step['mapping']['Balance'] = {
                        'col': balance_col, 
                        'method': 'balance_detection'
                    }
                    amount_cols.remove(balance_col)
                
                # Check for negative values (likely debits)
                for col in amount_cols:
                    col_values = df.iloc[:, col].astype(str)
                    if col_values.str.contains('-', na=False).any() or col_values.str.contains('debit', case=False, na=False).any():
                        column_mapping['Debits'] = col
                        mapping_step['mapping']['Debits'] = {
                            'col': col, 
                            'method': 'negative_values'
                        }
                        amount_cols.remove(col)
                        break
                
                # Assign remaining columns
                if amount_cols and column_mapping['Credits'] is None:
                    column_mapping['Credits'] = amount_cols[0]
                    mapping_step['mapping']['Credits'] = {
                        'col': amount_cols[0], 
                        'method': 'positive_values'
                    }
                    amount_cols.pop(0)
                
                # If we still don't have a balance column and have remaining amount columns
                if amount_cols and column_mapping['Balance'] is None:
                    column_mapping['Balance'] = amount_cols[0]
                    mapping_step['mapping']['Balance'] = {
                        'col': amount_cols[0], 
                        'method': 'remaining_amount'
                    }
            
            # Find date column after identifying amount columns
            if column_mapping['Date'] is None:
                for col in range(len(df.columns)):
                    if col in column_mapping.values():
                        continue
                    
                    col_values = df.iloc[:, col].astype(str)
                    
                    # More specific date pattern that avoids matching monetary values
                    # Look for patterns like DD/MM, DD-MM, YYYY-MM-DD
                    date_pattern = r'(\d{1,2}[/\-\.]\d{1,2}([/\-\.]\d{2,4})?|\d{4}-\d{2}-\d{2})'
                    
                    # Check if column contains date patterns
                    if col_values.str.contains(date_pattern, regex=True, na=False).any():
                        # Additional validation: ensure it's not a monetary value column
                        # by checking for currency indicators and decimal points with 2 digits
                        monetary_pattern = r'\d+\.\d{2}|\d+,\d{2}|-\d+|\d+\.\d{2}-|\d+,\d{2}-|[$€£¥]'
                        monetary_matches = col_values.str.contains(monetary_pattern, regex=True, na=False).sum()
                        date_matches = col_values.str.contains(date_pattern, regex=True, na=False).sum()
                        
                        # Only assign as date column if there are more date patterns than monetary patterns
                        if date_matches > monetary_matches:
                            column_mapping['Date'] = col
                            mapping_step['mapping']['Date'] = {
                                'col': col, 
                                'method': 'date_pattern',
                                'date_matches': int(date_matches),
                                'monetary_matches': int(monetary_matches)
                            }
                            break
            # Amount columns are already processed before date columns
            
            debug_info['column_mapping_steps'].append(mapping_step)
        
        debug_info['final_mapping'] = {k: v for k, v in column_mapping.items()}
        logger.info(f"Column mapping: {column_mapping}")
        
        # For lattice mode, add adjacent column indices to help with data extraction
        if extraction_mode == 'lattice':
            column_mapping['_lattice_mode'] = True
            column_mapping['_adjacent_columns'] = {}
            
            # For each identified column, add adjacent column if it exists and isn't mapped
            for field, col_idx in column_mapping.items():
                if field.startswith('_') or col_idx is None:
                    continue
                    
                # Check if next column is unmapped and could be part of this field
                if col_idx + 1 < len(df.columns) and (col_idx + 1) not in column_mapping.values():
                    column_mapping['_adjacent_columns'][field] = col_idx + 1
                    logger.info(f"Added adjacent column {col_idx + 1} for {field}")
        
        if debug:
            return (column_mapping, header_row), debug_info
        return column_mapping, header_row
    
    except Exception as e:
        error_msg = f"Error in identify_columns: {e}"
        logger.error(error_msg)
        debug_info['error'] = error_msg
        
        if debug:
            return (column_mapping, header_row), debug_info
        return column_mapping, header_row

def main():
    """Test function for direct module execution"""
    import argparse
    import json
    import sys
    
    # Add parent directory to path for imports
    sys.path.append('..')
    from modules.table_extractor import extract_tables_from_pdf
    
    parser = argparse.ArgumentParser(description='Identify transaction tables and columns')
    parser.add_argument('pdf_path', help='Path to the PDF file')
    parser.add_argument('--password', help='Password for encrypted PDF')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--output', help='Path to save debug info (JSON)')
    
    args = parser.parse_args()
    
    # Extract tables
    tables = extract_tables_from_pdf(args.pdf_path, args.password)
    
    results = []
    
    # Analyze each table
    for i, table in enumerate(tables):
        print(f"\nAnalyzing table {i+1}/{len(tables)}")
        
        if args.debug:
            is_tx, tx_debug = is_transaction_table(table, debug=True)
            
            if is_tx:
                col_result, col_debug = identify_columns(table, debug=True)
                
                result = {
                    'table_index': i,
                    'is_transaction_table': is_tx,
                    'transaction_debug': tx_debug,
                    'column_mapping': col_debug['final_mapping'],
                    'header_row': col_debug['header_row'],
                    'column_debug': col_debug
                }
            else:
                result = {
                    'table_index': i,
                    'is_transaction_table': is_tx,
                    'transaction_debug': tx_debug
                }
            
            results.append(result)
            print(f"Table {i+1}: {'Transaction table' if is_tx else 'Not a transaction table'}")
        else:
            is_tx = is_transaction_table(table)
            
            if is_tx:
                col_mapping, header_row = identify_columns(table)
                print(f"Table {i+1}: Transaction table")
                print(f"Column mapping: {col_mapping}")
                print(f"Header row: {header_row}")
            else:
                print(f"Table {i+1}: Not a transaction table")
    
    if args.debug and args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDebug info saved to {args.output}")

if __name__ == '__main__':
    main()
