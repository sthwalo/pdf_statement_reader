import re
import pandas as pd
import json
import logging
from datetime import datetime
import os

def clean_date(date_str, debug=False):
    """
    Clean and standardize date values to YYYY-MM-DD format
    
    Args:
        date_str: Date string to clean
        debug (bool): Enable debug mode
        
    Returns:
        str: Cleaned date in YYYY-MM-DD format
        dict: Debug info if debug=True
    """
    import re
    from datetime import datetime
    import pandas as pd
    
    debug_info = {
        'input': date_str,
        'output': '',
        'method': '',
        'error': None
    }
    
    if not date_str or pd.isna(date_str):
        debug_info['method'] = 'empty_input'
        debug_info['output'] = ''
        
        if debug:
            return "", debug_info
        return ""
    
    date_str = str(date_str).strip()
    debug_info['input'] = date_str
    
    try:
        # Skip if it looks like a monetary value (contains currency symbols, commas as thousand separators, or ends with '-')
        if re.search(r'[$€£¥]|\d{1,3},\d{3}\.\d{2}|\d+\.\d{2}-$', date_str) or date_str.endswith('-') or re.search(r'^\d+[,.]\d{2}$', date_str) or re.search(r'^\d+[,.]\d{2}-$', date_str):
            debug_info['method'] = 'skipped_monetary_value'
            debug_info['output'] = ''
            debug_info['error'] = 'Input appears to be a monetary value'
            
            if debug:
                return "", debug_info
            return ""
        
        # Special handling for bank statement format "MM DD" (e.g., "01 16")
        bank_date_match = re.search(r'^(\d{1,2})\s+(\d{1,2})$', date_str)
        if bank_date_match:
            month, day = map(int, bank_date_match.groups())
            year = datetime.now().year
            
            # Basic validation
            if 1 <= month <= 12 and 1 <= day <= 31:
                result = f"{year:04d}-{month:02d}-{day:02d}"
                debug_info['method'] = 'bank_statement_format'
                debug_info['output'] = result
                
                if debug:
                    return result, debug_info
                return result
        
        # Try to extract date in various formats
        date_match = re.search(r'(\d{1,2})[ /.-](\d{1,2})(?:[ /.-](\d{2,4}))?', date_str)
        if date_match:
            day_or_month1, day_or_month2, year = date_match.groups()
            
            # Try to determine if it's MM/DD or DD/MM format
            try:
                m1 = int(day_or_month1)
                m2 = int(day_or_month2)
                
                # For bank statements, if first number is 01-12, it's likely MM format
                if 1 <= m1 <= 12:
                    month = m1
                    day = m2
                    debug_info['method'] = 'mm_dd_pattern'
                # If first number is > 12, it's likely DD/MM
                elif m1 > 12 and m2 <= 12:
                    day, month = m1, m2
                    debug_info['method'] = 'dd_mm_pattern'
                # If second number is > 12, it's likely MM/DD
                elif m2 > 12 and m1 <= 12:
                    month, day = m1, m2
                    debug_info['method'] = 'mm_dd_pattern'
                # If both are <= 12, assume DD/MM format
                else:
                    day, month = m1, m2
                    debug_info['method'] = 'assumed_dd_mm'
                
                # Handle year
                if year:
                    year = int(year)
                    if year < 100:
                        year += 2000
                else:
                    # Default to current year if not provided
                    year = datetime.now().year
                
                # Basic validation of day and month
                if not (1 <= day <= 31 and 1 <= month <= 12):
                    debug_info['method'] = 'invalid_date_components'
                    debug_info['error'] = f"Invalid day or month: day={day}, month={month}"
                    
                    if debug:
                        return "", debug_info
                    return ""
                
                # Format as YYYY-MM-DD
                result = f"{year:04d}-{month:02d}-{day:02d}"
                debug_info['output'] = result
                
                if debug:
                    return result, debug_info
                return result
            except (ValueError, TypeError) as e:
                debug_info['error'] = f"Error parsing date components: {e}"
        else:
            debug_info['error'] = "No date pattern found"
    
    except Exception as e:
        debug_info['error'] = f"Error cleaning date: {e}"
    
    # Return empty string instead of original value if we couldn't parse it as a date
    debug_info['method'] = 'fallback_empty'
    debug_info['output'] = ''
    
    if debug:
        return "", debug_info
    return ""

def clean_numeric(value, debug=False):
    """
    Clean and standardize numeric values
    
    Args:
        value: Numeric value to clean
        debug (bool): Enable debug mode
        
    Returns:
        str: Cleaned numeric value as string with 2 decimal places
        dict: Debug info if debug=True
    """
    debug_info = {
        'input': value,
        'output': '',
        'method': '',
        'error': None
    }
    
    if not value or pd.isna(value):
        debug_info['method'] = 'empty_input'
        debug_info['output'] = ''
        
        if debug:
            return "", debug_info
        return ""
    
    value_str = str(value).strip()
    debug_info['input'] = value_str
    
    try:
        # Remove currency symbols
        value_str = re.sub(r'[$€£¥R]', '', value_str)
        
        # Handle negative values (either with - at end or beginning)
        is_negative = False
        if value_str.endswith('-'):
            is_negative = True
            value_str = value_str[:-1]
        elif value_str.startswith('-'):
            is_negative = True
            value_str = value_str[1:]
        
        # Remove commas and other formatting
        value_str = re.sub(r'[,\s]', '', value_str)
        
        # Convert to float
        try:
            numeric_value = float(value_str)
            if is_negative:
                numeric_value = -numeric_value
            
            # Format with 2 decimal places
            result = f"{numeric_value:.2f}"
            debug_info['method'] = 'standard_numeric'
            debug_info['output'] = result
            
            if debug:
                return result, debug_info
            return result
        except ValueError:
            debug_info['error'] = "Could not convert to float"
    except Exception as e:
        debug_info['error'] = f"Error cleaning numeric value: {e}"
    
    # Return empty string if we couldn't parse it as a number
    debug_info['method'] = 'fallback_empty'
    debug_info['output'] = ''
    
    if debug:
        return "", debug_info
    return ""

def propagate_dates(transactions, debug=False):
    """
    Fill in missing dates by propagating the last valid date
    
    Args:
        transactions (list): List of transaction dictionaries
        debug (bool): Enable debug mode
        
    Returns:
        list: Transactions with propagated dates
        dict: Debug info if debug=True
    """
    debug_info = {
        'input_count': len(transactions),
        'output_count': 0,
        'filled_dates_count': 0,
        'method': 'date_propagation',
        'error': None
    }
    
    if not transactions:
        debug_info['error'] = "No transactions provided"
        
        if debug:
            return [], debug_info
        return []
    
    try:
        current_date = None
        result = []
        
        for transaction in transactions:
            # Make a copy to avoid modifying the original
            tx = transaction.copy()
            
            # If this transaction has a date, update current_date
            if tx.get('Date') and tx['Date'].strip():
                current_date = tx['Date']
            # Otherwise, use the current_date if available
            elif current_date:
                tx['Date'] = current_date
                debug_info['filled_dates_count'] += 1
            
            result.append(tx)
        
        debug_info['output_count'] = len(result)
        
        if debug:
            return result, debug_info
        return result
    except Exception as e:
        debug_info['error'] = f"Error propagating dates: {e}"
        
        if debug:
            return transactions, debug_info
        return transactions

def ensure_balance_columns(transactions, debug=False):
    """
    Ensure balance columns are properly populated across all pages
    
    Args:
        transactions (list): List of transaction dictionaries
        debug (bool): Enable debug mode
        
    Returns:
        list: Transactions with properly populated balance columns
        dict: Debug info if debug=True
    """
    if not transactions:
        return transactions
    
    debug_info = {
        'balance_columns_fixed': 0,
        'balance_values_found': 0
    }
    
    # Find transactions with balance values
    transactions_with_balance = [tx for tx in transactions if tx.get('Balance')]
    debug_info['balance_values_found'] = len(transactions_with_balance)
    
    # If we have some transactions with balance values, use them to help identify balance columns
    if transactions_with_balance:
        # Group transactions by date to help with balance propagation
        date_groups = {}
        for i, tx in enumerate(transactions):
            date = tx.get('Date', '')
            if date:
                if date not in date_groups:
                    date_groups[date] = []
                date_groups[date].append(i)
        
        # For each date group, if any transaction has a balance, try to identify the balance column
        for date, indices in date_groups.items():
            # Find transactions with balance in this date group
            balance_indices = [i for i in indices if transactions[i].get('Balance')]
            
            # If we have balance values for this date, propagate to other transactions on same date
            if balance_indices:
                for i in indices:
                    if not transactions[i].get('Balance') and i not in balance_indices:
                        # Use the balance from the last transaction with balance for this date
                        last_balance_idx = balance_indices[-1]
                        transactions[i]['Balance'] = transactions[last_balance_idx].get('Balance', '')
                        debug_info['balance_columns_fixed'] += 1
    
    if debug:
        return transactions, debug_info
    return transactions

def clean_transactions(transactions, debug=False):
    """
    Clean all transaction data fields
    
    Args:
        transactions (list): List of transaction dictionaries
        debug (bool): Enable debug mode
        
    Returns:
        list: Cleaned transactions
        dict: Debug info if debug=True
    """
    cleaned = []
    debug_info = {
        'input_count': len(transactions),
        'output_count': 0,
        'cleaned': 0,
        'skipped': 0,
        'date_cleaned_count': 0,
        'debits_cleaned_count': 0,
        'credits_cleaned_count': 0,
        'balance_cleaned_count': 0,
        'service_fee_cleaned_count': 0,
        'method': 'transaction_cleaning',
        'error': None,
        'transaction_details': [],
        'balance_processing': {},
        'skipped_reasons': {}
    }
    
    if not transactions:
        debug_info['error'] = "No transactions provided"
        
        if debug:
            return [], debug_info
    try:
        # Process each transaction
        for tx in transactions:
            # Skip empty transactions
            if not tx:
                debug_info['skipped'] += 1
                debug_info['skipped_reasons']['empty'] = debug_info['skipped_reasons'].get('empty', 0) + 1
                continue
            
            # Clean and standardize fields
            cleaned_tx = {
                'Details': tx.get('Details', '').strip(),
                'ServiceFee': clean_numeric(tx.get('ServiceFee', '')),
                'Debits': clean_numeric(tx.get('Debits', '')),
                'Credits': clean_numeric(tx.get('Credits', '')),
                'Date': clean_date(tx.get('Date', '')),
                'Balance': clean_numeric(tx.get('Balance', ''))
            }
            
            # Skip transactions with no meaningful data
            if not cleaned_tx['Details'] and not cleaned_tx['ServiceFee'] and \
               not cleaned_tx['Debits'] and not cleaned_tx['Credits'] and \
               not cleaned_tx['Date']:
                debug_info['skipped'] += 1
                debug_info['skipped_reasons']['no_data'] = debug_info['skipped_reasons'].get('no_data', 0) + 1
                continue
            
            cleaned.append(cleaned_tx)
            debug_info['cleaned'] += 1
        
        # Propagate dates to rows that might be missing them
        cleaned = propagate_dates(cleaned)
        
        # Ensure balance columns are properly populated
        if debug:
            cleaned, balance_debug = ensure_balance_columns(cleaned, debug=True)
            debug_info['balance_processing'] = balance_debug
        else:
            cleaned = ensure_balance_columns(cleaned)
        
        if debug:
            return cleaned, debug_info
        return cleaned
    
    except Exception as e:
        error_msg = f"Error cleaning transactions: {e}"
        logger.error(error_msg)
        debug_info['error'] = error_msg
        
        if debug:
            return [], debug_info
        return []

def deduplicate_transactions(transactions, debug=False):
    """
    Remove duplicate transactions based on all fields
    
    Args:
        transactions (list): List of transaction dictionaries
        debug (bool): Enable debug mode
        
    Returns:
        list: Deduplicated transactions
        dict: Debug info if debug=True
    """
    debug_info = {
        'input_count': len(transactions),
        'output_count': 0,
        'duplicates_removed': 0,
        'method': 'deduplication',
        'error': None
    }
    
    if not transactions:
        debug_info['error'] = "No transactions provided"
        
        if debug:
            return [], debug_info
        return []
    
    try:
        # Convert each transaction to a tuple of items for hashability
        seen = set()
        result = []
        
        for transaction in transactions:
            # Create a hashable representation
            tx_tuple = tuple(sorted((k, str(v)) for k, v in transaction.items() if v))
            
            if tx_tuple not in seen:
                seen.add(tx_tuple)
                result.append(transaction)
        
        debug_info['output_count'] = len(result)
        debug_info['duplicates_removed'] = len(transactions) - len(result)
        
        if debug:
            return result, debug_info
        return result
    except Exception as e:
        debug_info['error'] = f"Error deduplicating transactions: {e}"
        
        if debug:
            return transactions, debug_info
        return transactions

def process_transactions(transactions, debug=False, debug_dir=None):
    """
    Process transactions: clean, propagate dates, and deduplicate
    
    Args:
        transactions (list): List of transaction dictionaries
        debug (bool): Enable debug mode
        debug_dir (str): Directory to save debug info
        
    Returns:
        list: Processed transactions
        dict: Debug info if debug=True
    """
    debug_info = {
        'input_count': len(transactions),
        'output_count': 0,
        'steps': [],
        'method': 'transaction_processing',
        'error': None
    }
    
    if not transactions:
        debug_info['error'] = "No transactions provided"
        
        if debug:
            return [], debug_info
        return []
    
    try:
        # Step 1: Clean transactions
        if debug:
            cleaned_transactions, cleaning_debug = clean_transactions(transactions, debug=True)
            debug_info['steps'].append({
                'step': 'clean_transactions',
                'input_count': len(transactions),
                'output_count': len(cleaned_transactions),
                'details': cleaning_debug
            })
            
            # Save cleaning debug info
            if debug_dir:
                os.makedirs(debug_dir, exist_ok=True)
                with open(os.path.join(debug_dir, 'cleaning_debug.json'), 'w') as f:
                    json.dump(cleaning_debug, f, indent=2)
        else:
            cleaned_transactions = clean_transactions(transactions)
        
        # Step 2: Propagate dates
        if debug:
            dated_transactions, date_debug = propagate_dates(cleaned_transactions, debug=True)
            debug_info['steps'].append({
                'step': 'propagate_dates',
                'input_count': len(cleaned_transactions),
                'output_count': len(dated_transactions),
                'details': date_debug
            })
            
            # Save date propagation debug info
            if debug_dir:
                with open(os.path.join(debug_dir, 'date_propagation_debug.json'), 'w') as f:
                    json.dump(date_debug, f, indent=2)
        else:
            dated_transactions = propagate_dates(cleaned_transactions)
        
        # Step 3: Deduplicate transactions
        if debug:
            final_transactions, dedup_debug = deduplicate_transactions(dated_transactions, debug=True)
            debug_info['steps'].append({
                'step': 'deduplicate_transactions',
                'input_count': len(dated_transactions),
                'output_count': len(final_transactions),
                'details': dedup_debug
            })
            
            # Save deduplication debug info
            if debug_dir:
                with open(os.path.join(debug_dir, 'deduplication_debug.json'), 'w') as f:
                    json.dump(dedup_debug, f, indent=2)
        else:
            final_transactions = deduplicate_transactions(dated_transactions)
        
        debug_info['output_count'] = len(final_transactions)
        
        if debug:
            return final_transactions, debug_info
        return final_transactions
    except Exception as e:
        debug_info['error'] = f"Error processing transactions: {e}"
        
        if debug:
            return transactions, debug_info
        return transactions

# For testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Test date cleaning
    test_dates = [
        "01/15/2023",
        "15/01/2023",
        "01-15-2023",
        "15-01-2023",
        "01.15.2023",
        "15.01.2023",
        "01 15",
        "15 01",
        "2,100.00-",
        "R500.00",
        ""
    ]
    
    logging.info("Testing date cleaning:")
    for date in test_dates:
        cleaned, debug = clean_date(date, debug=True)
        logging.info(f"Input: '{date}', Output: '{cleaned}', Method: {debug['method']}")
    
    # Test numeric cleaning
    test_numbers = [
        "1,234.56",
        "1234.56-",
        "-1234.56",
        "R1,234.56",
        "$1,234.56",
        "1 234.56",
        ""
    ]
    
    logging.info("\nTesting numeric cleaning:")
    for num in test_numbers:
        cleaned, debug = clean_numeric(num, debug=True)
        logging.info(f"Input: '{num}', Output: '{cleaned}', Method: {debug['method']}")
    
    # Test transaction processing
    test_transactions = [
        {"Details": "Transaction 1", "Date": "01/15/2023", "Debits": "1,234.56-", "Credits": "", "Balance": "5,000.00"},
        {"Details": "Transaction 2", "Date": "", "Debits": "", "Credits": "2,345.67", "Balance": "7,345.67"},
        {"Details": "Transaction 3", "Date": "", "Debits": "345.67-", "Credits": "", "Balance": "7,000.00"},
        {"Details": "Transaction 4", "Date": "01/20/2023", "Debits": "", "Credits": "1,000.00", "Balance": "8,000.00"},
        {"Details": "Transaction 4", "Date": "01/20/2023", "Debits": "", "Credits": "1,000.00", "Balance": "8,000.00"}  # Duplicate
    ]
    
    logging.info("\nTesting transaction processing:")
    processed, debug = process_transactions(test_transactions, debug=True)
    
    logging.info(f"Input count: {debug['input_count']}")
    logging.info(f"Output count: {debug['output_count']}")
    
    for i, tx in enumerate(processed):
        logging.info(f"Transaction {i+1}: {tx}")
