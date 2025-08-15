import pandas as pd
import os
import re
import openpyxl
from openpyxl.utils import get_column_letter

from datetime import datetime

def parse_date(date_str, statement_month=None, statement_year=None):
    """
    Parse date string into datetime object with proper year inference.
    
    Args:
        date_str: Date string to parse
        statement_month: Month of the statement (for inferring year)
        statement_year: Year of the statement
        
    Returns:
        datetime object or NaT if parsing fails
    """
    try:
        if pd.isna(date_str) or not isinstance(date_str, str):
            return pd.NaT
            
        # Clean the date string
        date_str = date_str.strip()
        
        # Check if it's already in a full date format (e.g., 2023-02-10)
        if re.match(r'\d{4}-\d{2}-\d{2}', date_str):
            return pd.to_datetime(date_str)
            
        # Handle DD/MM format
        if '/' in date_str:
            parts = date_str.split('/')
            if len(parts) >= 2:
                day = parts[0].strip().zfill(2)
                month = parts[1].strip().zfill(2)
                return pd.to_datetime(f"{day}/{month}/{statement_year}", format='%d/%m/%Y')
        
        # Handle just DD format (assuming same month as statement)
        if re.match(r'^\d{1,2}$', date_str):
            day = date_str.zfill(2)
            month = str(statement_month).zfill(2)
            return pd.to_datetime(f"{day}/{month}/{statement_year}", format='%d/%m/%Y')
            
        # Default fallback
        return pd.NaT
    except Exception as e:
        print(f"Error parsing date '{date_str}': {e}")
        return pd.NaT

def clean_numeric(value):
    """
    Clean numeric string values by removing currency symbols, commas, etc.
    
    Args:
        value: Value to clean
        
    Returns:
        float: Cleaned numeric value
    """
    try:
        if pd.isna(value):
            return 0.0
            
        # Convert to string if not already
        value = str(value)
        
        # Remove commas, currency symbols, and spaces
        value = value.replace(',', '').replace('R', '').replace(' ', '').replace('"', '')
        
        # Handle parentheses for negative numbers
        if '(' in value and ')' in value:
            value = value.replace('(', '').replace(')', '')
            return -float(value)
            
        # Handle empty or invalid values
        if value == '' or value.strip() == '.':
            return 0.0
            
        return float(value)
    except Exception as e:
        print(f"Error cleaning numeric value '{value}': {e}")
        return 0.0

def combine_csv_files(csv_files, output_path, expected_opening_balance=None, start_date=None, end_date=None):
    """
    Combine multiple CSV files into a single DataFrame and save to output path.
    
    Args:
        csv_files: List of CSV file paths to combine
        output_path: Path to save the combined CSV file
        expected_opening_balance: Expected opening balance for verification
        start_date: Start date for filtering transactions (optional)
        end_date: End date for filtering transactions (optional)
        
    Returns:
        DataFrame with combined transaction data
    """
    print(f"Combining {len(csv_files)} CSV files...")
    
    # Convert date strings to datetime objects if provided
    start_date_dt = pd.to_datetime(start_date) if start_date else None
    end_date_dt = pd.to_datetime(end_date) if end_date else None
    
    dfs = []
    statement_order_counter = 0
    
    for file_path in csv_files:
        try:
            filename = os.path.basename(file_path)
            print(f"Processing {filename}...")
            
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Add source filename
            df['source_file'] = filename
            
            # Parse statement month and year from filename
            month_match = re.search(r'\((\d{2})\)', filename)
            if month_match:
                statement_month = int(month_match.group(1))
                # Infer year based on fiscal year logic (March to February)
                statement_year = 2023 if statement_month >= 3 else 2024
                df['statement_month'] = statement_month
                df['statement_year'] = statement_year
                
                # Add statement order for sorting within same date
                statement_order_counter += 1
                df['statement_order'] = statement_order_counter
                
                # Parse dates
                df['date'] = df['Date'].apply(lambda x: parse_date(x, statement_month, statement_year))
                
                # Print date parsing statistics
                valid_dates = df['date'].notna().sum()
                total_dates = len(df)
                if total_dates > 0:
                    print(f"Successfully parsed {valid_dates}/{total_dates} dates ({valid_dates/total_dates*100:.1f}%)")
            else:
                # If can't extract month from filename, try to infer the date format
                print(f"Warning: Could not extract month from filename {filename}, attempting to infer dates")
                df['date'] = pd.to_datetime(df['Date'], errors='coerce')
                df['statement_month'] = None
                df['statement_year'] = None
                df['statement_order'] = statement_order_counter
            
            # Clean numeric columns
            for col in ['Debits', 'Credits', 'Balance']:
                if col in df.columns:
                    df[col] = df[col].apply(clean_numeric)
            
            # Ensure date column is datetime type
            if 'date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                # Replace NaT values with a default date outside our range
                df['date'] = df['date'].fillna(pd.Timestamp('1900-01-01'))
            
            # Filter by date if specified
            if start_date_dt and end_date_dt and 'date' in df.columns:
                before_count = len(df)
                df = df[df['date'].between(start_date_dt, end_date_dt)]
                after_count = len(df)
                print(f"  Filtered {filename} from {before_count} to {after_count} transactions within date range")
            
            # Add to list of dataframes if not empty
            if not df.empty:
                dfs.append(df)
                print(f"Added {len(df)} rows from {filename}")
            else:
                print(f"No valid data found in {filename}")
                
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    if not dfs:
        print("No valid CSV files to combine")
        return pd.DataFrame()  # Return empty DataFrame
    
    # Combine all DataFrames
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Combined {len(combined_df)} total rows from {len(dfs)} files")
    
    # Sort by date and statement_order to maintain chronological order
    if 'date' in combined_df.columns and 'statement_order' in combined_df.columns:
        print("Sorting by date and statement_order...")
        combined_df = combined_df.sort_values(['date', 'statement_order'])
    elif 'date' in combined_df.columns:
        print("Sorting by date only...")
        combined_df = combined_df.sort_values('date')
    
    # Recalculate balances if expected opening balance is provided
    if expected_opening_balance is not None and len(combined_df) > 0:
        try:
            # Import balance verification module
            from modules.balance_verification import recalculate_balances, verify_closing_balance
            
            # Get the first month in the data
            first_date = combined_df['date'].min()
            first_month = first_date.month
            first_year = first_date.year
            
            print(f"First transaction date: {first_date}, Month: {first_month}, Year: {first_year}")
            print(f"Setting opening balance to {expected_opening_balance}")
            
            # Recalculate all balances starting from the expected opening balance
            combined_df = recalculate_balances(combined_df, expected_opening_balance)
            
            # Verify the closing balance if we have transactions
            if len(combined_df) > 0:
                closing_balance = combined_df.iloc[-1]['Balance']
                print(f"Calculated closing balance: {closing_balance:.2f}")
                
                # Calculate expected closing based on opening balance and transaction amounts
                total_debits = combined_df['Debits'].sum()
                total_credits = combined_df['Credits'].sum()
                expected_closing = expected_opening_balance - total_debits + total_credits
                print(f"Expected closing balance based on transactions: {expected_closing:.2f}")
                print(f"Difference: {abs(closing_balance - expected_closing):.2f}")
        except ImportError:
            print("Warning: balance_verification module not found. Skipping balance recalculation.")
        except Exception as e:
            print(f"Error recalculating balances: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Save combined DataFrame to CSV
    combined_df.to_csv(output_path, index=False)
    print(f"Combined CSV saved to {output_path} with {len(combined_df)} rows")
    
    return combined_df

def clean_and_process_csv(df, expected_opening_balance=None, expected_closing_balance=None):
    """
    Clean and process the DataFrame to create a proper cashbook.
    Works with camelot CSV format (Date,Details,Debits,Credits,Balance,ServiceFee)
    
    Args:
        df: DataFrame with transaction data
        expected_opening_balance: Expected opening balance for verification
        expected_closing_balance: Expected closing balance for verification
        
    Returns:
        tuple: (processed DataFrame, balance info dictionary)
    """
    print("Processing combined data")
    
    # Import balance verification module if not already imported
    try:
        from modules.balance_verification import recalculate_balances, verify_closing_balance
    except ImportError:
        print("Warning: Could not import balance_verification module. Using built-in functions.")
    
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Clean numeric columns if not already cleaned
    for col in ['Debits', 'Credits', 'Balance']:
        if col in df.columns:
            # Check if column is already numeric
            if not pd.api.types.is_numeric_dtype(df[col]):
                # Handle various formats
                df[col] = df[col].astype(str)
                # Remove commas, currency symbols, and spaces
                df[col] = df[col].str.replace(',', '')
                df[col] = df[col].str.replace('R', '')
                df[col] = df[col].str.replace(' ', '')
                df[col] = df[col].str.replace('"', '')
                # Handle parentheses for negative numbers
                df[col] = df[col].apply(lambda x: 
                                       -float(x.replace('(', '').replace(')', '')) 
                                       if '(' in str(x) and ')' in str(x) 
                                       else (0 if x == '' or pd.isna(x) or x.strip() == '.' else float(x)))
    
    # Sort by date and statement_order if available
    if 'date' in df.columns:
        if 'statement_order' in df.columns:
            print("Sorting by date and statement_order")
            df = df.sort_values(['date', 'statement_order'])
        else:
            print("Sorting by date only")
            df = df.sort_values('date')
    
    # Fix opening balance if provided
    if expected_opening_balance is not None and len(df) > 0:
        # Check if we need to recalculate balances
        first_balance = df.iloc[0]['Balance'] if 'Balance' in df.columns else None
        
        if first_balance is None or abs(first_balance - expected_opening_balance) > 0.01:
            print(f"Setting opening balance to {expected_opening_balance} (was {first_balance})")
            
            try:
                # Use the balance verification module if available
                df = recalculate_balances(df, expected_opening_balance)
            except NameError:
                # Fall back to built-in recalculation
                df.iloc[0, df.columns.get_loc('Balance')] = expected_opening_balance
                
                # Recalculate all balances based on debits and credits
                print("Recalculating all balances...")
                current_balance = expected_opening_balance
                
                for i in range(len(df)):
                    # Update balance based on debits and credits
                    debit = df.iloc[i]['Debits'] if 'Debits' in df.columns else 0
                    credit = df.iloc[i]['Credits'] if 'Credits' in df.columns else 0
                    
                    # Handle NaN values
                    debit = 0 if pd.isna(debit) else debit
                    credit = 0 if pd.isna(credit) else credit
                    
                    # Apply transaction to balance
                    if i > 0:  # Skip first row as we already set its balance
                        current_balance = current_balance - debit + credit
                        df.iloc[i, df.columns.get_loc('Balance')] = current_balance
        else:
            print(f"Opening balance already matches expected value: {expected_opening_balance}")
    
    # Check closing balance if provided
    if expected_closing_balance is not None and len(df) > 0:
        try:
            # Use the balance verification module if available
            closing_balance, difference = verify_closing_balance(df, expected_closing_balance)
            print(f"Closing balance: {closing_balance}, Expected: {expected_closing_balance}, Difference: {difference}")
        except NameError:
            # Fall back to built-in verification
            final_balance = df.iloc[-1]['Balance']
            print(f"Final calculated balance: {final_balance}")
            print(f"Expected closing balance: {expected_closing_balance}")
            
            difference = abs(final_balance - expected_closing_balance)
            if difference > 0.01:
                print(f"Closing balance difference: {difference}")
            
    # Print summary of numeric data
    print(f"Debits range: {df['Debits'].min()} to {df['Debits'].max()}, non-zero: {(df['Debits'] != 0).sum()}")
    print(f"Credits range: {df['Credits'].min()} to {df['Credits'].max()}, non-zero: {(df['Credits'] != 0).sum()}")
    print(f"Balance range: {df['Balance'].min()} to {df['Balance'].max()}")
    
    # Determine transaction type based on Debits and Credits columns
    df['Type'] = df.apply(lambda x: 'Debit' if x['Debits'] > 0 else 'Credit' if x['Credits'] > 0 else 'Unknown', axis=1)
    
    # Create amount column (positive for credits, negative for debits)
    df['amount'] = df['Credits'] - df['Debits']
    
    # Create separate Debit and Credit columns (already exist in camelot format)
    # Just ensure they're properly formatted
    df['Debit'] = df['Debits']
    df['Credit'] = df['Credits']
    
    # Clean up the Balance column (already exists in camelot format)
    # Just ensure it's properly formatted
    df['balance'] = df['Balance']
    
    # Create bank_fee column based on ServiceFee
    if 'ServiceFee' in df.columns:
        df['bank_fee'] = df['ServiceFee'].apply(lambda x: 1 if x == 'Y' else 0)
    else:
        df['bank_fee'] = 0
    
    # Use Details column as description
    if 'Details' in df.columns:
        df['description'] = df['Details']
    
    # Categorize transactions
    df = categorize_transactions(df)
    
    # Create balance information dictionary for verification
    balance_info = {
        'opening_balance': df.iloc[0]['Balance'] if len(df) > 0 else None,
        'closing_balance': df.iloc[-1]['Balance'] if len(df) > 0 else None,
        'opening_balance_difference': abs(df.iloc[0]['Balance'] - expected_opening_balance) if expected_opening_balance is not None and len(df) > 0 else None,
        'closing_balance_difference': abs(df.iloc[-1]['Balance'] - expected_closing_balance) if expected_closing_balance is not None and len(df) > 0 else None
    }
    
    return df, balance_info

def categorize_transactions(df):
    """
    Categorize transactions based on description keywords.
    """
    # Define account mappings in a more structured way
    account_mappings = {
        # INCOME CATEGORIES
        'Income from Services': [
            'Service Income', 'Consulting Income', 'Freelance Income', 'FNB OB Pmt Nutri Feeds'
        ],
        'Investment Income': [
            'Interest Earned', 'Dividend'
        ],
        
        # EXPENSE CATEGORIES
        # Bank Related
        'Bank Charges': [
            'Penalty Interest', 'Service Fee', 'Bank Charges', 
            'Non FNB Withdrawal Fees', 'Monthly Fees', 'Service Fees',
            'Bank Charge', 'Account Fee', 'Item Fee', 'Fee', 'Fees',
            'Manual Reversal Fee', 'Unsuccessful F Declined', 'Unpaid No Funds',
            'Fee Tcib', 'Swift Commission', 'Replacement Fee', 'Card Fee',
            'Commission', 'Schd Trxn', 'Unpaid No Funds 01', 'Dr.Int.Rate',
            '!ERROR: unparsable description text!', 'Card POS Unsuccessful F Declined',
            'Bank Charges'
        ],
        
        # Transportation
        'Fuel Expense': [
            'Fuel', 'Engen', 'Total', 'Sasol', 'Trac Diamond Plaza', 'Doornpoort', 
            'Astron Energy', 'Petrol', 'BP', 'Caltex', 'Vw Parts', 'Engine Parts Luc','Fuel Purchase','Fuel Purchase Total','Fuel Purchase Engen','Fuel Purchase Sasol','Fuel Purchase Trac Diamond Plaza','Fuel Purchase','Fuel Purchase Astron Energy','Fuel Purchase BP','Fuel Purchase Caltex','Fuel Purchase Engen Linksfield Mo 428104*2012','Fuel Purchase Sasol Houghton 428104*2012 ','POS Purchase Total Boksburg Moto 428104*2012', 'Fuel Purchase Sasol Houghton 428104*2012','POS Purchase Astron Energy Cyril 428104*2012','POS Purchase Engen Bramley 428104*2012','POS Purchase Baobab Plaza 428104*2012','POS Purchase Doornpoort 428104*2012','POS Purchase Trac Diamond Plaza 428104*2012','POS Purchase Engen Linksfield Mo 428104*2012','POS Purchase Sasol Houghton 428104*2012 ','POS Purchase Total Boksburg Moto 428104*2012', 'Fuel Purchase Engen Linksfield Mo 428104*2012','Fuel Purchase Sasol Cosmo City 428104*2012'
        ],
        'Toll Fees': [
            'Plaza', 'Toll', 'Baobab Plaza', 'Capricorn Plaza', 'Kranskop Plaza',
            'Nyl Plaza', 'Carousel', 'Pumulani', 'Middleburg Tap','POS Purchase Middleburg Tap N Go', 'Doornkop Plaza', 'Toll Fees', 'Toll Fees Plaza', 'Toll Fees Trac Diamond Plaza','POS Purchase Kranskop Plaza 428104*2012','POS Purchase Capricorn Plaza 428104*2012','POS Purchase Nyl Plaza 428104*2012 ','POS Purchase Doornkop Plaza 428104*2012 '
        ],
        'Vehicle Maintenance': [
            'Parts', 'Engine Parts', 'Truck Spares', 'Vw Parts', 'Engine Parts Luc','FNB App Payment To Engine Parts Engine Parts Luc','FNB App Payment To Masikize Truck Spares'
        ],
        'Vehicle Hire': [
            'Car Hire', 'Car Rental', 'Truck Hire', 'Quantum Hire', 'Rentals', 
            'Outward Swift R024', 'Ez Truck Hire', 'Car Rental'
        ],
        'Travelling Expense': [
            'Uber', 'Taxi', 'Flight'
        ],
        
        # Office Expenses
        'Stationery and Printing': [
            'Game', 'Cenecon', 'Stationery', 'Office', 'Printing', 'Paper', 'POS Purchase Game',
            'Stationery', 'Printer Cartridges', 'Ink for Printers'
        ],
        'Telephone Expense': [
            'Airtime', 'Topup', 'Telkom', 'Vodacom', 'MTN', 'Cell C',
            'Telephone & Utilities'
        ],
        'Internet Expense': [
            'Wifi', 'Internet', 'Home Wifi', 'Fibre',
            'Internet ADSL & Hosting'
        ],
        'Business Equipment': [
            'Verimark', 'African Electro', 'Incredible', 'Hpy*', 
            'Electronic', 'POS Consultin', 'Ikh*E POS', 'Yoco', 'CSB Cosmo City','POS Purchase Bwh Northgate 428104*2012'
        ],
        
        # Personnel Expenses
        'Salaries and Wages': [
            'Driver', 'Salary', 'Tendai', 'Bonakele', 'Ncube', 'Ze', 'Send Money',
            'Salaries', 'Wages', 'Payroll', 'Staff Payment', 'Employee Payment','Settlement',
            'Salaries & Wages'
        ],
        'Director Remunerations': [
             'FNB App Payment To Lk', 'FNB App Payment To Lucky', 'FNB App Payment To Gu', 'FNB App Payment To Gr', 'FNB App Payment To G', 'FNB App Rtc Pmt To Lucky','FNB App Rtc Pmt To Lucky Nhlanhla','FNB App Rtc Pmt To Aunty Lucky','FNB App Rtc Pmt To Qiniso Nsele Lucky','FNB App Rtc Pmt To Lucky Mav Logistics','FNB App Payment To Luc Lucky','FNB App Rtc Pmt To Lucky Lucky Allowance'
        ],
        
        # Business Operations
        'Business Meetings': [
            'Nandos', 'Mcd', 'KFC', 'Chicken Licken', 'Tres Jolie', 'MCP', 'Lunch',
            'Steers', 'Galitos', 'Nizams', 'Newscafe', 'Tonys Ca', 'Snack Shop',
            'Rocky Liquor', 'Avianto',
            'Refreshments / Entertainment expences'
        ],
        'Outsourced Services': [
            'Transport', 'Masik', 'Luc Trs', 'Mas', 'Samantha Mas Logistics', 'Lucky Nikelo Logistics','FNB App Payment To Lucky Nikelo Logistics',
            'Outsourced Services'
        ],
        'Supplier Payments': [
            'Makhosin', 'Masikize', 'Supplier', 'Vendor'
        ],
        
        # Household Expenses
        'Household Expense': [
            'Grocery', 'Shoprite', 'Food', 'Ndoziz Buy', 'Beder Cash And Chic',
            'Diamond Hill', 'Checkers', 'Woolworths', 'PNP', 'Spar', 'Grocc',
            'Gro ', 'Makro', 'Edgars', 'Markham', 'Clicks', 'Dischem', 'Pharmacy',
            'BBW', 'Cotton Lounge', 'Crazy Store', 'Jet', 'MRP', 'Mrprice',
            'Euro Rivonia', 'Ok Minimark', 'Valueco', 'Csb Cosmo City',
            'Braamfontein Superm', 'Mall', 'British A', 'Cellini', 'Ladies Fash',
            'Cash Build', 'Butchery', 'Valley View', 'Eden Perfumes', 'Bramfontein Sup','POS Purchase S2S*Salamudestasupe 428104*2012 '
        ],
        
        # Personal Expenses
        'Drawings': [
            'ATM', 'Cash Advanc', 'Withdrawal', 'Cashback', 'Family Support'
        ],
        'Entertainment': [
            'DSTV', 'Ticketpros', 'Movies', 'Cinema', 'Liquorshop'
        ],
        'Cosmetics Expense': [
            'Cosmetics', 'Umumodi', 'Perfume'
        ],
        
        # Financial Expenses
        'Insurance': [
            'FNB Insure', 'Internal Debit Order', 'Insurance'
        ],
        'Interest Paid': [
            'Int On Debit Balance', 'Loan Interest'
        ],
        'Bond Payment': [
             'Mavondo', 'Rental', 'Rent', 'Mortgage', 'FNB App Transfer To','FNB App Payment To Flat'
        ],
        
        # Other Expenses
        'Educational Expenses': [
            'Sch Fees', 'School', 'Education', 'Computer Lessons', 'Extra Lessons', 'AmandaS Schoolfees',
            'Simphiwe', 'Pathfinder', 'Kids For Kids', 'School Fees', 'School Transport', 'School Uniform',
            'Educational Aids', 'Learner Support Material', 'Education and Training (Staff)'
        ],
        'Donations': [
            'Father\'S Day', 'Penlope Mail', 'Funeral', 'Donation',
            'Donations / Gifts'
        ],
        'Investment Expense': [
            'Invest', 'Investment', 'Shares'
        ],
        'Electricity': [
            'Electricity Prepaid', 'Eskom',
            'Electricity'
        ],
        
        # ASSET/LIABILITY CATEGORIES
        'Assets': [
            'Furniture', 'Equipment', 'Vehicle Purchase'
        ],
        'Liabilities': [
            'Loan', 'Debt', 'Credit Card', 'Payable'
        ],
        'Equity': [
            'Drawings', 'Retained Earnings', 'Capital'
        ],
        
        # TRANSFER CATEGORIES
        'Inter Account Transfers': [
            'Penlope Investments', 'Penelope Investments', 'Transfer Between Accounts'
        ],
        'Miscellaneous': [
            'Transfer To Trf', 'Transfer To Msu', 'Transfer To Ndu',
            'Transfer To Ukn', 'Transfer To Chantelle', 'Transfer To Sleeping Bag',
            'Transfer To Amn', 'Transfer To Mnc', 'Transfer To Sk', 'Liquorshop Cosmo',
            '4624616', 'Payment To Msu', 'Payment To Ndu',
            'S2S*Salamudestasupe', 'Steers Balfour',
            'POS Purchase S2S*Salamudestasupe 428104*2012 03 Sep','FNB App Transfer To N'
        ],
        
        # Additional new categories
        'Service Contracts': [
            'Service contract Copiers'
        ],
        'Professional Fees': [
            'Audit Fees', 'Accounting Services', 'Legal Fees'
        ],
        'Software Expenses': [
            'Computer Software and Licences'
        ],
        'Security Expenses': [
            'Security - Buildings/ Grounds'
        ],
        'Maintenance Expenses': [
            'Maintenance - Assets and Equiptment', 'Maintenance - Buildings',
            'Maintenance - Sport Facilities', 'Maintenance - Grounds'
        ],
        'Equipment': [
            'Tools / Equiptment', 'Protective Clothing'
        ],
        'Cleaning': [
            'Cleaning aids', 'Cleaning Material'
        ],
        'Affiliations': [
            'Affiliations', 'Badges'
        ],
        'Concert Expenses': [
            'Concert Facilitated', 'Concert Presented'
        ],
        'Compliance Fees': [
            'Compliance Fees (COIDA)'
        ],
        'Sporting Activities': [
            'Sporting Activities'
        ],
        'Trust Expenses': [
            'Trust Expenses'
        ],
        'Transportation Expenses': [
            'Fuel & Other Transport costs'
        ],
        'Rent': [
            'Rent'
        ],
        'Excursions': [
            'Excursions'
        ],
    }

    # Add Account column with default value
    df['Account'] = 'Uncategorized'
    
    # Case-insensitive categorization
    for account, keywords in account_mappings.items():
        for keyword in keywords:
            mask = (df['Account'] == 'Uncategorized') & df['description'].astype(str).str.lower().str.contains(keyword.lower(), na=False)
            df.loc[mask, 'Account'] = account
    
    # Additional categorization rules from audit analysis
    # Handle FNB App payments and transfers
    fnb_app_mask = df['description'].astype(str).str.contains('FNB App', case=False, na=False)
    df.loc[fnb_app_mask & df['description'].str.contains('Gro|Grocc', case=False, na=False), 'Account'] = 'Household Expense'
    df.loc[fnb_app_mask & df['description'].str.contains('Petrol', case=False, na=False), 'Account'] = 'Fuel Expense'
    df.loc[fnb_app_mask & df['description'].str.contains('Car|Rental|Hire', case=False, na=False), 'Account'] = 'Vehicle Hire'
    
    # Additional categorization rules
    credit_mask = (df['Type'] == 'Credit') & (~df['description'].astype(str).str.contains(
        'Int|Interest|Service Fee', case=False, na=False))
    df.loc[credit_mask & (df['Account'] == 'Uncategorized'), 'Account'] = 'Income from Services'
    
    # Define account types mapping
    account_types = {
        # Income
        'Income from Services': 'Income',
        'Investment Income': 'Income',
        
        # Expenses
        'Bank Charges': 'Expense',
        'Fuel Expense': 'Expense',
        'Toll Fees': 'Expense',
        'Vehicle Maintenance': 'Expense',
        'Vehicle Hire': 'Expense',
        'Travelling Expense': 'Expense',
        'Stationery and Printing': 'Expense',
        'Telephone Expense': 'Expense',
        'Internet Expense': 'Expense',
        'Business Equipment': 'Expense',
        'Salaries and Wages': 'Expense',
        'Director Remunerations': 'Expense',
        'Business Meetings': 'Expense',
        'Outsourced Services': 'Expense',
        'Supplier Payments': 'Expense',
        'Household Expense': 'Expense',
        'Drawings': 'Equity',
        'Entertainment': 'Expense',
        'Cosmetics Expense': 'Expense',
        'Insurance': 'Expense',
        'Interest Paid': 'Expense',
        'Bond Payment': 'Expense',
        'Educational Expenses': 'Expense',
        'Donations': 'Expense',
        'Investment Expense': 'Expense',
        'Electricity': 'Expense',
        'Service Contracts': 'Expense',
        'Professional Fees': 'Expense',
        'Software Expenses': 'Expense',
        'Security Expenses': 'Expense',
        'Maintenance Expenses': 'Expense',
        'Equipment': 'Expense',
        'Cleaning': 'Expense',
        'Affiliations': 'Expense',
        'Concert Expenses': 'Expense',
        'Compliance Fees': 'Expense',
        'Sporting Activities': 'Expense',
        'Trust Expenses': 'Expense',
        'Transportation Expenses': 'Expense',
        'Rent': 'Expense',
        'Excursions': 'Expense',

        # Assets/Liabilities/Equity
        'Assets': 'Asset',
        'Liabilities': 'Liability',
        'Equity': 'Equity',
        
        # Transfers
        'Inter Account Transfers': 'Transfer',
        
        # Default
        'Uncategorized': 'Unknown'
    }
    
    df['Account Type'] = df['Account'].map(account_types)
    
    return df

def generate_trial_balance(df):
    """
    Generate a trial balance from the categorized transactions.
    """
    trial_balance = df.groupby('Account').agg({
        'Debit': 'sum',
        'Credit': 'sum'
    }).reset_index()
    
    trial_balance['Net'] = trial_balance['Debit'] - trial_balance['Credit']
    trial_balance['Account Type'] = trial_balance['Account'].map(
        df.groupby('Account')['Account Type'].first())
    
    # Sort by account type and name
    trial_balance = trial_balance.sort_values(['Account Type', 'Account'])
    
    # Add totals row
    totals = pd.DataFrame({
        'Account': ['TOTAL'],
        'Debit': [trial_balance['Debit'].sum()],
        'Credit': [trial_balance['Credit'].sum()],
        'Net': [trial_balance['Net'].sum()],
        'Account Type': ['']
    })
    
    trial_balance = pd.concat([trial_balance, totals], ignore_index=True)
    
    return trial_balance

def generate_management_accounts(df):
    """
    Generate management accounts from the processed data.
    """
    # Calculate totals by account type
    account_totals = df.groupby(['Account Type', 'Account']).agg({
        'Debit': 'sum',
        'Credit': 'sum'
    }).reset_index()
    
    account_totals['Net'] = account_totals['Debit'] - account_totals['Credit']
    
    # Income Statement
    income_accounts = account_totals[account_totals['Account Type'] == 'Income']
    expense_accounts = account_totals[account_totals['Account Type'] == 'Expense']
    
    total_income = abs(income_accounts['Credit'].sum() - income_accounts['Debit'].sum())
    total_expenses = expense_accounts['Debit'].sum() - expense_accounts['Credit'].sum()
    net_profit = total_income - total_expenses
    
    income_statement = pd.DataFrame({
        'Category': ['Revenue', 'Less: Expenses', 'Net Profit/(Loss)'],
        'Amount': [total_income, total_expenses, net_profit]
    })
    
    # Balance Sheet
    assets = account_totals[account_totals['Account Type'] == 'Asset']['Net'].sum()
    liabilities = account_totals[account_totals['Account Type'] == 'Liability']['Net'].sum()
    equity = account_totals[account_totals['Account Type'] == 'Equity']['Net'].sum()
    
    balance_sheet = pd.DataFrame({
        'Category': ['Assets', 'Liabilities', 'Equity', 'Retained Earnings'],
        'Amount': [assets, liabilities, equity, net_profit]
    })
    
    return income_statement, balance_sheet

def generate_cashbook_excel(df, output_path):
    """
    Generate an Excel cashbook from the processed DataFrame.
    
    Args:
        df: DataFrame with processed transaction data
        output_path: Path to save the Excel file
        
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"Generating Excel cashbook at {output_path}")
    
    try:
        # Create a copy of the DataFrame to avoid modifying the original
        df_excel = df.copy()
        
        # Add account categories if not already present
        if 'Account' not in df_excel.columns and 'Details' in df_excel.columns:
            print("Categorizing transactions...")
            df_excel['Account'] = df_excel['Details'].apply(lambda x: categorize_transaction(x))
        
        # Calculate totals
        total_debits = df_excel['Debits'].sum()
        total_credits = df_excel['Credits'].sum()
        opening_balance = df_excel.iloc[0]['Balance'] if len(df_excel) > 0 else 0
        closing_balance = df_excel.iloc[-1]['Balance'] if len(df_excel) > 0 else 0
        calculated_closing = opening_balance - total_debits + total_credits
        
        # Print balance verification
        print(f"Opening balance: {opening_balance:.2f}")
        print(f"Total debits: {total_debits:.2f}")
        print(f"Total credits: {total_credits:.2f}")
        print(f"Calculated closing balance: {calculated_closing:.2f}")
        print(f"Actual closing balance: {closing_balance:.2f}")
        print(f"Difference: {abs(calculated_closing - closing_balance):.2f}")
        
        # Create Excel writer
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Detailed transactions sheet
            # Format date column if present
            if 'date' in df_excel.columns:
                df_excel['Date_Formatted'] = df_excel['date'].dt.strftime('%Y-%m-%d')
                # Reorder columns to put date first
                cols = df_excel.columns.tolist()
                if 'Date_Formatted' in cols:
                    cols.remove('Date_Formatted')
                    cols.insert(0, 'Date_Formatted')
                    df_excel = df_excel[cols]
            
            # Add a balance verification row at the top of the detailed transactions sheet
            verification_df = pd.DataFrame({
                'Date_Formatted': ['BALANCE VERIFICATION'],
                'Details': [f'Opening: {opening_balance:.2f}, Debits: {total_debits:.2f}, Credits: {total_credits:.2f}, Calculated Closing: {calculated_closing:.2f}, Actual Closing: {closing_balance:.2f}, Difference: {abs(calculated_closing - closing_balance):.2f}'],
                'Debits': [0],
                'Credits': [0],
                'Balance': [0]
            })
            
            # Combine verification row with transaction data
            df_excel_with_verification = pd.concat([verification_df, df_excel], ignore_index=True)
            
            # Write to Excel
            df_excel_with_verification.to_excel(writer, sheet_name='Detailed Transactions', index=False)
            
            # Monthly summary sheet
            if 'date' in df_excel.columns:
                print("Generating monthly summary...")
                # Add month and year columns if not present
                df_excel['month'] = df_excel['date'].dt.month
                df_excel['year'] = df_excel['date'].dt.year
                
                # Group by month and year
                monthly_summary = df_excel.groupby(['year', 'month']).agg({
                    'Debits': 'sum',
                    'Credits': 'sum',
                    'Balance': 'last'
                }).reset_index()
                
                # Add month name
                month_names = {
                    1: 'January', 2: 'February', 3: 'March', 4: 'April',
                    5: 'May', 6: 'June', 7: 'July', 8: 'August',
                    9: 'September', 10: 'October', 11: 'November', 12: 'December'
                }
                monthly_summary['Month'] = monthly_summary['month'].map(month_names)
                
                # Calculate net movement
                monthly_summary['Net'] = monthly_summary['Credits'] - monthly_summary['Debits']
                
                # Reorder columns
                monthly_summary = monthly_summary[['year', 'month', 'Month', 'Debits', 'Credits', 'Net', 'Balance']]
                
                # Add totals row
                totals = pd.DataFrame({
                    'year': ['Total'],
                    'month': [''],
                    'Month': [''],
                    'Debits': [monthly_summary['Debits'].sum()],
                    'Credits': [monthly_summary['Credits'].sum()],
                    'Net': [monthly_summary['Credits'].sum() - monthly_summary['Debits'].sum()],
                    'Balance': [monthly_summary['Balance'].iloc[-1] if len(monthly_summary) > 0 else 0]
                })
                monthly_summary = pd.concat([monthly_summary, totals])
                
                # Write to Excel
                monthly_summary.to_excel(writer, sheet_name='Monthly Summary', index=False)
            
            # Trial balance sheet
            if 'Account' in df_excel.columns:
                print("Generating trial balance...")
                # Group by account
                trial_balance = df_excel.groupby('Account').agg({
                    'Debits': 'sum',
                    'Credits': 'sum'
                }).reset_index()
                
                # Calculate net amount
                trial_balance['Net'] = trial_balance['Credits'] - trial_balance['Debits']
                
                # Add totals row
                totals = pd.DataFrame({
                    'Account': ['Total'],
                    'Debits': [trial_balance['Debits'].sum()],
                    'Credits': [trial_balance['Credits'].sum()],
                    'Net': [trial_balance['Credits'].sum() - trial_balance['Debits'].sum()]
                })
                trial_balance = pd.concat([trial_balance, totals])
                
                # Write to Excel
                trial_balance.to_excel(writer, sheet_name='Trial Balance', index=False)
                
            # Balance verification sheet
            print("Generating balance verification sheet...")
            verification_data = {
                'Item': ['Opening Balance', 'Total Debits', 'Total Credits', 'Net Movement', 'Calculated Closing Balance', 'Actual Closing Balance', 'Difference'],
                'Amount': [
                    opening_balance,
                    total_debits,
                    total_credits,
                    total_credits - total_debits,
                    calculated_closing,
                    closing_balance,
                    abs(calculated_closing - closing_balance)
                ]
            }
            verification_df = pd.DataFrame(verification_data)
            verification_df.to_excel(writer, sheet_name='Balance Verification', index=False)
        
        print(f"Excel cashbook generated successfully at {output_path}")
        return True
    
    except Exception as e:
        print(f"Error generating Excel cashbook: {str(e)}")
        import traceback
        traceback.print_exc()

def categorize_transaction(details_text):
    """
    Categorize a transaction based on its details text.
    
    Args:
        details_text: String containing transaction details
        
    Returns:
        str: Account category
    """
    details = str(details_text).upper()
    
    # Define categories based on keywords
    categories = {
        'BANK CHARGES': ['BANK CHARGES', 'SERVICE FEE', 'ADMIN FEE', 'ACCOUNT FEE', 'TRANSACTION FEE'],
        'INTEREST': ['INTEREST', 'INT ON', 'EXCESS INTEREST'],
        'TRANSFERS': ['TRANSFER', 'PAYMENT', 'IMMEDIATE PAYMENT', 'INTERNET PMT'],
        'CARD PAYMENTS': ['CARD', 'POS PURCHASE', 'CARD PURCHASE', 'DEBIT CARD'],
        'CASH': ['ATM', 'CASH', 'WITHDRAWAL'],
        'SALARY': ['SALARY', 'WAGES', 'PAYROLL'],
        'UTILITIES': ['ELECTRICITY', 'WATER', 'UTILITY', 'MUNICIPAL'],
        'RENT': ['RENT', 'LEASE', 'PROPERTY'],
        'INSURANCE': ['INSURANCE', 'ASSURANCE', 'POLICY'],
        'SUBSCRIPTIONS': ['SUBSCRIPTION', 'MEMBERSHIP'],
        'FUEL': ['FUEL', 'PETROL', 'DIESEL', 'ENGEN', 'CALTEX', 'SHELL', 'BP'],
        'GROCERIES': ['GROCERY', 'FOOD', 'SPAR', 'CHECKERS', 'SHOPRITE', 'PICK N PAY', 'WOOLWORTHS'],
        'ONLINE SERVICES': ['NETFLIX', 'SPOTIFY', 'APPLE', 'GOOGLE', 'MICROSOFT', 'AMAZON', 'FACEBOOK'],
        'TELECOMMUNICATIONS': ['TELKOM', 'MTN', 'VODACOM', 'CELL C', 'TELEPHONE', 'AIRTIME', 'DATA'],
        'MEDICAL': ['MEDICAL', 'DOCTOR', 'PHARMACY', 'HOSPITAL', 'CLINIC', 'HEALTH'],
        'EDUCATION': ['SCHOOL', 'COLLEGE', 'UNIVERSITY', 'TUITION', 'EDUCATION'],
        'ENTERTAINMENT': ['RESTAURANT', 'CAFE', 'CINEMA', 'MOVIE', 'ENTERTAINMENT'],
        'TRAVEL': ['FLIGHT', 'AIRLINE', 'HOTEL', 'ACCOMMODATION', 'TRAVEL', 'UBER', 'BOLT']
    }
    
    # Check for matches in transaction details
    for category, keywords in categories.items():
        for keyword in keywords:
            if keyword in details:
                return category
    
    # Default category if no match found
    return 'UNCATEGORIZED'

def clean_and_process_csv(df, expected_opening_balance=None, expected_closing_balance=None):
    """
    Clean and process the CSV data with focus on proper sorting and natural balance calculation.
    
    Args:
        df: DataFrame with transaction data
        expected_opening_balance: Expected opening balance (for verification only, not forced)
        expected_closing_balance: Expected closing balance (for verification only, not forced)
        
    Returns:
        tuple: (DataFrame with cleaned data, dict with balance info)
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Initialize balance info dictionary
    balance_info = {
        'opening_balance': None,
        'closing_balance': None,
        'expected_opening_balance': expected_opening_balance,
        'expected_closing_balance': expected_closing_balance,
        'opening_balance_difference': None,
        'closing_balance_difference': None
    }
    
    # Ensure numeric columns are properly formatted
    for col in ['Debits', 'Credits', 'Balance']:
        if col in df.columns:
            # Convert to string first to handle any non-numeric values
            df[col] = df[col].astype(str)
            
            # Remove currency symbols, commas, and spaces
            df[col] = df[col].str.replace('R', '', regex=False)
            df[col] = df[col].str.replace(',', '', regex=False)
            df[col] = df[col].str.replace(' ', '', regex=False)
            
            # Handle parentheses for negative numbers
            df[col] = df[col].apply(lambda x: 
                                   -float(x.strip('()')) if '(' in x and ')' in x 
                                   else float(x) if x.strip() and x.strip() != 'nan' 
                                   else 0)
    
    # Extract source file information for better sorting
    df['source_file'] = df.get('source_file', '')
    
    # Extract month and year from source file for better sorting
    def extract_month_year(source_file):
        import re
        month_match = re.search(r'\((\d{2})\)', str(source_file))
        if month_match:
            month = int(month_match.group(1))
            # Determine fiscal year based on month (assuming fiscal year starts in March)
            year = 2024 if 3 <= month <= 12 else 2025
            return month, year
        return None, None
    
    # Add month and year columns for better sorting
    df['statement_month'], df['statement_year'] = zip(*df['source_file'].apply(extract_month_year))
    
    # Sort by date, then by statement_year, statement_month if date is the same
    df = df.sort_values(['date', 'statement_year', 'statement_month'])
    
    # Store the original opening balance
    original_opening_balance = df.iloc[0]['Balance']
    balance_info['opening_balance'] = original_opening_balance
    
    # Compare with expected opening balance if provided
    if expected_opening_balance is not None:
        balance_info['opening_balance_difference'] = original_opening_balance - expected_opening_balance
        print(f"Original opening balance: {original_opening_balance}")
        print(f"Expected opening balance: {expected_opening_balance}")
        print(f"Difference: {balance_info['opening_balance_difference']}")
    
    # Recalculate all balances sequentially based on the original opening balance
    print("Recalculating all balances sequentially...")
    running_balance = original_opening_balance
    
    # Update the first row's balance to match the original opening balance
    df.iloc[0, df.columns.get_loc('Balance')] = running_balance
    
    # Recalculate all subsequent balances
    for i in range(1, len(df)):
        # Apply debits (negative) and credits (positive)
        running_balance = running_balance - df.iloc[i]['Debits'] + df.iloc[i]['Credits']
        df.iloc[i, df.columns.get_loc('Balance')] = running_balance
    
    # Store the calculated closing balance
    final_balance = df.iloc[-1]['Balance']
    balance_info['closing_balance'] = final_balance
    print(f"Final calculated balance: {final_balance}")
    
    # Compare with expected closing balance if provided
    if expected_closing_balance is not None:
        balance_info['closing_balance_difference'] = final_balance - expected_closing_balance
        print(f"Expected closing balance: {expected_closing_balance}")
        print(f"Difference: {balance_info['closing_balance_difference']}")
    
    # Print ranges for debugging
    print(f"Debits range: {df['Debits'].min()} to {df['Debits'].max()}, non-zero: {(df['Debits'] != 0).sum()}")
    print(f"Credits range: {df['Credits'].min()} to {df['Credits'].max()}, non-zero: {(df['Credits'] != 0).sum()}")
    print(f"Balance range: {df['Balance'].min()} to {df['Balance'].max()}")
    
    # Drop temporary sorting columns
    if 'statement_month' in df.columns:
        df = df.drop(['statement_month', 'statement_year'], axis=1)
    
    return df, balance_info

def process_cashbook(input_dir, output_file, start_date, end_date, use_existing_combined=False, expected_opening_balance=None, expected_closing_balance=None):
    """
    Process CSV files to generate a cashbook for the specified date range.
    
    Args:
        input_dir (str): Directory containing CSV files
        output_file (str): Path to save the output Excel file
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        use_existing_combined (bool): Whether to use existing combined CSV file if available
        expected_opening_balance (float): Expected opening balance for verification
        expected_closing_balance (float): Expected closing balance for verification
        
    Returns:
        dict: Result with success status, output path, transaction count, and balance adjustment info
    """
    print(f"Processing cashbook for period {start_date} to {end_date}")
    
    try:
        # Import balance verification module
        from modules.balance_verification import identify_opening_balance, recalculate_balances, verify_closing_balance
        
        # Convert date strings to datetime objects
        start_date_dt = pd.to_datetime(start_date)
        end_date_dt = pd.to_datetime(end_date)
        
        # Check if combined CSV file exists
        combined_csv_path = os.path.join(input_dir, "combined_transactions.csv")
        
        if use_existing_combined and os.path.exists(combined_csv_path):
            print(f"Using existing combined CSV file: {combined_csv_path}")
            df = pd.read_csv(combined_csv_path)
            print(f"Read {len(df)} rows from combined CSV file")
        else:
            # Find all CSV files in the input directory
            csv_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.csv') and f != "combined_transactions.csv"]
            
            if not csv_files:
                print("No CSV files found in the input directory")
                return {"success": False, "output_path": None, "error": "No CSV files found", "transaction_count": 0}
            
            print(f"Found {len(csv_files)} CSV files")
            
            # Combine CSV files
            df = combine_csv_files(csv_files, combined_csv_path, expected_opening_balance, start_date, end_date)
            
            if df is None or df.empty:
                print("No data found in CSV files")
                return {"success": False, "output_path": None, "error": "No data found", "transaction_count": 0}
        
        # Filter by date range if not already done in combine_csv_files
        if 'date' in df.columns:
            # Ensure date column is datetime type
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                print("Converting date column to datetime...")
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                # Replace NaT values with a default date outside our range
                df['date'] = df['date'].fillna(pd.Timestamp('1900-01-01'))
            
            df_filtered = df[df['date'].between(start_date_dt, end_date_dt)]
            print(f"Filtered to {len(df_filtered)} transactions within date range {start_date_dt.date()} to {end_date_dt.date()}")
        else:
            print("Warning: No date column found in the data. Cannot filter by date range.")
            df_filtered = df
        
        # Print month distribution in filtered data
        if 'date' in df_filtered.columns:
            month_counts = df_filtered.groupby(df_filtered['date'].dt.strftime('%Y-%m')).size()
            print("\nMonth distribution in filtered data:")
            for month, count in month_counts.items():
                print(f"{month}: {count} transactions")
            
            # Get the date range of transactions
            min_date = df_filtered['date'].min()
            max_date = df_filtered['date'].max()
            print(f"Processing transactions from {min_date} to {max_date}")
        
        # Sort by date and statement_order if available
        if 'date' in df_filtered.columns and 'statement_order' in df_filtered.columns:
            print("Sorting by date and statement_order...")
            df_filtered = df_filtered.sort_values(['date', 'statement_order'])
        elif 'date' in df_filtered.columns:
            print("No statement_order column found. Sorting by date only.")
            df_filtered = df_filtered.sort_values('date')
        
        # Initialize balance info dictionary
        balance_info = {
            'opening_balance': None,
            'closing_balance': None,
            'opening_balance_difference': None,
            'closing_balance_difference': None
        }
        
        # Check if we have transactions to process
        if len(df_filtered) > 0:
            # Ensure all numeric columns are properly converted
            for col in ['Debits', 'Credits', 'Balance']:
                if col in df_filtered.columns and not pd.api.types.is_numeric_dtype(df_filtered[col]):
                    print(f"Converting {col} column to numeric...")
                    df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')
                    df_filtered[col] = df_filtered[col].fillna(0.0)
            
            # Check if the opening balance matches the expected value
            if expected_opening_balance is not None:
                try:
                    first_balance = float(df_filtered.iloc[0]['Balance'])
                    
                    if abs(first_balance - expected_opening_balance) > 0.01:
                        print(f"Warning: First transaction balance {first_balance:.2f} does not match expected opening balance {expected_opening_balance:.2f}")
                        print("Recalculating balances...")
                        
                        # Recalculate balances using the balance_verification module
                        df_filtered = recalculate_balances(df_filtered, expected_opening_balance)
                        
                        # Verify the recalculated balances
                        new_first_balance = float(df_filtered.iloc[0]['Balance'])
                        new_last_balance = float(df_filtered.iloc[-1]['Balance'])
                        print(f"Recalculated opening balance: {new_first_balance:.2f}")
                        print(f"Recalculated closing balance: {new_last_balance:.2f}")
                    else:
                        print(f"First transaction balance {first_balance:.2f} matches expected opening balance {expected_opening_balance:.2f}")
                        new_last_balance = float(df_filtered.iloc[-1]['Balance'])
                        print(f"Current closing balance: {new_last_balance:.2f}")
                    
                    # Verify closing balance if provided
                    if expected_closing_balance is not None:
                        closing_diff = abs(float(df_filtered.iloc[-1]['Balance']) - expected_closing_balance)
                        print(f"Expected closing balance: {expected_closing_balance:.2f}")
                        print(f"Difference: {closing_diff:.2f}")
                        
                        if closing_diff > 0.01:
                            print("Warning: Closing balance does not match expected closing balance")
                            print("This could indicate missing transactions or calculation errors")
                            
                            # Calculate what the closing balance should be based on opening balance and transactions
                            total_debits = df_filtered['Debits'].sum()
                            total_credits = df_filtered['Credits'].sum()
                            calculated_closing = expected_opening_balance - total_debits + total_credits
                            print(f"Calculated closing balance based on transactions: {calculated_closing:.2f}")
                            print(f"Difference from expected: {abs(calculated_closing - expected_closing_balance):.2f}")
                        
                        # Update balance info
                        balance_info['closing_balance_difference'] = closing_diff
                    
                    # Update balance info
                    balance_info['opening_balance'] = float(df_filtered.iloc[0]['Balance'])
                    balance_info['closing_balance'] = float(df_filtered.iloc[-1]['Balance'])
                    balance_info['opening_balance_difference'] = abs(float(df_filtered.iloc[0]['Balance']) - expected_opening_balance)
                    
                except Exception as e:
                    print(f"Error verifying balances: {str(e)}")
                    import traceback
                    traceback.print_exc()
            
            # Generate Excel cashbook
            excel_result = generate_cashbook_excel(df_filtered, output_file)
            
            if excel_result:
                print(f"Successfully generated Excel cashbook at {output_file}")
            else:
                print(f"Failed to generate Excel cashbook")
            
            # Return success result
            return {
                "success": True, 
                "output_path": output_file,
                "transaction_count": len(df_filtered),
                "opening_balance": balance_info.get('opening_balance'),
                "closing_balance": balance_info.get('closing_balance'),
                "opening_balance_difference": balance_info.get('opening_balance_difference'),
                "closing_balance_difference": balance_info.get('closing_balance_difference')
            }
        else:
            print("No transactions found in the filtered data. Cannot verify balances.")
            if expected_opening_balance is not None:
                print(f"Expected opening balance was: {expected_opening_balance:.2f}")
            if expected_closing_balance is not None:
                print(f"Expected closing balance was: {expected_closing_balance:.2f}")
            
            return {
                "success": False,
                "output_path": None,
                "error": "No transactions found in the specified date range",
                "transaction_count": 0
            }
    
    except Exception as e:
        print(f"Error processing cashbook: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            'success': False,
            'output_path': None,
            'error': str(e),
            'transaction_count': 0
        }

def main():
    """
    Main function when running as a script.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Process bank statement CSV files into a cashbook')
    parser.add_argument('--input', required=True, help='Directory containing CSV transaction files')
    parser.add_argument('--output', required=True, help='Path to save the Excel cashbook')
    parser.add_argument('--start-date', default='2024-03-01', help='Start date of fiscal year (YYYY-MM-DD)')
    parser.add_argument('--end-date', default='2025-02-28', help='End date of fiscal year (YYYY-MM-DD)')
    parser.add_argument('--debug', action='store_true', help='Print debug information')
    
    args = parser.parse_args()
    
    result = process_cashbook(
        args.input,
        args.output,
        args.start_date,
        args.end_date,
        args.debug
    )
    
    if result['success']:
        print(f"\nCashbook created successfully with {result['transaction_count']} transactions")
        print(f"Output file: {result['output_path']}")
        sys.exit(0)
    else:
        print(f"\nFailed to create cashbook: {result['error']}")
        sys.exit(1)

if __name__ == "__main__":
    main()