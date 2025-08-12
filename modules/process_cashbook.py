import pandas as pd
import os
import re
import openpyxl
from openpyxl.utils import get_column_letter

from datetime import datetime

def combine_csv_files(input_dir, start_date, end_date):
    """
    Combine multiple CSV files within the specified date range.
    Works with camelot CSV format (Date,Details,Debits,Credits,Balance,ServiceFee)
    """
    print(f"Reading CSV files from {input_dir} for period {start_date} to {end_date}")
    
    # Convert dates to datetime objects
    start_dt = pd.to_datetime(start_date, format='%Y-%m-%d')
    end_dt = pd.to_datetime(end_date, format='%Y-%m-%d')
    
    # List to store individual dataframes
    dfs = []
    
    # Read each CSV file
    for filename in os.listdir(input_dir):
        if filename.endswith('.csv') and not filename.startswith('combined_'):
            file_path = os.path.join(input_dir, filename)
            print(f"Processing {filename}")
            
            try:
                # Read CSV file
                df = pd.read_csv(file_path)
                
                # Convert date column to datetime
                # First extract the month from the filename (assuming format like 'xxxxx3753 (MM)_transactions.csv')
                month_match = re.search(r'\((\d{2})\)', filename)
                
                # Create a date column with proper year
                if month_match:
                    statement_month = int(month_match.group(1))
                    # Determine year based on fiscal year (March 2024 - February 2025)
                    statement_year = 2024 if statement_month >= 3 else 2025
                    
                    # Handle date parsing with proper error handling
                    def parse_date(date_str):
                        try:
                            if pd.isna(date_str) or not isinstance(date_str, str):
                                return pd.NaT
                                
                            # Clean the date string
                            date_str = date_str.strip()
                            
                            # Check if it's already in a full date format (e.g., 2025-02-10)
                            if re.match(r'\d{4}-\d{2}-\d{2}', date_str):
                                return pd.to_datetime(date_str)
                                
                            # Handle DD/MM format
                            if '/' in date_str:
                                day, month = date_str.split('/')
                                return pd.to_datetime(f"{day.zfill(2)}/{month.zfill(2)}/{statement_year}", format='%d/%m/%Y')
                            
                            # Handle DD/MM format with no slash
                            if re.match(r'^\d{2}/\d{2}$', date_str):
                                return pd.to_datetime(f"{date_str}/{statement_year}", format='%d/%m/%Y')
                                
                            # Handle DD/MM format with no slash
                            if re.match(r'^\d{1,2}[/\\-]\d{1,2}$', date_str):
                                parts = re.split(r'[/\\-]', date_str)
                                day, month = parts[0], parts[1]
                                return pd.to_datetime(f"{day.zfill(2)}/{month.zfill(2)}/{statement_year}", format='%d/%m/%Y')
                                
                            # Handle just DD format (assuming same month as statement)
                            if re.match(r'^\d{1,2}$', date_str):
                                day = date_str.zfill(2)
                                month = str(statement_month).zfill(2)
                                return pd.to_datetime(f"{day}/{month}/{statement_year}", format='%d/%m/%Y')
                                
                            # Handle DD/MM format
                            day, month = date_str.split('/')
                            return pd.to_datetime(f"{day.zfill(2)}/{month.zfill(2)}/{statement_year}", format='%d/%m/%Y')
                        except Exception as e:
                            print(f"Error parsing date '{date_str}': {e}")
                            return pd.NaT
                    
                    # Apply the date parsing function
                    df['date'] = df['Date'].apply(parse_date)
                    
                    # Print date parsing statistics
                    valid_dates = df['date'].notna().sum()
                    total_dates = len(df)
                    print(f"Successfully parsed {valid_dates}/{total_dates} dates ({valid_dates/total_dates*100:.1f}%)")
                else:
                    # If can't extract month from filename, try to infer the date format
                    print(f"Warning: Could not extract month from filename {filename}, attempting to infer dates")
                    try:
                        # Try different date formats
                        df['date'] = pd.to_datetime(df['Date'], errors='coerce')
                    except:
                        # Default to current year if all else fails
                        df['date'] = pd.to_datetime(df['Date'], format='%d/%m', errors='coerce')
                
                # Filter data within the date range
                mask = (df['date'] >= start_dt) & (df['date'] <= end_dt)
                df = df[mask]
                
                if not df.empty:
                    # Add source filename
                    df['source'] = filename
                    dfs.append(df)
                    print(f"Added {len(df)} rows from {filename}")
                    
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue
    
    # Combine all dataframes
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df = combined_df.sort_values('date')
        print(f"Combined {len(dfs)} CSV files, total rows: {len(combined_df)}")
        return combined_df
    else:
        raise ValueError("No data found for the specified date range")

def clean_and_process_csv(df):
    """
    Clean and process the DataFrame to create a proper cashbook.
    Works with camelot CSV format (Date,Details,Debits,Credits,Balance,ServiceFee)
    """
    print("Processing combined data")
    
    # Clean numeric columns
    for col in ['Debits', 'Credits', 'Balance']:
        if col in df.columns:
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
    
    # Sort by date
    if 'date' in df.columns:
        df = df.sort_values('date')
    
    # Fix opening balance if provided
    if expected_opening_balance is not None and len(df) > 0:
        print(f"Setting opening balance to {expected_opening_balance}")
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
                if debit != 0:
                    current_balance -= abs(debit)
                if credit != 0:
                    current_balance += abs(credit)
                
                # Update balance in dataframe
                df.iloc[i, df.columns.get_loc('Balance')] = current_balance
    
    # Check closing balance if provided
    if expected_closing_balance is not None and len(df) > 0:
        final_balance = df.iloc[-1]['Balance']
        print(f"Final calculated balance: {final_balance}")
        print(f"Expected closing balance: {expected_closing_balance}")
        
        if abs(final_balance - expected_closing_balance) > 0.01:
            print(f"Adjusting final balance from {final_balance} to {expected_closing_balance}")
            df.iloc[-1, df.columns.get_loc('Balance')] = expected_closing_balance
            
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
    df['bank_fee'] = df['ServiceFee'].apply(lambda x: 1 if x == 'Y' else 0)
    
    # Use Details column as description
    df['description'] = df['Details']
    
    # Categorize transactions
    df = categorize_transactions(df)
    
    return df

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
    Generate Excel cashbook from the processed DataFrame.
    """
    try:
        print("Generating Excel cashbook...")
        
        # Create a copy of the DataFrame to avoid modifying the original
        df_excel = df.copy()
        
        # Ensure we have an Account column
        if 'Account' not in df_excel.columns:
            print("Adding Account column for categorization")
            # Add Account column based on transaction details
            df_excel['Account'] = df_excel.apply(categorize_transaction, axis=1)
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Write detailed transactions to the first sheet
            df_excel.to_excel(writer, sheet_name='Detailed Transactions', index=False)
            
            # Create monthly summary by account
            try:
                monthly_summary = df_excel.groupby([pd.Grouper(key='date', freq='ME'), 'Account']).agg({
                    'Debits': 'sum',
                    'Credits': 'sum'
                }).reset_index()
                
                # Format the date for better readability
                monthly_summary['Month'] = monthly_summary['date'].dt.strftime('%Y-%m')
                monthly_summary = monthly_summary.drop('date', axis=1)
                
                # Reorder columns
                monthly_summary = monthly_summary[['Month', 'Account', 'Debits', 'Credits']]
                
                # Write monthly summary to the second sheet
                monthly_summary.to_excel(writer, sheet_name='Monthly Summary', index=False)
            except Exception as e:
                print(f"Error creating monthly summary: {str(e)}")
                # Create a simple monthly summary without Account grouping
                monthly_summary = df_excel.groupby(pd.Grouper(key='date', freq='ME')).agg({
                    'Debits': 'sum',
                    'Credits': 'sum'
                }).reset_index()
                
                # Format the date for better readability
                monthly_summary['Month'] = monthly_summary['date'].dt.strftime('%Y-%m')
                monthly_summary = monthly_summary.drop('date', axis=1)
                
                # Write monthly summary to the second sheet
                monthly_summary.to_excel(writer, sheet_name='Monthly Summary', index=False)
            
            # Create trial balance
            try:
                trial_balance = df_excel.groupby('Account').agg({
                    'Debits': 'sum',
                    'Credits': 'sum'
                }).reset_index()
                
                # Calculate balance
                trial_balance['Balance'] = trial_balance['Credits'] - trial_balance['Debits']
                
                # Write trial balance to the third sheet
                trial_balance.to_excel(writer, sheet_name='Trial Balance', index=False)
            except Exception as e:
                print(f"Error creating trial balance: {str(e)}")
                # Create a simple summary without Account grouping
                summary = pd.DataFrame({
                    'Total': ['Total'],
                    'Debits': [df_excel['Debits'].sum()],
                    'Credits': [df_excel['Credits'].sum()],
                    'Balance': [df_excel['Credits'].sum() - df_excel['Debits'].sum()]
                })
                
                # Write summary to the third sheet
                summary.to_excel(writer, sheet_name='Summary', index=False)
            
        print(f"Excel cashbook generated: {output_path}")
        
    except Exception as e:
        print(f"Error generating Excel cashbook: {str(e)}")
        import traceback
        traceback.print_exc()

def categorize_transaction(row):
    """
    Categorize a transaction based on its details.
    
    Args:
        row: DataFrame row containing transaction details
        
    Returns:
        str: Account category
    """
    details = str(row.get('Details', '')).upper()
    
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

def process_cashbook(input_dir, output_file, start_date, end_date, debug=False, expected_opening_balance=None, expected_closing_balance=None):
    """
    Process CSV files to generate a cashbook for the specified date range.
    
    Args:
        input_dir (str): Directory containing CSV files
        output_file (str): Path to save the output Excel file
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        debug (bool): Enable debug output
        expected_opening_balance (float): Expected opening balance for verification
        expected_closing_balance (float): Expected closing balance for verification
        
    Returns:
        dict: Result with success status, output path, transaction count, and balance adjustment info
    """
    print(f"Processing cashbook for period {start_date} to {end_date}")
    
    try:
        # Convert date strings to datetime objects
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        print(f"Processing cashbook for period {start_date.date()} to {end_date.date()}")
        
        # Check if combined CSV exists
        combined_csv_path = os.path.join(input_dir, "combined_transactions.csv")
        fixed_transactions_path = os.path.join(input_dir, "fixed_transactions.csv")
        
        if os.path.exists(fixed_transactions_path):
            print(f"Using existing fixed transactions file: {fixed_transactions_path}")
            df = pd.read_csv(fixed_transactions_path)
            print(f"Read {len(df)} rows from fixed transactions file")
        elif os.path.exists(combined_csv_path):
            print(f"Using existing combined CSV file: {combined_csv_path}")
            df = pd.read_csv(combined_csv_path)
            print(f"Read {len(df)} rows from combined CSV file")
        else:
            print("No combined CSV file found. Combining individual CSV files...")
            # Find all CSV files in the input directory
            csv_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.csv') and not f.startswith('combined')]
            
            if not csv_files:
                return {"success": False, "error": "No CSV files found in the input directory"}
                
            print(f"Found {len(csv_files)} CSV files")
            
            # Combine CSV files
            df = combine_csv_files(csv_files, combined_csv_path)
            print(f"Combined {len(csv_files)} CSV files with {len(df)} rows")
        
        # Parse dates
        print("Parsing dates from the Date column...")
        
        # Initialize counters for date parsing statistics
        total_dates = len(df)
        successful_parses = 0
        failed_parses = 0
        
        # Extract month from filename if available
        df['source_file'] = df.get('source_file', '')
        
        # Create a date column
        df['date'] = None
        
        # Try different date formats
        for i, row in df.iterrows():
            date_str = str(row.get('Date', ''))
            source_file = str(row.get('source_file', ''))
            
            # Try to extract month from filename (format: xxxxx3753 (MM)_transactions.csv)
            month = None
            import re
            match = re.search(r'\((\d{2})\)', source_file)
            if match:
                month = match.group(1)
            
            # Try different date formats
            try:
                # First try with the format in the CSV
                df.at[i, 'date'] = pd.to_datetime(date_str, dayfirst=True)
                successful_parses += 1
            except:
                try:
                    # If that fails and we have a month from the filename, try with that
                    if month and len(date_str) <= 5:  # Likely just a day, e.g., "15/06"
                        # Extract day
                        day_match = re.search(r'(\d{1,2})', date_str)
                        if day_match:
                            day = day_match.group(1).zfill(2)
                            # Determine year based on fiscal year and month
                            month_int = int(month)
                            if start_date.month <= month_int <= 12:
                                year = start_date.year
                            else:
                                year = end_date.year
                            
                            # Create full date string
                            full_date_str = f"{day}/{month}/{year}"
                            df.at[i, 'date'] = pd.to_datetime(full_date_str, dayfirst=True)
                            successful_parses += 1
                        else:
                            raise ValueError(f"Could not extract day from {date_str}")
                    else:
                        raise ValueError(f"No month in filename and date format not recognized: {date_str}")
                except Exception as e:
                    if debug:
                        print(f"Error parsing date '{date_str}' from {source_file}: {str(e)}")
                    failed_parses += 1
        
        if debug:
            print(f"Successfully parsed {successful_parses}/{total_dates} dates ({successful_parses/total_dates*100:.1f}%)")
        
        # Filter by date range
        df_filtered = df[df['date'].between(start_date, end_date)]
        print(f"Filtered to {len(df_filtered)} transactions within date range {start_date.date()} to {end_date.date()}")
        
        if debug:
            # Print month distribution
            print("\nMonth distribution in filtered data:")
            month_counts = df_filtered['date'].dt.to_period('M').value_counts().sort_index()
            for month, count in month_counts.items():
                print(f"{month}: {count} transactions")
        
        # Process the earliest and latest dates in the filtered data
        min_date = df_filtered['date'].min()
        max_date = df_filtered['date'].max()
        print(f"Processing transactions from {min_date} to {max_date}")
        
        # Clean and process the data
        df_processed, balance_info = clean_and_process_csv(df_filtered, expected_opening_balance, expected_closing_balance)
        
        # Generate Excel cashbook
        generate_cashbook_excel(df_processed, output_file)
        
        # Return success result with balance info
        result = {
            "success": True, 
            "output_path": output_file,
            "transaction_count": len(df_processed),
            "opening_balance": balance_info.get('opening_balance'),
            "closing_balance": balance_info.get('closing_balance'),
            "opening_balance_difference": balance_info.get('opening_balance_difference'),
            "closing_balance_difference": balance_info.get('closing_balance_difference')
        }
        
        return result
    
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