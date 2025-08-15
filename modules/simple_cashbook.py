#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

def categorize_security_transaction(details_text):
    """
    Categorize a transaction based on its details text for security industry.
    
    Args:
        details_text: String containing transaction details
        
    Returns:
        str: Account category
    """
    details = str(details_text).upper()
    
    # Define categories based on keywords for security industry
    categories = {
        # Income categories
        'SECURITY SERVICES': ['SECURITY', 'GUARD', 'PATROL', 'MONITORING', 'SURVEILLANCE', 'PROTECTION'],
        'ALARM MONITORING': ['ALARM', 'MONITOR', 'RESPONSE'],
        'EQUIPMENT SALES': ['EQUIPMENT', 'CAMERA', 'CCTV', 'INSTALL', 'SYSTEM SALE'],
        'CONSULTING FEES': ['CONSULT', 'ASSESSMENT', 'AUDIT', 'RISK ASSESSMENT'],
        
        # Expense categories
        'SALARIES AND WAGES': ['SALARY', 'WAGE', 'PAYROLL', 'STAFF PAYMENT'],
        'VEHICLE EXPENSES': ['FUEL', 'PETROL', 'DIESEL', 'VEHICLE', 'CAR', 'MAINTENANCE', 'REPAIR'],
        'EQUIPMENT PURCHASES': ['PURCHASE', 'EQUIPMENT', 'UNIFORM', 'RADIO', 'WEAPON'],
        'INSURANCE': ['INSURANCE', 'ASSURANCE', 'POLICY', 'COVER'],
        'TRAINING': ['TRAINING', 'COURSE', 'CERTIFICATION', 'LICENSE'],
        'OFFICE EXPENSES': ['RENT', 'LEASE', 'OFFICE', 'UTILITIES', 'ELECTRICITY', 'WATER'],
        'TELECOMMUNICATIONS': ['TELKOM', 'MTN', 'VODACOM', 'CELL C', 'TELEPHONE', 'AIRTIME', 'DATA'],
        
        # Financial categories
        'BANK CHARGES': ['BANK CHARGES', 'SERVICE FEE', 'ADMIN FEE', 'ACCOUNT FEE', 'TRANSACTION FEE'],
        'INTEREST': ['INTEREST', 'INT ON', 'EXCESS INTEREST'],
        'LOAN REPAYMENTS': ['LOAN', 'REPAYMENT', 'INSTALLMENT'],
        
        # Asset/Liability categories
        'ASSET PURCHASES': ['ASSET', 'PROPERTY', 'BUILDING'],
        'TAXES': ['TAX', 'VAT', 'PAYE', 'UIF', 'SDL'],
        
        # Transfer categories
        'TRANSFERS': ['TRANSFER', 'PAYMENT', 'IMMEDIATE PAYMENT', 'INTERNET PMT'],
    }
    
    # Check for matches in transaction details
    for category, keywords in categories.items():
        for keyword in keywords:
            if keyword in details:
                return category
    
    # Default category if no match found
    return 'UNCATEGORIZED'

def process_simple_cashbook(input_csv, output_file, fiscal_start_month=3, fiscal_start_day=1):
    """
    Process a combined CSV file to generate a simplified cashbook with fiscal year organization.
    
    Args:
        input_csv (str): Path to the combined CSV file
        output_file (str): Path to save the output Excel file
        fiscal_start_month (int): Month when fiscal year starts (1-12)
        fiscal_start_day (int): Day when fiscal year starts (1-31)
        
    Returns:
        dict: Result with success status, output path, transaction count
    """
    logger.info(f"Processing simplified cashbook from {input_csv}")
    
    try:
        # Read the combined CSV file
        df = pd.read_csv(input_csv)
        logger.info(f"Read {len(df)} rows from {input_csv}")
        
        # Ensure we have a date column for processing
        logger.info("Processing date information...")
        
        # Check for various date column names
        date_columns = ['date', 'Date', 'DATE', 'Transaction Date', 'TransactionDate']
        found_date_col = None
        
        for col in date_columns:
            if col in df.columns:
                found_date_col = col
                break
        
        if found_date_col:
            logger.info(f"Found date column: {found_date_col}")
            
            # Create a working copy of the date column
            df['original_date'] = df[found_date_col].copy()
            
            # Check if Date column is already in datetime format
            if pd.api.types.is_datetime64_any_dtype(df[found_date_col]):
                df['date'] = df[found_date_col]
                logger.info("Date column is already in datetime format")
                converted = True
            
            # Try to parse dates directly first with multiple formats
            date_formats = ['%d/%m/%Y', '%Y-%m-%d', '%d-%m-%Y', '%d %b %Y', '%d %B %Y', '%d/%m']
            converted = False
            
            for date_format in date_formats:
                try:
                    # Handle special case for DD/MM format
                    if date_format == '%d/%m' and 'StatementPeriod' in df.columns:
                        # Extract year from StatementPeriod
                        df['TempYear'] = df['StatementPeriod'].str.extract(r'(\d{4})').iloc[:, 0]
                        # Combine with Date
                        df['FullDate'] = df.apply(
                            lambda row: f"{row[found_date_col]}/{row['TempYear']}" if pd.notna(row[found_date_col]) else None, 
                            axis=1
                        )
                        df['date'] = pd.to_datetime(df['FullDate'], format='%d/%m/%Y', errors='coerce')
                        df.drop(['TempYear', 'FullDate'], axis=1, inplace=True)
                    else:
                        df['date'] = pd.to_datetime(df[found_date_col], format=date_format, errors='coerce')
                    
                    if not df['date'].isna().all():
                        converted = True
                        logger.info(f"Successfully parsed dates with format {date_format}")
                        break
                except Exception as e:
                    logger.debug(f"Failed to parse dates with format {date_format}: {e}")
            
            # If all specific formats fail, try pandas' flexible parser
            if not converted:
                try:
                    df['date'] = pd.to_datetime(df[found_date_col], errors='coerce', dayfirst=True)
                    if not df['date'].isna().all():
                        converted = True
                        logger.info("Parsed dates using pandas' flexible parser")
                except Exception as e:
                    logger.debug(f"Failed to parse dates with flexible parser: {e}")
            
            # If direct parsing failed, try using StatementPeriod for inference
            if not converted and 'StatementPeriod' in df.columns:
                logger.info("Using StatementPeriod to help with date inference")
                
                # Extract statement period information
                # Format is typically "14 December 2024 - 16 January 2025"
                def extract_statement_period(period_str):
                    try:
                        if not isinstance(period_str, str):
                            return None, None
                            
                        # Split by dash to get start and end dates
                        parts = period_str.split('-')
                        if len(parts) != 2:
                            return None, None
                            
                        start_date_str = parts[0].strip()
                        end_date_str = parts[1].strip()
                        
                        # Parse dates
                        import re
                        
                        # Extract month and year from start date
                        start_match = re.search(r'(\d+)\s+(\w+)\s+(\d{4})', start_date_str)
                        if not start_match:
                            return None, None
                            
                        start_month = start_match.group(2)
                        start_year = int(start_match.group(3))
                        
                        # Extract month and year from end date
                        end_match = re.search(r'(\d+)\s+(\w+)\s+(\d{4})', end_date_str)
                        if not end_match:
                            return None, None
                            
                        end_month = end_match.group(2)
                        end_year = int(end_match.group(3))
                        
                        return (start_month, start_year), (end_month, end_year)
                    except Exception as e:
                        logger.debug(f"Error extracting statement period: {e}")
                        return None, None
                
                # Get unique statement periods
                statement_periods = df['StatementPeriod'].unique()
                period_info = {}
                
                for period in statement_periods:
                    start_info, end_info = extract_statement_period(period)
                    if start_info and end_info:
                        period_info[period] = (start_info, end_info)
                
                # Create a function to infer full dates from DD/MM format
                def infer_full_date(date_str, statement_period):
                    try:
                        if not isinstance(date_str, str):
                            return None
                            
                        # Parse DD/MM format
                        parts = date_str.split('/')
                        if len(parts) != 2:
                            return None
                            
                        day = int(parts[0])
                        month = int(parts[1])
                        
                        # Get statement period info
                        if statement_period not in period_info:
                            return None
                            
                        (start_month_name, start_year), (end_month_name, end_year) = period_info[statement_period]
                        
                        # Convert month name to number
                        from datetime import datetime
                        month_names = {
                            'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
                            'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
                        }
                        
                        start_month_num = None
                        for month_name, month_num in month_names.items():
                            if month_name.lower() in start_month_name.lower():
                                start_month_num = month_num
                                break
                                
                        end_month_num = None
                        for month_name, month_num in month_names.items():
                            if month_name.lower() in end_month_name.lower():
                                end_month_num = month_num
                                break
                        
                        if start_month_num is None or end_month_num is None:
                            # Use the transaction's month directly
                            return datetime(start_year, month, day)
                        
                        # Determine year based on statement period
                        year = start_year
                        
                        # If statement crosses year boundary (Dec-Jan)
                        if start_month_num > end_month_num:
                            # If transaction month is in the second part of the period
                            if month < start_month_num:
                                year = end_year
                        
                        return datetime(year, month, day)
                    except Exception as e:
                        logger.debug(f"Error inferring full date: {e}")
                        return None
                
                # Apply date inference
                df['date'] = df.apply(
                    lambda row: infer_full_date(row[found_date_col], row['StatementPeriod']), 
                    axis=1
                )
                
                # Log results
                valid_dates = df['date'].notna().sum()
                total_dates = len(df)
                logger.info(f"Inferred {valid_dates} dates out of {total_dates} from DD/MM format using statement periods")
                
                # If we still have missing dates, try one more time with a fallback approach
                if valid_dates < total_dates:
                    logger.info("Some dates could not be parsed, applying fallback method")
                    
                    # For rows with missing dates, try a more aggressive approach
                    def fallback_date_parser(row):
                        if pd.notna(row['date']):
                            return row['date']  # Keep already parsed dates
                            
                        date_str = row[found_date_col]
                        if not isinstance(date_str, str):
                            return None
                            
                        # Try to extract day and month
                        try:
                            parts = date_str.split('/')
                            if len(parts) == 2:
                                day = int(parts[0])
                                month = int(parts[1])
                                
                                # Use first year from statement period if available
                                if row['StatementPeriod'] and isinstance(row['StatementPeriod'], str):
                                    year_match = re.search(r'\d{4}', row['StatementPeriod'])
                                    if year_match:
                                        year = int(year_match.group(0))
                                        from datetime import datetime
                                        return datetime(year, month, day)
                        except Exception as e:
                            logger.debug(f"Fallback parsing failed: {e}")
                        return None
                    
                    # Apply fallback parser
                    df['date'] = df.apply(fallback_date_parser, axis=1)
                    
                    # Log updated results
                    valid_dates_after_fallback = df['date'].notna().sum()
                    logger.info(f"After fallback: {valid_dates_after_fallback} dates out of {total_dates} parsed successfully")
                
            else:
                # Fallback to simpler date parsing
                logger.info("No StatementPeriod found, using basic date parsing")
                
                # For DD/MM format, assume current year
                current_year = datetime.now().year
                
                def parse_simple_date(date_str):
                    try:
                        if not isinstance(date_str, str):
                            return None
                            
                        # Parse DD/MM format
                        parts = date_str.split('/')
                        if len(parts) != 2:
                            return None
                            
                        day = int(parts[0])
                        month = int(parts[1])
                        
                        # Use current year as default
                        from datetime import datetime
                        return datetime(current_year, month, day)
                    except Exception as e:
                        logger.debug(f"Error parsing simple date: {e}")
                        return None
                
                df['date'] = df[found_date_col].apply(parse_simple_date)
            
            # Fill any remaining NaT values with a default date that's clearly identifiable
            missing_dates = df['date'].isna().sum()
            if missing_dates > 0:
                logger.warning(f"Could not parse {missing_dates} dates, using default date")
                df['date'] = df['date'].fillna(pd.Timestamp('2099-12-31'))
                
            logger.info(f"Date parsing complete. Sample dates: {df['date'].head(5).tolist()}")
        else:
            logger.warning("No date column found. Creating placeholder date column.")
            # Create a placeholder date column
            df['date'] = pd.Timestamp('2099-12-31')
        
        # Use existing FiscalPeriod if available, otherwise calculate it
        if 'FiscalPeriod' in df.columns:
            logger.info("Using existing FiscalPeriod column from CSV")
        elif 'date' in df.columns:
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
                
                # Add fiscal period column
                def get_fiscal_period(date_val):
                    if pd.isna(date_val):
                        return None
                        
                    month = date_val.month
                    day = date_val.day
                    year = date_val.year
                    
                    # Check if date is on or after the fiscal start date in the same year
                    if (month > fiscal_start_month) or (month == fiscal_start_month and day >= fiscal_start_day):
                        fiscal_year = f"{year}-{year+1}"
                    else:
                        fiscal_year = f"{year-1}-{year}"
                        
                    return fiscal_year
                    
                df['FiscalPeriod'] = df['date'].apply(get_fiscal_period)
                
                # Add fiscal quarter
                def get_fiscal_quarter(date_val):
                    if pd.isna(date_val):
                        return None
                        
                    month = date_val.month
                    day = date_val.day
                    
                    # Calculate fiscal month (1-12, with fiscal_start_month being month 1)
                    if (month > fiscal_start_month) or (month == fiscal_start_month and day >= fiscal_start_day):
                        fiscal_month = (month - fiscal_start_month) % 12 + 1
                    else:
                        fiscal_month = (month + 12 - fiscal_start_month) % 12 + 1
                    
                    # Determine quarter (1-4)
                    quarter = (fiscal_month - 1) // 3 + 1
                    
                    return f"Q{quarter}"
                    
                df['FiscalQuarter'] = df['date'].apply(get_fiscal_quarter)
                logger.info("Added fiscal quarters to transactions")
        
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
        
        # Add transaction categories
        if 'Details' in df.columns:
            df['Category'] = df['Details'].apply(categorize_security_transaction)
            logger.info("Added security industry transaction categories")
        
        # Sort by fiscal period, then by date
        if 'FiscalPeriod' in df.columns and 'date' in df.columns:
            df = df.sort_values(['FiscalPeriod', 'date'])
            logger.info("Sorted transactions by fiscal period and date")
        elif 'date' in df.columns:
            df = df.sort_values('date')
            logger.info("Sorted transactions by date only")
        
        # Generate Excel cashbook
        generate_simple_cashbook_excel(df, output_file)
        
        return {
            "success": True,
            "output_path": output_file,
            "transaction_count": len(df)
        }
    
    except Exception as e:
        logger.error(f"Error processing cashbook: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            'success': False,
            'output_path': None,
            'error': str(e),
            'transaction_count': 0
        }

def generate_simple_cashbook_excel(df, output_path):
    """
    Generate a simplified Excel cashbook from the processed DataFrame.
    
    Args:
        df: DataFrame with processed transaction data
        output_path: Path to save the Excel file
    """
    logger.info(f"Generating simplified Excel cashbook at {output_path}")
    
    try:
        # Create a copy of the DataFrame to avoid modifying the original
        df_excel = df.copy()
        
        # Calculate totals
        total_debits = df_excel['Debits'].sum()
        total_credits = df_excel['Credits'].sum()
        opening_balance = df_excel.iloc[0]['Balance'] if len(df_excel) > 0 else 0
        closing_balance = df_excel.iloc[-1]['Balance'] if len(df_excel) > 0 else 0
        
        # Print balance verification
        logger.info(f"Opening balance: {opening_balance:.2f}")
        logger.info(f"Total debits: {total_debits:.2f}")
        logger.info(f"Total credits: {total_credits:.2f}")
        logger.info(f"Closing balance: {closing_balance:.2f}")
        
        # Create Excel writer
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # 1. Detailed transactions sheet
            logger.info("Generating detailed transactions sheet...")
            
            # Format date column if present
            if 'date' in df_excel.columns:
                # Keep original date format if available
                if 'original_date' in df_excel.columns:
                    df_excel['Date_Formatted'] = df_excel['original_date']
                else:
                    # Handle NaN values in date column
                    df_excel['Date_Formatted'] = df_excel['date'].apply(
                        lambda x: x.strftime('%d %b %Y') if pd.notna(x) else ''
                    )
            
            # Select and reorder columns for detailed transactions
            columns_to_include = []
            if 'FiscalPeriod' in df_excel.columns:
                columns_to_include.append('FiscalPeriod')
            if 'FiscalQuarter' in df_excel.columns:
                columns_to_include.append('FiscalQuarter')
            if 'Date_Formatted' in df_excel.columns:
                columns_to_include.append('Date_Formatted')
            # Include original transaction number if available
            if 'TransactionNumber' in df_excel.columns:
                columns_to_include.append('TransactionNumber')
            if 'Details' in df_excel.columns:
                columns_to_include.append('Details')
            if 'Description' in df_excel.columns and 'Details' not in df_excel.columns:
                columns_to_include.append('Description')
            if 'Category' in df_excel.columns:
                columns_to_include.append('Category')
            
            # Always include financial columns
            columns_to_include.extend(['Debits', 'Credits', 'Balance'])
            
            # Filter to only include selected columns that exist
            columns_to_include = [col for col in columns_to_include if col in df_excel.columns]
            
            # Create detailed transactions sheet
            detailed_df = df_excel[columns_to_include].copy()
            
            # Add a balance verification row at the top
            verification_data = {col: [''] for col in columns_to_include}
            verification_data[columns_to_include[0]] = ['BALANCE VERIFICATION']
            if 'Details' in columns_to_include:
                idx = columns_to_include.index('Details')
                verification_data[columns_to_include[idx]] = [f'Opening: {opening_balance:.2f}, Debits: {total_debits:.2f}, Credits: {total_credits:.2f}, Closing: {closing_balance:.2f}']
            
            verification_df = pd.DataFrame(verification_data)
            
            # Combine verification row with transaction data
            detailed_with_verification = pd.concat([verification_df, detailed_df], ignore_index=True)
            
            # Write to Excel
            detailed_with_verification.to_excel(writer, sheet_name='Detailed Transactions', index=False)
            
            # 2. Monthly summary sheet
            if 'date' in df_excel.columns:
                logger.info("Generating monthly summary by fiscal period...")
                
                # Add month and year columns
                df_excel['month'] = df_excel['date'].dt.month
                df_excel['year'] = df_excel['date'].dt.year
                df_excel['month_name'] = df_excel['date'].dt.strftime('%B')
                
                # Group by fiscal period and month
                if 'FiscalPeriod' in df_excel.columns:
                    monthly_summary = df_excel.groupby(['FiscalPeriod', 'year', 'month', 'month_name']).agg({
                        'Debits': 'sum',
                        'Credits': 'sum',
                        'Balance': 'last'
                    }).reset_index()
                    
                    # Calculate net movement
                    monthly_summary['Net'] = monthly_summary['Credits'] - monthly_summary['Debits']
                    
                    # Sort by fiscal period, year, and month
                    monthly_summary = monthly_summary.sort_values(['FiscalPeriod', 'year', 'month'])
                    
                    # Add subtotals for each fiscal period
                    fiscal_totals = df_excel.groupby(['FiscalPeriod']).agg({
                        'Debits': 'sum',
                        'Credits': 'sum',
                        'Balance': 'last'
                    }).reset_index()
                    
                    fiscal_totals['Net'] = fiscal_totals['Credits'] - fiscal_totals['Debits']
                    fiscal_totals['year'] = ''
                    fiscal_totals['month'] = ''
                    fiscal_totals['month_name'] = 'FISCAL YEAR TOTAL'
                    
                    # Combine monthly summary with fiscal totals
                    monthly_summary = pd.concat([monthly_summary, fiscal_totals], ignore_index=True)
                    
                    # Sort again to ensure fiscal totals appear at the end of each fiscal period
                    monthly_summary = monthly_summary.sort_values(['FiscalPeriod', 'year', 'month'], 
                                                               na_position='last')
                    
                    # Reorder columns
                    monthly_summary = monthly_summary[['FiscalPeriod', 'year', 'month', 'month_name', 
                                                    'Debits', 'Credits', 'Net', 'Balance']]
                else:
                    # If no fiscal period, just group by year and month
                    monthly_summary = df_excel.groupby(['year', 'month', 'month_name']).agg({
                        'Debits': 'sum',
                        'Credits': 'sum',
                        'Balance': 'last'
                    }).reset_index()
                    
                    # Calculate net movement
                    monthly_summary['Net'] = monthly_summary['Credits'] - monthly_summary['Debits']
                    
                    # Sort by year and month
                    monthly_summary = monthly_summary.sort_values(['year', 'month'])
                    
                    # Reorder columns
                    monthly_summary = monthly_summary[['year', 'month', 'month_name', 
                                                    'Debits', 'Credits', 'Net', 'Balance']]
                
                # Add grand total row
                total_row_data = {col: '' for col in monthly_summary.columns}
                total_row_data[monthly_summary.columns[0]] = 'GRAND TOTAL'
                total_row_data['Debits'] = monthly_summary['Debits'].sum()
                total_row_data['Credits'] = monthly_summary['Credits'].sum()
                total_row_data['Net'] = monthly_summary['Credits'].sum() - monthly_summary['Debits'].sum()
                total_row_data['Balance'] = monthly_summary['Balance'].iloc[-1] if len(monthly_summary) > 0 else 0
                
                total_row = pd.DataFrame([total_row_data])
                monthly_summary = pd.concat([monthly_summary, total_row], ignore_index=True)
                
                # Write to Excel
                monthly_summary.to_excel(writer, sheet_name='Monthly Summary', index=False)
            
            # 3. Trial balance sheet
            if 'Category' in df_excel.columns:
                logger.info("Generating trial balance by category...")
                
                # Group by category
                trial_balance = df_excel.groupby('Category').agg({
                    'Debits': 'sum',
                    'Credits': 'sum'
                }).reset_index()
                
                # Calculate net amount
                trial_balance['Net'] = trial_balance['Credits'] - trial_balance['Debits']
                
                # Determine account type based on category
                def determine_account_type(category):
                    income_categories = ['SECURITY SERVICES', 'ALARM MONITORING', 'EQUIPMENT SALES', 'CONSULTING FEES']
                    expense_categories = ['SALARIES AND WAGES', 'VEHICLE EXPENSES', 'EQUIPMENT PURCHASES', 
                                         'INSURANCE', 'TRAINING', 'OFFICE EXPENSES', 'TELECOMMUNICATIONS',
                                         'BANK CHARGES']
                    asset_categories = ['ASSET PURCHASES']
                    liability_categories = ['TAXES', 'LOAN REPAYMENTS']
                    
                    if category in income_categories:
                        return 'Income'
                    elif category in expense_categories:
                        return 'Expense'
                    elif category in asset_categories:
                        return 'Asset'
                    elif category in liability_categories:
                        return 'Liability'
                    else:
                        return 'Other'
                
                trial_balance['Account Type'] = trial_balance['Category'].apply(determine_account_type)
                
                # Sort by account type and category
                trial_balance = trial_balance.sort_values(['Account Type', 'Category'])
                
                # Add subtotals by account type
                account_type_totals = trial_balance.groupby('Account Type').agg({
                    'Debits': 'sum',
                    'Credits': 'sum',
                    'Net': 'sum'
                }).reset_index()
                
                account_type_totals['Category'] = account_type_totals['Account Type'] + ' TOTAL'
                
                # Combine trial balance with account type totals
                trial_balance = pd.concat([trial_balance, account_type_totals], ignore_index=True)
                
                # Add grand total row
                total_row_data = {
                    'Category': 'GRAND TOTAL',
                    'Account Type': '',
                    'Debits': trial_balance['Debits'].sum(),
                    'Credits': trial_balance['Credits'].sum(),
                    'Net': trial_balance['Credits'].sum() - trial_balance['Debits'].sum()
                }
                
                total_row = pd.DataFrame([total_row_data])
                trial_balance = pd.concat([trial_balance, total_row], ignore_index=True)
                
                # Reorder columns
                trial_balance = trial_balance[['Account Type', 'Category', 'Debits', 'Credits', 'Net']]
                
                # Write to Excel
                trial_balance.to_excel(writer, sheet_name='Trial Balance', index=False)
            
            # 4. Balance verification sheet
            logger.info("Generating balance verification sheet...")
            verification_data = {
                'Item': ['Opening Balance', 'Total Debits', 'Total Credits', 'Net Movement', 'Calculated Closing Balance', 'Actual Closing Balance'],
                'Amount': [
                    opening_balance,
                    total_debits,
                    total_credits,
                    total_credits - total_debits,
                    opening_balance - total_debits + total_credits,
                    closing_balance
                ]
            }
            verification_df = pd.DataFrame(verification_data)
            verification_df.to_excel(writer, sheet_name='Balance Verification', index=False)
        
        logger.info(f"Excel cashbook generated successfully at {output_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error generating Excel cashbook: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    Main function when running as a script.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate a simplified cashbook with fiscal year organization')
    parser.add_argument('--input', required=True, help='Path to the combined CSV file')
    parser.add_argument('--output', required=True, help='Path to save the Excel cashbook')
    parser.add_argument('--fiscal-start-month', type=int, default=3, help='Month when fiscal year starts (1-12)')
    parser.add_argument('--fiscal-start-day', type=int, default=1, help='Day when fiscal year starts (1-31)')
    
    args = parser.parse_args()
    
    result = process_simple_cashbook(
        args.input,
        args.output,
        args.fiscal_start_month,
        args.fiscal_start_day
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
