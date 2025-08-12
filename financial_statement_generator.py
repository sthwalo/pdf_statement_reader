#!/usr/bin/env python3
"""
Financial Statement Generator
Generates Income Statement, Balance Sheet, and Cash Flow Statement from cashbook data.
"""
import pandas as pd
from pathlib import Path
from datetime import datetime
import numpy as np

class FinancialStatementGenerator:
    def __init__(self, cashbook_path=None):
        """Initialize the Financial Statement Generator."""
        default_path = next(Path("data").glob("Annual_Cashbook_*.xlsx"), Path("data/Annual_Cashbook.2023.xlsx"))
        self.cashbook_path = Path(cashbook_path) if cashbook_path else default_path
        self.transactions_df = None
        self.trial_balance_df = None
        self.reporting_period = None
        self.income_accounts = [
            'Income from Services',
            'Interest Income',
            'Other Income',
            'Sales Revenue',
            'Consulting Income',
            'Tuition'
        ]
        self.expense_accounts = [
            'Bank Charges',
            'Communication',
            'Rent',
            'Salaries and Wages',
            'Staff Welfare',
            'Stationery',
            'Telephone',
            'Training',
            'Transport',
            'Uniforms',
            'Outsourced Services',
            'Miscellanious',
            'Investment Expense',
            'Printing and Designs',
            'Business Meetings',
            'Graduation',
            'Compliance',
            'Entertainment',
            'Excursions',
            'Consulting and Professional Fees',
            'Refunds'
        ]
        self.asset_accounts = [
            'Cash and Bank',
            'Accounts Receivable',
            'Equipment',
            'Investments',
            'Petty Cash'
        ]
        self.liability_accounts = [
            'Accounts Payable',
            'Loans Payable',
            'Accrued Expenses'
        ]
        
    def load_data(self):
        """Load transaction data and trial balance from the cashbook."""
        try:
            # Load detailed transactions
            self.transactions_df = pd.read_excel(self.cashbook_path, sheet_name='Detailed Transactions')
            self.transactions_df['date'] = pd.to_datetime(self.transactions_df['date'])
            
            # Load trial balance
            self.trial_balance_df = pd.read_excel(self.cashbook_path, sheet_name='Trial Balance')
            
            # Set reporting period
            self.reporting_period = {
                'start': self.transactions_df['date'].min(),
                'end': self.transactions_df['date'].max()
            }
            return True
        except Exception as e:
            print(f"❌ Error loading cashbook: {e}")
            return False
            
    def generate_income_statement(self, start_date=None, end_date=None):
        """Generate Income Statement (Profit & Loss)."""
        if start_date is None:
            start_date = self.reporting_period['start']
        if end_date is None:
            end_date = self.reporting_period['end']
            
        # Filter transactions for the period
        mask = (self.transactions_df['date'] >= start_date) & (self.transactions_df['date'] <= end_date)
        period_transactions = self.transactions_df[mask]
        
        # Calculate totals for each income and expense account
        income_statement = []
        
        # Revenue section
        total_revenue = 0
        income_section = []
        for account in self.income_accounts:
            amount = period_transactions[period_transactions['Account'] == account]['amount'].sum()
            if amount != 0:  # Only include accounts with activity
                income_section.append({
                    'Account': account,
                    'Amount': abs(amount)  # Income is typically stored as negative in the DB
                })
                total_revenue += abs(amount)
                
        # Expenses section
        total_expenses = 0
        expense_section = []
        for account in self.expense_accounts:
            amount = period_transactions[period_transactions['Account'] == account]['amount'].sum()
            if amount != 0:  # Only include accounts with activity
                expense_section.append({
                    'Account': account,
                    'Amount': abs(amount)  # Make expenses positive for display
                })
                total_expenses += abs(amount)
        
        # Calculate net income
        net_income = total_revenue - total_expenses
        
        return {
            'period_start': start_date,
            'period_end': end_date,
            'income_section': income_section,
            'expense_section': expense_section,
            'total_revenue': total_revenue,
            'total_expenses': total_expenses,
            'net_income': net_income
        }
    
    def generate_balance_sheet(self, as_of_date=None):
        """Generate Balance Sheet using trial balance data."""
        if as_of_date is None:
            as_of_date = self.reporting_period['end']
            
        # Calculate asset balances from trial balance
        assets = []
        total_assets = 0
        for account in self.asset_accounts:
            # Filter trial balance for this account
            account_data = self.trial_balance_df[self.trial_balance_df['Account'] == account]
            if not account_data.empty:
                balance = account_data['Net'].sum()
                if balance != 0:  # Only include accounts with balances
                    assets.append({
                        'Account': account,
                        'Amount': balance
                    })
                    total_assets += balance
                
        # Calculate liability balances from trial balance
        liabilities = []
        total_liabilities = 0
        for account in self.liability_accounts:
            # Filter trial balance for this account
            account_data = self.trial_balance_df[self.trial_balance_df['Account'] == account]
            if not account_data.empty:
                balance = account_data['Net'].sum()
                if balance != 0:  # Only include accounts with balances
                    liabilities.append({
                        'Account': account,
                        'Amount': abs(balance)  # Make liabilities positive for display
                    })
                    total_liabilities += abs(balance)
        
        # Get drawings amount from trial balance
        # Get drawings balance from trial balance
        drawings_data = self.trial_balance_df[self.trial_balance_df['Account'] == 'Drawings']
        drawings_balance = drawings_data['Net'].sum() if not drawings_data.empty else 0
        
        # Calculate retained earnings (net income minus drawings)
        retained_earnings = total_assets - total_liabilities - drawings_balance
        
        return {
            'as_of_date': as_of_date,
            'assets': assets,
            'liabilities': liabilities,
            'total_assets': total_assets,
            'total_liabilities': total_liabilities,
            'retained_earnings': retained_earnings
        }
    
    def generate_cash_flow_statement(self, start_date=None, end_date=None):
        """Generate Cash Flow Statement."""
        if start_date is None:
            start_date = self.reporting_period['start']
        if end_date is None:
            end_date = self.reporting_period['end']
            
        # Filter transactions for the period
        mask = (self.transactions_df['date'] >= start_date) & (self.transactions_df['date'] <= end_date)
        period_transactions = self.transactions_df[mask]
        
        # Operating activities
        operating_cash_flows = []
        
        # Net income from operations
        income_stmt = self.generate_income_statement(start_date, end_date)
        operating_cash_flows.append({
            'Description': 'Net Income',
            'Amount': income_stmt['net_income']
        })
        
        # Changes in working capital
        start_balance_sheet = self.generate_balance_sheet(start_date)
        end_balance_sheet = self.generate_balance_sheet(end_date)
        
        # Calculate changes in working capital accounts
        working_capital_accounts = {
            'Accounts Receivable': 'operating',
            'Accounts Payable': 'operating',
            'Accrued Expenses': 'operating'
        }
        
        def get_account_balance(balance_sheet, account):
            """Helper to get account balance from balance sheet data."""
            for item in balance_sheet['assets'] + balance_sheet['liabilities']:
                if item['Account'] == account:
                    return item['Amount']
            return 0
        
        working_capital_changes = []
        for account, activity_type in working_capital_accounts.items():
            start_balance = get_account_balance(start_balance_sheet, account)
            end_balance = get_account_balance(end_balance_sheet, account)
            change = end_balance - start_balance
            if change != 0:
                working_capital_changes.append({
                    'Account': account,
                    'Change': change,
                    'Type': activity_type
                })
        
        # Investing activities
        investing_activities = []
        investing_accounts = ['Equipment', 'Investments']
        for account in investing_accounts:
            amount = period_transactions[period_transactions['Account'] == account]['amount'].sum()
            if amount != 0:
                investing_activities.append({
                    'Description': f'Changes in {account}',
                    'Amount': amount
                })
        
        # Financing activities
        financing_activities = []
        financing_accounts = ['Loans Payable']
        for account in financing_accounts:
            amount = period_transactions[period_transactions['Account'] == account]['amount'].sum()
            if amount != 0:
                financing_activities.append({
                    'Description': f'Changes in {account}',
                    'Amount': amount
                })
        
        # Calculate net cash flows
        operating_total = sum(item['Amount'] for item in operating_cash_flows)
        investing_total = sum(item['Amount'] for item in investing_activities)
        financing_total = sum(item['Amount'] for item in financing_activities)
        
        # Get opening and closing cash balances
        cash_accounts = ['Cash and Bank', 'Petty Cash']
        opening_cash = sum(get_account_balance(start_balance_sheet, account) for account in cash_accounts)
        closing_cash = sum(get_account_balance(end_balance_sheet, account) for account in cash_accounts)
        
        return {
            'period_start': start_date,
            'period_end': end_date,
            'operating_activities': operating_cash_flows,
            'working_capital_changes': working_capital_changes,
            'investing_activities': investing_activities,
            'financing_activities': financing_activities,
            'operating_total': operating_total,
            'investing_total': investing_total,
            'financing_total': financing_total,
            'opening_cash': opening_cash,
            'closing_cash': closing_cash,
            'net_cash_change': closing_cash - opening_cash
        }
    
    def export_to_excel(self, output_path=None):
        """Export all financial statements to Excel."""
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = Path(f"reports/Financial_Statements_{timestamp}.xlsx")
        else:
            output_path = Path(output_path)
            
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate statements
        income_stmt = self.generate_income_statement()
        balance_sheet = self.generate_balance_sheet()
        cash_flow = self.generate_cash_flow_statement()
        
        # Create Excel writer
        with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
            workbook = writer.book
            
            # Format definitions
            header_format = workbook.add_format({
                'bold': True,
                'font_size': 12,
                'align': 'center',
                'border': 1
            })
            subheader_format = workbook.add_format({
                'bold': True,
                'font_size': 11,
                'align': 'left'
            })
            money_format = workbook.add_format({
                'num_format': '#,##0.00',
                'align': 'right'
            })
            total_format = workbook.add_format({
                'bold': True,
                'num_format': '#,##0.00',
                'align': 'right',
                'top': 1
            })
            
            # Income Statement
            ws_income = workbook.add_worksheet('Income Statement')
            ws_income.set_column('A:A', 40)
            ws_income.set_column('B:B', 15)
            
            # Header
            ws_income.merge_range('A1:B1', 'Income Statement', header_format)
            ws_income.write('A2', f"For the period {income_stmt['period_start'].strftime('%Y-%m-%d')} to {income_stmt['period_end'].strftime('%Y-%m-%d')}")
            
            current_row = 4
            
            # Revenue section
            ws_income.write(current_row, 0, 'Revenue', subheader_format)
            current_row += 1
            
            for item in income_stmt['income_section']:
                ws_income.write(current_row, 0, item['Account'])
                ws_income.write(current_row, 1, item['Amount'], money_format)
                current_row += 1
                
            ws_income.write(current_row, 0, 'Total Revenue', subheader_format)
            ws_income.write(current_row, 1, income_stmt['total_revenue'], total_format)
            current_row += 2
            
            # Expenses section
            ws_income.write(current_row, 0, 'Expenses', subheader_format)
            current_row += 1
            
            for item in income_stmt['expense_section']:
                ws_income.write(current_row, 0, item['Account'])
                ws_income.write(current_row, 1, item['Amount'], money_format)
                current_row += 1
                
            ws_income.write(current_row, 0, 'Total Expenses', subheader_format)
            ws_income.write(current_row, 1, income_stmt['total_expenses'], total_format)
            current_row += 2
            
            # Net Income
            ws_income.write(current_row, 0, 'Net Income', subheader_format)
            ws_income.write(current_row, 1, income_stmt['net_income'], total_format)
            
            # Balance Sheet
            ws_balance = workbook.add_worksheet('Balance Sheet')
            ws_balance.set_column('A:A', 40)
            ws_balance.set_column('B:B', 15)
            
            # Header
            ws_balance.merge_range('A1:B1', 'Balance Sheet', header_format)
            ws_balance.write('A2', f"As of {balance_sheet['as_of_date'].strftime('%Y-%m-%d')}")
            
            current_row = 4
            
            # Assets section
            ws_balance.write(current_row, 0, 'Assets', subheader_format)
            current_row += 1
            
            for item in balance_sheet['assets']:
                ws_balance.write(current_row, 0, item['Account'])
                ws_balance.write(current_row, 1, item['Amount'], money_format)
                current_row += 1
                
            ws_balance.write(current_row, 0, 'Total Assets', subheader_format)
            ws_balance.write(current_row, 1, balance_sheet['total_assets'], total_format)
            current_row += 2
            
            # Liabilities section
            ws_balance.write(current_row, 0, 'Liabilities', subheader_format)
            current_row += 1
            
            for item in balance_sheet['liabilities']:
                ws_balance.write(current_row, 0, item['Account'])
                ws_balance.write(current_row, 1, item['Amount'], money_format)
                current_row += 1
                
            ws_balance.write(current_row, 0, 'Total Liabilities', subheader_format)
            ws_balance.write(current_row, 1, balance_sheet['total_liabilities'], total_format)
            current_row += 2
            
            # Equity section
            ws_balance.write(current_row, 0, 'Equity', subheader_format)
            current_row += 1
            
            # Add Drawings first (as a reduction to equity)
            # Get drawings from trial balance
            drawings_data = self.trial_balance_df[self.trial_balance_df['Account'] == 'Drawings']
            drawings_balance = drawings_data['Net'].sum() if not drawings_data.empty else 0
            if drawings_balance != 0:
                ws_balance.write(current_row, 0, 'Less: Drawings')
                ws_balance.write(current_row, 1, abs(drawings_balance), money_format)
                current_row += 1
            
            ws_balance.write(current_row, 0, 'Retained Earnings')
            ws_balance.write(current_row, 1, balance_sheet['retained_earnings'], money_format)
            current_row += 1
            
            # Total Equity is retained earnings after drawings
            total_equity = balance_sheet['retained_earnings']
            ws_balance.write(current_row, 0, 'Total Equity', subheader_format)
            ws_balance.write(current_row, 1, total_equity, total_format)
            
            # Cash Flow Statement
            ws_cash = workbook.add_worksheet('Cash Flow')
            ws_cash.set_column('A:A', 40)
            ws_cash.set_column('B:B', 15)
            
            # Header
            ws_cash.merge_range('A1:B1', 'Cash Flow Statement', header_format)
            ws_cash.write('A2', f"For the period {cash_flow['period_start'].strftime('%Y-%m-%d')} to {cash_flow['period_end'].strftime('%Y-%m-%d')}")
            
            current_row = 4
            
            # Operating activities
            ws_cash.write(current_row, 0, 'Operating Activities', subheader_format)
            current_row += 1
            
            for item in cash_flow['operating_activities']:
                ws_cash.write(current_row, 0, item['Description'])
                ws_cash.write(current_row, 1, item['Amount'], money_format)
                current_row += 1
            
            # Working capital changes
            for item in cash_flow['working_capital_changes']:
                ws_cash.write(current_row, 0, f"Changes in {item['Account']}")
                ws_cash.write(current_row, 1, item['Change'], money_format)
                current_row += 1
                
            ws_cash.write(current_row, 0, 'Net Cash from Operations', subheader_format)
            ws_cash.write(current_row, 1, cash_flow['operating_total'], total_format)
            current_row += 2
            
            # Investing activities
            if cash_flow['investing_activities']:
                ws_cash.write(current_row, 0, 'Investing Activities', subheader_format)
                current_row += 1
                
                for item in cash_flow['investing_activities']:
                    ws_cash.write(current_row, 0, item['Description'])
                    ws_cash.write(current_row, 1, item['Amount'], money_format)
                    current_row += 1
                    
                ws_cash.write(current_row, 0, 'Net Cash from Investing', subheader_format)
                ws_cash.write(current_row, 1, cash_flow['investing_total'], total_format)
                current_row += 2
            
            # Financing activities
            if cash_flow['financing_activities']:
                ws_cash.write(current_row, 0, 'Financing Activities', subheader_format)
                current_row += 1
                
                for item in cash_flow['financing_activities']:
                    ws_cash.write(current_row, 0, item['Description'])
                    ws_cash.write(current_row, 1, item['Amount'], money_format)
                    current_row += 1
                    
                ws_cash.write(current_row, 0, 'Net Cash from Financing', subheader_format)
                ws_cash.write(current_row, 1, cash_flow['financing_total'], total_format)
                current_row += 2
            
            # Summary
            ws_cash.write(current_row, 0, 'Opening Cash Balance')
            ws_cash.write(current_row, 1, cash_flow['opening_cash'], money_format)
            current_row += 1
            
            ws_cash.write(current_row, 0, 'Net Change in Cash')
            ws_cash.write(current_row, 1, cash_flow['net_cash_change'], money_format)
            current_row += 1
            
            ws_cash.write(current_row, 0, 'Closing Cash Balance', subheader_format)
            ws_cash.write(current_row, 1, cash_flow['closing_cash'], total_format)
            
        print(f"✅ Financial statements exported to: {output_path}")
        return output_path

def determine_financial_year(transactions_df):
    """Determine the financial year based on transaction dates."""
    min_date = transactions_df['date'].min()
    max_date = transactions_df['date'].max()
    
    # If most transactions fall in a particular year, use that as the base year
    year_counts = transactions_df['date'].dt.year.value_counts()
    base_year = year_counts.index[0]  # Most common year
    
    # Financial year typically starts in March
    start_date = pd.Timestamp(f"{base_year}-03-01")
    
    # Determine if the next year is a leap year
    next_year = base_year + 1
    if next_year % 4 == 0 and (next_year % 100 != 0 or next_year % 400 == 0):
        # Leap year - use February 29
        end_date = pd.Timestamp(f"{next_year}-02-29")
    else:
        # Not a leap year - use February 28
        end_date = pd.Timestamp(f"{next_year}-02-28")
    
    # Adjust if transactions suggest a different period
    if min_date > start_date:
        start_date = min_date
    if max_date < end_date:
        end_date = max_date
    
    return start_date, end_date

def main():
    """Generate financial statements from cashbook data."""
    # Get cashbook path from command line if provided
    import sys
    cashbook_path = sys.argv[1] if len(sys.argv) > 1 else None
    
    generator = FinancialStatementGenerator(cashbook_path)
    if generator.load_data():
        # Determine financial year from transaction data
        start_date, end_date = determine_financial_year(generator.transactions_df)
        
        # Format dates for filename
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        # Create descriptive filename
        year_str = f"FY{start_date.year}-{end_date.year}"
        output_path = generator.export_to_excel(f"reports/Financial_Statements_{year_str}_{start_str}_{end_str}.xlsx")
        
        print(f"\n💰 Financial statements have been generated successfully!")
        print(f"📊 Period: {start_str} to {end_str}")
        print(f"📊 You can find the reports at: {output_path}")
    else:
        print("❌ Failed to generate financial statements")

if __name__ == "__main__":
    main()
