#!/usr/bin/env python3
"""
Interactive 2024 Cashbook Audit System
Focused audit and account allocation for Annual_Cashbook_2024.xlsx
"""

import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import json
import shutil
import sys
import shutil
import json  
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class Interactive2024Audit:
    def __init__(self):
        self.file_path = Path("data/output/2025/Annual_Cashbook_2025.xlsx")  # Updated to 2024 file
        self.transactions_df = None
        self.config_file = Path("data/account_categories.json")
        self.changes_made = []
        
        # Load account categories from config file or use defaults
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                self.account_categories = json.load(f)
        else:
            self.account_categories = {
                'Income': [
                    'Income from Services',
                    'Interest Income',
                    'Other Income',
                    'Sales Revenue',
                    'Tuition'
                ],
                'Expenses': [
                    'Bank Charges',
                    'Communication',
                    'Rent',
                    'Salaries',
                    'Staff Welfare',
                    'Stationery',
                    'Telephone',
                    'Training',
                    'Transport',
                    'Uniforms',
                    'Outsourced Services',
                    'Miscellanious',
                    'Investment Expense',
                    'Professional Fees'
                ],
                'Assets': [
                    'Cash and Bank',
                    'Accounts Receivable',
                    'Equipment',
                    'Investments'
                ],
                'Liabilities': [
                    'Accounts Payable',
                    'Loans Payable',
                    'Accrued Expenses'
                ]
            }
            # Save default categories
            self.save_account_categories()
        
        self.changes_made = []
        
    def load_data(self):
        """Load the 2024 cashbook data."""
        print("="*80)
        print("INTERACTIVE 2024 CASHBOOK AUDIT SYSTEM")
        print("="*80)
        print(f"Loading: {self.file_path.name}")
        
        if not self.file_path.exists():
            print(f"❌ Error: File {self.file_path} not found!")
            return False
            
        try:
            # Load detailed transactions
            self.transactions_df = pd.read_excel(self.file_path, sheet_name='Detailed Transactions')
            
            # Convert date column to datetime with error handling
            try:
                # Try multiple date formats
                for date_format in ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%d-%m-%Y', '%Y/%m/%d']:
                    try:
                        self.transactions_df['date'] = pd.to_datetime(
                            self.transactions_df['date'], 
                            format=date_format,
                            errors='raise'
                        )
                        # If we get here, the format worked
                        print(f"✅ Successfully parsed dates using format: {date_format}")
                        break
                    except (ValueError, TypeError):
                        continue
                
                # If none of the formats worked, try the flexible parser
                if not pd.api.types.is_datetime64_any_dtype(self.transactions_df['date']):
                    self.transactions_df['date'] = pd.to_datetime(
                        self.transactions_df['date'], 
                        infer_datetime_format=True,
                        errors='raise'
                    )
                    print("✅ Successfully parsed dates using flexible format")
                    
            except Exception as e:
                print(f"⚠️  Error parsing dates: {str(e)}")
                print("❌ Please ensure date column contains valid dates")
            
            # Verify no 1970 dates
            if any(self.transactions_df['date'].dt.year == 1970):
                print("⚠️  Warning: Found dates from 1970 - possible parsing issue")
                print("Please check the date format in your Excel file")
            
            print(f"✅ Loaded {len(self.transactions_df)} transactions")
            
            # Show file overview
            print(f"\nFile Overview:")
            print(f"Date Range: {self.transactions_df['date'].min().strftime('%Y-%m-%d')} to {self.transactions_df['date'].max().strftime('%Y-%m-%d')}")
            print(f"Balance Range: {self.transactions_df['balance'].min():,.2f} to {self.transactions_df['balance'].max():,.2f}")
            
            # Show current account distribution
            account_dist = self.transactions_df['Account'].value_counts()
            uncategorized_count = account_dist.get('Uncategorized', 0)
            
            print(f"\nCurrent Account Allocation:")
            print(f"  Uncategorized: {uncategorized_count} transactions")
            print(f"  Categorized: {len(self.transactions_df) - uncategorized_count} transactions")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading file: {e}")
            return False
    
    def display_transaction(self, idx, transaction):
        """Display a single transaction with full details."""
        print("\n" + "="*80)
        print(f"TRANSACTION #{idx + 1} of {len(self.transactions_df)}")
        print("="*80)
        
        print(f"📅 Date:        {transaction['date'].strftime('%Y-%m-%d')}")
        print(f"📝 Description: {transaction['description']}")
        print(f"💰 Amount:      {transaction['amount']:>10,.2f}")
        print(f"🏦 Balance:     {transaction['balance']:>10,.2f}")
        print(f"💳 Bank Fee:    {transaction['bank_fee']:>10,.2f}")
        print(f"📊 Type:        {transaction['Type']}")
        print(f"💸 Debit:       {transaction['Debit']:>10,.2f}")
        print(f"💵 Credit:      {transaction['Credit']:>10,.2f}")
        print(f"🏷️  Account:     {transaction['Account']}")
        print(f"📂 Acc Type:    {transaction['Account Type']}")
        
        # Show context (previous and next transactions)
        print(f"\n📋 Context:")
        if idx > 0:
            prev_tx = self.transactions_df.iloc[idx-1]
            print(f"  Previous: {prev_tx['date'].strftime('%m-%d')} | {prev_tx['description'][:40]:<40} | {prev_tx['amount']:>8,.0f}")
        
        print(f"  Current:  {transaction['date'].strftime('%m-%d')} | {transaction['description'][:40]:<40} | {transaction['amount']:>8,.0f} ⭐")
        
        if idx < len(self.transactions_df) - 1:
            next_tx = self.transactions_df.iloc[idx+1]
            print(f"  Next:     {next_tx['date'].strftime('%m-%d')} | {next_tx['description'][:40]:<40} | {next_tx['amount']:>8,.0f}")
    
    def show_account_menu(self):
        """Display the account allocation menu."""
        print(f"\n🎯 ACCOUNT ALLOCATION MENU")
        print("-" * 50)
        
        menu_items = []
        item_num = 1
        
        for category, accounts in self.account_categories.items():
            print(f"\n📁 {category.upper()}:")
            for account in accounts:
                print(f"  {item_num:2d}. {account}")
                menu_items.append((category, account))
                item_num += 1
        
        print(f"\n🔧 ACTIONS:")
        print(f"  {item_num:2d}. Create New Account")
        menu_items.append(("ACTION", "CREATE_NEW"))
        item_num += 1
        
        print(f"  {item_num:2d}. Skip This Transaction")
        menu_items.append(("ACTION", "SKIP"))
        item_num += 1
        
        print(f"  {item_num:2d}. Save & Exit")
        menu_items.append(("ACTION", "SAVE_EXIT"))
        item_num += 1
        
        print(f"  {item_num:2d}. Exit Without Saving")
        menu_items.append(("ACTION", "EXIT"))
        item_num += 1
        
        print(f"  {item_num:2d}. Auto-Categorize Similar")
        menu_items.append(("ACTION", "AUTO_CATEGORIZE"))
        item_num += 1
        
        print(f"  {item_num:2d}. Analyze Account Allocations")  # New Analysis action
        menu_items.append(("ACTION", "ANALYZE_ACCOUNTS"))
        
        return menu_items
    
    def get_user_choice(self, menu_items, transaction):
        """Get user's account allocation choice."""
        while True:
            try:
                print(f"\n❓ Select account for this transaction (1-{len(menu_items)}):")
                choice = input("Choice: ").strip()
                
                if not choice:
                    continue
                    
                choice_num = int(choice)
                if 1 <= choice_num <= len(menu_items):
                    category, account = menu_items[choice_num - 1]
                    
                    if account == "AUTO_CATEGORIZE":
                        if len(self.changes_made) > 0:
                            last_change = self.changes_made[-1]
                            self.auto_categorize_similar(transaction, last_change['new_account'])
                        else:
                            print("❌ No recent categorization to apply")
                        return None
                    elif account == "ANALYZE_ACCOUNTS":
                        self.analyze_account_allocations()
                        return None
                    elif account == "CREATE_NEW":
                        return self.create_new_account()
                    elif account == "SKIP":
                        print("⏭️  Skipping transaction...")
                        return None
                    elif account == "SAVE_EXIT":
                        return "SAVE_EXIT"
                    elif account == "EXIT":
                        return "EXIT"
                    else:
                        # Confirm the choice
                        print(f"\n✅ Selected: {account} ({category})")
                        confirm = input("Confirm? (y/n): ").strip().lower()
                        if confirm in ['y', 'yes']:
                            return account
                        else:
                            print("Selection cancelled.")
                            continue
                else:
                    print(f"❌ Invalid choice. Please enter 1-{len(menu_items)}")
                    
            except ValueError:
                print("❌ Please enter a valid number")
            except KeyboardInterrupt:
                print("\n\n🛑 Interrupted by user")
                return "EXIT"
    
    def save_account_categories(self):
        """Save account categories to config file."""
        self.config_file.parent.mkdir(exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(self.account_categories, f, indent=4)
    
    def create_new_account(self):
        """Allow user to create a new account."""
        print(f"\n🆕 CREATE NEW ACCOUNT")
        print("-" * 30)
        
        account_name = input("Enter new account name: ").strip()
        if not account_name:
            print("❌ Account name cannot be empty")
            return None
            
        print(f"\nSelect category for '{account_name}':")
        categories = list(self.account_categories.keys())
        for i, cat in enumerate(categories, 1):
            print(f"  {i}. {cat}")
            
        try:
            cat_choice = int(input("Category (1-4): ").strip())
            if 1 <= cat_choice <= len(categories):
                category = categories[cat_choice - 1]
                
                # Add to our categories
                self.account_categories[category].append(account_name)
                
                # Save updated categories to config file
                self.save_account_categories()
                
                print(f"✅ Created new account: {account_name} ({category})")
                return account_name
            else:
                print("❌ Invalid category choice")
                return None
        except ValueError:
            print("❌ Invalid input")
            return None
    
    def update_transaction(self, idx, new_account):
        """Update a transaction's account allocation."""
        old_account = self.transactions_df.at[idx, 'Account']
        self.transactions_df.at[idx, 'Account'] = new_account
        
        # Determine account type
        account_type = "Unknown"
        for category, accounts in self.account_categories.items():
            if new_account in accounts:
                if category == 'Income':
                    account_type = 'Income'
                elif category == 'Expenses':
                    account_type = 'Expense'
                elif category == 'Assets':
                    account_type = 'Asset'
                elif category == 'Liabilities':
                    account_type = 'Liability'
                break
        
        self.transactions_df.at[idx, 'Account Type'] = account_type
        
        # Log the change
        change_record = {
            'transaction_idx': idx,
            'date': self.transactions_df.at[idx, 'date'],
            'description': self.transactions_df.at[idx, 'description'],
            'amount': self.transactions_df.at[idx, 'amount'],
            'old_account': old_account,
            'new_account': new_account,
            'timestamp': datetime.now()
        }
        self.changes_made.append(change_record)
        
        print(f"✅ Updated: {old_account} → {new_account}")
    


    def save_changes(self, output_path):
        """Save the changes to a new Excel file."""
        if not self.changes_made:
            print("ℹ️  No changes to save")
            return

        try:
            # Create backup with timestamp to prevent overwrites
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = self.file_path.parent / f"{self.file_path.stem}.backup_{timestamp}.xlsx"
            print(f"📋 Creating backup: {backup_path.name}")
            
            # Copy original file to backup
            shutil.copy2(self.file_path, backup_path)
            
            # First, read all sheets from original file
            original_sheets = {}
            with pd.ExcelFile(self.file_path) as xls:
                sheet_names = xls.sheet_names
                for sheet_name in sheet_names:
                    original_sheets[sheet_name] = pd.read_excel(xls, sheet_name=sheet_name)
            
            # Update the Detailed Transactions sheet with our changes
            original_sheets['Detailed Transactions'] = self.transactions_df
            
            # Recalculate Monthly Summary
            monthly_summary = self.transactions_df.groupby([pd.Grouper(key='date', freq='ME'), 'Account']).agg({
                'Debit': 'sum',
                'Credit': 'sum'
            }).reset_index()
            monthly_summary['Net'] = monthly_summary['Debit'] - monthly_summary['Credit']
            original_sheets['Monthly Summary'] = monthly_summary
            
            # Recalculate Trial Balance
            trial_balance = self.transactions_df.groupby('Account').agg({
                'Debit': 'sum',
                'Credit': 'sum'
            }).reset_index()
            trial_balance['Net'] = trial_balance['Debit'] - trial_balance['Credit']
            trial_balance['Account Type'] = trial_balance['Account'].map(
                self.transactions_df.groupby('Account')['Account Type'].first())
            
            # Add totals row to trial balance
            totals = pd.DataFrame({
                'Account': ['TOTAL'],
                'Debit': [trial_balance['Debit'].sum()],
                'Credit': [trial_balance['Credit'].sum()],
                'Net': [trial_balance['Net'].sum()],
                'Account Type': ['']
            })
            trial_balance = pd.concat([trial_balance, totals], ignore_index=True)
            original_sheets['Trial Balance'] = trial_balance
            
            # Generate Analysis Summary
            analysis_results = []
            detailed_sheets = {}  # Store detailed transaction data by account
            
            # Analyze each account
            for account in trial_balance['Account'].unique():
                if account == 'TOTAL':
                    continue
                    
                # Get trial balance amounts
                trial_bal_row = trial_balance[trial_balance['Account'] == account].iloc[0]
                trial_bal_debit = trial_bal_row['Debit']
                trial_bal_credit = trial_bal_row['Credit']
                trial_bal_net = trial_bal_row['Net']
                
                # Get monthly totals
                account_monthly = monthly_summary[monthly_summary['Account'] == account]
                monthly_debit = account_monthly['Debit'].sum()
                monthly_credit = account_monthly['Credit'].sum()
                monthly_net = account_monthly['Net'].sum()
                
                # Get detailed transactions
                account_transactions = self.transactions_df[
                    self.transactions_df['Account'] == account
                ]
                detailed_debit = account_transactions['Debit'].fillna(0).sum()
                detailed_credit = account_transactions['Credit'].fillna(0).sum()
                
                # Store results
                analysis_results.append({
                    'Account': account,
                    'Trial Balance Net': trial_bal_net,
                    'Trial Balance Debit': trial_bal_debit,
                    'Trial Balance Credit': trial_bal_credit,
                    'Monthly Net': monthly_net,
                    'Detailed Debit': detailed_debit,
                    'Detailed Credit': detailed_credit,
                    'Difference': trial_bal_net - (detailed_credit - detailed_debit),
                    'Reconciled': abs(trial_bal_net - (detailed_credit - detailed_debit)) < 0.01
                })
                
                # Store detailed transactions for this account
                if len(account_transactions) > 0:
                    # Use standardized account name for sheet name
                    standardized_account = self.get_standardized_account_name(account)
                    
                    # Create an abbreviated but unique sheet name
                    words = standardized_account.split()
                    if len(words) > 1:
                        # For multi-word names, use first letters of each word + first word
                        abbrev = ''.join(word[0].upper() for word in words[1:])
                        sheet_name = f"{words[0][:12]}_{abbrev}_dtl"
                    else:
                        # For single word names, use more characters
                        sheet_name = f"{standardized_account[:15]}_dtl"
                    
                    # Clean up the sheet name
                    sheet_name = sheet_name.replace('/', '_').replace(' ', '_')
                    
                    # If sheet already exists, concatenate the transactions
                    if sheet_name in detailed_sheets:
                        detailed_sheets[sheet_name] = pd.concat([
                            detailed_sheets[sheet_name],
                            account_transactions
                        ]).drop_duplicates().sort_values('date')
                    else:
                        detailed_sheets[sheet_name] = account_transactions
            
            # Add Analysis Summary sheet
            original_sheets['Analysis Summary'] = pd.DataFrame(analysis_results)
            
            # Add detailed sheets for each account
            for sheet_name, df in detailed_sheets.items():
                original_sheets[sheet_name] = df
            
            # Try with xlsxwriter first, fall back to openpyxl if needed
            try:
                engine = 'xlsxwriter'
                temp_path = self.file_path.parent / f"temp_{timestamp}.xlsx"
                with pd.ExcelWriter(temp_path, engine=engine) as writer:
                    for sheet_name, df in original_sheets.items():
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
            except ImportError:
                engine = 'openpyxl'
                # Write directly to the original file since openpyxl can handle it
                with pd.ExcelWriter(self.file_path, engine=engine) as writer:
                    for sheet_name, df in original_sheets.items():
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # If we used temp file and it exists, replace the original
            if engine == 'xlsxwriter' and temp_path.exists():
                shutil.move(str(temp_path), str(self.file_path))
            
            # Save change log to consolidated file
            log_dir = Path("reconciliation_reports")
            log_dir.mkdir(exist_ok=True)
            consolidated_log_path = log_dir / "consolidated_audit_changes.xlsx"
            
            # Convert changes to DataFrame
            changes_df = pd.DataFrame(self.changes_made)
            changes_df['audit_session'] = timestamp  # Add session identifier
            
            try:
                # If consolidated file exists, read and append
                if consolidated_log_path.exists():
                    existing_changes = pd.read_excel(consolidated_log_path)
                    combined_changes = pd.concat([existing_changes, changes_df], ignore_index=True)
                else:
                    combined_changes = changes_df
                
                # Save consolidated changes
                with pd.ExcelWriter(consolidated_log_path, engine=engine) as writer:
                    # Write main changes sheet
                    combined_changes.to_excel(writer, sheet_name='All Changes', index=False)
                    
                    # Add a summary sheet grouping by audit session
                    summary = combined_changes.groupby('audit_session').agg({
                        'transaction_idx': 'count',
                        'timestamp': 'first'
                    }).reset_index()
                    summary.columns = ['Audit Session', 'Changes Made', 'Timestamp']
                    summary.to_excel(writer, sheet_name='Session Summary', index=False)
                
                print(f"✅ Saved {len(self.changes_made)} changes to {self.file_path.name}")
                print(f"📝 Change log appended to: {consolidated_log_path.name}")
                print(f"📊 Total recorded changes: {len(combined_changes)}")
            except Exception as e:
                print(f"⚠️  Warning: Could not update consolidated log: {e}")
                # Fallback to single file if consolidated save fails
                fallback_path = log_dir / f"audit_changes_{timestamp}.xlsx"
                changes_df.to_excel(fallback_path, index=False, engine=engine)
                print(f"📝 Changes saved to individual log: {fallback_path.name}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving changes: {e}")
            if 'temp_path' in locals() and temp_path.exists():
                temp_path.unlink()  # Clean up temp file if it exists
            return False
    
    def get_standardized_account_name(self, account_name):
        """Standardize account names to handle variations."""
        # Define mappings for similar account names
        account_mappings = {
            'Salaries': 'Salaries and Wages',
            'Salaries and Wages': 'Salaries and Wages',
            'Stationery': 'Stationery and Printing',
            'Stationery and Printing': 'Stationery and Printing',
            'Consulting and Professional Fees': 'Professional Fees',
            # Add more mappings as needed
        }
        
        return account_mappings.get(account_name, account_name)
    
    def analyze_account_allocations(self):
        """Analyze transactions in each account for potential misallocations."""
        print("\n📊 ACCOUNT ALLOCATION ANALYSIS")
        print("="*80)
        
        # Get transactions with standardized account names
        df = self.transactions_df.copy()
        df['standardized_account'] = df['Account'].apply(self.get_standardized_account_name)
        
        # Group accounts by type
        account_types = {}
        for category, accounts in self.account_categories.items():
            for account in accounts:
                standardized_account = self.get_standardized_account_name(account)
                account_types[standardized_account] = category
                
        # Show standardization info
        original_accounts = df['Account'].unique()
        standardized_accounts = df['standardized_account'].unique()
        if len(original_accounts) != len(standardized_accounts):
            print("\n🔄 Account Standardization:")
            for orig in original_accounts:
                std = self.get_standardized_account_name(orig)
                if std != orig:
                    print(f"  • {orig} → {std}")
        
        # Organize accounts by type
        accounts_by_type = {
            'Income': [],
            'Expenses': [],
            'Assets': [],
            'Liabilities': []
        }
        
        for account in sorted(df['standardized_account'].unique()):
            if account != 'Uncategorized':
                account_type = account_types.get(account)
                if account_type:
                    accounts_by_type[account_type].append(account)
                    
        # Show transaction counts by standardized account
        print("\n📊 Transaction Counts by Account:")
        for account_type, accounts in accounts_by_type.items():
            if accounts:
                print(f"\n{account_type}:")
                for account in accounts:
                    count = len(df[df['standardized_account'] == account])
                    total = df[df['standardized_account'] == account]['amount'].sum()
                    print(f"  • {account}: {count} transactions, Total: {total:,.2f}")
        
        current_type_idx = 0
        current_account_idx = 0
        current_transaction_idx = 0
        
        types_list = list(accounts_by_type.keys())
        
        while True:
            current_type = types_list[current_type_idx]
            accounts = accounts_by_type[current_type]
            
            if not accounts:
                current_type_idx = (current_type_idx + 1) % len(types_list)
                continue
                
            current_account = accounts[current_account_idx]
            
            # Get transactions for current standardized account
            account_mask = (df['standardized_account'].str.strip() == current_account.strip())
            account_transactions = df[account_mask].copy()
            
            # Add original account name for reference
            account_transactions['original_account'] = account_transactions['Account']
            account_transactions['Account'] = account_transactions['standardized_account']
            
            # Reset index to ensure proper navigation
            account_transactions = account_transactions.reset_index(drop=False)
            
            if len(account_transactions) == 0:
                print(f"\nNo transactions found for {current_account}")
                current_account_idx = (current_account_idx + 1) % len(accounts)
                continue
            
            # Show account summary
            original_accounts = account_transactions['original_account'].unique()
            if len(original_accounts) > 1:
                print(f"\n🔄 This standardized account contains transactions from:")
                for orig in original_accounts:
                    count = len(account_transactions[account_transactions['original_account'] == orig])
                    print(f"  • {orig}: {count} transactions")
            
            # Clear screen with newlines
            print("\n" * 2)
            print(f"📊 ACCOUNT TYPE: {current_type}")
            print(f"📁 ANALYZING: {current_account}")
            print(f"📈 Progress: Account {current_account_idx + 1} of {len(accounts)}")
            print("-" * 50)
            
            # Get transaction
            transaction = account_transactions.iloc[current_transaction_idx]
            
            # Display transaction
            # Show transaction details with better formatting
            total_transactions = len(account_transactions)
            print(f"\n📊 Transaction Details ({current_transaction_idx + 1} of {total_transactions}):")
            print("-" * 40)
            print(f"📅 Date:        {pd.to_datetime(transaction['date']).strftime('%Y-%m-%d')}")
            print(f"📝 Description: {transaction['description']}")
            print(f"💰 Amount:      {transaction['amount']:>10,.2f}")
            print(f"💸 Debit:       {transaction['Debit']:>10,.2f}")
            print(f"💵 Credit:      {transaction['Credit']:>10,.2f}")
            print(f"🏷️  Account:     {transaction['standardized_account']}")
            if transaction['original_account'] != transaction['standardized_account']:
                print(f"   Original:    {transaction['original_account']}")
            
            # Show context (previous/next transactions if available)
            if total_transactions > 1:
                print("\n📋 Context:")
                if current_transaction_idx > 0:
                    prev_tx = account_transactions.iloc[current_transaction_idx - 1]
                    print(f"  ⬆️  Previous: {pd.to_datetime(prev_tx['date']).strftime('%m-%d')} | {prev_tx['description'][:40]:<40} | {prev_tx['amount']:>8,.2f}")
                print(f"  ⭐ Current:  {pd.to_datetime(transaction['date']).strftime('%m-%d')} | {transaction['description'][:40]:<40} | {transaction['amount']:>8,.2f}")
                if current_transaction_idx < total_transactions - 1:
                    next_tx = account_transactions.iloc[current_transaction_idx + 1]
                    print(f"  ⬇️  Next:     {pd.to_datetime(next_tx['date']).strftime('%m-%d')} | {next_tx['description'][:40]:<40} | {next_tx['amount']:>8,.2f}")
            
            # Show navigation options
            print("\nNAVIGATION:")
            print("p - Previous transaction    n - Next transaction")
            print("b - Previous account        f - Next account")
            print("[ - Previous account type   ] - Next account type")
            print("r - Reallocate transaction  s - Save and exit")
            print("q - Exit without saving")
            
            choice = input("\nChoice: ").strip().lower()
            
            if choice == 'p':  # Previous transaction
                if current_transaction_idx > 0:
                    current_transaction_idx -= 1
                else:
                    print("\n⚠️ Already at first transaction")
            elif choice == 'n':  # Next transaction
                if current_transaction_idx < len(account_transactions) - 1:
                    current_transaction_idx += 1
                else:
                    print("\n⚠️ Already at last transaction")
            elif choice == 'b':  # Previous account
                current_account_idx = (current_account_idx - 1) % len(accounts)
                current_transaction_idx = 0
            elif choice == 'f':  # Next account
                current_account_idx = (current_account_idx + 1) % len(accounts)
                current_transaction_idx = 0
            elif choice == '[':  # Previous account type
                current_type_idx = (current_type_idx - 1) % len(types_list)
                current_account_idx = 0
                current_transaction_idx = 0
            elif choice == ']':  # Next account type
                current_type_idx = (current_type_idx + 1) % len(types_list)
                current_account_idx = 0
                current_transaction_idx = 0
            elif choice == 'r':  # Reallocate
                print("\nSelect new account for this transaction:")
                menu_items = self.show_account_menu()
                new_account = self.get_user_choice(menu_items, transaction)
                
                if new_account and new_account not in ["EXIT", "SAVE_EXIT", "SKIP"]:
                    self.update_transaction(transaction.name, new_account)
                    print(f"✅ Updated transaction categorization")
            elif choice == 's':  # Save and exit
                if self.changes_made:
                    save = input("Save changes before exiting? (y/n): ").strip().lower()
                    if save in ['y', 'yes']:
                        self.save_changes(self.file_path)
                return
            elif choice == 'q':  # Quit without saving
                if self.changes_made:
                    confirm = input("You have unsaved changes. Really quit? (y/n): ").strip().lower()
                    if confirm in ['y', 'yes']:
                        return
                else:
                    return
            else:
                print("Invalid choice. Please try again.")
            
        if self.changes_made:
            save = input("\nSave changes made during analysis? (y/n): ").strip().lower()
            if save in ['y', 'yes']:
                self.save_changes(self.file_path)
            else:
                print("Changes discarded")
    
    def auto_categorize_similar(self, transaction, new_account):
        """Auto-categorize similar transactions based on description patterns."""
        description = transaction['description']
        original_desc = description
        
        # Create search patterns
        patterns = [
            description,  # Exact match
            ' '.join(description.split()[:-1]),  # All but last word
            ' '.join(description.split()[:3]),   # First three words
            description.split('To ')[-1] if 'To ' in description else None,  # After "To"
            description.split('From ')[-1] if 'From ' in description else None,  # After "From"
        ]
        patterns = [p for p in patterns if p]  # Remove None values
        
        # Find similar uncategorized transactions
        similar_found = 0
        for pattern in patterns:
            mask = (
                (self.transactions_df['Account'] == 'Uncategorized') &
                (self.transactions_df['description'].str.contains(pattern, case=False, na=False))
            )
            similar_txns = self.transactions_df[mask]
            
            if not similar_txns.empty:
                print(f"\n🔍 Found {len(similar_txns)} similar transactions matching: '{pattern}'")
                confirm = input("Auto-categorize these? (y/n): ").strip().lower()
                
                if confirm in ['y', 'yes']:
                    for idx, similar_txn in similar_txns.iterrows():
                        self.update_transaction(idx, new_account)
                        similar_found += 1
        
        if similar_found:
            print(f"\n✨ Auto-categorized {similar_found} similar transactions as {new_account}")
        else:
            print("\n❌ No similar uncategorized transactions found")
    
    def review_transactions(self, transactions_to_review):
        """Review and categorize a set of transactions."""
        for i, (idx, transaction) in enumerate(transactions_to_review.iterrows()):
            self.display_transaction(i, transaction)
            menu_items = self.show_account_menu()
            
            choice = self.get_user_choice(menu_items, transaction)
            
            if choice == "EXIT":
                print("🛑 Exiting without saving...")
                break
            elif choice == "SAVE_EXIT":
                self.save_changes(self.file_path)
                print("💾 Saved and exiting...")
                break
            elif choice and choice not in ["SKIP"]:
                self.update_transaction(idx, choice)
                self.auto_categorize_similar(transaction, choice)
    
    def run_interactive_audit(self):
        """Run the main interactive audit process."""
        if not self.load_data():
            return
            
        print(f"\n🚀 Starting Interactive Audit")
        
        # Initialize needs_attention as None
        needs_attention = None
        
        # Show main menu for actions
        while True:
            print("\n" + "="*50)
            print("MAIN AUDIT MENU")
            print("="*50)
            print("1. Review Uncategorized Transactions")
            print("2. Analyze Account Allocations")
            print("3. Save Changes")
            print("4. Exit")
            
            try:
                choice = input("\nSelect action (1-4): ").strip()
                
                if choice == '1':
                    # Filter transactions that need attention
                    needs_attention = self.transactions_df[
                        (self.transactions_df['Account'] == 'Uncategorized') |
                        (self.transactions_df['Account Type'] == 'Unknown')
                    ].copy()
                    
                    if len(needs_attention) == 0:
                        print("\n✅ All transactions are properly categorized!")
                        continue
                        
                    print(f"\n📊 Found {len(needs_attention)} transactions needing attention")
                    self.review_transactions(needs_attention)
                    
                elif choice == '2':
                    self.analyze_account_allocations()
                    
                elif choice == '3':
                    if self.changes_made:
                        self.save_changes(self.file_path)
                        print("\n✅ Changes saved successfully!")
                    else:
                        print("\nℹ️  No changes to save")
                        
                elif choice == '4':
                    if self.changes_made:
                        save = input("\nSave changes before exiting? (y/n): ").strip().lower()
                        if save in ['y', 'yes']:
                            self.save_changes(self.file_path)
                    print("\n👋 Goodbye!")
                    break
                    
                else:
                    print("\n❌ Invalid choice. Please enter 1-4.")
                    
            except KeyboardInterrupt:
                print("\n\n🛑 Interrupted by user")
                if self.changes_made:
                    save = input("\nSave changes before exiting? (y/n): ").strip().lower()
                    if save in ['y', 'yes']:
                        self.save_changes(self.file_path)
                break
        
        # Only process needs_attention if it exists and has rows
        if needs_attention is not None and not needs_attention.empty:
            for i, (idx, transaction) in enumerate(needs_attention.iterrows()):
                self.display_transaction(i, transaction)
                menu_items = self.show_account_menu()
                
                choice = self.get_user_choice(menu_items, transaction)
                
                if choice == "EXIT":
                    print("🛑 Exiting without saving...")
                    break
                elif choice == "SAVE_EXIT":
                    self.save_changes(self.file_path)  # Add file_path parameter
                    print("💾 Saved and exiting...")
                    break
                elif choice and choice not in ["SKIP"]:
                    self.update_transaction(idx, choice)
                    self.auto_categorize_similar(transaction, choice)  # Auto-categorize similar transactions
        
        # Final summary
        print(f"\n" + "="*80)
        print(f"AUDIT SESSION SUMMARY")
        print(f"="*80)
        print(f"Transactions reviewed: {len(needs_attention) if needs_attention is not None else 0}")
        print(f"Changes made: {len(self.changes_made)}")
        
        if self.changes_made:
            print(f"\nLast 5 changes made:")
            for change in self.changes_made[-5:]:
                print(f"  • {change['date'].strftime('%Y-%m-%d')} - {change['description'][:50]}")
                print(f"    {change['old_account']} → {change['new_account']}")
            
            save_choice = input(f"\nSave changes? (y/n): ").strip().lower()
            if save_choice in ['y', 'yes']:
                self.save_changes(self.file_path)
                print("💾 Changes saved successfully!")
            else:
                print("❌ Changes discarded")

def main():
    """Main function."""
    audit = Interactive2024Audit()
    audit.run_interactive_audit()

if __name__ == "__main__":
    main()
