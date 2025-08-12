#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Lattice Extraction Module

Specialized module for extracting transaction data from PDFs using tabula's lattice mode.
This module focuses specifically on handling the unique challenges of lattice extraction
such as split fields across adjacent columns and different column detection patterns.
"""

import os
import re
import json
import logging
import pandas as pd
import tabula
import PyPDF2
from typing import List, Dict, Tuple, Optional, Any, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class LatticeExtractor:
    """
    Specialized class for extracting and processing transaction data from PDFs using lattice mode.
    """
    
    def __init__(self, debug: bool = False):
        """
        Initialize the LatticeExtractor
        
        Args:
            debug (bool): Enable debug mode with extra logging and information
        """
        self.debug = debug
        self.debug_info = {
            'pdf_path': None,
            'total_pages': 0,
            'tables_extracted': 0,
            'transaction_tables': 0,
            'total_transactions': 0,
            'page_errors': [],
            'table_results': []
        }
    
    def extract_tables_from_pdf(self, pdf_path: str, password: Optional[str] = None) -> List[pd.DataFrame]:
        """
        Extract tables from PDF using lattice mode only
        
        Args:
            pdf_path (str): Path to the PDF file
            password (str, optional): Password for encrypted PDF
            
        Returns:
            list: List of pandas DataFrames containing tables
        """
        tables = []
        self.debug_info['pdf_path'] = pdf_path
        
        try:
            # Get total pages
            with open(pdf_path, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                total_pages = len(pdf_reader.pages)
                logger.info(f"PDF has {total_pages} pages")
                self.debug_info['total_pages'] = total_pages
            
            # Extract with lattice mode
            try:
                lattice_tables = tabula.read_pdf(
                    pdf_path,
                    pages='all',
                    password=password,
                    multiple_tables=True,
                    lattice=True,
                    pandas_options={'header': None}
                )
                logger.info(f"Lattice mode extracted {len(lattice_tables)} tables")
                
                # Add metadata to each table
                for table in lattice_tables:
                    # Add extraction mode as a DataFrame attribute
                    table._extraction_mode = 'lattice'
                    
                tables.extend(lattice_tables)
                self.debug_info['tables_extracted'] = len(tables)
                
            except Exception as e:
                error_msg = f"Error in lattice mode extraction: {e}"
                logger.error(error_msg)
                self.debug_info['page_errors'].append(error_msg)
                
                # Try page by page as fallback
                for page in range(1, total_pages + 1):
                    try:
                        page_tables = tabula.read_pdf(
                            pdf_path,
                            pages=str(page),
                            password=password,
                            multiple_tables=True,
                            lattice=True,
                            pandas_options={'header': None}
                        )
                        logger.info(f"Lattice mode page {page} extracted {len(page_tables)} tables")
                        
                        # Add metadata to each table
                        for table in page_tables:
                            table._extraction_mode = 'lattice'
                            table._page_number = page
                        
                        tables.extend(page_tables)
                    except Exception as page_error:
                        error_msg = f"Error extracting page {page} with lattice mode: {page_error}"
                        logger.error(error_msg)
                        self.debug_info['page_errors'].append(error_msg)
            
            logger.info(f"Total lattice tables extracted: {len(tables)}")
            return tables
            
        except Exception as e:
            error_msg = f"Error extracting tables: {e}"
            logger.error(error_msg)
            self.debug_info['page_errors'].append(error_msg)
            return []
    
    def is_transaction_table(self, df: pd.DataFrame) -> bool:
        """
        Identify if a dataframe contains transaction data based on content patterns
        Specialized for lattice mode tables
        
        Args:
            df (DataFrame): Pandas DataFrame to analyze
            
        Returns:
            bool: True if table contains transaction data
        """
        table_debug = {
            'has_date': False,
            'has_amount': False,
            'has_details': False,
            'date_columns': [],
            'amount_columns': [],
            'details_columns': [],
            'row_count': len(df) if df is not None else 0,
            'column_count': len(df.columns) if df is not None else 0,
            'potential_transaction_rows': 0
        }
        
        if df is None or df.empty or len(df) < 2:  # Need at least header + one row
            return False
        
        try:
            # Convert all values to string for pattern matching
            df_str = df.astype(str)
            
            # Look for column headers or content patterns
            has_date = False
            has_amount = False
            has_details = False
            potential_transaction_rows = 0
            
            # Define more inclusive patterns for transaction data
            date_pattern = r'\d{1,2}[/\-\.]\d{1,2}[/\-\.]?(\d{2}|\d{4})?|\d{4}-\d{2}-\d{2}'
            amount_pattern = r'\d+[\.,]\d{2}|-\d+[\.,]\d{2}|\d+[\.,]\d{3}[\.,]\d{2}'
            details_pattern = r'payment|transfer|debit|credit|withdrawal|deposit|fee|charge|interest|card|purchase|salary|loan|balance|opening|closing'
            
            # Check each column for patterns
            for col_idx in range(len(df.columns)):
                col_values = df_str.iloc[:, col_idx].str.lower()
                
                # Check for date patterns
                if col_values.str.contains(date_pattern, regex=True, na=False).any():
                    has_date = True
                    table_debug['date_columns'].append(col_idx)
                    logger.info(f"Found date pattern in column {col_idx}")
                    
                # Check for amount patterns
                if col_values.str.contains(amount_pattern, regex=True, na=False).any():
                    has_amount = True
                    table_debug['amount_columns'].append(col_idx)
                    logger.info(f"Found amount pattern in column {col_idx}")
                    
                # Check for transaction details
                # Look for columns with text content that's likely transaction details
                if col_values.str.contains(details_pattern, case=False, regex=True, na=False).any() or \
                   (col_values.str.len().mean() > 5 and col_values.str.contains('[a-z]', regex=True, na=False).mean() > 0.3):
                    has_details = True
                    table_debug['details_columns'].append(col_idx)
                    logger.info(f"Found transaction details in column {col_idx}")
            
            # For lattice mode, also check combined adjacent columns
            for col_idx in range(len(df.columns) - 1):
                # Combine values from adjacent columns
                combined_values = df_str.iloc[:, col_idx] + ' ' + df_str.iloc[:, col_idx + 1]
                combined_values = combined_values.str.lower()
                
                # Check for date patterns in combined columns
                if not has_date and combined_values.str.contains(date_pattern, regex=True, na=False).any():
                    has_date = True
                    table_debug['date_columns'].extend([col_idx, col_idx + 1])
                    logger.info(f"Found date pattern in combined columns {col_idx} and {col_idx + 1}")
                
                # Check for amount patterns in combined columns
                if not has_amount and combined_values.str.contains(amount_pattern, regex=True, na=False).any():
                    has_amount = True
                    table_debug['amount_columns'].extend([col_idx, col_idx + 1])
                    logger.info(f"Found amount pattern in combined columns {col_idx} and {col_idx + 1}")
                
                # Check for transaction details in combined columns
                if not has_details and combined_values.str.contains(details_pattern, case=False, regex=True, na=False).any():
                    has_details = True
                    table_debug['details_columns'].extend([col_idx, col_idx + 1])
                    logger.info(f"Found transaction details in combined columns {col_idx} and {col_idx + 1}")
            
            # Count potential transaction rows (rows with dates or amounts)
            for row_idx in range(len(df)):
                row_values = df_str.iloc[row_idx, :].astype(str)
                has_row_date = row_values.str.contains(date_pattern, regex=True, na=False).any()
                has_row_amount = row_values.str.contains(amount_pattern, regex=True, na=False).any()
                has_row_details = row_values.str.contains(details_pattern, case=False, regex=True, na=False).any()
                
                # If row has date or amount, it's potentially a transaction row
                if has_row_date or has_row_amount or has_row_details:
                    potential_transaction_rows += 1
            
            table_debug['potential_transaction_rows'] = potential_transaction_rows
            
            # For lattice mode, be more lenient in identifying transaction tables
            # If we have at least two of the three features and at least one potential transaction row,
            # it's likely a transaction table
            features_count = sum([has_date, has_amount, has_details])
            is_tx_table = (features_count >= 1 and potential_transaction_rows >= 2) or \
                          (features_count >= 2 and potential_transaction_rows >= 1)
            
            table_debug['has_date'] = has_date
            table_debug['has_amount'] = has_amount
            table_debug['has_details'] = has_details
            
            logger.info(f"Lattice table analysis - has_date: {has_date}, has_amount: {has_amount}, "
                       f"has_details: {has_details}, potential_rows: {potential_transaction_rows}, "
                       f"is_transaction_table: {is_tx_table}")
            
            return is_tx_table
        
        except Exception as e:
            error_msg = f"Error in is_transaction_table: {e}"
            logger.error(error_msg)
            return False
    
    def identify_columns(self, df: pd.DataFrame) -> Tuple[Dict[str, int], Optional[int]]:
        """
        Identify which columns correspond to our target fields
        Specialized for lattice mode tables
        
        Args:
            df (DataFrame): Pandas DataFrame to analyze
            
        Returns:
            tuple: (column_mapping, header_row)
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
        
        try:
            # Convert all values to string for pattern matching
            df_str = df.astype(str)
            
            # First, try to identify header row more aggressively
            header_terms = ['date', 'description', 'details', 'debit', 'credit', 'balance', 
                           'transaction', 'particulars', 'amount', 'reference', 'withdrawal', 'deposit']
            
            for i in range(min(10, len(df))):  # Check first 10 rows
                row_values = df_str.iloc[i].str.lower()
                
                # Check if this row contains common header terms
                header_match_count = 0
                for term in header_terms:
                    if row_values.str.contains(term, case=False, regex=False, na=False).any():
                        header_match_count += 1
                
                if header_match_count >= 2:
                    header_row = i
                    logger.info(f"Found header row at index {i} with {header_match_count} header terms")
                    break
            
            # Define patterns for better identification
            date_pattern = r'\d{1,2}[/\-\.]\d{1,2}[/\-\.]?(\d{2}|\d{4})?|\d{4}-\d{2}-\d{2}'
            amount_pattern = r'\d+[\.,]\d{2}|-\d+[\.,]\d{2}|\d+[\.,]\d{3}[\.,]\d{2}'
            details_pattern = r'payment|transfer|debit|credit|withdrawal|deposit|fee|charge|interest|purchase|salary|atm|card|cheque|check|direct|online|cash|loan|mortgage|insurance'
            
            # Track columns with their scores for different field types
            column_scores = {col_idx: {'details': 0, 'date': 0, 'debit': 0, 'credit': 0, 'balance': 0, 'service_fee': 0} 
                            for col_idx in range(len(df.columns))}
            
            # Analyze column headers if header row was found
            if header_row is not None:
                header_values = df_str.iloc[header_row].str.lower()
                
                for col_idx, header in enumerate(header_values):
                    if pd.isna(header) or not header:
                        continue
                        
                    header = str(header).lower()
                    
                    # Score based on header text
                    if any(term in header for term in ['detail', 'description', 'particular', 'narration', 'transaction']):
                        column_scores[col_idx]['details'] += 10
                    
                    if any(term in header for term in ['date', 'day']):
                        column_scores[col_idx]['date'] += 10
                    
                    if any(term in header for term in ['debit', 'withdrawal', 'payment', 'out']):
                        column_scores[col_idx]['debit'] += 10
                    
                    if any(term in header for term in ['credit', 'deposit', 'in']):
                        column_scores[col_idx]['credit'] += 10
                    
                    if any(term in header for term in ['balance', 'closing', 'opening']):
                        column_scores[col_idx]['balance'] += 10
                    
                    if any(term in header for term in ['fee', 'charge', 'service']):
                        column_scores[col_idx]['service_fee'] += 10
            
            # Analyze column content
            # Skip header row if identified
            start_row = header_row + 1 if header_row is not None else 0
            
            # Check each column for patterns
            for col_idx in range(len(df.columns)):
                # Get column values, skipping header
                col_values = df_str.iloc[start_row:, col_idx].str.lower()
                
                # Skip empty columns
                if col_values.isna().all() or (col_values == '').all() or (col_values == 'nan').all():
                    continue
                
                # Check for date patterns
                date_match_count = col_values.str.contains(date_pattern, regex=True, na=False).sum()
                if date_match_count > 0:
                    # Score based on percentage of rows that match date pattern
                    # Ensure minimum score of 5 if any matches are found
                    date_score = max(5, min(10, int(date_match_count / len(col_values) * 10)))
                    column_scores[col_idx]['date'] += date_score
                    logger.info(f"Found date pattern in column {col_idx} (score: {date_score})")
                
                # Check for amount patterns
                amount_match_count = col_values.str.contains(amount_pattern, regex=True, na=False).sum()
                if amount_match_count > 0:
                    # Determine if it's likely debit, credit, balance or fee
                    # Ensure minimum score of 5 if any matches are found
                    amount_score = max(5, min(10, int(amount_match_count / len(col_values) * 10)))
                    
                    # Look for negative values (often debits)
                    negative_count = col_values.str.contains(r'-\d+[\.,]\d{2}', regex=True, na=False).sum()
                    
                    # Check for specific keywords in the column or adjacent columns
                    if col_values.str.contains('balance|closing|opening', case=False, regex=True, na=False).any():
                        column_scores[col_idx]['balance'] += amount_score
                        logger.info(f"Found balance pattern in column {col_idx} (score: {amount_score})")
                    elif col_values.str.contains('fee|charge|service', case=False, regex=True, na=False).any():
                        column_scores[col_idx]['service_fee'] += amount_score
                        logger.info(f"Found service fee pattern in column {col_idx} (score: {amount_score})")
                    elif col_values.str.contains('debit|payment|withdrawal', case=False, regex=True, na=False).any() or negative_count > 0:
                        column_scores[col_idx]['debit'] += amount_score
                        logger.info(f"Found debit pattern in column {col_idx} (score: {amount_score})")
                    elif col_values.str.contains('credit|deposit|interest', case=False, regex=True, na=False).any():
                        column_scores[col_idx]['credit'] += amount_score
                        logger.info(f"Found credit pattern in column {col_idx} (score: {amount_score})")
                    else:
                        # If no specific indicator, add to both potential debit and credit with lower score
                        # But still ensure a minimum score
                        min_score = max(3, amount_score // 2)
                        column_scores[col_idx]['debit'] += min_score
                        column_scores[col_idx]['credit'] += min_score
                        logger.info(f"Found amount pattern in column {col_idx} (score: {min_score} for both debit/credit)")
                
                # Check for transaction details
                details_match_count = col_values.str.contains(details_pattern, case=False, regex=True, na=False).sum()
                if details_match_count > 0:
                    # Ensure minimum score of 5 if any matches are found
                    details_score = max(5, min(10, int(details_match_count / len(col_values) * 10)))
                    column_scores[col_idx]['details'] += details_score
                    logger.info(f"Found transaction details in column {col_idx} (score: {details_score})")
                
                # Additional heuristic: leftmost columns are more likely to be details
                if col_idx == 0 or col_idx == 1:
                    column_scores[col_idx]['details'] += 3
                    logger.info(f"Added position bonus for details to column {col_idx}")
                
                # Additional heuristic: rightmost columns are more likely to be balance
                if col_idx == len(df.columns) - 1 or col_idx == len(df.columns) - 2:
                    column_scores[col_idx]['balance'] += 3
                    logger.info(f"Added position bonus for balance to column {col_idx}")
                
                # If column has any numeric values, give it a minimum score for amount fields
                if col_values.str.contains(r'\d+', regex=True, na=False).any():
                    for field in ['debit', 'credit', 'balance']:
                        if column_scores[col_idx][field] == 0:
                            column_scores[col_idx][field] = 1
                            logger.info(f"Added minimum numeric score for {field} to column {col_idx}")
            
            # Now check combined adjacent columns for lattice mode
            adjacent_columns_candidates = {'date': [], 'amount': [], 'details': []}
            
            for col_idx in range(len(df.columns) - 1):
                # Skip if either column is empty
                if (df_str.iloc[start_row:, col_idx].isna().all() or 
                    df_str.iloc[start_row:, col_idx + 1].isna().all()):
                    continue
                    
                # Combine adjacent columns
                combined_values = df_str.iloc[start_row:, col_idx].str.cat(
                    df_str.iloc[start_row:, col_idx + 1], sep=' ', na_rep='')
                
                # Check for date patterns in combined columns
                date_match_count = combined_values.str.contains(date_pattern, regex=True, na=False).sum()
                col1_date_count = df_str.iloc[start_row:, col_idx].str.contains(date_pattern, regex=True, na=False).sum()
                col2_date_count = df_str.iloc[start_row:, col_idx + 1].str.contains(date_pattern, regex=True, na=False).sum()
                
                # Consider adjacent columns if combined has more matches OR if both columns have some matches
                if (date_match_count > 0 and date_match_count > max(col1_date_count, col2_date_count)) or \
                   (col1_date_count > 0 and col2_date_count > 0):
                    # This indicates a date might be split across columns
                    adjacent_columns_candidates['date'].append((col_idx, col_idx + 1))
                    logger.info(f"Found potential split date across columns {col_idx} and {col_idx + 1}")
                
                # Check for amount patterns in combined columns
                amount_match_count = combined_values.str.contains(amount_pattern, regex=True, na=False).sum()
                col1_amount_count = df_str.iloc[start_row:, col_idx].str.contains(amount_pattern, regex=True, na=False).sum()
                col2_amount_count = df_str.iloc[start_row:, col_idx + 1].str.contains(amount_pattern, regex=True, na=False).sum()
                
                # Consider adjacent columns if combined has more matches OR if both columns have some matches
                if (amount_match_count > 0 and amount_match_count > max(col1_amount_count, col2_amount_count)) or \
                   (col1_amount_count > 0 and col2_amount_count > 0):
                    # This indicates an amount might be split across columns
                    adjacent_columns_candidates['amount'].append((col_idx, col_idx + 1))
                    logger.info(f"Found potential split amount across columns {col_idx} and {col_idx + 1}")
                
                # Check for details patterns in combined columns
                details_match_count = combined_values.str.contains(details_pattern, regex=True, na=False).sum()
                col1_details_count = df_str.iloc[start_row:, col_idx].str.contains(details_pattern, regex=True, na=False).sum()
                col2_details_count = df_str.iloc[start_row:, col_idx + 1].str.contains(details_pattern, regex=True, na=False).sum()
                
                # Consider adjacent columns if combined has more matches OR if both columns have some matches
                # For details, we're more aggressive in combining columns
                if (details_match_count > 0) or (col1_details_count > 0 and col2_details_count > 0):
                    # This indicates details might be split across columns
                    adjacent_columns_candidates['details'].append((col_idx, col_idx + 1))
                    logger.info(f"Found potential split details across columns {col_idx} and {col_idx + 1}")
                    
                # Also check if one column has text and the adjacent has numbers - common pattern for details+amount
                has_text1 = df_str.iloc[start_row:, col_idx].str.contains(r'[a-zA-Z]{3,}', regex=True, na=False).any()
                has_text2 = df_str.iloc[start_row:, col_idx + 1].str.contains(r'[a-zA-Z]{3,}', regex=True, na=False).any()
                has_num1 = df_str.iloc[start_row:, col_idx].str.contains(r'\d+[\.,]\d{2}', regex=True, na=False).any()
                has_num2 = df_str.iloc[start_row:, col_idx + 1].str.contains(r'\d+[\.,]\d{2}', regex=True, na=False).any()
                
                if (has_text1 and has_num2) or (has_text2 and has_num1):
                    # This indicates a potential details+amount pair
                    if (has_text1 and has_num2) and (col_idx, col_idx + 1) not in adjacent_columns_candidates['details']:
                        adjacent_columns_candidates['details'].append((col_idx, col_idx + 1))
                        logger.info(f"Found potential text+number pair across columns {col_idx} and {col_idx + 1}")
                    elif (has_text2 and has_num1) and (col_idx, col_idx + 1) not in adjacent_columns_candidates['details']:
                        adjacent_columns_candidates['details'].append((col_idx, col_idx + 1))
                        logger.info(f"Found potential number+text pair across columns {col_idx} and {col_idx + 1}")
            
            # Process adjacent column candidates to boost scores
            for field_type, candidates in adjacent_columns_candidates.items():
                for col_pair in candidates:
                    col_idx, adj_idx = col_pair
                    
                    # Boost scores for both columns in the pair
                    if field_type == 'date':
                        column_scores[col_idx]['date'] += 3
                        column_scores[adj_idx]['date'] += 3
                        logger.info(f"Boosted date scores for columns {col_idx} and {adj_idx}")
                    elif field_type == 'amount':
                        # For amount fields, boost all amount-related scores
                        for amount_field in ['debit', 'credit', 'balance', 'service_fee']:
                            column_scores[col_idx][amount_field] += 2
                            column_scores[adj_idx][amount_field] += 2
                        logger.info(f"Boosted amount scores for columns {col_idx} and {adj_idx}")
                    elif field_type == 'details':
                        column_scores[col_idx]['details'] += 3
                        column_scores[adj_idx]['details'] += 3
                        logger.info(f"Boosted details scores for columns {col_idx} and {adj_idx}")
            
            # Assign columns based on scores
            field_assignments = {}
            
            # First, find the best column for each field
            for field in ['details', 'date', 'debit', 'credit', 'balance', 'service_fee']:
                best_col = None
                best_score = 0
                
                for col_idx, scores in column_scores.items():
                    if scores[field] > best_score:
                        best_score = scores[field]
                        best_col = col_idx
                
                # Use a lower minimum threshold for assignment to be more inclusive
                # This helps capture more potential transaction columns in lattice mode
                min_threshold = 1  # Accept any score above 1 for all fields
                if best_score >= min_threshold:
                    field_assignments[field] = (best_col, best_score)
                    logger.info(f"Assigned {field} to column {best_col} with score {best_score}")
            
            # Map field assignments to column_mapping
            if 'details' in field_assignments:
                column_mapping['Details'] = field_assignments['details'][0]
            
            if 'date' in field_assignments:
                column_mapping['Date'] = field_assignments['date'][0]
            
            if 'debit' in field_assignments:
                column_mapping['Debits'] = field_assignments['debit'][0]
            
            if 'credit' in field_assignments:
                # Make sure it's not the same as Debits
                if 'Debits' not in column_mapping or column_mapping['Debits'] != field_assignments['credit'][0]:
                    column_mapping['Credits'] = field_assignments['credit'][0]
            
            if 'balance' in field_assignments:
                column_mapping['Balance'] = field_assignments['balance'][0]
            
            if 'service_fee' in field_assignments:
                # Make sure it's not already assigned to another field
                if not any(field_assignments['service_fee'][0] == column_mapping.get(field) 
                          for field in ['Debits', 'Credits', 'Balance'] if field in column_mapping):
                    column_mapping['ServiceFee'] = field_assignments['service_fee'][0]
            
            # Add adjacent column information for lattice mode
            column_mapping['_lattice_mode'] = True
            column_mapping['_adjacent_columns'] = {}
            
            # Use the adjacent column candidates to populate the adjacent columns mapping
            for field, col_idx in column_mapping.items():
                if field.startswith('_') or col_idx is None:
                    continue
                
                field_lower = field.lower()
                if field_lower == 'details':
                    # Check if this column is part of a details pair
                    for col_pair in adjacent_columns_candidates['details']:
                        if col_idx in col_pair:
                            # Get the other column in the pair
                            adj_idx = col_pair[0] if col_pair[1] == col_idx else col_pair[1]
                            
                            # Skip if this adjacent column is already mapped to a different field
                            if not any(adj_idx == mapped_idx for mapped_field, mapped_idx in column_mapping.items() 
                                    if not mapped_field.startswith('_') and mapped_field != field and mapped_idx is not None):
                                column_mapping['_adjacent_columns'][field] = adj_idx
                                logger.info(f"Added adjacent column {adj_idx} for {field} from candidates")
                                break
                
                elif field_lower == 'date':
                    # Check if this column is part of a date pair
                    for col_pair in adjacent_columns_candidates['date']:
                        if col_idx in col_pair:
                            # Get the other column in the pair
                            adj_idx = col_pair[0] if col_pair[1] == col_idx else col_pair[1]
                            
                            # Skip if this adjacent column is already mapped to a different field
                            if not any(adj_idx == mapped_idx for mapped_field, mapped_idx in column_mapping.items() 
                                    if not mapped_field.startswith('_') and mapped_field != field and mapped_idx is not None):
                                column_mapping['_adjacent_columns'][field] = adj_idx
                                logger.info(f"Added adjacent column {adj_idx} for {field} from candidates")
                                break
                
                elif field_lower in ['debits', 'credits', 'balance', 'servicefee']:
                    # Check if this column is part of an amount pair
                    for col_pair in adjacent_columns_candidates['amount']:
                        if col_idx in col_pair:
                            # Get the other column in the pair
                            adj_idx = col_pair[0] if col_pair[1] == col_idx else col_pair[1]
                            
                            # Skip if this adjacent column is already mapped to a different field
                            if not any(adj_idx == mapped_idx for mapped_field, mapped_idx in column_mapping.items() 
                                    if not mapped_field.startswith('_') and mapped_field != field and mapped_idx is not None):
                                column_mapping['_adjacent_columns'][field] = adj_idx
                                logger.info(f"Added adjacent column {adj_idx} for {field} from candidates")
                                break
                
                # If no adjacent column was found from candidates, check adjacent columns
                if field not in column_mapping['_adjacent_columns']:
                    for adj_idx in [col_idx - 1, col_idx + 1]:
                        if adj_idx < 0 or adj_idx >= len(df.columns):
                            continue
                        
                        # Skip if this adjacent column is already mapped to a different field
                        if any(adj_idx == mapped_idx for mapped_field, mapped_idx in column_mapping.items() 
                              if not mapped_field.startswith('_') and mapped_field != field and mapped_idx is not None):
                            continue
                        
                        # Check if adjacent column has similar content type
                        if field == 'Details' and column_scores[adj_idx]['details'] > 0:
                            column_mapping['_adjacent_columns'][field] = adj_idx
                            logger.info(f"Added adjacent column {adj_idx} for {field}")
                            break
                        elif field == 'Date' and column_scores[adj_idx]['date'] > 0:
                            column_mapping['_adjacent_columns'][field] = adj_idx
                            logger.info(f"Added adjacent column {adj_idx} for {field}")
                            break
                        elif field == 'Debits' and column_scores[adj_idx]['debit'] > 0:
                            column_mapping['_adjacent_columns'][field] = adj_idx
                            logger.info(f"Added adjacent column {adj_idx} for {field}")
                            break
                        elif field == 'Credits' and column_scores[adj_idx]['credit'] > 0:
                            column_mapping['_adjacent_columns'][field] = adj_idx
                            logger.info(f"Added adjacent column {adj_idx} for {field}")
                            break
                        elif field == 'Balance' and column_scores[adj_idx]['balance'] > 0:
                            column_mapping['_adjacent_columns'][field] = adj_idx
                            logger.info(f"Added adjacent column {adj_idx} for {field}")
                            break
                        elif field == 'ServiceFee' and column_scores[adj_idx]['service_fee'] > 0:
                            column_mapping['_adjacent_columns'][field] = adj_idx
                            logger.info(f"Added adjacent column {adj_idx} for {field}")
                            break
            
            logger.info(f"Lattice column mapping: {column_mapping}")
            return column_mapping, header_row
        
        except Exception as e:
            error_msg = f"Error in identify_columns: {e}"
            logger.error(error_msg)
            return column_mapping, header_row
    
    def extract_json_text(self, value: Any) -> str:
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
    
    def has_meaningful_data(self, row_data: Dict[str, str]) -> bool:
        """Check if a row has meaningful transaction data with strict validation"""
        try:
            # Skip rows that are likely headers
            if row_data.get('Details', '').lower() in ['details', 'description'] and \
               (row_data.get('Debits', '').lower() == 'debits' or row_data.get('Credits', '').lower() == 'credits'):
                return False
            
            # Special case for balance brought forward - always keep these
            if 'balance brought forward' in row_data.get('Details', '').lower():
                return True
            
            # Define patterns for validation
            date_pattern = r'\d{1,2}[/\-\.]\d{1,2}[/\-\.]?(\d{2}|\d{4})?|\d{4}-\d{2}-\d{2}'
            amount_pattern = r'\d+[\.,]\d{2}|-\d+[\.,]\d{2}|\d+[\.,]\d{3}[\.,]\d{2}'
            
            # Check if we have details field
            has_details = bool(row_data.get('Details', '').strip())
            
            # Check if we have a valid date
            date_value = row_data.get('Date', '').strip()
            has_valid_date = bool(date_value and re.search(date_pattern, date_value))
            
            # Check if we have at least one valid amount field
            has_debits = bool(row_data.get('Debits', '').strip())
            has_credits = bool(row_data.get('Credits', '').strip())
            has_service_fee = bool(row_data.get('ServiceFee', '').strip())
            has_balance = bool(row_data.get('Balance', '').strip())
            
            has_valid_amount = False
            for field in ['Debits', 'Credits', 'Balance', 'ServiceFee']:
                amount_value = row_data.get(field, '').strip()
                if amount_value and re.search(amount_pattern, amount_value):
                    has_valid_amount = True
                    break
            
            # For a complete transaction, we need:
            # 1. Either a valid date or meaningful details
            # 2. At least one valid amount field
            
            # Check for basic requirements
            basic_requirements_met = (has_details or has_valid_date) and has_valid_amount
            
            if not basic_requirements_met:
                # For lattice mode, be more lenient
                # If we have at least two fields with data, consider it a potential transaction
                filled_fields = sum([has_details, has_valid_date, has_debits, has_credits, has_service_fee, has_balance])
                if filled_fields >= 2:  # At least two fields with data
                    return True
                return False
            
            # Additional validation for field alignment
            # If we have both Debits and Credits, that's suspicious unless it's a special case
            if has_debits and has_credits:
                # Check if this is a valid case where both fields might be populated
                # For example, a transaction with both a fee and a deposit
                details_lower = row_data.get('Details', '').lower()
                if not any(term in details_lower for term in ['fee', 'charge', 'interest', 'adjustment', 'correction']):
                    # This is likely a data alignment issue - log it but don't reject outright
                    logger.warning(f"Transaction has both Debits and Credits: {row_data}")
            
            # Validate that amounts are properly aligned with transaction details
            if has_details and has_valid_amount:
                details_lower = row_data.get('Details', '').lower()
                
                # Check for misalignment between details and amount fields
                debit_terms = ['payment', 'withdrawal', 'purchase', 'fee', 'charge']
                credit_terms = ['deposit', 'credit', 'interest', 'salary', 'refund']
                
                has_debit_term = any(term in details_lower for term in debit_terms)
                has_credit_term = any(term in details_lower for term in credit_terms)
                
                # If details suggest a debit but we only have a credit amount (or vice versa), log a warning
                if has_debit_term and not has_debits and has_credits:
                    logger.warning(f"Possible field misalignment: Debit terms in details but only Credits value: {row_data}")
                
                if has_credit_term and has_debits and not has_credits:
                    logger.warning(f"Possible field misalignment: Credit terms in details but only Debits value: {row_data}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in has_meaningful_data: {e}")
            return False
    
    def extract_transactions_from_table(self, df: pd.DataFrame, column_mapping: Dict[str, int], 
                                           header_row: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Extract transaction data from a table based on column mapping
        Specialized for lattice mode tables
        
        Args:
            df (DataFrame): Pandas DataFrame containing the table
            column_mapping (dict): Mapping of column indices to field names
            header_row (int, optional): Index of header row to skip
            
        Returns:
            list: List of transaction dictionaries
        """
        transactions = []
        potential_transactions = []
        prev_transaction = None
        transactions_in_table = 0
        
        # Define patterns for validation
        date_pattern = r'\d{1,2}[/\-\.]\d{1,2}[/\-\.]?(\d{2}|\d{4})?|\d{4}-\d{2}-\d{2}'
        amount_pattern = r'\d+[\.,]\d{2}|-\d+[\.,]\d{2}|\d+[\.,]\d{3}[\.,]\d{2}'
        
        try:
            # First pass: extract potential transaction rows
            start_row = header_row + 1 if header_row is not None else 0
            
            for i in range(start_row, len(df)):
                row_data = {
                    'Details': '',
                    'ServiceFee': '',
                    'Debits': '',
                    'Credits': '',
                    'Date': '',
                    'Balance': '',
                    'row_index': i,
                    'is_continuation': False
                }
                
                found_data = False
                
                # Extract data from each column based on mapping
                for field, col_idx in column_mapping.items():
                    # Skip metadata fields (those starting with underscore)
                    if field.startswith('_'):
                        continue
                        
                    if col_idx is not None and col_idx < len(df.columns):
                        value = df.iloc[i, col_idx]
                        # Extract text from JSON-like strings
                        extracted_value = self.extract_json_text(value)
                        
                        # For lattice mode, check adjacent columns if they exist
                        if column_mapping.get('_lattice_mode') and field in column_mapping.get('_adjacent_columns', {}):
                            adj_col_idx = column_mapping['_adjacent_columns'][field]
                            if adj_col_idx is not None and adj_col_idx < len(df.columns):
                                adj_value = df.iloc[i, adj_col_idx]
                                adj_extracted = self.extract_json_text(adj_value)
                                
                                # Combine values based on field type
                                if field == 'Details':
                                    # For details, concatenate with space if both have content
                                    if extracted_value and adj_extracted:
                                        # Remove duplicates if the adjacent column contains the same text
                                        if adj_extracted.strip() in extracted_value:
                                            pass  # Keep original value
                                        elif extracted_value.strip() in adj_extracted:
                                            extracted_value = adj_extracted  # Use adjacent value if it's more complete
                                        else:
                                            extracted_value = f"{extracted_value} {adj_extracted}"
                                    elif adj_extracted:
                                        extracted_value = adj_extracted
                                elif field == 'Date':
                                    # Use the value that looks like a date
                                    if re.search(date_pattern, adj_extracted) and not re.search(date_pattern, extracted_value):
                                        extracted_value = adj_extracted
                                    # If both have date patterns, use the more complete one
                                    elif re.search(date_pattern, adj_extracted) and re.search(date_pattern, extracted_value):
                                        if len(adj_extracted) > len(extracted_value):
                                            extracted_value = adj_extracted
                                elif field in ['Debits', 'Credits', 'Balance', 'ServiceFee']:
                                    # Use the value that looks like an amount
                                    if re.search(amount_pattern, adj_extracted) and not re.search(amount_pattern, extracted_value):
                                        extracted_value = adj_extracted
                                    # If both have amount patterns, use the non-zero one or the larger one
                                    elif re.search(amount_pattern, adj_extracted) and re.search(amount_pattern, extracted_value):
                                        # Try to extract numeric values for comparison
                                        try:
                                            # Clean and convert to float for comparison
                                            val1 = float(re.search(r'\d+[\.,]\d+', extracted_value).group(0).replace(',', '.'))
                                            val2 = float(re.search(r'\d+[\.,]\d+', adj_extracted).group(0).replace(',', '.'))
                                            
                                            # Use the non-zero value if one is zero
                                            if val1 == 0 and val2 != 0:
                                                extracted_value = adj_extracted
                                            elif val1 != 0 and val2 == 0:
                                                pass  # Keep original value
                                            # Otherwise use the one with more digits
                                            elif len(adj_extracted) > len(extracted_value):
                                                extracted_value = adj_extracted
                                        except Exception:
                                            # If comparison fails, use the longer string
                                            if len(adj_extracted) > len(extracted_value):
                                                extracted_value = adj_extracted
                        
                        # Clean up the extracted value
                        if field in ['Debits', 'Credits', 'Balance', 'ServiceFee']:
                            # Ensure we have just the amount
                            amount_match = re.search(amount_pattern, extracted_value)
                            if amount_match:
                                extracted_value = amount_match.group(0)
                        elif field == 'Date':
                            # Try to standardize date format
                            date_match = re.search(date_pattern, extracted_value)
                            if date_match:
                                extracted_value = date_match.group(0)
                        
                        row_data[field] = extracted_value
                        if extracted_value.strip():
                            found_data = True
                
                # Check if this row has meaningful data
                if found_data:
                    # Check if this is likely a continuation row (has details but missing other key fields)
                    if row_data['Details'] and not row_data['Date'] and not row_data['Debits'] and not row_data['Credits']:
                        row_data['is_continuation'] = True
                    
                    potential_transactions.append(row_data)
            
            # Second pass: process potential transactions and handle multi-row transactions
            idx = 0
            while idx < len(potential_transactions):
                row_data = potential_transactions[idx]
                row_index = row_data.pop('row_index', None)
                is_continuation = row_data.pop('is_continuation', False)
                
                # If this is a continuation row and we have a previous transaction
                if is_continuation and prev_transaction:
                    # Append details to previous transaction
                    if row_data['Details']:
                        prev_transaction['Details'] += ' ' + row_data['Details']
                    
                    # Copy any missing fields from continuation row to main transaction
                    for field in ['Balance', 'ServiceFee']:
                        if row_data[field] and not prev_transaction[field]:
                            prev_transaction[field] = row_data[field]
                    
                    idx += 1
                    continue
                
                # This is a main transaction row
                if self.has_meaningful_data(row_data):
                    # Look ahead for continuation rows and merge them
                    look_ahead_idx = idx + 1
                    while look_ahead_idx < len(potential_transactions):
                        next_row = potential_transactions[look_ahead_idx]
                        # If next row is a continuation (has details but no date/amounts)
                        if next_row.get('is_continuation', False) or \
                           (next_row['Details'] and not next_row['Date'] and \
                            not next_row['Debits'] and not next_row['Credits']):
                            # Append details
                            if next_row['Details']:
                                row_data['Details'] += ' ' + next_row['Details']
                            
                            # Copy any missing fields
                            for field in ['Balance', 'ServiceFee']:
                                if next_row[field] and not row_data[field]:
                                    row_data[field] = next_row[field]
                            
                            look_ahead_idx += 1
                        else:
                            break
                    
                    # Skip the continuation rows we've merged
                    idx = look_ahead_idx
                    
                    # Ensure proper field alignment
                    # If we have both Debits and Credits populated, determine which is correct
                    if row_data['Debits'] and row_data['Credits']:
                        # Try to determine which one is correct based on context
                        # For example, if Details contains keywords like "payment" or "withdrawal",
                        # it's likely a debit; if it contains "deposit" or "credit", it's likely a credit
                        details_lower = row_data['Details'].lower()
                        if any(term in details_lower for term in ['payment', 'withdrawal', 'purchase', 'fee', 'charge']):
                            # This is likely a debit transaction
                            row_data['Credits'] = ''
                        elif any(term in details_lower for term in ['deposit', 'credit', 'interest', 'salary']):
                            # This is likely a credit transaction
                            row_data['Debits'] = ''
                    
                    # Add the processed transaction
                    transactions.append(row_data)
                    prev_transaction = row_data
                    transactions_in_table += 1
                else:
                    # Not a valid transaction, move to next row
                    idx += 1
            
            logger.info(f"Extracted {transactions_in_table} transactions from table")
            return transactions
        
        except Exception as e:
            error_msg = f"Error extracting transactions from table: {e}"
            logger.error(error_msg)
            return []
    
    def process_pdf(self, pdf_path: str, password: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Process a PDF file and extract transactions using lattice mode
        
        Args:
            pdf_path (str): Path to the PDF file
            password (str, optional): Password for encrypted PDF
            
        Returns:
            list: List of transaction dictionaries
        """
        all_transactions = []
        self.debug_info = {
            'pdf_path': pdf_path,
            'total_pages': 0,
            'tables_extracted': 0,
            'transaction_tables': 0,
            'total_transactions': 0,
            'page_errors': [],
            'table_results': []
        }
        
        try:
            # Extract tables using lattice mode
            tables = self.extract_tables_from_pdf(pdf_path, password)
            
            # Process each table
            for i, table in enumerate(tables):
                table_debug = {
                    'table_index': i,
                    'row_count': len(table),
                    'column_count': len(table.columns),
                    'is_transaction_table': False,
                    'transactions_extracted': 0
                }
                
                logger.info(f"Processing table {i+1}/{len(tables)}")
                
                # Check if this is a transaction table
                # For lattice mode, be more lenient in considering tables as transaction tables
                if self.is_transaction_table(table):
                    table_debug['is_transaction_table'] = True
                    self.debug_info['transaction_tables'] += 1
                    
                    # Get page number if available
                    page_number = getattr(table, '_page_number', None)
                    if page_number:
                        logger.info(f"Table {i+1} is from page {page_number}")
                        table_debug['page_number'] = page_number
                    
                    # Identify columns
                    column_mapping, header_row = self.identify_columns(table)
                    table_debug['column_mapping'] = column_mapping
                    table_debug['header_row'] = header_row
                    
                    # Extract transactions
                    transactions = self.extract_transactions_from_table(table, column_mapping, header_row)
                    
                    # Only add tables that actually yielded transactions
                    if transactions:
                        all_transactions.extend(transactions)
                        table_debug['transactions_extracted'] = len(transactions)
                        self.debug_info['total_transactions'] += len(transactions)
                        logger.info(f"Extracted {len(transactions)} transactions from table {i+1}")
                    else:
                        logger.info(f"No transactions extracted from table {i+1} despite being identified as a transaction table")
                else:
                    # Even if not identified as a transaction table, try with a more lenient approach
                    # This helps catch tables that might have been misclassified
                    if len(table) >= 3:  # Table has at least 3 rows (potential header + transactions)
                        logger.info(f"Table {i+1} not initially identified as transaction table, trying with lenient approach")
                        
                        # Try to identify columns anyway
                        column_mapping, header_row = self.identify_columns(table)
                        
                        # Check if we found any potential transaction columns
                        if any(not k.startswith('_') for k, v in column_mapping.items() if v is not None):
                            # Extract transactions with lenient settings
                            transactions = self.extract_transactions_from_table(table, column_mapping, header_row)
                            
                            if transactions:
                                logger.info(f"Lenient approach found {len(transactions)} transactions in table {i+1}")
                                all_transactions.extend(transactions)
                                table_debug['is_transaction_table'] = True  # Mark as transaction table retroactively
                                table_debug['transactions_extracted'] = len(transactions)
                                self.debug_info['total_transactions'] += len(transactions)
                                self.debug_info['transaction_tables'] += 1
                            else:
                                logger.info(f"Lenient approach found no transactions in table {i+1}")
                        else:
                            logger.info(f"Table {i+1} has no identifiable transaction columns, skipping")
                    else:
                        logger.info(f"Table {i+1} is not a transaction table, skipping")
                
                self.debug_info['table_results'].append(table_debug)
            
            # Post-process transactions to ensure consistency
            if all_transactions:
                all_transactions = self.post_process_transactions(all_transactions)
            
            logger.info(f"Extracted {len(all_transactions)} total transactions from PDF using lattice mode")
            return all_transactions
        except Exception as e:
            error_msg = f"Error processing PDF: {e}"
            logger.error(error_msg)
            self.debug_info['error'] = error_msg
            return []
            
    def post_process_transactions(self, transactions: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Post-process extracted transactions to ensure consistency and quality
        
        Args:
            transactions (list): List of transaction dictionaries
            
        Returns:
            list: List of processed transaction dictionaries
        """
        processed_transactions = []
        seen_transactions = set()
        
        # Define patterns for validation
        date_pattern = r'\d{1,2}[/\-\.]\d{1,2}[/\-\.]?(\d{2}|\d{4})?|\d{4}-\d{2}-\d{2}'
        amount_pattern = r'\d+[\.,]\d{2}|-\d+[\.,]\d{2}|\d+[\.,]\d{3}[\.,]\d{2}'
        
        try:
            for tx in transactions:
                # Clean up transaction fields
                clean_tx = {
                    'Date': tx.get('Date', '').strip(),
                    'Details': tx.get('Details', '').strip(),
                    'Debits': tx.get('Debits', '').strip(),
                    'Credits': tx.get('Credits', '').strip(),
                    'Balance': tx.get('Balance', '').strip(),
                    'ServiceFee': tx.get('ServiceFee', '').strip()
                }
                
                # Standardize date format if possible
                if clean_tx['Date']:
                    date_match = re.search(date_pattern, clean_tx['Date'])
                    if date_match:
                        clean_tx['Date'] = date_match.group(0)
                
                # Standardize amount formats
                for field in ['Debits', 'Credits', 'Balance', 'ServiceFee']:
                    if clean_tx[field]:
                        amount_match = re.search(amount_pattern, clean_tx[field])
                        if amount_match:
                            clean_tx[field] = amount_match.group(0)
                
                # Fix field alignment issues
                # If we have both Debits and Credits populated, determine which is correct
                if clean_tx['Debits'] and clean_tx['Credits']:
                    # Try to determine which one is correct based on context
                    details_lower = clean_tx['Details'].lower()
                    if any(term in details_lower for term in ['payment', 'withdrawal', 'purchase', 'fee', 'charge']):
                        # This is likely a debit transaction
                        clean_tx['Credits'] = ''
                    elif any(term in details_lower for term in ['deposit', 'credit', 'interest', 'salary']):
                        # This is likely a credit transaction
                        clean_tx['Debits'] = ''
                    else:
                        # If we can't determine from details, keep the larger value
                        try:
                            debit_val = float(clean_tx['Debits'].replace(',', '.').replace(' ', ''))
                            credit_val = float(clean_tx['Credits'].replace(',', '.').replace(' ', ''))
                            if debit_val > credit_val:
                                clean_tx['Credits'] = ''
                            else:
                                clean_tx['Debits'] = ''
                        except Exception:
                            # If conversion fails, keep both values
                            pass
                
                # Check if Details field contains amount values that should be in other fields
                if clean_tx['Details']:
                    # Look for amount patterns in details that might be misplaced
                    amount_matches = re.findall(amount_pattern, clean_tx['Details'])
                    if amount_matches:
                        # If we found amounts in details and no amounts in debit/credit fields
                        if not clean_tx['Debits'] and not clean_tx['Credits']:
                            # Try to determine if it's debit or credit based on context
                            details_lower = clean_tx['Details'].lower()
                            if any(term in details_lower for term in ['payment', 'withdrawal', 'purchase', 'fee', 'charge']):
                                clean_tx['Debits'] = amount_matches[0]
                                # Remove the amount from details
                                clean_tx['Details'] = re.sub(re.escape(amount_matches[0]), '', clean_tx['Details']).strip()
                            elif any(term in details_lower for term in ['deposit', 'credit', 'interest', 'salary']):
                                clean_tx['Credits'] = amount_matches[0]
                                # Remove the amount from details
                                clean_tx['Details'] = re.sub(re.escape(amount_matches[0]), '', clean_tx['Details']).strip()
                
                # Remove very long details that are likely parsing errors
                if len(clean_tx['Details']) > 200:
                    clean_tx['Details'] = clean_tx['Details'][:197] + '...'
                
                # Clean up details field - remove excess whitespace and normalize
                clean_tx['Details'] = re.sub(r'\s+', ' ', clean_tx['Details']).strip()
                
                # Create a fingerprint to detect duplicates
                # Use a combination of date, details, and amounts
                fingerprint = f"{clean_tx['Date']}|{clean_tx['Details'][:50]}|{clean_tx['Debits']}|{clean_tx['Credits']}"
                
                # Only add if it's not a duplicate and has meaningful data
                if fingerprint not in seen_transactions and self.has_meaningful_data(clean_tx):
                    seen_transactions.add(fingerprint)
                    processed_transactions.append(clean_tx)
                    
            logger.info(f"Post-processing reduced {len(transactions)} transactions to {len(processed_transactions)} unique transactions")
            return processed_transactions
            
        except Exception as e:
            error_msg = f"Error in post_process_transactions: {e}"
            logger.error(error_msg)
            if hasattr(self, 'debug_info'):
                self.debug_info['error'] = error_msg
            return transactions  # Return original transactions if processing fails
            
    def export_to_csv(self, transactions: List[Dict[str, str]], output_path: str) -> bool:
        """
        Export transactions to CSV file
        
        Args:
            transactions (list): List of transaction dictionaries
            output_path (str): Path to save the CSV file
            
        Returns:
            bool: True if successful
        """
        try:
            if not transactions:
                logger.warning("No transactions to export")
                return False
            
            # Convert to DataFrame
            df = pd.DataFrame(transactions)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save to CSV
            df.to_csv(output_path, index=False)
            logger.info(f"Saved {len(transactions)} transactions to {output_path}")
            return True
            
        except Exception as e:
            error_msg = f"Error exporting to CSV: {e}"
            logger.error(error_msg)
            return False
    
    def get_debug_info(self) -> Dict[str, Any]:
        """
        Get debug information
        
        Returns:
            dict: Debug information
        """
        return self.debug_info


def main():
    """Command line interface for the lattice extractor"""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='Extract transactions from PDF using lattice mode')
    parser.add_argument('pdf_path', help='Path to the PDF file')
    parser.add_argument('--password', help='Password for encrypted PDF')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--output', help='Path to save transactions CSV')
    parser.add_argument('--debug-info', help='Path to save debug info (JSON)')
    
    args = parser.parse_args()
    
    # Create extractor
    extractor = LatticeExtractor(debug=args.debug)
    
    # Process PDF
    transactions = extractor.process_pdf(args.pdf_path, args.password)
    
    print(f"Extracted {len(transactions)} transactions using lattice mode")
    
    # Export to CSV if output path provided
    if args.output:
        if extractor.export_to_csv(transactions, args.output):
            print(f"Transactions saved to {args.output}")
    
    # Save debug info if requested
    if args.debug and args.debug_info:
        debug_info = extractor.get_debug_info()
        with open(args.debug_info, 'w') as f:
            json.dump(debug_info, f, indent=2)
        print(f"Debug info saved to {args.debug_info}")


if __name__ == '__main__':
    main()
