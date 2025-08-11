#!/usr/bin/env python3
import os
import sys
import json
import logging
import pandas as pd
import numpy as np
import tabula
import PyPDF2
from modules.transaction_identifier import identify_columns, is_transaction_table

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def analyze_pdf_structure(pdf_path):
    """
    Analyze the structure of each page in a PDF file and identify differences
    between pages, especially focusing on header rows and column structures.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        dict: Analysis results
    """
    logger.info(f"Analyzing PDF structure: {pdf_path}")
    
    # Extract tables from each page separately
    tables_by_page = {}
    page_count = 0
    
    # First determine page count
    try:
        # Use PyPDF2 to get page count
        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            page_count = len(pdf_reader.pages)
            logger.info(f"PDF has {page_count} pages")
    except Exception as e:
        logger.error(f"Error determining page count: {e}")
        return {"error": str(e)}
    
    # Extract tables page by page
    for page in range(1, page_count + 1):
        try:
            # Try both stream and lattice modes
            stream_tables = tabula.read_pdf(
                pdf_path,
                pages=str(page),
                multiple_tables=True,
                stream=True,
                guess=True,
                pandas_options={'header': None}
            )
            
            lattice_tables = tabula.read_pdf(
                pdf_path,
                pages=str(page),
                multiple_tables=True,
                lattice=True,
                pandas_options={'header': None}
            )
            
            # Combine tables, removing duplicates
            page_tables = stream_tables.copy()
            for table in lattice_tables:
                if not any(table.equals(t) for t in page_tables):
                    page_tables.append(table)
            
            tables_by_page[page] = page_tables
            logger.info(f"Page {page}: Extracted {len(page_tables)} tables")
        except Exception as e:
            logger.error(f"Error extracting tables from page {page}: {e}")
            tables_by_page[page] = []
    
    # Analyze each page's structure
    page_analysis = {}
    
    for page, tables in tables_by_page.items():
        page_analysis[page] = {
            "table_count": len(tables),
            "tables": []
        }
        
        for i, table in enumerate(tables):
            # Convert to DataFrame for analysis
            df = pd.DataFrame(table)
            
            # Check if this is a transaction table
            is_transaction, transaction_debug = is_transaction_table(df, debug=True)
            
            # Identify columns if it's a transaction table
            column_mapping = {}
            column_debug = {}
            if is_transaction:
                column_mapping_result = identify_columns(df, debug=True)
                if isinstance(column_mapping_result, tuple) and len(column_mapping_result) == 2:
                    column_mapping, column_debug = column_mapping_result
                else:
                    column_mapping = column_mapping_result
            
            # Analyze column structure
            column_types = {}
            for col in range(len(df.columns)):
                col_values = df.iloc[:, col].astype(str)
                
                # Check for date patterns
                date_pattern_count = sum(1 for val in col_values if 
                                        any(pattern in val.lower() for pattern in 
                                           ["date", "/"]) and 
                                        len(val) < 20)
                
                # Check for numeric patterns
                numeric_pattern_count = sum(1 for val in col_values if 
                                          any(c.isdigit() for c in val) and 
                                          any(c in val for c in ".,") and
                                          not any(c in val.lower() for c in "date/"))
                
                # Check for large numbers (potential balance column)
                large_numbers = []
                for val in col_values:
                    # Clean the value to extract numeric part
                    clean_val = ''.join(c for c in val if c.isdigit() or c in '.-')
                    try:
                        if clean_val:
                            num = float(clean_val)
                            if abs(num) > 1000:
                                large_numbers.append(num)
                    except ValueError:
                        pass
                
                large_number_count = len(large_numbers)
                
                # Check for text content
                text_content_count = sum(1 for val in col_values if 
                                       len(val) > 5 and 
                                       not val.isdigit() and
                                       "/" not in val and
                                       not any(c in val for c in ".,"))
                
                # Determine column type based on patterns
                if date_pattern_count > len(df) * 0.3:
                    col_type = "date"
                elif large_number_count > len(df) * 0.3:
                    col_type = "balance"
                elif numeric_pattern_count > len(df) * 0.3:
                    col_type = "amount"
                elif text_content_count > len(df) * 0.3:
                    col_type = "details"
                else:
                    col_type = "unknown"
                
                column_types[col] = {
                    "type": col_type,
                    "date_pattern_count": date_pattern_count,
                    "numeric_pattern_count": numeric_pattern_count,
                    "large_number_count": large_number_count,
                    "text_content_count": text_content_count
                }
            
            # Check for header rows
            potential_headers = []
            for i in range(min(5, len(df))):
                row = df.iloc[i].astype(str)
                header_keywords = sum(1 for val in row if 
                                    any(keyword in val.lower() for keyword in 
                                       ["date", "details", "description", "debit", "credit", 
                                        "amount", "balance", "reference"]))
                if header_keywords >= 2:
                    potential_headers.append({
                        "row": i,
                        "keywords": header_keywords,
                        "values": row.tolist()
                    })
            
            # Check for balance column keywords
            balance_keywords = []
            for r in range(min(10, len(df))):
                for c in range(len(df.columns)):
                    cell_val = str(df.iloc[r, c]).lower()
                    if any(keyword in cell_val for keyword in ["balance", "bal", "closing", "opening", "b/fwd", "c/fwd"]):
                        balance_keywords.append({
                            "row": r,
                            "col": c,
                            "value": cell_val
                        })
            
            # Store table analysis
            table_analysis = {
                "shape": df.shape,
                "is_transaction_table": is_transaction,
                "column_types": column_types,
                "potential_headers": potential_headers,
                "balance_keywords": balance_keywords,
                "transaction_debug": transaction_debug,
                "column_mapping": column_mapping,
                "column_debug": column_debug if column_debug else None
            }
            
            page_analysis[page]["tables"].append(table_analysis)
    
    # Compare pages to identify structural differences
    page_comparison = {
        "total_pages": page_count,
        "pages_with_tables": len(tables_by_page),
        "page_differences": [],
        "balance_column_analysis": {
            "pages_with_balance_column": [],
            "pages_without_balance_column": []
        }
    }
    
    # Compare each page with the first page
    first_page_structure = page_analysis.get(1, {})
    
    for page, analysis in page_analysis.items():
        if page == 1:
            continue
        
        differences = []
        
        # Compare table count
        if analysis["table_count"] != first_page_structure.get("table_count", 0):
            differences.append(f"Different table count: Page 1 has {first_page_structure.get('table_count', 0)}, Page {page} has {analysis['table_count']}")
        
        # Compare table structures
        for i, table in enumerate(analysis.get("tables", [])):
            if i < len(first_page_structure.get("tables", [])):
                first_table = first_page_structure["tables"][i]
                
                # Compare shape
                if table["shape"] != first_table["shape"]:
                    differences.append(f"Table {i+1} has different shape: Page 1 is {first_table['shape']}, Page {page} is {table['shape']}")
                
                # Compare column types
                for col, col_type in table["column_types"].items():
                    if col < len(first_table["column_types"]) and col_type["type"] != first_table["column_types"][col]["type"]:
                        differences.append(f"Column {col} has different type: Page 1 is {first_table['column_types'][col]['type']}, Page {page} is {col_type['type']}")
                
                # Compare header rows
                if len(table["potential_headers"]) != len(first_table["potential_headers"]):
                    differences.append(f"Different number of potential headers: Page 1 has {len(first_table['potential_headers'])}, Page {page} has {len(table['potential_headers'])}")
                
                # Compare balance keywords
                if len(table["balance_keywords"]) != len(first_table["balance_keywords"]):
                    differences.append(f"Different number of balance keywords: Page 1 has {len(first_table['balance_keywords'])}, Page {page} has {len(table['balance_keywords'])}")
            else:
                differences.append(f"Page {page} has extra table {i+1} not present on Page 1")
        
        # Check if this page has a balance column
        has_balance_column = False
        for table in analysis.get("tables", []):
            if table["column_mapping"] and isinstance(table["column_mapping"], dict) and table["column_mapping"].get("Balance") is not None:
                has_balance_column = True
                break
            
            # Also check column types
            for col, col_type in table["column_types"].items():
                if col_type["type"] == "balance":
                    has_balance_column = True
                    break
        
        if has_balance_column:
            page_comparison["balance_column_analysis"]["pages_with_balance_column"].append(page)
        else:
            page_comparison["balance_column_analysis"]["pages_without_balance_column"].append(page)
        
        if differences:
            page_comparison["page_differences"].append({
                "page": page,
                "differences": differences
            })
    
    # Check if first page has a balance column
    has_balance_column_first_page = False
    for table in first_page_structure.get("tables", []):
        if table.get("column_mapping") and isinstance(table["column_mapping"], dict) and table["column_mapping"].get("Balance") is not None:
            has_balance_column_first_page = True
            break
        
        # Also check column types
        for col, col_type in table.get("column_types", {}).items():
            if col_type["type"] == "balance":
                has_balance_column_first_page = True
                break
    
    if has_balance_column_first_page:
        page_comparison["balance_column_analysis"]["pages_with_balance_column"].append(1)
    else:
        page_comparison["balance_column_analysis"]["pages_without_balance_column"].append(1)
    
    # Summarize findings
    page_comparison["summary"] = {
        "pages_with_different_structure": len(page_comparison["page_differences"]),
        "first_page_has_balance_column": has_balance_column_first_page,
        "all_pages_have_balance_column": len(page_comparison["balance_column_analysis"]["pages_without_balance_column"]) == 0,
        "pages_missing_balance_column": page_comparison["balance_column_analysis"]["pages_without_balance_column"]
    }
    
    return {
        "page_analysis": page_analysis,
        "page_comparison": page_comparison
    }

def main():
    # Check if PDF file path is provided
    if len(sys.argv) < 2:
        logger.error("Please provide the path to a PDF file as an argument")
        print("Usage: python analyze_pdf_structure.py <pdf_file_path>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "structure_analysis")
    os.makedirs(output_dir, exist_ok=True)
    
    # Analyze PDF structure
    analysis = analyze_pdf_structure(pdf_path)
    
    # Save analysis to JSON
    output_file = os.path.join(output_dir, f"{os.path.basename(pdf_path).split('.')[0]}_structure_analysis.json")
    with open(output_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    logger.info(f"Analysis saved to {output_file}")
    
    # Print summary
    if "page_comparison" in analysis and "summary" in analysis["page_comparison"]:
        summary = analysis["page_comparison"]["summary"]
        logger.info(f"PDF has {analysis['page_comparison']['total_pages']} pages")
        logger.info(f"Pages with different structure from page 1: {summary['pages_with_different_structure']}")
        logger.info(f"First page has balance column: {summary['first_page_has_balance_column']}")
        logger.info(f"All pages have balance column: {summary['all_pages_have_balance_column']}")
        if not summary['all_pages_have_balance_column']:
            logger.info(f"Pages missing balance column: {summary['pages_missing_balance_column']}")
    
    # Print detailed differences
    if "page_comparison" in analysis and "page_differences" in analysis["page_comparison"]:
        for page_diff in analysis["page_comparison"]["page_differences"]:
            logger.info(f"\nDifferences on Page {page_diff['page']}:")
            for diff in page_diff["differences"]:
                logger.info(f"  - {diff}")

if __name__ == "__main__":
    main()
