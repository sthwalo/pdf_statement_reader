#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Table Extraction Module

Responsible for extracting tables from PDF files using tabula.
This module handles the raw table extraction only.
"""

import os
import logging
import tabula
import PyPDF2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def extract_tables_from_pdf(pdf_path, password=None, debug=False):
    """
    Extract tables from PDF using both stream and lattice modes
    
    Args:
        pdf_path (str): Path to the PDF file
        password (str, optional): Password for encrypted PDF
        debug (bool, optional): Enable debug mode with extra logging
        
    Returns:
        list: List of pandas DataFrames containing tables
        dict: Debug info if debug=True
    """
    tables = []
    debug_info = {
        'pdf_path': pdf_path,
        'total_pages': 0,
        'stream_tables': 0,
        'lattice_tables': 0,
        'special_tables': 0,
        'page_errors': [],
        'extraction_mode': []
    }
    
    try:
        # Get total pages
        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            total_pages = len(pdf_reader.pages)
            logger.info(f"PDF has {total_pages} pages")
            debug_info['total_pages'] = total_pages
        
        # Extract with stream mode
        try:
            stream_tables = tabula.read_pdf(
                pdf_path,
                pages='all',
                password=password,
                multiple_tables=True,
                stream=True,
                guess=True,
                pandas_options={'header': None}
            )
            logger.info(f"Stream mode extracted {len(stream_tables)} tables")
            debug_info['stream_tables'] = len(stream_tables)
            debug_info['extraction_mode'].append('stream_all')
            
            # Add metadata to each table to identify extraction mode
            for table in stream_tables:
                # Add extraction mode as a DataFrame attribute
                table._extraction_mode = 'stream'
                
            tables.extend(stream_tables)
        except Exception as e:
            error_msg = f"Error in stream mode extraction: {e}"
            logger.error(error_msg)
            debug_info['page_errors'].append(error_msg)
            
            # Try page by page as fallback
            for page in range(1, total_pages + 1):
                try:
                    page_tables = tabula.read_pdf(
                        pdf_path,
                        pages=str(page),
                        password=password,
                        multiple_tables=True,
                        stream=True,
                        guess=True,
                        pandas_options={'header': None}
                    )
                    logger.info(f"Stream mode page {page} extracted {len(page_tables)} tables")
                    debug_info['extraction_mode'].append(f'stream_page_{page}')
                    
                    # Add metadata to each table
                    for table in page_tables:
                        table._extraction_mode = 'stream'
                        table._page_number = page
                        
                    tables.extend(page_tables)
                except Exception as page_error:
                    error_msg = f"Error extracting page {page} with stream mode: {page_error}"
                    logger.error(error_msg)
                    debug_info['page_errors'].append(error_msg)
        
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
            debug_info['lattice_tables'] = len(lattice_tables)
            debug_info['extraction_mode'].append('lattice_all')
            
            # Add metadata to each table to identify extraction mode
            for table in lattice_tables:
                # Add extraction mode as a DataFrame attribute
                table._extraction_mode = 'lattice'
            
            # Add unique lattice tables
            for table in lattice_tables:
                if not any(table.equals(t) for t in tables):
                    tables.append(table)
        except Exception as e:
            error_msg = f"Error in lattice mode extraction: {e}"
            logger.error(error_msg)
            debug_info['page_errors'].append(error_msg)
            
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
                    debug_info['extraction_mode'].append(f'lattice_page_{page}')
                    
                    # Add metadata to each table
                    for table in page_tables:
                        table._extraction_mode = 'lattice'
                        table._page_number = page
                    
                    # Add unique tables
                    for table in page_tables:
                        if not any(table.equals(t) for t in tables):
                            tables.append(table)
                except Exception as page_error:
                    error_msg = f"Error extracting page {page} with lattice mode: {page_error}"
                    logger.error(error_msg)
                    debug_info['page_errors'].append(error_msg)
        
        # Special handling for problematic pages (e.g., page 17)
        try:
            special_tables = tabula.read_pdf(
                pdf_path,
                pages='17',
                password=password,
                multiple_tables=True,
                stream=True,
                guess=False,
                pandas_options={'header': None}
            )
            logger.info(f"Special handling for page 17 extracted {len(special_tables)} tables")
            debug_info['special_tables'] = len(special_tables)
            debug_info['extraction_mode'].append('special_page_17')
            
            for table in special_tables:
                if not any(table.equals(t) for t in tables):
                    tables.append(table)
        except:
            logger.info("No page 17 or error processing it specifically")
        
        logger.info(f"Total tables extracted: {len(tables)}")
        
        if debug:
            return tables, debug_info
        return tables
    
    except Exception as e:
        error_msg = f"Error extracting tables: {e}"
        logger.error(error_msg)
        debug_info['page_errors'].append(error_msg)
        
        if debug:
            return [], debug_info
        return []

def main():
    """Test function for direct module execution"""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='Extract tables from PDF')
    parser.add_argument('pdf_path', help='Path to the PDF file')
    parser.add_argument('--password', help='Password for encrypted PDF')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--output', help='Path to save debug info (JSON)')
    
    args = parser.parse_args()
    
    if args.debug:
        tables, debug_info = extract_tables_from_pdf(args.pdf_path, args.password, debug=True)
        print(f"Extracted {len(tables)} tables")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(debug_info, f, indent=2)
            print(f"Debug info saved to {args.output}")
        else:
            print(json.dumps(debug_info, indent=2))
    else:
        tables = extract_tables_from_pdf(args.pdf_path, args.password)
        print(f"Extracted {len(tables)} tables")

if __name__ == '__main__':
    main()
