#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Page Analyzer Module

Analyzes PDF pages to detect structural differences and explain why data extraction
works on some pages but fails on others. Provides recommendations to improve extraction
robustness across varying layouts.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import tabula
import PyPDF2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    logger.warning("pdfplumber not available. Some features will be limited.")
    PDFPLUMBER_AVAILABLE = False

try:
    import camelot
    CAMELOT_AVAILABLE = True
except ImportError:
    logger.warning("camelot-py not available. Some features will be limited.")
    CAMELOT_AVAILABLE = False


class PageAnalyzer:
    """
    Analyzes PDF pages to detect structural differences and explain why data extraction
    works on some pages but fails on others.
    """
    
    def __init__(self, pdf_path: str):
        """
        Initialize the PageAnalyzer with a PDF file path.
        
        Args:
            pdf_path (str): Path to the PDF file to analyze
        """
        self.pdf_path = pdf_path
        self.pages = []  # Stores analysis of each page
        self.total_pages = 0
        self.successful_pages = []
        self.failed_pages = []
        self.page_comparisons = {}
        
        # Check if the PDF exists
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Get total pages
        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            self.total_pages = len(pdf_reader.pages)
            logger.info(f"PDF has {self.total_pages} pages")
    
    def analyze(self, pages: List[int] = None) -> Dict:
        """
        Main function: analyzes specified pages or all pages and returns a report.
        
        Args:
            pages (List[int], optional): List of page numbers to analyze. If None, analyzes all pages.
            
        Returns:
            Dict: Analysis report
        """
        if pages is None:
            pages = list(range(1, self.total_pages + 1))
        
        logger.info(f"Analyzing pages: {pages}")
        
        # Analyze each page
        for page_num in pages:
            if page_num < 1 or page_num > self.total_pages:
                logger.warning(f"Page {page_num} is out of range (1-{self.total_pages}). Skipping.")
                continue
            
            page_data = self._analyze_page(page_num)
            self.pages.append(page_data)
        
        # Generate report
        report = {
            "pdf_path": self.pdf_path,
            "total_pages": self.total_pages,
            "analyzed_pages": len(self.pages),
            "page_details": self.pages,
            "successful_extraction_pages": self.successful_pages,
            "failed_extraction_pages": self.failed_pages,
            "comparisons": self.page_comparisons,
            "global_recommendations": self._generate_recommendations()
        }
        
        return report
    
    def compare_pages(self, page_a: int, page_b: int) -> Dict:
        """
        Compare two pages to find structural differences.
        
        Args:
            page_a (int): First page number
            page_b (int): Second page number
            
        Returns:
            Dict: Comparison report
        """
        # Ensure both pages are analyzed
        a_idx = next((i for i, p in enumerate(self.pages) if p["page"] == page_a), None)
        b_idx = next((i for i, p in enumerate(self.pages) if p["page"] == page_b), None)
        
        if a_idx is None:
            self.pages.append(self._analyze_page(page_a))
            a_idx = len(self.pages) - 1
        
        if b_idx is None:
            self.pages.append(self._analyze_page(page_b))
            b_idx = len(self.pages) - 1
        
        # Compare pages
        comparison = self._compare_pages(self.pages[a_idx], self.pages[b_idx])
        
        # Store comparison
        comparison_key = f"page_{page_a}_vs_{page_b}"
        self.page_comparisons[comparison_key] = comparison
        
        return comparison
    
    def _analyze_page(self, page_num: int) -> Dict:
        """
        Extract key structural features from a single page.
        
        Args:
            page_num (int): Page number to analyze (1-indexed)
            
        Returns:
            Dict: Page analysis data
        """
        logger.info(f"Analyzing page {page_num}")
        
        page_data = {
            "page": page_num,
            "tables": {
                "tabula_stream": [],
                "tabula_lattice": [],
                "camelot_stream": [],
                "camelot_lattice": []
            },
            "text_blocks": [],
            "lines": [],
            "extraction_success": False,
            "layout_fingerprint": {}
        }
        
        # Extract tables using tabula (stream mode)
        try:
            stream_tables = tabula.read_pdf(
                self.pdf_path,
                pages=str(page_num),
                multiple_tables=True,
                stream=True,
                guess=True,
                pandas_options={'header': None}
            )
            page_data["tables"]["tabula_stream"] = self._process_tables(stream_tables)
            logger.info(f"Page {page_num}: Extracted {len(stream_tables)} tables with tabula stream mode")
        except Exception as e:
            logger.error(f"Error extracting tables from page {page_num} with tabula stream mode: {e}")
        
        # Extract tables using tabula (lattice mode)
        try:
            lattice_tables = tabula.read_pdf(
                self.pdf_path,
                pages=str(page_num),
                multiple_tables=True,
                lattice=True,
                pandas_options={'header': None}
            )
            page_data["tables"]["tabula_lattice"] = self._process_tables(lattice_tables)
            logger.info(f"Page {page_num}: Extracted {len(lattice_tables)} tables with tabula lattice mode")
        except Exception as e:
            logger.error(f"Error extracting tables from page {page_num} with tabula lattice mode: {e}")
        
        # Extract tables using camelot if available
        if CAMELOT_AVAILABLE:
            try:
                # Stream mode
                camelot_stream_tables = camelot.read_pdf(
                    self.pdf_path,
                    pages=str(page_num),
                    flavor='stream'
                )
                page_data["tables"]["camelot_stream"] = [
                    {
                        "rows": table.df.shape[0],
                        "columns": table.df.shape[1],
                        "accuracy": table.accuracy,
                        "whitespace": table.whitespace,
                        "sample_data": table.df.head(3).to_dict() if not table.df.empty else {}
                    }
                    for table in camelot_stream_tables
                ]
                logger.info(f"Page {page_num}: Extracted {len(camelot_stream_tables)} tables with camelot stream mode")
            except Exception as e:
                logger.error(f"Error extracting tables from page {page_num} with camelot stream mode: {e}")
            
            try:
                # Lattice mode
                camelot_lattice_tables = camelot.read_pdf(
                    self.pdf_path,
                    pages=str(page_num),
                    flavor='lattice'
                )
                page_data["tables"]["camelot_lattice"] = [
                    {
                        "rows": table.df.shape[0],
                        "columns": table.df.shape[1],
                        "accuracy": table.accuracy,
                        "whitespace": table.whitespace,
                        "sample_data": table.df.head(3).to_dict() if not table.df.empty else {}
                    }
                    for table in camelot_lattice_tables
                ]
                logger.info(f"Page {page_num}: Extracted {len(camelot_lattice_tables)} tables with camelot lattice mode")
            except Exception as e:
                logger.error(f"Error extracting tables from page {page_num} with camelot lattice mode: {e}")
        
        # Extract text blocks and lines using pdfplumber if available
        if PDFPLUMBER_AVAILABLE:
            try:
                with pdfplumber.open(self.pdf_path) as pdf:
                    plumber_page = pdf.pages[page_num - 1]  # pdfplumber uses 0-indexed pages
                    
                    # Extract text blocks
                    page_data["text_blocks"] = [
                        {
                            "text": block["text"],
                            "x0": block["x0"],
                            "y0": block["top"],
                            "x1": block["x1"],
                            "y1": block["bottom"],
                            "width": block["width"],
                            "height": block["height"]
                        }
                        for block in plumber_page.extract_words()
                    ]
                    
                    # Extract lines
                    page_data["lines"] = [
                        {
                            "x0": line["x0"],
                            "y0": line["top"],
                            "x1": line["x1"],
                            "y1": line["bottom"],
                            "width": line["width"],
                            "height": line["height"]
                        }
                        for line in plumber_page.lines
                    ]
                    
                    # Generate layout fingerprint
                    page_data["layout_fingerprint"] = self._generate_layout_fingerprint(plumber_page)
                    
                    logger.info(f"Page {page_num}: Extracted {len(page_data['text_blocks'])} text blocks and {len(page_data['lines'])} lines")
            except Exception as e:
                logger.error(f"Error extracting text and lines from page {page_num} with pdfplumber: {e}")
        
        # Determine if extraction was successful
        has_tables = (
            len(page_data["tables"]["tabula_stream"]) > 0 or
            len(page_data["tables"]["tabula_lattice"]) > 0 or
            len(page_data["tables"]["camelot_stream"]) > 0 or
            len(page_data["tables"]["camelot_lattice"]) > 0
        )
        
        has_text = len(page_data["text_blocks"]) > 0
        
        page_data["extraction_success"] = has_tables or has_text
        
        if page_data["extraction_success"]:
            self.successful_pages.append(page_num)
        else:
            self.failed_pages.append(page_num)
        
        return page_data
    
    def _process_tables(self, tables: List) -> List[Dict]:
        """
        Process extracted tables into a standard format for analysis.
        
        Args:
            tables (List): List of pandas DataFrames
            
        Returns:
            List[Dict]: Processed table data
        """
        processed = []
        
        for table in tables:
            if isinstance(table, pd.DataFrame) and not table.empty:
                # Calculate basic statistics
                num_rows, num_cols = table.shape
                non_empty_cells = table.count().sum()
                total_cells = num_rows * num_cols
                fill_rate = non_empty_cells / total_cells if total_cells > 0 else 0
                
                # Check for common patterns in financial statements
                has_date_column = any(
                    table[col].astype(str).str.contains(r'\d{1,2}[/\-\.]\d{1,2}', regex=True).any()
                    for col in table.columns
                )
                
                has_amount_column = any(
                    table[col].astype(str).str.contains(r'\d+\.\d{2}|\d+,\d{2}', regex=True).any()
                    for col in table.columns
                )
                
                processed.append({
                    "rows": num_rows,
                    "columns": num_cols,
                    "fill_rate": fill_rate,
                    "has_date_column": has_date_column,
                    "has_amount_column": has_amount_column,
                    "sample_data": table.head(3).to_dict() if not table.empty else {}
                })
        
        return processed
    
    def _generate_layout_fingerprint(self, page) -> Dict:
        """
        Generate a fingerprint of the page layout for comparison.
        
        Args:
            page: pdfplumber page object
            
        Returns:
            Dict: Layout fingerprint
        """
        if not PDFPLUMBER_AVAILABLE:
            return {}
        
        # Text density and distribution
        words = page.extract_words()
        if not words:
            return {
                "text_density": 0,
                "has_tables": False,
                "has_lines": False,
                "text_columns": 0
            }
        
        # Calculate text density
        page_height = page.height
        page_width = page.width
        page_area = page_height * page_width
        text_area = sum(word["width"] * word["height"] for word in words)
        text_density = text_area / page_area if page_area > 0 else 0
        
        # Detect number of text columns
        x_positions = [word["x0"] for word in words]
        if not x_positions:
            text_columns = 0
        else:
            # Use histogram to detect columns
            hist, bin_edges = np.histogram(x_positions, bins=10)
            peaks = [i for i, val in enumerate(hist) if val > np.mean(hist) * 1.5]
            text_columns = len(peaks) if peaks else 1
        
        return {
            "text_density": text_density,
            "has_tables": len(page.find_tables()) > 0,
            "has_lines": len(page.lines) > 0,
            "text_columns": text_columns
        }
    
    def _compare_pages(self, page_a: Dict, page_b: Dict) -> Dict:
        """
        Compare two pages to find structural differences.
        
        Args:
            page_a (Dict): First page data
            page_b (Dict): Second page data
            
        Returns:
            Dict: Comparison report
        """
        page_a_num = page_a["page"]
        page_b_num = page_b["page"]
        
        # Compare table extraction success
        a_tables_stream = len(page_a["tables"]["tabula_stream"])
        b_tables_stream = len(page_b["tables"]["tabula_stream"])
        
        a_tables_lattice = len(page_a["tables"]["tabula_lattice"])
        b_tables_lattice = len(page_b["tables"]["tabula_lattice"])
        
        # Compare text blocks
        a_text_blocks = len(page_a["text_blocks"])
        b_text_blocks = len(page_b["text_blocks"])
        
        # Compare layout fingerprints
        a_fingerprint = page_a["layout_fingerprint"]
        b_fingerprint = page_b["layout_fingerprint"]
        
        # Identify key differences
        differences = []
        
        if a_tables_stream > 0 and b_tables_stream == 0:
            differences.append(f"Page {page_a_num} has {a_tables_stream} tables with stream mode, Page {page_b_num} has none")
        elif a_tables_stream == 0 and b_tables_stream > 0:
            differences.append(f"Page {page_a_num} has no tables with stream mode, Page {page_b_num} has {b_tables_stream}")
        
        if a_tables_lattice > 0 and b_tables_lattice == 0:
            differences.append(f"Page {page_a_num} has {a_tables_lattice} tables with lattice mode, Page {page_b_num} has none")
        elif a_tables_lattice == 0 and b_tables_lattice > 0:
            differences.append(f"Page {page_a_num} has no tables with lattice mode, Page {page_b_num} has {b_tables_lattice}")
        
        if a_text_blocks > 0 and b_text_blocks == 0:
            differences.append(f"Page {page_a_num} has {a_text_blocks} text blocks, Page {page_b_num} has none")
        elif a_text_blocks == 0 and b_text_blocks > 0:
            differences.append(f"Page {page_a_num} has no text blocks, Page {page_b_num} has {b_text_blocks}")
        
        # Compare fingerprints if available
        if a_fingerprint and b_fingerprint:
            if a_fingerprint.get("text_density", 0) > b_fingerprint.get("text_density", 0) * 1.5:
                differences.append(f"Page {page_a_num} has much higher text density than Page {page_b_num}")
            elif b_fingerprint.get("text_density", 0) > a_fingerprint.get("text_density", 0) * 1.5:
                differences.append(f"Page {page_b_num} has much higher text density than Page {page_a_num}")
            
            if a_fingerprint.get("text_columns", 0) != b_fingerprint.get("text_columns", 0):
                differences.append(f"Page {page_a_num} has {a_fingerprint.get('text_columns', 0)} text columns, Page {page_b_num} has {b_fingerprint.get('text_columns', 0)}")
            
            if a_fingerprint.get("has_tables", False) != b_fingerprint.get("has_tables", False):
                differences.append(f"Page {page_a_num} {'has' if a_fingerprint.get('has_tables', False) else 'does not have'} tables detected by pdfplumber, Page {page_b_num} {'has' if b_fingerprint.get('has_tables', False) else 'does not have'}")
        
        # Generate recommendations based on differences
        recommendations = self._generate_comparison_recommendations(page_a, page_b, differences)
        
        return {
            "page_a": page_a_num,
            "page_b": page_b_num,
            "differences": differences,
            "recommendations": recommendations
        }
    
    def _generate_comparison_recommendations(self, page_a: Dict, page_b: Dict, differences: List[str]) -> List[str]:
        """
        Generate recommendations based on page comparison.
        
        Args:
            page_a (Dict): First page data
            page_b (Dict): Second page data
            differences (List[str]): List of identified differences
            
        Returns:
            List[str]: Recommendations
        """
        recommendations = []
        
        # Check if one page has tables and the other doesn't
        a_has_tables = any(len(tables) > 0 for tables in page_a["tables"].values())
        b_has_tables = any(len(tables) > 0 for tables in page_b["tables"].values())
        
        if a_has_tables and not b_has_tables:
            recommendations.append(f"For Page {page_b['page']}, try text-based extraction instead of table extraction")
        elif not a_has_tables and b_has_tables:
            recommendations.append(f"For Page {page_a['page']}, try text-based extraction instead of table extraction")
        
        # Check if stream mode works better than lattice or vice versa
        a_stream_tables = len(page_a["tables"]["tabula_stream"])
        a_lattice_tables = len(page_a["tables"]["tabula_lattice"])
        b_stream_tables = len(page_b["tables"]["tabula_stream"])
        b_lattice_tables = len(page_b["tables"]["tabula_lattice"])
        
        if a_stream_tables > a_lattice_tables and b_lattice_tables > b_stream_tables:
            recommendations.append(f"Use stream mode for Page {page_a['page']} and lattice mode for Page {page_b['page']}")
        elif a_lattice_tables > a_stream_tables and b_stream_tables > b_lattice_tables:
            recommendations.append(f"Use lattice mode for Page {page_a['page']} and stream mode for Page {page_b['page']}")
        
        # Check if camelot might work better
        if CAMELOT_AVAILABLE:
            a_camelot_tables = len(page_a["tables"]["camelot_stream"]) + len(page_a["tables"]["camelot_lattice"])
            b_camelot_tables = len(page_b["tables"]["camelot_stream"]) + len(page_b["tables"]["camelot_lattice"])
            a_tabula_tables = a_stream_tables + a_lattice_tables
            b_tabula_tables = b_stream_tables + b_lattice_tables
            
            if a_camelot_tables > a_tabula_tables:
                recommendations.append(f"Consider using camelot instead of tabula for Page {page_a['page']}")
            if b_camelot_tables > b_tabula_tables:
                recommendations.append(f"Consider using camelot instead of tabula for Page {page_b['page']}")
        
        # Check for multi-column layout
        if page_a["layout_fingerprint"].get("text_columns", 0) > 1:
            recommendations.append(f"Page {page_a['page']} has a multi-column layout. Consider column detection before extraction.")
        if page_b["layout_fingerprint"].get("text_columns", 0) > 1:
            recommendations.append(f"Page {page_b['page']} has a multi-column layout. Consider column detection before extraction.")
        
        return recommendations
    
    def _generate_recommendations(self) -> List[str]:
        """
        Generate global recommendations based on all page analyses.
        
        Returns:
            List[str]: Global recommendations
        """
        recommendations = []
        
        # Check if some pages work better with different extraction methods
        stream_success_pages = []
        lattice_success_pages = []
        
        for page_data in self.pages:
            if len(page_data["tables"]["tabula_stream"]) > len(page_data["tables"]["tabula_lattice"]):
                stream_success_pages.append(page_data["page"])
            elif len(page_data["tables"]["tabula_lattice"]) > len(page_data["tables"]["tabula_stream"]):
                lattice_success_pages.append(page_data["page"])
        
        if stream_success_pages and lattice_success_pages:
            recommendations.append(f"Use adaptive extraction: stream mode for pages {stream_success_pages}, lattice mode for pages {lattice_success_pages}")
        
        # Check if camelot might work better overall
        if CAMELOT_AVAILABLE:
            camelot_better_pages = []
            
            for page_data in self.pages:
                camelot_tables = len(page_data["tables"]["camelot_stream"]) + len(page_data["tables"]["camelot_lattice"])
                tabula_tables = len(page_data["tables"]["tabula_stream"]) + len(page_data["tables"]["tabula_lattice"])
                
                if camelot_tables > tabula_tables:
                    camelot_better_pages.append(page_data["page"])
            
            if camelot_better_pages:
                recommendations.append(f"Consider using camelot instead of tabula for pages {camelot_better_pages}")
        
        # Check for multi-column layouts
        multi_column_pages = [
            page_data["page"]
            for page_data in self.pages
            if page_data["layout_fingerprint"].get("text_columns", 0) > 1
        ]
        
        if multi_column_pages:
            recommendations.append(f"Pages {multi_column_pages} have multi-column layouts. Consider column detection before extraction.")
        
        # Add general recommendations
        if self.failed_pages:
            recommendations.append("Add fallback text parsing when table extraction fails")
        
        if len(self.pages) > 1:
            recommendations.append("Implement page-specific extraction strategies based on detected layout")
        
        return recommendations


def analyze_pdf_pages(pdf_path: str, pages: List[int] = None, compare_pages: List[Tuple[int, int]] = None) -> Dict:
    """
    Analyze pages in a PDF file and generate a report.
    
    Args:
        pdf_path (str): Path to the PDF file
        pages (List[int], optional): List of page numbers to analyze. If None, analyzes all pages.
        compare_pages (List[Tuple[int, int]], optional): List of page pairs to compare
        
    Returns:
        Dict: Analysis report
    """
    analyzer = PageAnalyzer(pdf_path)
    report = analyzer.analyze(pages)
    
    # Compare specific pages if requested
    if compare_pages:
        for page_a, page_b in compare_pages:
            analyzer.compare_pages(page_a, page_b)
    
    return report


def main():
    """Command-line interface for the page analyzer."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze PDF pages for structural differences')
    parser.add_argument('pdf_path', help='Path to the PDF file')
    parser.add_argument('--pages', help='Comma-separated list of pages to analyze (e.g., "1,5,10")')
    parser.add_argument('--compare', help='Comma-separated list of page pairs to compare (e.g., "1-2,5-10")')
    parser.add_argument('--output', help='Path to save the analysis report (JSON)')
    
    args = parser.parse_args()
    
    # Parse pages to analyze
    pages = None
    if args.pages:
        try:
            pages = [int(p) for p in args.pages.split(',')]
        except ValueError:
            logger.error("Invalid page numbers. Use comma-separated integers (e.g., '1,5,10')")
            return 1
    
    # Parse page pairs to compare
    compare_pages = []
    if args.compare:
        try:
            for pair in args.compare.split(','):
                a, b = map(int, pair.split('-'))
                compare_pages.append((a, b))
        except (ValueError, IndexError):
            logger.error("Invalid page pairs. Use comma-separated pairs (e.g., '1-2,5-10')")
            return 1
    
    # Analyze PDF
    report = analyze_pdf_pages(args.pdf_path, pages, compare_pages)
    
    # Save report if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Analysis report saved to {args.output}")
    else:
        # Print summary to console
        print(f"\nPDF Analysis Summary: {args.pdf_path}")
        print(f"Total pages: {report['total_pages']}")
        print(f"Analyzed pages: {report['analyzed_pages']}")
        print(f"Successful extraction pages: {report['successful_extraction_pages']}")
        print(f"Failed extraction pages: {report['failed_extraction_pages']}")
        
        if report['comparisons']:
            print("\nPage Comparisons:")
            for key, comparison in report['comparisons'].items():
                print(f"\n{key.replace('_', ' ').title()}:")
                for diff in comparison['differences']:
                    print(f"  - {diff}")
                print("  Recommendations:")
                for rec in comparison['recommendations']:
                    print(f"  - {rec}")
        
        print("\nGlobal Recommendations:")
        for rec in report['global_recommendations']:
            print(f"  - {rec}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
