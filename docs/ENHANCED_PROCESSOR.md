# Enhanced Bank Statement Processor

This document describes the enhanced bank statement processing tools that have been developed to improve the extraction of transaction data from PDF bank statements, with a particular focus on balance column detection and multiline transaction handling.

## Overview

The enhanced processor builds upon the existing PDF Statement Reader functionality with the following improvements:

1. **Improved Balance Column Detection**
   - Uses keyword proximity heuristics to find balance columns
   - Analyzes numeric values to identify likely balance columns
   - Detects consistent increasing/decreasing trends in columns

2. **Enhanced Multiline Transaction Handling**
   - Merges continuation rows into previous transaction details
   - Special handling for rows with the same date but only details
   - Preserves "balance brought forward" rows

3. **Balance Value Propagation**
   - Fills missing balance values within date groups
   - Propagates balance values across pages
   - Maintains running balance consistency

4. **Robust Error Handling**
   - Detailed debug information at each processing step
   - Filtering of malformed rows
   - Comprehensive logging

## Available Tools

### 1. Enhanced Statement Processor (`enhanced_statement_processor.py`)

A comprehensive tool that can process individual PDF files or entire directories with improved algorithms.

**Features:**
- Process single files or entire directories
- Generate detailed debug information
- Combine multiple CSV outputs into a single file
- Parallel processing for faster batch operations

**Usage:**
```bash
# Process a single PDF file
python enhanced_statement_processor.py path/to/statement.pdf

# Process all PDFs in a directory
python enhanced_statement_processor.py path/to/statements/

# Process with debug information
python enhanced_statement_processor.py path/to/statements/ --debug

# Process and combine all results into a single CSV
python enhanced_statement_processor.py path/to/statements/ --combine

# Specify output directory
python enhanced_statement_processor.py path/to/statements/ --output-dir path/to/output/

# Control parallel processing
python enhanced_statement_processor.py path/to/statements/ --workers 8
```

### 2. Enhanced Bulk Processor (`enhanced_bulk_process.py`)

A simplified tool focused on batch processing multiple PDF files.

**Usage:**
```bash
# Process all PDFs in a directory
python enhanced_bulk_process.py path/to/statements/

# Process with debug information
python enhanced_bulk_process.py path/to/statements/ --debug

# Specify output directory
python enhanced_bulk_process.py path/to/statements/ --output-dir path/to/output/

# Control parallel processing
python enhanced_bulk_process.py path/to/statements/ --workers 8
```

## Debug Information

When running with the `--debug` flag, the processor generates detailed JSON files in a `debug` subdirectory:

1. `*_table_extraction.json` - Information about tables extracted from the PDF
2. `*_transaction_extraction.json` - Details about transaction extraction from tables
3. `*_cleaning_debug.json` - Information about data cleaning and balance propagation
4. `*_csv_export.json` - Details about the CSV export process

These files are valuable for diagnosing issues with specific PDFs or understanding how the processor is handling different statement formats.

## Command Line Arguments

### Enhanced Statement Processor

```
usage: enhanced_statement_processor.py [-h] [--output-dir OUTPUT_DIR] [--workers WORKERS] [--debug] [--combine] input

Enhanced processing for bank statement PDFs

positional arguments:
  input                 PDF file or directory containing PDF files to process

optional arguments:
  -h, --help            show this help message and exit
  --output-dir OUTPUT_DIR, -o OUTPUT_DIR
                        Output directory for CSV files
  --workers WORKERS, -w WORKERS
                        Maximum number of parallel workers
  --debug, -d           Enable debug mode with detailed logging
  --combine, -c         Combine all CSV files into one after processing
```

### Enhanced Bulk Processor

```
usage: enhanced_bulk_process.py [-h] [--output-dir OUTPUT_DIR] [--workers WORKERS] [--debug] input_dir

Enhanced batch processing for bank statement PDFs

positional arguments:
  input_dir             Directory containing PDF files to process

optional arguments:
  -h, --help            show this help message and exit
  --output-dir OUTPUT_DIR, -o OUTPUT_DIR
                        Output directory for CSV files
  --workers WORKERS, -w WORKERS
                        Maximum number of parallel workers
  --debug, -d           Enable debug mode with detailed logging
```

## Output Format

The processor generates CSV files with the following columns:
- `Date` - Transaction date in YYYY-MM-DD format
- `Details` - Transaction description
- `Debits` - Debit amounts (negative values)
- `Credits` - Credit amounts (positive values)
- `Balance` - Running balance after the transaction

When using the `--combine` option, an additional `Source` column is added to indicate which PDF file each transaction came from.

## Integration with Existing Code

The enhanced processor uses the core modules from the PDF Statement Reader project:
- `modules/table_extractor.py` - For extracting tables from PDFs
- `modules/transaction_extractor.py` - For extracting transactions from tables
- `modules/data_cleaner.py` - For cleaning and processing transaction data
- `modules/csv_exporter.py` - For exporting transactions to CSV

The improvements have been made to these core modules to enhance their functionality while maintaining compatibility with the existing codebase.
