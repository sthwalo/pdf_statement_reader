# Improved Bank Statement Batch Processing Guide

This guide explains how to use the improved batch processing system for extracting transaction data from bank statement PDFs.

## Overview

The improved system consists of two main components:

1. `extract_transactions_focused.py` - A focused transaction extraction module that specifically targets transaction data while ignoring header and footer sections
2. `batch_process_improved.py` - A batch processing script that processes multiple PDFs in parallel

The system is designed to extract only the financial transaction data from bank statements, focusing on the following columns:
- Details
- ServiceFee
- Debits
- Credits
- Date
- Balance

## Key Improvements

- **Transaction-only focus**: The system now better identifies and extracts only transaction tables, ignoring header/footer sections
- **Improved column identification**: Better detection of transaction columns using both header analysis and content patterns
- **Parallel processing**: Process multiple PDFs simultaneously for faster batch processing
- **Combined output option**: Ability to combine all extracted transactions into a single CSV file
- **Better date handling**: Improved date detection and propagation
- **Enhanced logging**: Comprehensive logging for troubleshooting

## Usage

### Single PDF Processing

To process a single PDF file:

```bash
python extract_transactions_focused.py path/to/statement.pdf --output output.csv
```

Options:
- `--output`: Specify the output CSV file path (optional)
- `--password`: Provide password for encrypted PDFs (optional)

### Batch Processing

To process multiple PDF files in a directory:

```bash
python batch_process_improved.py --input input_directory --output output_directory
```

Options:
- `--input` or `-i`: Directory containing PDF files to process (required)
- `--output` or `-o`: Directory where CSV files will be saved (required)
- `--password` or `-p`: Password for encrypted PDFs (optional)
- `--workers` or `-w`: Maximum number of worker processes (optional)
- `--combine` or `-c`: Combine all extracted transactions into a single CSV file (optional)

Example:
```bash
python batch_process_improved.py --input statements/ --output extracted/ --combine
```

## How It Works

1. **Table Extraction**: Uses tabula-py to extract tables from PDFs using both stream and lattice modes
2. **Transaction Table Identification**: Identifies which tables contain transaction data using pattern recognition
3. **Column Mapping**: Maps columns to the expected transaction fields
4. **Transaction Extraction**: Extracts only transaction rows, skipping headers and footers
5. **Data Cleaning**: Cleans and normalizes dates, amounts, and other fields
6. **Deduplication**: Removes duplicate transactions
7. **CSV Output**: Saves transactions to CSV files

## Troubleshooting

- Check the `batch_processing.log` file for detailed information about the processing
- If specific PDFs fail to process, try processing them individually with `extract_transactions_focused.py`
- For PDFs with unusual layouts, you may need to adjust the extraction parameters

## Requirements

- Python 3.6+
- pandas
- tabula-py
- PyPDF2
- Java Runtime Environment (required by tabula-py)

## Installation

```bash
pip install pandas tabula-py PyPDF2
```

Make sure Java is installed on your system, as it's required by tabula-py.
