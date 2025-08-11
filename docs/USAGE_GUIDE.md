# Bank Statement Processor Usage Guide

This guide explains how to use the enhanced bank statement extraction and analysis tools.

## Overview

The bank statement processor consists of three main scripts:

1. **extract_statements_combined.py**: Extracts transaction data from bank statements with proper handling of multi-line descriptions
2. **analyze_combined.py**: Analyzes the extracted CSV files and generates a comprehensive financial report
3. **statement_processor.py**: A unified CLI tool to extract and analyze bank statements in one go

## Using the Statement Processor CLI

The `statement_processor.py` script provides a unified interface for extraction and analysis:

```bash
# Extract data from bank statement PDFs
./statement_processor.py extract /path/to/pdf/file_or_directory

# Analyze extracted CSV files
./statement_processor.py analyze /path/to/csv/directory

# Extract and analyze in one go
./statement_processor.py process /path/to/pdf/file_or_directory
```

### Command Options

#### Extract Command
```bash
./statement_processor.py extract INPUT [--output-dir OUTPUT_DIR]
```
- `INPUT`: PDF file or directory containing PDF files
- `--output-dir`: Directory to save CSV files (default: same as input)

#### Analyze Command
```bash
./statement_processor.py analyze INPUT [--output OUTPUT_FILE]
```
- `INPUT`: Directory containing CSV files
- `--output`: Path to save the report (default: financial_report.txt in input directory)

#### Process Command
```bash
./statement_processor.py process INPUT [--output-dir OUTPUT_DIR]
```
- `INPUT`: PDF file or directory containing PDF files
- `--output-dir`: Directory to save CSV files and report (default: same as input)

## Direct Script Usage

You can also use the individual scripts directly:

```bash
# Extract data from bank statements
python extract_statements_combined.py /path/to/pdf/file_or_directory

# Analyze extracted CSV files
python analyze_combined.py /path/to/csv/directory
```

python extract_tabula_improved.py /Users/sthwalonyoni/pdf_statement_reader/data/fwstatement/xxxxx3753\ \(01\).pdf

python extract_tabula_improved.py /Users/sthwalonyoni/pdf_statement_reader/data/fwstatement/xxxxx3753\ \(01\).pdf --output /Users/sthwalonyoni/pdf_statement_reader/data/fwstatement/xxxxx3753\ \(01\)_transactions.csv

/Users/sthwalonyoni/pdf_statement_reader/.venv/bin/python /Users/sthwalonyoni/pdf_statement_reader/standardbankv1.py /Users/sthwalonyoni/pdf_statement_reader/data/fwstatement/xxxxx3753\ \(01\).pdf


## Features of the Enhanced Extraction

- **Multi-line Description Handling**: Preserves complete transaction descriptions that span multiple lines
- **Complex Column Structure Support**: Handles bank statements with specific column layouts
- **Transaction Boundary Detection**: Correctly identifies where one transaction ends and another begins
- **Clean CSV Output**: Produces standardized CSV files with consistent column names

## Features of the Financial Analysis

- **Transaction Categorization**: Automatically categorizes transactions based on their descriptions
- **Monthly Cash Flow Analysis**: Breaks down income and expenses by month
- **Top Spending Categories**: Identifies where most money is being spent
- **Income Source Analysis**: Highlights the main sources of income
- **Comprehensive Financial Report**: Generates a detailed report with key financial metrics

## Output Files

- **Extraction**: Creates CSV files with the suffix `_combined.csv`
- **Analysis**: Generates a `financial_report.txt` file with the analysis results

## Troubleshooting

### Common Issues

1. **Invalid Date Formats**: The script will handle invalid dates gracefully, but they will be marked as "Invalid date" in the report
2. **Missing Columns**: If certain columns are missing in the PDF, the script will attempt to infer them or use empty values
3. **Categorization Errors**: If transactions are not categorized correctly, you may need to update the category patterns in the `analyze_combined.py` script

### Improving Results

- For better categorization, you can modify the category patterns in the `categorize_transactions` function
- If extraction is missing data, you may need to adjust the column positions in the `extract_statements_combined.py` script
- For more accurate date parsing, ensure the date format in the bank statement is recognized by the script
