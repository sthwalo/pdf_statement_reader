# Modular Bank Statement PDF Extraction System

This documentation explains the modular system for extracting transaction data from bank statement PDFs. The system is designed with a focus on debugging, logging, and modular components that can be tested independently.

## System Architecture

The system is divided into the following modules:

1. **Table Extractor** (`modules/table_extractor.py`): Extracts tables from PDFs using tabula with stream and lattice modes, including fallback per-page extraction and special page handling.

2. **Transaction Identifier** (`modules/transaction_identifier.py`): Detects transaction tables and maps columns to transaction fields using heuristics.

3. **Transaction Extractor** (`modules/transaction_extractor.py`): Extracts transaction data from identified tables based on column mappings.

4. **Data Cleaner** (`modules/data_cleaner.py`): Cleans and normalizes transaction data, including date formatting, numeric value cleaning, and data deduplication.

5. **CSV Exporter** (`modules/csv_exporter.py`): Saves transaction data to CSV files and combines multiple CSV files into one.

6. **Batch Processor** (`modules/batch_processor.py`): Coordinates the entire extraction process for multiple PDF files.

7. **Main Script** (`extract_statements.py`): Provides a simple command-line interface to use the modular system.

## Usage

### Basic Usage

```bash
# Process a single PDF file
python extract_statements.py single --pdf statement.pdf --output output/

# Process all PDFs in a directory
python extract_statements.py batch --input statements/ --output output/

# Process with debug information
python extract_statements.py batch --input statements/ --output output/ --debug

# Combine all extracted CSVs into one file
python extract_statements.py batch --input statements/ --output output/ --combine
```

### Advanced Usage

```bash
# Extract tables only
python extract_statements.py extract-tables --pdf statement.pdf --output tables.json

# Identify transaction tables only
python extract_statements.py identify --pdf statement.pdf --output identification.json --debug
```

## Module Details

### Table Extractor

The table extractor module is responsible for extracting tables from PDFs using tabula-py. It includes:

- Multiple extraction modes (stream and lattice)
- Fallback to per-page extraction if whole-document extraction fails
- Special handling for problematic pages
- Detailed logging and debug information

```python
from modules.table_extractor import extract_tables_from_pdf

# Basic usage
tables = extract_tables_from_pdf('statement.pdf')

# With debug information
tables, debug_info = extract_tables_from_pdf('statement.pdf', debug=True)
```

### Transaction Identifier

The transaction identifier module detects transaction tables and maps columns to transaction fields. It includes:

- Heuristics for detecting transaction tables based on content patterns
- Column mapping based on header row detection and content pattern heuristics
- Detailed logging and debug information

```python
from modules.transaction_identifier import is_transaction_table, identify_columns

# Check if a table contains transactions
is_tx = is_transaction_table(table)

# Identify column mappings
column_mapping, header_row = identify_columns(table)
```

### Transaction Extractor

The transaction extractor module extracts transaction data from identified tables. It includes:

- Extraction based on column mappings
- Handling of JSON-like strings
- Filtering of non-transaction rows
- Detailed logging and debug information

```python
from modules.transaction_extractor import extract_transactions_from_table, extract_transactions_from_pdf

# Extract transactions from a single table
transactions = extract_transactions_from_table(table, column_mapping, header_row)

# Extract transactions from all tables in a PDF
transactions = extract_transactions_from_pdf(tables)
```

### Data Cleaner

The data cleaner module cleans and normalizes transaction data. It includes:

- Date formatting
- Numeric value cleaning
- Date propagation
- Transaction deduplication
- Detailed logging and debug information

```python
from modules.data_cleaner import clean_transactions, deduplicate_transactions, propagate_dates

# Clean transactions
cleaned_transactions = clean_transactions(transactions)

# Deduplicate transactions
deduplicated_transactions = deduplicate_transactions(cleaned_transactions)

# Propagate dates
transactions_with_dates = propagate_dates(transactions)
```

### CSV Exporter

The CSV exporter module saves transaction data to CSV files. It includes:

- CSV file creation
- Combining multiple CSV files into one
- Detailed logging and debug information

```python
from modules.csv_exporter import save_to_csv, combine_csv_files

# Save transactions to CSV
save_to_csv(transactions, 'output.csv')

# Combine multiple CSV files
combine_csv_files(['file1.csv', 'file2.csv'], 'combined.csv')
```

### Batch Processor

The batch processor module coordinates the entire extraction process. It includes:

- Single PDF processing
- Directory processing with parallel execution
- CSV combining
- Detailed logging and debug information

```python
from modules.batch_processor import process_single_pdf, process_directory

# Process a single PDF
result = process_single_pdf('statement.pdf', 'output/')

# Process a directory of PDFs
result = process_directory('statements/', 'output/', combine=True)
```

## Debugging

The system includes extensive debugging capabilities. When the `--debug` flag is used, each module generates detailed debug information that is saved to JSON files in the `debug` directory.

Debug information includes:

- Table extraction details
- Transaction table identification heuristics
- Column mapping analysis
- Transaction extraction details
- Data cleaning steps
- CSV export details

## Error Handling

The system includes robust error handling at each stage of the process. Errors are logged and propagated up the call stack, allowing for detailed error reporting.

## Extending the System

The modular design makes it easy to extend the system. To add support for a new bank statement format:

1. Update the transaction identification heuristics in `transaction_identifier.py`
2. Add new date or numeric value cleaning functions in `data_cleaner.py`
3. Update the column mapping logic in `transaction_identifier.py`

## Requirements

- Python 3.x
- pandas
- tabula-py
- PyPDF2
- Java Runtime Environment (for tabula-java)

## Troubleshooting

### Common Issues

1. **No tables extracted**: Check if the PDF is scanned or has security restrictions.
2. **No transaction tables identified**: Check the transaction identification heuristics.
3. **Incorrect column mappings**: Check the column mapping logic.
4. **Date parsing errors**: Check the date cleaning functions.

### Debugging Tips

1. Use the `--debug` flag to generate detailed debug information.
2. Check the log file for error messages.
3. Use the `extract-tables` and `identify` commands to debug specific stages of the process.
4. Examine the debug JSON files for detailed information about each stage.

## Conclusion

This modular system provides a robust and flexible approach to extracting transaction data from bank statement PDFs. The focus on debugging, logging, and modular components makes it easy to identify and fix issues, and to extend the system to support new bank statement formats.
