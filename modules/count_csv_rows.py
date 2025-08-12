#!/usr/bin/env python3
import os
import pandas as pd
import glob

def count_csv_rows():
    """
    Count rows in each CSV file and compare with combined CSV.
    """
    input_dir = os.path.join("data", "output", "camelot")
    combined_file = os.path.join(input_dir, "combined_transactions.csv")
    
    # Get all individual CSV files
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    csv_files = [f for f in csv_files if not os.path.basename(f).startswith("combined_")]
    
    if not csv_files:
        print("No CSV files found in the directory.")
        return
    
    print(f"Found {len(csv_files)} CSV files to analyze.")
    
    # Count rows in each file
    total_rows = 0
    file_counts = {}
    
    for file in sorted(csv_files):
        try:
            df = pd.read_csv(file)
            row_count = len(df)
            file_name = os.path.basename(file)
            file_counts[file_name] = row_count
            total_rows += row_count
            print(f"{file_name}: {row_count} rows")
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
    
    print(f"\nTotal rows across all individual files: {total_rows}")
    
    # Check combined file
    if os.path.exists(combined_file):
        try:
            combined_df = pd.read_csv(combined_file)
            combined_rows = len(combined_df)
            print(f"\nCombined CSV file: {combined_rows} rows")
            
            if combined_rows < total_rows:
                print(f"WARNING: Combined file has {total_rows - combined_rows} fewer rows than the sum of individual files!")
            elif combined_rows > total_rows:
                print(f"WARNING: Combined file has {combined_rows - total_rows} more rows than the sum of individual files!")
            else:
                print("✅ Combined file has the same number of rows as the sum of individual files.")
                
            # Check for source file distribution in combined file
            if 'source_file' in combined_df.columns:
                print("\nRows by source file in combined CSV:")
                source_counts = combined_df['source_file'].value_counts().to_dict()
                
                # Compare with individual file counts
                print("\nComparison of row counts:")
                print("File Name | Individual | Combined | Difference")
                print("---------|------------|----------|------------")
                for file_name, count in file_counts.items():
                    combined_count = source_counts.get(file_name, 0)
                    diff = count - combined_count
                    status = "✅" if diff == 0 else "❌"
                    print(f"{file_name} | {count} | {combined_count} | {diff} {status}")
        except Exception as e:
            print(f"Error processing combined file: {str(e)}")
    else:
        print("\nCombined CSV file not found.")

if __name__ == "__main__":
    count_csv_rows()
