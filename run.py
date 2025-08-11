#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PDF Statement Reader - Entry Point

This script serves as the entry point for the PDF Statement Reader application.
It runs the main processor from the core directory.
"""

import os
import sys
import argparse

def main():
    """Main entry point function"""
    # Get the absolute path to the core directory
    core_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'core')
    
    # Add the project root to the path to ensure modules can be found
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # Import the main processor
    sys.path.append(core_dir)
    from pdf_statement_processor import main as processor_main
    
    # Run the main processor
    processor_main()

if __name__ == "__main__":
    # If arguments were passed, forward them to the main processor
    if len(sys.argv) > 1:
        # Get the path to the main processor
        processor_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            'core', 
            'pdf_statement_processor.py'
        )
        
        # Execute the main processor with the arguments
        os.execv(sys.executable, [sys.executable, processor_path] + sys.argv[1:])
    else:
        main()
