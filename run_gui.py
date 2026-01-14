#!/usr/bin/env python3
"""
Launch the Boltz-2 GUI Application.

Usage:
    python run_gui.py
    
This will start a local web server and open the GUI in your browser.
"""

import sys
import os

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui.app import main

if __name__ == "__main__":
    print("🧬 Starting Boltz-2 GUI...")
    print("   The GUI will open in your browser automatically.")
    print("   Press Ctrl+C to stop the server.\n")
    main()
