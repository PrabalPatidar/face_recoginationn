#!/usr/bin/env python3
"""
Main entry point for the Face Scan Project.
"""

import sys
import argparse
import logging
from pathlib import Path

# Add config to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "config"))

from config.settings import config
from .app import create_app
from .gui.main_window import MainWindow


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/app.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def run_web_app(host='0.0.0.0', port=5000, debug=False):
    """Run the web application."""
    logger = logging.getLogger(__name__)
    logger.info("Starting Face Scan web application")
    
    app = create_app()
    app.run(host=host, port=port, debug=debug)


def run_gui_app():
    """Run the GUI application."""
    logger = logging.getLogger(__name__)
    logger.info("Starting Face Scan GUI application")
    
    try:
        import tkinter as tk
        from tkinter import messagebox
        
        root = tk.Tk()
        app = MainWindow(root)
        root.mainloop()
        
    except ImportError:
        logger.error("GUI dependencies not available. Please install tkinter.")
        sys.exit(1)


def run_cli_app():
    """Run the CLI application."""
    logger = logging.getLogger(__name__)
    logger.info("Starting Face Scan CLI application")
    
    # CLI functionality would be implemented here
    print("Face Scan CLI - Command line interface")
    print("Use --help for available commands")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Face Scan Project')
    parser.add_argument('--mode', choices=['web', 'gui', 'cli'], default='web',
                       help='Application mode (default: web)')
    parser.add_argument('--host', default='0.0.0.0', help='Host for web server')
    parser.add_argument('--port', type=int, default=5000, help='Port for web server')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--config', help='Configuration file path')
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging()
    logger.info(f"Starting Face Scan Project in {args.mode} mode")
    
    try:
        if args.mode == 'web':
            run_web_app(args.host, args.port, args.debug)
        elif args.mode == 'gui':
            run_gui_app()
        elif args.mode == 'cli':
            run_cli_app()
            
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
