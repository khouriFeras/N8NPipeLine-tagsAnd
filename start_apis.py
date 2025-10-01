#!/usr/bin/env python3
"""
Start both translation and tagging APIs on different ports
"""

import os
import sys
import threading
import time

def start_translation_api():
    """Start translation API on port 5000"""
    os.environ['API_MODE'] = 'translation'
    from unified_api import app
    print("Starting Translation API on port 5000...")
    app.run(host='0.0.0.0', port=5000, debug=False)

def start_tagging_api():
    """Start tagging API on port 5001"""
    os.environ['API_MODE'] = 'tagging'
    from unified_api import app
    print("Starting Tagging API on port 5001...")
    app.run(host='0.0.0.0', port=5001, debug=False)

if __name__ == '__main__':
    print("Starting both Translation and Tagging APIs...")
    
    # Start translation API in a separate thread
    translation_thread = threading.Thread(target=start_translation_api)
    translation_thread.daemon = True
    translation_thread.start()
    
    # Wait a moment for the first API to start
    time.sleep(2)
    
    # Start tagging API in the main thread
    start_tagging_api()
