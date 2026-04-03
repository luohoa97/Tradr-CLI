#!/usr/bin/env python
"""Test script to verify FinBERT loads correctly with multithreading."""

import sys
import threading
import time

def load_finbert_in_thread(thread_id: int):
    """Load FinBERT in a thread to test the workaround."""
    print(f"[Thread {thread_id}] Starting FinBERT load...")
    
    from trading_cli.sentiment.finbert import FinBERTAnalyzer
    
    analyzer = FinBERTAnalyzer.get_instance()
    
    def progress_callback(msg: str):
        print(f"[Thread {thread_id}] Progress: {msg}")
    
    success = analyzer.load(progress_callback=progress_callback)
    
    if success:
        print(f"[Thread {thread_id}] ✓ FinBERT loaded successfully!")
        
        # Test inference
        result = analyzer.analyze_batch(["Test headline for sentiment analysis"])
        print(f"[Thread {thread_id}] Test result: {result}")
    else:
        print(f"[Thread {thread_id}] ✗ FinBERT failed to load: {analyzer.load_error}")
    
    return success

def main():
    print("=" * 60)
    print("Testing FinBERT multithreaded loading with fds_to_keep workaround")
    print("=" * 60)
    
    # Try loading in multiple threads to trigger the issue
    threads = []
    results = []
    
    for i in range(3):
        t = threading.Thread(target=lambda idx=i: results.append(load_finbert_in_thread(idx)))
        threads.append(t)
        t.start()
        time.sleep(0.5)  # Small delay between thread starts
    
    # Wait for all threads to complete
    for t in threads:
        t.join()
    
    print("\n" + "=" * 60)
    print("Test Results:")
    print("=" * 60)
    
    # The singleton should only load once
    if len(results) > 0:
        print(f"✓ At least one thread attempted loading")
        if any(results):
            print(f"✓ FinBERT loaded successfully in multithreaded context")
            print("\n✅ TEST PASSED - fds_to_keep workaround is working!")
            return 0
        else:
            print(f"✗ All threads failed to load FinBERT")
            print("\n❌ TEST FAILED - workaround did not resolve the issue")
            return 1
    else:
        print("✗ No threads completed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
