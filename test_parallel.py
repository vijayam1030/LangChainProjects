#!/usr/bin/env python3
"""
Simple test to verify parallel processing is working correctly
"""
import time
import concurrent.futures
import queue

def fake_model(model_name, delay):
    """Simulate a model call with different delays"""
    print(f"🚀 Starting {model_name} (delay: {delay}s)")
    time.sleep(delay)
    result = f"Response from {model_name} after {delay}s"
    print(f"✅ Completed {model_name}")
    return result

def test_parallel_processing():
    """Test that models run in parallel, not sequentially"""
    models = [
        ("Model A", 3),
        ("Model B", 1),
        ("Model C", 2),
        ("Model D", 1.5)
    ]
    
    start_time = time.time()
    results = []
    
    print("🚀 Starting parallel processing test...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(fake_model, name, delay): name 
            for name, delay in models
        }
        
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            elapsed = time.time() - start_time
            print(f"📦 Result received after {elapsed:.1f}s: {result}")
            results.append((elapsed, result))
    
    total_time = time.time() - start_time
    print(f"\n🎉 Total time: {total_time:.1f}s")
    
    # Check if it was truly parallel
    expected_sequential_time = sum(delay for _, delay in models)
    print(f"⏱️ Sequential would have taken: {expected_sequential_time}s")
    
    if total_time < expected_sequential_time * 0.8:
        print("✅ SUCCESS: Models ran in parallel!")
    else:
        print("❌ FAILED: Models seem to have run sequentially")
    
    # Check completion order
    print("\n📋 Completion order:")
    for elapsed, result in results:
        print(f"  {elapsed:.1f}s: {result}")

if __name__ == "__main__":
    test_parallel_processing()
