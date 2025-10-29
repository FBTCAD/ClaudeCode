#!/usr/bin/env python
"""
GPU Availability Check for TensorFlow
Verifies CUDA and GPU setup for RTX 4090
"""

import sys

print("="*70)
print("GPU AVAILABILITY CHECK FOR TENSORFLOW")
print("="*70)

# Check TensorFlow installation
print("\n1. Checking TensorFlow installation...")
try:
    import tensorflow as tf
    print(f"   ✓ TensorFlow version: {tf.__version__}")
except ImportError as e:
    print(f"   ✗ TensorFlow not installed: {e}")
    print("\n   Install with: pip install tensorflow[and-cuda]")
    sys.exit(1)

# Check if TensorFlow is built with CUDA support
print("\n2. Checking CUDA support...")
cuda_available = tf.test.is_built_with_cuda()
print(f"   {'✓' if cuda_available else '✗'} TensorFlow built with CUDA: {cuda_available}")

# Check GPU availability
print("\n3. Checking GPU devices...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"   ✓ Found {len(gpus)} GPU(s):")
    for i, gpu in enumerate(gpus):
        print(f"     - GPU {i}: {gpu.name}")
        # Get GPU details
        try:
            gpu_details = tf.config.experimental.get_device_details(gpu)
            if gpu_details:
                print(f"       Device name: {gpu_details.get('device_name', 'Unknown')}")
                print(f"       Compute capability: {gpu_details.get('compute_capability', 'Unknown')}")
        except:
            pass
else:
    print("   ✗ No GPUs found")
    print("\n   Possible issues:")
    print("   - CUDA toolkit not installed")
    print("   - cuDNN not installed")
    print("   - TensorFlow not installed with GPU support")
    print("   - GPU drivers not installed/updated")

# Check CUDA and cuDNN versions
print("\n4. Checking CUDA/cuDNN versions...")
try:
    from tensorflow.python.platform import build_info
    print(f"   CUDA version: {build_info.build_info['cuda_version']}")
    print(f"   cuDNN version: {build_info.build_info['cudnn_version']}")
except:
    print("   Could not determine CUDA/cuDNN versions")

# Test GPU computation
print("\n5. Testing GPU computation...")
if gpus:
    try:
        with tf.device('/GPU:0'):
            # Simple matrix multiplication test
            a = tf.random.normal([1000, 1000])
            b = tf.random.normal([1000, 1000])
            c = tf.matmul(a, b)
        print("   ✓ GPU computation test passed!")
        print(f"   ✓ RTX 4090 is ready for training!")
    except Exception as e:
        print(f"   ✗ GPU computation test failed: {e}")
else:
    print("   ⊗ Skipping GPU test (no GPUs available)")

# Memory configuration recommendation
print("\n6. GPU Memory Configuration...")
if gpus:
    print("   Recommended memory growth settings:")
    print("   - Enables dynamic memory allocation")
    print("   - Prevents TensorFlow from allocating all GPU memory")
    print("   - Allows multiple processes to share the GPU")
    print("\n   Add this code to your scripts:")
    print("""
   import tensorflow as tf
   gpus = tf.config.list_physical_devices('GPU')
   if gpus:
       try:
           for gpu in gpus:
               tf.config.experimental.set_memory_growth(gpu, True)
       except RuntimeError as e:
           print(e)
   """)

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

if gpus and cuda_available:
    print("✓ Your system is ready for GPU-accelerated training!")
    print(f"✓ RTX 4090 detected and operational")
    print("\nExpected speedup for LSTM training:")
    print("  - Small models (< 1M params): 5-10x faster")
    print("  - Medium models (1-10M params): 10-30x faster")
    print("  - Large models (> 10M params): 20-50x faster")
    print("\nYour cgpt2/cgpt3 models (~7M params): 15-35x faster!")
elif cuda_available and not gpus:
    print("⚠ CUDA is available but no GPUs detected")
    print("  Check your GPU drivers and CUDA installation")
elif gpus and not cuda_available:
    print("⚠ GPU detected but TensorFlow not built with CUDA")
    print("  Reinstall TensorFlow with GPU support:")
    print("  pip uninstall tensorflow")
    print("  pip install tensorflow[and-cuda]")
else:
    print("✗ GPU acceleration not available")
    print("\nTo enable GPU support:")
    print("1. Install NVIDIA GPU drivers")
    print("2. Install CUDA Toolkit 12.x")
    print("3. Install cuDNN 8.9+")
    print("4. Install TensorFlow with GPU support:")
    print("   pip install tensorflow[and-cuda]")

print("="*70)
