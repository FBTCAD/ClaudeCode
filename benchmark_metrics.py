#!/usr/bin/env python
"""
Benchmark to demonstrate the performance impact of adding metrics=['mae']
to model.compile() in TensorFlow/Keras
"""

import numpy as np
import tensorflow as tf
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

def create_model():
    """Create a simple LSTM model for testing"""
    model = Sequential([
        LSTM(128, input_shape=(50, 1), return_sequences=True),
        LSTM(128, return_sequences=True),
        LSTM(128, return_sequences=False),
        Dense(64),
        Dense(32),
        Dense(5)  # Predict 5 days
    ])
    return model

def generate_data(n_samples=1000, seq_length=50):
    """Generate synthetic training data"""
    X = np.random.randn(n_samples, seq_length, 1).astype(np.float32)
    y = np.random.randn(n_samples, 5).astype(np.float32)
    return X, y

def benchmark_without_metrics(X, y, epochs=10):
    """Benchmark training WITHOUT metrics"""
    print("\n" + "="*70)
    print("BENCHMARK 1: WITHOUT metrics (like cgpt3.py)")
    print("="*70)

    model = create_model()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    print(f"Training on {len(X)} samples for {epochs} epochs...")
    start_time = time.time()

    history = model.fit(
        X, y,
        batch_size=32,
        epochs=epochs,
        verbose=1,
        validation_split=0.2
    )

    elapsed_time = time.time() - start_time

    print(f"\n✓ Training completed in {elapsed_time:.2f} seconds")
    print(f"  Average time per epoch: {elapsed_time/epochs:.2f} seconds")

    return elapsed_time

def benchmark_with_mae(X, y, epochs=10):
    """Benchmark training WITH metrics=['mae']"""
    print("\n" + "="*70)
    print("BENCHMARK 2: WITH metrics=['mae'] (like cgpt2.py)")
    print("="*70)

    model = create_model()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

    print(f"Training on {len(X)} samples for {epochs} epochs...")
    start_time = time.time()

    history = model.fit(
        X, y,
        batch_size=32,
        epochs=epochs,
        verbose=1,
        validation_split=0.2
    )

    elapsed_time = time.time() - start_time

    print(f"\n✓ Training completed in {elapsed_time:.2f} seconds")
    print(f"  Average time per epoch: {elapsed_time/epochs:.2f} seconds")

    return elapsed_time

def main():
    print("\n" + "="*70)
    print("PERFORMANCE BENCHMARK: Impact of metrics=['mae']")
    print("="*70)
    print("\nThis benchmark demonstrates why cgpt2.py is slower than cgpt3.py")
    print("by comparing training with and without the MAE metric.\n")

    # Generate test data
    print("Generating synthetic data...")
    X, y = generate_data(n_samples=1000, seq_length=50)
    print(f"Data shape: X={X.shape}, y={y.shape}")

    epochs = 10

    # Run benchmarks
    time_without_metrics = benchmark_without_metrics(X, y, epochs)
    time_with_mae = benchmark_with_mae(X, y, epochs)

    # Compare results
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON RESULTS")
    print("="*70)
    print(f"WITHOUT metrics (cgpt3.py style): {time_without_metrics:.2f} seconds")
    print(f"WITH metrics=['mae'] (cgpt2.py):  {time_with_mae:.2f} seconds")
    print(f"\nSlowdown: {((time_with_mae / time_without_metrics - 1) * 100):.1f}%")
    print(f"Extra time: {(time_with_mae - time_without_metrics):.2f} seconds")
    print("="*70)

    # Detailed explanation
    print("\n" + "="*70)
    print("WHY DOES metrics=['mae'] SLOW DOWN TRAINING?")
    print("="*70)
    print("""
1. EXTRA COMPUTATION PER BATCH:
   - Loss (MSE): mean((y_true - y_pred)²)     ← Used for backprop
   - Metric (MAE): mean(|y_true - y_pred|)    ← Only for monitoring

2. MEMORY OVERHEAD:
   - MAE values must be stored for each batch
   - Aggregated across training and validation sets
   - Kept in history for plotting/logging

3. I/O OVERHEAD:
   - MAE is displayed in the progress bar on every batch
   - Written to logs and callbacks
   - Slows down the training loop iteration

4. NO GRADIENT BENEFIT:
   - MAE is NOT used for backpropagation
   - It's purely for monitoring/logging
   - All computation is "wasted" for optimization purposes

CONCLUSION:
If you don't need MAE for monitoring, removing it provides free speedup!
    """)

    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    print("""
To speed up cgpt2.py:
1. Remove metrics=['mae'] from line 614
2. If you need MAE for evaluation, compute it ONCE after training:

   # After training
   predictions = model.predict(X_test)
   mae = np.mean(np.abs(y_test - predictions))
   print(f"Test MAE: {mae}")

This gives you the MAE value without slowing down training!
    """)

if __name__ == "__main__":
    main()
