#!/usr/bin/env python
"""
Combined Stock Price Prediction Script - Fixed Version (NaN Prevention)
========================================================================

Fixed issues:
- Proper handling of create_dataset 3-value returns
- Added bounds checking for array indices
- Safe inverse transform for all predictions
- Added model.summary() display
- Better error handling and data validation
- Consistent index handling for plotting

Features:
- Downloads stock data from yfinance with retry logic and rate limiting
- Robust CSV caching with proper date/timezone handling
- Optionally applies moving average smoothing
- Two training modes: single and multiple
- Two network architectures: simple and multires
- Proper overlay of predictions on actual data

Usage example:
    python newcgpt1_fixed.py --mode multiple --architecture multires --pred_horizon 5 --multi_seq_lengths "15,30,45,55" --use_smoothing --smooth_window 100 --filter_start 2020-01-01 --symbols MSFT
"""

import os
import warnings
import logging 
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Bidirectional, LSTM, TimeDistributed, Concatenate, Attention, GaussianNoise, ReLU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import yfinance as yf
from functools import partial
from datetime import datetime
import time
import random
import json

# Set up logging configuration
logging.basicConfig(
    level=logging.WARNING,
    format='%(message)s'
)

# Suppress various warnings and info messages
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR

# Suppress absl logging before importing TensorFlow
try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
except ImportError:
    pass

# ====================
# GPU Configuration for RTX 4090
# ====================

def setup_gpu():
    """Configure GPU for optimal performance on RTX 4090"""
    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        try:
            # Enable memory growth to prevent TensorFlow from allocating all GPU memory
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            print(f"\n{'='*70}")
            print("GPU CONFIGURATION")
            print(f"{'='*70}")
            print(f"✓ Found {len(gpus)} GPU(s)")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")

            # Enable mixed precision for RTX 4090 (uses Tensor Cores for faster training)
            # This can provide 2-3x speedup on RTX 4090
            try:
                from tensorflow.keras import mixed_precision
                policy = mixed_precision.Policy('mixed_float16')
                mixed_precision.set_global_policy(policy)
                print(f"✓ Mixed precision enabled (float16) - optimized for RTX 4090 Tensor Cores")
                print(f"  Expected additional speedup: 2-3x on top of GPU acceleration")
            except Exception as e:
                print(f"⚠ Mixed precision not enabled: {e}")

            # Enable XLA compilation for additional performance
            # tf.config.optimizer.set_jit(True)
            # print(f"✓ XLA JIT compilation enabled")

            print(f"{'='*70}\n")
            return True

        except RuntimeError as e:
            print(f"⚠ GPU configuration error: {e}")
            return False
    else:
        print("\n⚠ No GPU detected - training will use CPU (much slower)")
        print("  For RTX 4090 support, ensure:")
        print("  1. NVIDIA drivers are installed")
        print("  2. CUDA Toolkit is installed")
        print("  3. TensorFlow with GPU support: pip install tensorflow[and-cuda]\n")
        return False

# Setup GPU before any model operations
GPU_AVAILABLE = setup_gpu()

# ====================
# Configuration
# ====================

# Rate limiting configuration
RATE_LIMIT_CALLS_PER_MIN = 60
RATE_LIMIT_INTERVAL = 60.0
MAX_RETRY_ATTEMPTS = 3
RETRY_BASE_WAIT = 5
RETRY_MAX_WAIT = 30

# Storage configuration
CACHE_DIR = "stock_cache"
NAMES_CACHE_FILE = os.path.join(CACHE_DIR, "names_cache.json")

# ====================
# Utility Functions
# ====================

def ensure_cache_dir():
    """Create cache directory if it doesn't exist"""
    os.makedirs(CACHE_DIR, exist_ok=True)

def safe_inverse_transform(scaler, data, data_name="data"):
    """Safely inverse transform data, handling shape issues to prevent NaN"""
    try:
        if data is None:
            print(f"[WARNING] {data_name} is None, skipping inverse transform")
            return np.array([])
        
        original_shape = data.shape
        print(f"[DEBUG] {data_name} shape before inverse_transform: {original_shape}")
        
        # Handle different dimensions
        if len(data.shape) == 3:
            # 3D array - reshape to 2D for transformation
            n_samples = data.shape[0]
            n_features = data.shape[1] * data.shape[2] if data.shape[2] > 1 else data.shape[1]
            data_2d = data.reshape(n_samples, n_features)
            result = scaler.inverse_transform(data_2d)
            # For predictions, we usually want (n_samples, n_predictions)
            if data.shape[2] == 1:
                result = result.reshape(n_samples, data.shape[1])
        elif len(data.shape) == 2:
            # 2D array - transform directly
            result = scaler.inverse_transform(data)
        elif len(data.shape) == 1:
            # 1D array - reshape to 2D, transform, then flatten
            data_2d = data.reshape(-1, 1)
            result = scaler.inverse_transform(data_2d).flatten()
        else:
            print(f"[ERROR] Unexpected shape for {data_name}: {original_shape}")
            return data
        
        # Check for NaN in result
        if np.any(np.isnan(result)):
            print(f"[WARNING] NaN detected in {data_name} after inverse transform!")
            result = np.nan_to_num(result, nan=0.0)
        
        print(f"[DEBUG] {data_name} shape after inverse_transform: {result.shape}")
        return result
        
    except Exception as e:
        print(f"[ERROR] Failed to inverse transform {data_name}: {e}")
        return data

def normalize_df_index(df):
    """
    Normalize DataFrame index to ensure consistent datetime handling.
    - Ensure DatetimeIndex
    - Handle timezone conversion
    - Sort and deduplicate
    """
    if df is None or df.empty:
        return df
    
    df = df.copy()
    
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        # Check if 'Date' column exists
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        else:
            # Try to convert existing index to datetime
            try:
                df.index = pd.to_datetime(df.index)
            except Exception:
                return df
    
    # Handle timezone - convert to UTC then remove timezone info
    try:
        if df.index.tz is not None:
            df.index = df.index.tz_convert('UTC').tz_localize(None)
        else:
            # If no timezone, try to localize to UTC then remove
            try:
                df.index = df.index.tz_localize('UTC').tz_localize(None)
            except:
                # If already tz-naive, leave as is
                pass
    except Exception:
        # If any timezone operation fails, continue without it
        pass
    
    # Sort index and remove duplicates
    try:
        df = df[~df.index.duplicated(keep='first')].sort_index()
    except Exception:
        pass
    
    # Remove any NaN values
    df = df.dropna()
    
    return df

def load_cached_stock(symbol, filter_start=None, filter_end=None):
    """
    Load cached stock data from CSV if available.
    Returns filtered data if filter dates are provided.
    """
    csv_path = os.path.join(CACHE_DIR, f"{symbol}.csv")
    
    if os.path.exists(csv_path):
        try:
            # Read with proper index handling
            df = pd.read_csv(csv_path, parse_dates=True, index_col=0)
            df = normalize_df_index(df)
            
            if df is not None and not df.empty and 'Close' in df.columns:
                # Apply date filter if specified
                if filter_start:
                    filter_start_date = pd.Timestamp(filter_start)
                    if filter_end:
                        filter_end_date = pd.Timestamp(filter_end)
                        df = df[(df.index >= filter_start_date) & (df.index <= filter_end_date)]
                    else:
                        df = df[df.index >= filter_start_date]
                
                print(f"  [OK] Loaded cached data for {symbol} ({len(df)} rows after filtering)")
                return df
        except Exception as e:
            print(f"  [WARNING] Failed to load cached CSV for {symbol}: {e}")
    
    return None

def save_stock_cache(symbol, df):
    """Save stock data to CSV cache"""
    ensure_cache_dir()
    csv_path = os.path.join(CACHE_DIR, f"{symbol}.csv")
    
    try:
        df = normalize_df_index(df)
        df.to_csv(csv_path)
        print(f"  [OK] Saved {symbol} data to cache: {csv_path}")
        return True
    except Exception as e:
        print(f"  [WARNING] Failed to save cache for {symbol}: {e}")
        return False

def load_names_cache():
    """Load cached stock names from JSON"""
    try:
        if os.path.exists(NAMES_CACHE_FILE):
            with open(NAMES_CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def save_names_cache(cache):
    """Save stock names cache to JSON"""
    ensure_cache_dir()
    try:
        with open(NAMES_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        print(f"  [WARNING] Failed saving names cache: {e}")

def rate_limit_pause(min_pause=0.5):
    """Simple rate limiting pause"""
    time.sleep(min_pause + random.uniform(0, 0.5))

def download_stock_with_retry(symbol, start_date='2010-01-01', max_retries=MAX_RETRY_ATTEMPTS):
    """
    Download stock data with retry logic.
    Returns DataFrame or None if failed.
    """
    for attempt in range(max_retries):
        try:
            rate_limit_pause()
            
            print(f"  -> Downloading {symbol} (attempt {attempt + 1}/{max_retries})...")
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, period='max', auto_adjust=False)
            
            if df is not None and not df.empty:
                df = normalize_df_index(df)
                print(f"  [OK] Successfully downloaded {symbol} ({len(df)} rows)")
                return df
            else:
                raise Exception("Empty data returned")
                
        except Exception as e:
            print(f"  [ERROR] Attempt {attempt + 1} failed for {symbol}: {e}")
            
            if attempt < max_retries - 1:
                wait_time = min(RETRY_BASE_WAIT * (2 ** attempt), RETRY_MAX_WAIT)
                wait_time += random.uniform(0, 2)
                print(f"  -> Waiting {wait_time:.1f} seconds before retry...")
                time.sleep(wait_time)
    
    print(f"  [FAILED] Could not download {symbol} after {max_retries} attempts")
    return None

def download_and_filter_data(symbol, results_dir, filter_start, filter_end=None, force_download=False):
    """
    Enhanced download and filter function with proper caching and error handling.
    """
    ensure_cache_dir()
    
    # Try to load from cache first (unless force_download is True)
    if not force_download:
        df = load_cached_stock(symbol, filter_start, filter_end)
        if df is not None:
            return df
    
    # Download fresh data
    print(f"Downloading fresh data for {symbol}...")
    df = download_stock_with_retry(symbol, start_date='2010-01-01')
    
    if df is None:
        print(f"[ERROR] Failed to download data for {symbol}")
        return None
    
    # Save to cache
    save_stock_cache(symbol, df)
    
    # Apply filtering
    if filter_start:
        filter_start_date = pd.Timestamp(filter_start)
        if filter_end:
            filter_end_date = pd.Timestamp(filter_end)
            df = df[(df.index >= filter_start_date) & (df.index <= filter_end_date)]
        else:
            df = df[df.index >= filter_start_date]
        
        print(f"  [OK] Filtered data for {symbol}: {len(df)} rows from {filter_start} to {filter_end or 'present'}")
    
    return df

def get_stock_name(symbol, names_cache=None):
    """Get stock name with caching"""
    if names_cache is not None and symbol in names_cache:
        return names_cache[symbol]
    
    try:
        rate_limit_pause()
        ticker = yf.Ticker(symbol)
        info = ticker.info
        name = info.get('longName') or info.get('shortName') or info.get('name') or symbol
        
        if names_cache is not None:
            names_cache[symbol] = name
            save_names_cache(names_cache)
        
        return name
    except Exception:
        return symbol

def create_dataset(dataset, time_step, num_predict):
    """
    Creates input sequences (X) and corresponding targets (Y) from dataset.
    Returns X, Y, and the indices of Y relative to the original array.
    
    FIXED: Consistent return of 3 values with proper error handling
    """
    X, Y, Y_indices = [], [], []
    
    # Validate inputs
    if len(dataset) < time_step + num_predict:
        print(f"[WARNING] Dataset too small: {len(dataset)} < {time_step + num_predict}")
        return np.array([]), np.array([]), np.array([])
    
    for i in range(len(dataset) - time_step - num_predict + 1):
        a = dataset[i:(i + time_step)]
        X.append(a)
        Y.append(dataset[i + time_step : i + time_step + num_predict])
        Y_indices.append(i + time_step)
    
    return np.array(X), np.array(Y), np.array(Y_indices)

# ====================
# Model Building Functions
# ====================

def build_simple_model(input_shape, num_predict):
    """
    Builds a single-input (simple) model with bidirectional LSTM layers and an attention block.
    KEPT: Zero regularization as in original (since claude.py works with it)
    """
    actf = {'activation': 'tanh', 'recurrent_activation': 'sigmoid'}
    reg_params = {
        'kernel_regularizer': tf.keras.regularizers.l2(0.001),
        'recurrent_regularizer': tf.keras.regularizers.l2(0.001),
        'dropout': 0.0,
        'recurrent_dropout': 0.0
    }
    LSTM_partial = partial(LSTM, units=128, return_sequences=True, **actf, **reg_params)
    
    def build_layer(inp0, inp1):
        noisy = GaussianNoise(0.1)(inp0)
        noisy = ReLU()(noisy)
        x = Bidirectional(LSTM_partial(), merge_mode='concat')(inp1)
        x = Concatenate()([noisy, x])
        return x

    def tdatt(x_in):
        x = Bidirectional(LSTM_partial(), merge_mode='concat')(x_in)
        query = TimeDistributed(Dense(256))(x)
        att_out = Attention()([query, x, x])
        att_out = TimeDistributed(Dense(64))(att_out)
        att_out = TimeDistributed(Dense(1))(att_out)
        return att_out

    inputs = Input(shape=input_shape)
    x = build_layer(inputs, inputs)
    x = build_layer(inputs, x)
    x = build_layer(inputs, x)
    x = build_layer(inputs, x)
    
    x = tdatt(x)
    x = Concatenate()([inputs, x])
    x1 = LSTM_partial(return_sequences=False)(x)
    
    x = Dense(128)(x1)
    x = Dense(64)(x)
    x = Dense(32)(x)
    x = Dense(16)(x)
    x = Dense(8)(x)
    x = Dense(4)(x)
    x = Concatenate()([x1, x])
    outputs = Dense(num_predict)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

def build_multires_model(input_shapes, num_predict):
    """
    Builds a multi-input (multi-resolution) model that processes different sequence lengths.
    KEPT: Zero regularization as in original
    """
    actf = {'activation': 'tanh', 'recurrent_activation': 'sigmoid'}
    reg_params = {
        'kernel_regularizer': tf.keras.regularizers.l2(0.001),
        'recurrent_regularizer': tf.keras.regularizers.l2(0.001),
        'dropout': 0.01,
        'recurrent_dropout': 0.01
    }
    LSTM_partial = partial(LSTM, units=128, return_sequences=True, **actf, **reg_params)
    
    def build_branch(inp):
        x = Bidirectional(LSTM_partial(), merge_mode='concat')(inp)
        x = Bidirectional(LSTM_partial(), merge_mode='concat')(x)
        x = Bidirectional(LSTM_partial(), merge_mode='concat')(x)
        x = Bidirectional(LSTM_partial(), merge_mode='concat')(x)
        query = TimeDistributed(Dense(256))(x)
        att_out = Attention()([query, x, x])
        att_out = TimeDistributed(Dense(64))(att_out)
        att_out = TimeDistributed(Dense(1))(att_out)
        x = Concatenate()([inp, att_out])
        x = LSTM_partial(return_sequences=False)(x)
        return x

    inputs = [Input(shape=ishape) for ishape in input_shapes]
    branches = [build_branch(inp) for inp in inputs]
    
    x = Concatenate()(branches)
    x = Dense(128)(x)
    x = Dense(64)(x)
    x = Dense(32)(x)
    x = Dense(16)(x)
    x = Dense(8)(x)
    x = Dense(4)(x)
    x = Concatenate()([x] + branches)
    outputs = Dense(num_predict)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# ====================
# Main Script
# ====================

def main(args):
    # Configuration Options
    MODE = args.mode                  # 'single' or 'multiple'
    ARCH = args.architecture          # 'simple' or 'multires'
    NUM_PRED_DAYS = args.pred_horizon # 1 to 5
    SEQ_LENGTH = args.sequence_length # used if MODE == 'single'
    MULTI_SEQ_LENGTHS = [int(x) for x in args.multi_seq_lengths.split(',')]
    USE_SMOOTHING = args.use_smoothing
    SMOOTH_WINDOW = args.smooth_window
    FILTER_START = args.filter_start
    FILTER_END = args.filter_end
    SYMBOLS = args.symbols.split(',')
    RESULTS_DIR = args.results_dir
    FORCE_DOWNLOAD = args.force_download

    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Load names cache
    names_cache = load_names_cache()
    
    # Print configuration
    print("\n" + "="*60)
    print("STOCK PRICE PREDICTION - CONFIGURATION")
    print("="*60)
    print(f"Mode: {MODE}")
    print(f"Architecture: {ARCH}")
    print(f"Prediction Horizon: {NUM_PRED_DAYS} days")
    print(f"Sequence Lengths: {MULTI_SEQ_LENGTHS}")
    print(f"Smoothing: {'Yes' if USE_SMOOTHING else 'No'}" + (f" (window={SMOOTH_WINDOW})" if USE_SMOOTHING else ""))
    print(f"Date Range: {FILTER_START} to {FILTER_END or 'present'}")
    print(f"Force Download: {FORCE_DOWNLOAD}")
    print(f"Symbols: {', '.join(SYMBOLS)}")
    print(f"Results Directory: {RESULTS_DIR}")
    print(f"Cache Directory: {CACHE_DIR}")
    print("="*60 + "\n")

    # Process each symbol
    successful_symbols = []
    failed_symbols = []
    
    for symbol in SYMBOLS:
        print(f"\n{'='*50}")
        print(f"Processing: {symbol}")
        print(f"{'='*50}")
        
        try:
            # Get stock name
            stock_name = get_stock_name(symbol, names_cache)
            print(f"Stock Name: {stock_name}")
            
            # Download/load and filter data
            df = download_and_filter_data(
                symbol, 
                RESULTS_DIR, 
                FILTER_START, 
                FILTER_END,
                force_download=FORCE_DOWNLOAD
            )
            
            if df is None or df.empty:
                print(f"[ERROR] No data available for {symbol}")
                failed_symbols.append(symbol)
                continue
            
            # FIXED: Data validation
            min_required = max(MULTI_SEQ_LENGTHS) + NUM_PRED_DAYS + 10
            if len(df) < min_required:
                print(f"[ERROR] Not enough data! Have {len(df)} rows, need at least {min_required}")
                failed_symbols.append(symbol)
                continue
            
            # Optionally apply smoothing (moving average)
            if USE_SMOOTHING:
                df_smoothed = df[['Close']].rolling(window=SMOOTH_WINDOW, min_periods=1).mean()
                print(f"Applied {SMOOTH_WINDOW}-day moving average smoothing")
            else:
                df_smoothed = df[['Close']]
            
            # Check for NaN in data
            if df_smoothed.isnull().any().any():
                print(f"[WARNING] NaN detected in data, filling with forward fill")
                df_smoothed = df_smoothed.fillna(method='ffill').fillna(method='bfill')
            
            # Normalize the data
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(df_smoothed.values.reshape(-1, 1))
            print(f"Data normalized to range [0, 1]")
            
            # Train/Test Split
            time_step = SEQ_LENGTH if MODE == 'single' else max(MULTI_SEQ_LENGTHS)
            train_size = int(len(scaled_data) * 0.8)
            train_set = scaled_data[:train_size]
            test_set = scaled_data[train_size - time_step:]
            
            print(f"Train/Test Split: {train_size}/{len(scaled_data) - train_size} samples")
            
            # FIXED: Consistent handling of 3-value returns from create_dataset
            if MODE == 'single':
                # Build datasets for each specified sequence length
                X_train_list, y_train_list, train_indices_list = [], [], []
                X_test_list, y_test_list, test_indices_list = [], [], []
                
                for seq in MULTI_SEQ_LENGTHS:
                    Xtr, ytr, train_idx = create_dataset(train_set, seq, NUM_PRED_DAYS)
                    Xte, yte, test_idx = create_dataset(test_set, seq, NUM_PRED_DAYS)
                    
                    if len(Xtr) == 0 or len(Xte) == 0:
                        print(f"[ERROR] Empty dataset for sequence {seq}")
                        continue
                    
                    Xtr = Xtr.reshape((Xtr.shape[0], seq, 1))
                    Xte = Xte.reshape((Xte.shape[0], seq, 1))
                    
                    X_train_list.append(Xtr)
                    y_train_list.append(ytr)
                    train_indices_list.append(train_idx)
                    X_test_list.append(Xte)
                    y_test_list.append(yte)
                    test_indices_list.append(test_idx)
                    
                    print(f"  Sequence {seq}: Train shape {Xtr.shape}, Test shape {Xte.shape}")
                
                if not X_train_list:
                    print(f"[ERROR] No valid datasets created")
                    failed_symbols.append(symbol)
                    continue
                    
            else:
                # In multiple mode, create a common dataset using the maximum sequence length
                X_train, y_train, train_indices = create_dataset(train_set, max(MULTI_SEQ_LENGTHS), NUM_PRED_DAYS)
                X_test, y_test, test_indices = create_dataset(test_set, max(MULTI_SEQ_LENGTHS), NUM_PRED_DAYS)
                
                if len(X_train) == 0 or len(X_test) == 0:
                    print(f"[ERROR] Empty dataset created")
                    failed_symbols.append(symbol)
                    continue
                
                X_train = X_train.reshape((X_train.shape[0], max(MULTI_SEQ_LENGTHS), 1))
                X_test = X_test.reshape((X_test.shape[0], max(MULTI_SEQ_LENGTHS), 1))
                print(f"Dataset shapes - X_train: {X_train.shape}, X_test: {X_test.shape}")
            
            # Build the model
            print("\nBuilding model...")
            if ARCH == 'simple':
                if MODE == 'single':
                    model = build_simple_model(input_shape=(X_train_list[0].shape[1], 1), num_predict=NUM_PRED_DAYS)
                else:
                    model = build_simple_model(input_shape=(X_train.shape[1], 1), num_predict=NUM_PRED_DAYS)
            elif ARCH == 'multires':
                input_shapes = [(seq, 1) for seq in MULTI_SEQ_LENGTHS]
                model = build_multires_model(input_shapes=input_shapes, num_predict=NUM_PRED_DAYS)
            else:
                print("[ERROR] Invalid architecture option!")
                failed_symbols.append(symbol)
                continue
            
            adamopt = tf.keras.optimizers.Adam(learning_rate=1e-03, beta_1=0.95, beta_2=0.99, epsilon=1e-08)
            model.compile(optimizer=adamopt, loss='mean_squared_error', metrics=['mae'])  # Added MAE metric
            
            # FIXED: Display model summary
            print("\n" + "="*70)
            print("MODEL ARCHITECTURE")
            print("="*70)
            model.summary()
            print(f"\nTotal parameters: {model.count_params():,}")
            print("="*70 + "\n")
            
            # Setup directories and callbacks
            symbol_results_dir = os.path.join(RESULTS_DIR, symbol)
            os.makedirs(symbol_results_dir, exist_ok=True)
            best_model_path = os.path.join(symbol_results_dir, 'best_model.h5')
            model_checkpoint = ModelCheckpoint(
                filepath=best_model_path, 
                monitor='val_loss', 
                save_best_only=True, 
                verbose=1
            )
            early_stopping = EarlyStopping(
                monitor='val_loss', 
                patience=40, 
                restore_best_weights=True,
                verbose=1
            )
            reduce_lr = ReduceLROnPlateau(
                monitor='loss', 
                factor=0.90, 
                patience=10, 
                min_lr=1e-07, 
                cooldown=3, 
                verbose=1
            )
            
            # Training
            print("\nTraining model...")
            if MODE == 'single':
                history_list = []
                initial_epoch = 0
                epochs = 10
                
                # Track the indices for the first sequence (used for plotting)
                train_indices = train_indices_list[0]
                test_indices = test_indices_list[0]
                
                # In single mode, iterate over each sequence length's dataset
                for i, (seq, Xtr, ytr) in enumerate(zip(MULTI_SEQ_LENGTHS, X_train_list, y_train_list)):
                    print(f"\nTraining with sequence length: {seq} (Phase {i+1}/{len(MULTI_SEQ_LENGTHS)})")
                    history = model.fit(
                        Xtr, ytr, 
                        batch_size=32, 
                        epochs=epochs, 
                        initial_epoch=initial_epoch,
                        validation_data=(X_test_list[0], y_test_list[0]),
                        callbacks=[reduce_lr, early_stopping, model_checkpoint],
                        verbose=1
                    )
                    initial_epoch += epochs
                    epochs += 5
                    history_list.append(history.history)
                    
                    if early_stopping.stopped_epoch > 0:
                        print(f"Early stopping triggered at epoch {early_stopping.stopped_epoch}")
                        break
                
                # Combine histories
                if history_list:
                    combined_history = {'loss': [], 'val_loss': [], 'mae': [], 'val_mae': []}
                    for h in history_list:
                        for key in combined_history.keys():
                            if key in h:
                                combined_history[key].extend(h[key])
                    history = combined_history
                else:
                    history = {}
                    
            else:
                # In multiple mode, prepare appropriate inputs
                if ARCH == 'multires':
                    train_inputs = [X_train[:, :seq, :] for seq in MULTI_SEQ_LENGTHS]
                    test_inputs = [X_test[:, :seq, :] for seq in MULTI_SEQ_LENGTHS]
                    history = model.fit(
                        train_inputs, y_train, 
                        batch_size=32, 
                        epochs=50,
                        validation_data=(test_inputs, y_test),
                        callbacks=[reduce_lr, early_stopping, model_checkpoint],
                        verbose=1
                    )
                else:
                    history = model.fit(
                        X_train, y_train, 
                        batch_size=32, 
                        epochs=50,
                        validation_data=(X_test, y_test),
                        callbacks=[reduce_lr, early_stopping, model_checkpoint],
                        verbose=1
                    )
                history = history.history
            
            # Plot training history
            print("\nPlotting training history...")
            plt.figure(figsize=(10, 6))
            if 'loss' in history and len(history['loss']) > 0:
                plt.subplot(1, 2, 1)
                plt.plot(history['loss'], label='Train Loss')
                if 'val_loss' in history:
                    plt.plot(history['val_loss'], label='Val Loss')
                plt.title(f'{symbol} Loss')
                plt.yscale('log')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                
                plt.subplot(1, 2, 2)
                if 'mae' in history:
                    plt.plot(history['mae'], label='Train MAE')
                if 'val_mae' in history:
                    plt.plot(history['val_mae'], label='Val MAE')
                plt.title(f'{symbol} MAE')
                plt.xlabel('Epoch')
                plt.ylabel('MAE')
                plt.legend()
                
            plt.tight_layout()
            plt.savefig(os.path.join(symbol_results_dir, f'{symbol}_history.pdf'))
            plt.close()
            print(f"Training history saved to {symbol}_history.pdf")
            
            # Load best model for prediction
            print("\nLoading best model for predictions...")
            best_model = tf.keras.models.load_model(best_model_path)
            
            # FIXED: Prepare data for predictions with consistent variable assignment
            if MODE == 'single':
                X_train_pred = X_train_list[0]
                X_test_pred = X_test_list[0]
                y_train_pred = y_train_list[0]
                y_test_pred = y_test_list[0]
                # Indices already assigned above
            else:
                X_train_pred = X_train
                X_test_pred = X_test
                y_train_pred = y_train
                y_test_pred = y_test
                # train_indices and test_indices already assigned
            
            # Make predictions
            print("Making predictions...")
            if ARCH == 'multires':
                def prepare_multires(X, seq_lengths):
                    return [X[:, :seq, :] for seq in seq_lengths]
                train_inputs = prepare_multires(X_train_pred, MULTI_SEQ_LENGTHS)
                test_inputs = prepare_multires(X_test_pred, MULTI_SEQ_LENGTHS)
                train_predict = best_model.predict(train_inputs, verbose=0)
                test_predict = best_model.predict(test_inputs, verbose=0)
            else:
                train_predict = best_model.predict(X_train_pred, verbose=0)
                test_predict = best_model.predict(X_test_pred, verbose=0)
            
            # FIXED: Use safe inverse transform
            train_predict = safe_inverse_transform(scaler, train_predict, "train_predict")
            test_predict = safe_inverse_transform(scaler, test_predict, "test_predict")
            actual_prices = scaler.inverse_transform(scaled_data)
            
            # Future prediction loop
            print("\nGenerating future predictions...")
            future_steps = 10
            X_future = X_test_pred[-1].reshape(1, X_test_pred.shape[1], 1)
            future_predict = []
            
            for step in range(future_steps):
                if ARCH == 'multires':
                    inp = [X_future[:, :seq, :] for seq in MULTI_SEQ_LENGTHS]
                    future_price = best_model.predict(inp, verbose=0)
                else:
                    future_price = best_model.predict(X_future, verbose=0)
                
                # Check for NaN in predictions
                if np.any(np.isnan(future_price)):
                    print(f"[WARNING] NaN detected in future prediction step {step}, stopping future predictions")
                    break
                    
                future_predict.append(future_price[0])
                # Update X_future: remove the first time step and append the first predicted value
                X_future = np.append(X_future[:, 1:, :], future_price[0, 0].reshape(1, 1, 1), axis=1)
            
            if future_predict:
                future_predict = np.array(future_predict).reshape(-1, NUM_PRED_DAYS)
                future_predict = safe_inverse_transform(scaler, future_predict, "future_predict")
            else:
                print("[WARNING] No valid future predictions generated")
                future_predict = np.array([])
            
            # Create future dates for plotting
            last_date = df.index[-1]
            if len(future_predict) > 0:
                future_dates = pd.date_range(start=last_date, periods=len(future_predict) + NUM_PRED_DAYS, inclusive='right')
            else:
                future_dates = pd.date_range(start=last_date, periods=2, inclusive='right')
            
            # FIXED: Plotting with bounds checking
            print("\nCreating prediction plots...")
            plt.figure(figsize=(16, 8))
            plt.plot(df.index, actual_prices, label='Actual Stock Price', color='blue', linewidth=1.5)

            # FIXED: Safe calculation of plot indices with bounds checking
            max_df_index = len(df.index) - 1
            
            # Train predictions - indices are relative to the training set
            train_indices_plot = np.clip(train_indices, 0, max_df_index)
            valid_train_mask = train_indices_plot < len(df.index)
            
            if np.any(valid_train_mask):
                train_plot_indices = train_indices_plot[valid_train_mask]
                train_plot_values = train_predict[:len(train_plot_indices), 0] if train_predict.ndim > 1 else train_predict[:len(train_plot_indices)]
                plt.plot(df.index[train_plot_indices], train_plot_values, 
                        label='Train Predict (Day 1)', color='red', alpha=0.7)
            
            # Test predictions - offset by the training size minus the time_step
            test_indices_adjusted = test_indices + (train_size - time_step)
            test_indices_plot = np.clip(test_indices_adjusted, 0, max_df_index)
            valid_test_mask = test_indices_plot < len(df.index)
            
            if np.any(valid_test_mask):
                test_plot_indices = test_indices_plot[valid_test_mask]
                test_plot_values = test_predict[:len(test_plot_indices), 0] if test_predict.ndim > 1 else test_predict[:len(test_plot_indices)]
                plt.plot(df.index[test_plot_indices], test_plot_values, 
                        label='Test Predict (Day 1)', color='orange', alpha=0.7)
            
            # Future predictions
            if len(future_predict) > 0:
                for i in range(min(NUM_PRED_DAYS, future_predict.shape[1] if future_predict.ndim > 1 else 1)):
                    future_values = future_predict[:, i] if future_predict.ndim > 1 else future_predict
                    plt.plot(future_dates[i:i+len(future_values)], future_values,
                            label=f'Future Predictions (Day {i+1})', 
                            color=plt.cm.rainbow(i/max(NUM_PRED_DAYS, 1)),
                            linestyle='--', marker='o', markersize=3)
            
            # Add split lines
            if train_size < len(df.index):
                plt.axvline(x=df.index[train_size], color='gray', linestyle='--', 
                           alpha=0.5, label='Train/Test Split')
            plt.axvline(x=last_date, color='gray', linestyle=':', alpha=0.5, label='Test/Future Split')
            
            plt.title(f'Stock Price Prediction for {symbol} ({stock_name})')
            plt.xlabel('Date')
            plt.ylabel('Stock Price ($)')
            plt.legend(loc='best', fontsize=8)
            plt.grid(True, alpha=0.3)
            plt.gcf().autofmt_xdate()
            
            plt.tight_layout()
            plt.savefig(os.path.join(symbol_results_dir, f'{symbol}_predictions.pdf'))
            plt.close()
            print(f"Prediction plot saved to {symbol}_predictions.pdf")
            
            # Save future predictions to text file
            with open(os.path.join(symbol_results_dir, f'{symbol}_results.txt'), 'w') as f:
                f.write(f"Stock: {symbol} ({stock_name})\n")
                f.write(f"Model Configuration:\n")
                f.write(f"  Architecture: {ARCH}, Mode: {MODE}\n")
                f.write(f"  Sequence Lengths: {MULTI_SEQ_LENGTHS}\n")
                f.write(f"  Prediction Horizon: {NUM_PRED_DAYS} days\n")
                f.write(f"  Date Range: {FILTER_START} to {FILTER_END or 'present'}\n")
                f.write(f"  Total Parameters: {model.count_params():,}\n")
                f.write(f"\nTraining Results:\n")
                if 'loss' in history and history['loss']:
                    f.write(f"  Final Training Loss: {history['loss'][-1]:.6f}\n")
                if 'val_loss' in history and history['val_loss']:
                    f.write(f"  Final Validation Loss: {history['val_loss'][-1]:.6f}\n")
                if 'mae' in history and history['mae']:
                    f.write(f"  Final Training MAE: {history['mae'][-1]:.6f}\n")
                if 'val_mae' in history and history['val_mae']:
                    f.write(f"  Final Validation MAE: {history['val_mae'][-1]:.6f}\n")
                f.write("-" * 60 + "\n")
                
                if len(future_predict) > 0:
                    f.write("\nFuture Predictions:\n")
                    f.write("Date" + " " * 8 + " ".join([f"Day {i+1:>7}" for i in range(NUM_PRED_DAYS)]) + "\n")
                    for i in range(min(10, len(future_predict))):
                        date_str = future_dates[i].strftime('%Y-%m-%d')
                        values = []
                        for j in range(NUM_PRED_DAYS):
                            if future_predict.ndim > 1 and j < future_predict.shape[1]:
                                values.append(f"{future_predict[i, j]:>10.4f}")
                            else:
                                values.append(f"{'N/A':>8}")
                        f.write(f"{date_str}: {' '.join(values)}\n")
                else:
                    f.write("\n[No future predictions generated due to errors]\n")
            
            print(f"Results saved to {symbol}_results.txt")
            print(f"\n[SUCCESS] Processing for {symbol} completed successfully")
            successful_symbols.append(symbol)
            
        except Exception as e:
            print(f"\n[ERROR] Failed to process {symbol}: {str(e)}")
            import traceback
            traceback.print_exc()
            failed_symbols.append(symbol)
    
    # Save names cache
    save_names_cache(names_cache)
    
    # Print summary
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Total Symbols: {len(SYMBOLS)}")
    print(f"Successful: {len(successful_symbols)}")
    print(f"Failed: {len(failed_symbols)}")
    
    if successful_symbols:
        print(f"\nSuccessfully processed: {', '.join(successful_symbols)}")
    
    if failed_symbols:
        print(f"\nFailed to process: {', '.join(failed_symbols)}")
    
    print(f"\nResults saved to: {RESULTS_DIR}/")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stock Price Prediction - Fixed Version (NaN Prevention)")
    parser.add_argument("--mode", type=str, default="multiple", choices=["single", "multiple"],
                        help="Training mode: 'single' for one sequence, 'multiple' for multiple sequence training")
    parser.add_argument("--architecture", type=str, default="multires", choices=["simple", "multires"],
                        help="Network architecture: 'simple' for single-input, 'multires' for multi-resolution")
    parser.add_argument("--pred_horizon", type=int, default=5, help="Number of days to forecast (1-5)")
    parser.add_argument("--sequence_length", type=int, default=60, help="Sequence length if mode is 'single'")
    parser.add_argument("--multi_seq_lengths", type=str, default="15,30,45,55",
                        help="Comma-separated sequence lengths for multiple sequence mode")
    parser.add_argument("--use_smoothing", action="store_true", help="Apply moving average smoothing")
    parser.add_argument("--smooth_window", type=int, default=100, help="Window size for moving average smoothing")
    parser.add_argument("--filter_start", type=str, default="2020-01-01", help="Start date for filtering data")
    ##parser.add_argument("--filter_end", type=str, default=None, help="End date for filtering data (optional)")
    parser.add_argument("--filter_end", type=str, default="2025-10-20", help="End date for filtering data (optional)")
    # Read symbols from portfolio.txt file
    portfolio_file = "portfolio.txt"
    default_symbols = "MSFT,AAPL"  # Fallback if file doesn't exist
    try:
        if os.path.exists(portfolio_file):
            with open(portfolio_file, 'r') as f:
                symbols_from_file = [line.strip() for line in f if line.strip()]
                default_symbols = ','.join(symbols_from_file)
    except Exception as e:
        print(f"[WARNING] Could not read {portfolio_file}: {e}. Using fallback symbols.")
    
    parser.add_argument("--symbols", type=str, default=default_symbols, help="Comma-separated list of stock symbols") 
    parser.add_argument("--results_dir", type=str, default="CGPT2_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), 
                        help="Directory to store results")
    parser.add_argument("--force_download", action="store_true", help="Force download fresh data (ignore cache)")
    
    args = parser.parse_args()
    main(args)