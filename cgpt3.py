#!/usr/bin/env python
"""
Enhanced Stock Price Prediction Script - NaN Protected Version
==============================================================

Features:
- Downloads stock data from yfinance with retry logic and rate limiting
- Robust CSV caching with proper date/timezone handling
- Enhanced NaN prevention from ND.py while maintaining newcgpt1.py alignment
- Optionally applies moving average smoothing
- Two training modes:
    * "single": Train using one fixed sequence length (80/20 split) and build datasets for each sequence length.
    * "multiple": Train on a common dataset generated with the maximum sequence length and then slice it to create multi-resolution inputs.
- Forecast horizon can be set from 1 to 5 days
- Two network architectures:
    * "simple": Single-input model
    * "multires": Multi-resolution (multi-input) model that fuses inputs from different sequence lengths
- FIXED: Proper data/model alignment with comprehensive NaN protection

Usage example:
    python enhanced_newcgpt1.py --mode multiple --architecture multires --pred_horizon 5 --multi_seq_lengths "15,30,45,55" --use_smoothing --smooth_window 100 --filter_start 2020-01-01 --symbols MSFT,GOOGL
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
# Enhanced Utility Functions (From ND.py)
# ====================

def ensure_cache_dir():
    """Create cache directory if it doesn't exist"""
    os.makedirs(CACHE_DIR, exist_ok=True)

def safe_inverse_transform(scaler, data, data_name="data"):
    """
    Safely inverse transform data, handling shape issues that could cause NaN.
    Enhanced version from ND.py to prevent transformation errors.
    """
    print(f"[DEBUG] {data_name} shape before inverse_transform: {data.shape}")
    
    if len(data.shape) == 3:
        original_shape = data.shape
        if data.shape[2] == 1:
            data_2d = data.reshape(data.shape[0], data.shape[1])
        else:
            data_2d = data.reshape(-1, data.shape[-1])
        print(f"[DEBUG] Reshaped {data_name} from {original_shape} to {data_2d.shape}")
        result = scaler.inverse_transform(data_2d)
        if data.shape[2] == 1 and len(original_shape) == 3:
            result = result.reshape(original_shape[0], original_shape[1])
        return result
    elif len(data.shape) == 2:
        return scaler.inverse_transform(data)
    elif len(data.shape) == 1:
        data_2d = data.reshape(-1, 1)
        print(f"[DEBUG] Reshaped {data_name} from {data.shape} to {data_2d.shape}")
        return scaler.inverse_transform(data_2d)
    else:
        raise ValueError(f"Unexpected data shape for {data_name}: {data.shape}")

def normalize_df_index(df):
    """
    Enhanced normalize DataFrame index to ensure consistent datetime handling and prevent NaN issues.
    Enhanced version from ND.py with comprehensive data cleanup.
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
    
    # Sort index and remove duplicates (prevents alignment issues)
    try:
        df = df[~df.index.duplicated(keep='first')].sort_index()
    except Exception:
        pass
    
    # Enhanced NaN cleanup - fill any NaN values that might cause issues
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    return df

def comprehensive_nan_cleanup(df):
    """
    Comprehensive NaN cleanup function combining strategies from ND.py.
    Applies multiple fallback strategies to ensure no NaN values remain.
    """
    if df is None or df.empty:
        return df
    
    print(f"[NaN CLEANUP] Starting cleanup on {len(df)} rows, {len(df.columns)} columns")
    
    # Count initial NaN values
    initial_nan_count = df.isnull().sum().sum()
    print(f"[NaN CLEANUP] Initial NaN count: {initial_nan_count}")
    
    # Strategy 1: Forward fill then backward fill
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # Strategy 2: For any remaining NaN, use column means
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = df[col].fillna(0)
    
    # Strategy 3: Final fallback - replace any remaining NaN with 0
    df = df.fillna(0)
    
    # Verify cleanup
    final_nan_count = df.isnull().sum().sum()
    print(f"[NaN CLEANUP] Final NaN count: {final_nan_count}")
    print(f"[NaN CLEANUP] Cleaned {initial_nan_count - final_nan_count} NaN values")
    
    return df

def load_cached_stock(symbol, filter_start=None, filter_end=None):
    """
    Load cached stock data from CSV if available.
    Enhanced with comprehensive NaN prevention.
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
                
                # Apply comprehensive NaN cleanup
                df = comprehensive_nan_cleanup(df)
                
                print(f"  [OK] Loaded cached data for {symbol} ({len(df)} rows after filtering and cleanup)")
                return df
        except Exception as e:
            print(f"  [WARNING] Failed to load cached CSV for {symbol}: {e}")
    
    return None

def save_stock_cache(symbol, df):
    """Save stock data to CSV cache with NaN cleanup"""
    ensure_cache_dir()
    csv_path = os.path.join(CACHE_DIR, f"{symbol}.csv")
    
    try:
        df = normalize_df_index(df)
        df = comprehensive_nan_cleanup(df)
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
    Download stock data with retry logic and enhanced validation.
    Enhanced from ND.py with better error handling.
    """
    for attempt in range(max_retries):
        try:
            rate_limit_pause()
            
            print(f"  -> Downloading {symbol} (attempt {attempt + 1}/{max_retries})...")
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, period='max', auto_adjust=False)
            
            if df is not None and not df.empty:
                df = normalize_df_index(df)
                df = comprehensive_nan_cleanup(df)
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
    Enhanced download and filter function with comprehensive NaN protection.
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
    
    # Final NaN cleanup after filtering
    df = comprehensive_nan_cleanup(df)
    
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
    IMPORTANT: Keeps the Y_indices tracking from newcgpt1.py that ensures proper alignment!
    Enhanced with NaN validation.
    """
    print(f"[DATASET] Creating dataset with time_step={time_step}, num_predict={num_predict}")
    print(f"[DATASET] Input dataset shape: {dataset.shape}")
    
    # Validate input for NaN values
    if np.isnan(dataset).any():
        print(f"[WARNING] Found NaN values in input dataset, applying cleanup...")
        dataset = np.nan_to_num(dataset, nan=0.0, posinf=0.0, neginf=0.0)
    
    X, Y, Y_indices = [], [], []
    for i in range(len(dataset) - time_step - num_predict + 1):
        a = dataset[i:(i + time_step)]
        y = dataset[i + time_step : i + time_step + num_predict]
        
        # Validate sequences for NaN
        if np.isnan(a).any() or np.isnan(y).any():
            print(f"[WARNING] Skipping sequence {i} due to NaN values")
            continue
            
        X.append(a)
        Y.append(y)
        Y_indices.append(i + time_step)
    
    print(f"[DATASET] Created {len(X)} sequences")
    return np.array(X), np.array(Y), np.array(Y_indices)

def apply_enhanced_smoothing(df, use_smoothing, smooth_window):
    """
    Apply smoothing with comprehensive NaN protection.
    Enhanced version that prevents NaN introduction during smoothing.
    """
    if use_smoothing:
        print(f"[SMOOTHING] Applying {smooth_window}-day moving average with NaN protection")
        # Use min_periods=1 to prevent NaN gaps at boundaries
        df_smoothed = df[['Close']].rolling(window=smooth_window, min_periods=1).mean()
        # Ensure no NaN values remain
        df_smoothed = df_smoothed.fillna(method='ffill').fillna(method='bfill').fillna(0)
        print(f"[SMOOTHING] Smoothing complete, NaN count: {df_smoothed.isnull().sum().sum()}")
    else:
        df_smoothed = df[['Close']].copy()
        # Ensure source data has no NaN
        df_smoothed = df_smoothed.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    return df_smoothed

# ====================
# Model Building Functions (Unchanged from newcgpt1.py)
# ====================

def build_simple_model(input_shape, num_predict):
    """
    Builds a single-input (simple) model with bidirectional LSTM layers and an attention block.
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
    """
    actf = {'activation': 'tanh', 'recurrent_activation': 'sigmoid'}
    reg_params = {
        'kernel_regularizer': tf.keras.regularizers.l2(0.001),
        'recurrent_regularizer': tf.keras.regularizers.l2(0.001),
        'dropout': 0.0,
        'recurrent_dropout': 0.0
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
# Main Script with Enhanced NaN Protection
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
    print("\n" + "="*70)
    print("ENHANCED STOCK PRICE PREDICTION - NaN PROTECTED VERSION")
    print("="*70)
    print(f"Mode: {MODE}")
    print(f"Architecture: {ARCH}")
    print(f"Prediction Horizon: {NUM_PRED_DAYS} days")
    print(f"Sequence Lengths: {MULTI_SEQ_LENGTHS}")
    print(f"NaN Protection: Enhanced (from ND.py)")
    print(f"Data Alignment: Maintained (from newcgpt1.py)")
    print(f"Smoothing: {'Yes' if USE_SMOOTHING else 'No'}" + (f" (window={SMOOTH_WINDOW})" if USE_SMOOTHING else ""))
    print(f"Date Range: {FILTER_START} to {FILTER_END or 'present'}")
    print(f"Force Download: {FORCE_DOWNLOAD}")
    print(f"Symbols: {', '.join(SYMBOLS)}")
    print(f"Results Directory: {RESULTS_DIR}")
    print(f"Cache Directory: {CACHE_DIR}")
    print("="*70 + "\n")

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
            
            # Apply enhanced smoothing with NaN protection
            df_smoothed = apply_enhanced_smoothing(df, USE_SMOOTHING, SMOOTH_WINDOW)
            
            # Normalize the data with validation
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(df_smoothed.values.reshape(-1, 1))
            
            # Validate scaled data for NaN
            if np.isnan(scaled_data).any():
                print(f"[WARNING] Found NaN in scaled data, applying cleanup...")
                scaled_data = np.nan_to_num(scaled_data, nan=0.0)
            
            print(f"Data normalized to range [0, 1], NaN count: {np.isnan(scaled_data).sum()}")
            
            # Train/Test Split
            time_step = SEQ_LENGTH if MODE == 'single' else max(MULTI_SEQ_LENGTHS)
            train_size = int(len(scaled_data) * 0.8)
            train_set = scaled_data[:train_size]
            test_set = scaled_data[train_size - time_step:]
            
            print(f"Train/Test Split: {train_size}/{len(scaled_data) - train_size} samples")
            
            # Create datasets; **MAINTAINS INDEX TRACKING FROM newcgpt1.py**
            if MODE == 'single':
                # Build datasets for each specified sequence length
                X_train_list, y_train_list, train_indices_list = [], [], []
                X_test_list, y_test_list, test_indices_list = [], [], []
                for seq in MULTI_SEQ_LENGTHS:
                    Xtr, ytr, train_indices = create_dataset(train_set, seq, NUM_PRED_DAYS)
                    Xte, yte, test_indices = create_dataset(test_set, seq, NUM_PRED_DAYS)
                    Xtr = Xtr.reshape((Xtr.shape[0], seq, 1))
                    Xte = Xte.reshape((Xte.shape[0], seq, 1))
                    X_train_list.append(Xtr)
                    y_train_list.append(ytr)
                    train_indices_list.append(train_indices)
                    X_test_list.append(Xte)
                    y_test_list.append(yte)
                    test_indices_list.append(test_indices)
                    print(f"  Sequence {seq}: Train shape {Xtr.shape}, Test shape {Xte.shape}")
            else:
                # In multiple mode, create a common dataset using the maximum sequence length
                X_train, y_train, train_indices = create_dataset(train_set, max(MULTI_SEQ_LENGTHS), NUM_PRED_DAYS)
                X_train = X_train.reshape((X_train.shape[0], max(MULTI_SEQ_LENGTHS), 1))
                X_test, y_test, test_indices = create_dataset(test_set, max(MULTI_SEQ_LENGTHS), NUM_PRED_DAYS)
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
            model.compile(optimizer=adamopt, loss='mean_squared_error')
            print(f"Model compiled with {model.count_params():,} parameters")
            
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
                restore_best_weights=True
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
                # In single mode, iterate over each sequence length's dataset
                for seq, Xtr, ytr in zip(MULTI_SEQ_LENGTHS, X_train_list, y_train_list):
                    print(f"Training with sequence length: {seq}")
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
                history = history_list[0] if history_list else {}
                # Use the first train/test indices for plotting
                train_indices = train_indices_list[0]
                test_indices = test_indices_list[0]
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
            
            # Plot training history
            print("\nPlotting training history...")
            plt.figure(figsize=(8, 5))
            train_loss = history['loss'] if isinstance(history, dict) else history.history['loss']
            val_loss = history['val_loss'] if isinstance(history, dict) else history.history['val_loss']
            plt.plot(train_loss, label='Train')
            plt.plot(val_loss, label='Test')
            plt.title(f'{symbol} Training Loss')
            plt.yscale('log')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend(loc='upper right')
            plt.savefig(os.path.join(symbol_results_dir, f'{symbol}_history.pdf'))
            plt.close()
            print(f"Training history saved to {symbol}_history.pdf")
            
            # Load best model for prediction
            print("\nLoading best model for predictions...")
            best_model = tf.keras.models.load_model(best_model_path)
            
            # Prediction with enhanced error handling
            if MODE == 'single':
                # Use the first dataset for predictions in single mode
                X_train_pred = X_train_list[0]
                X_test_pred = X_test_list[0]
                y_train_pred = y_train_list[0]
                y_test_pred = y_test_list[0]
            else:
                X_train_pred = X_train
                X_test_pred = X_test
                y_train_pred = y_train
                y_test_pred = y_test
            
            if ARCH == 'multires':
                def prepare_multires(X, seq_lengths):
                    return [X[:, :seq, :] for seq in seq_lengths]
                train_inputs = prepare_multires(X_train_pred, MULTI_SEQ_LENGTHS)
                test_inputs = prepare_multires(X_test_pred, MULTI_SEQ_LENGTHS)
                train_predict = best_model.predict(train_inputs)
                test_predict = best_model.predict(test_inputs)
            else:
                train_predict = best_model.predict(X_train_pred)
                test_predict = best_model.predict(X_test_pred)
            
            # Use safe inverse transform to prevent NaN issues
            train_predict = safe_inverse_transform(scaler, train_predict, "train_predict")
            test_predict = safe_inverse_transform(scaler, test_predict, "test_predict")
            actual_prices = scaler.inverse_transform(scaled_data)
            
            # Future prediction loop with NaN protection
            print("\nGenerating future predictions...")
            future_steps = 10
            X_future = X_test_pred[-1].reshape(1, X_test_pred.shape[1], 1)
            future_predict = []
            
            for step in range(future_steps):
                try:
                    if ARCH == 'multires':
                        inp = [X_future[:, :seq, :] for seq in MULTI_SEQ_LENGTHS]
                        future_price = best_model.predict(inp, verbose=0)
                    else:
                        future_price = best_model.predict(X_future, verbose=0)
                    
                    # Validate prediction for NaN
                    if np.isnan(future_price).any() or np.isinf(future_price).any():
                        print(f"[WARNING] Invalid prediction at step {step}, using fallback")
                        future_price = np.array([[actual_prices[-1]]])  # Use last known price
                    
                    future_predict.append(future_price[0])
                    # Update X_future: remove the first time step and append the first predicted value
                    X_future = np.append(X_future[:, 1:, :], future_price[0, 0].reshape(1, 1, 1), axis=1)
                
                except Exception as e:
                    print(f"[ERROR] Future prediction failed at step {step}: {e}")
                    break
            
            if len(future_predict) > 0:
                future_predict = np.array(future_predict).reshape(-1, NUM_PRED_DAYS)
                future_predict = safe_inverse_transform(scaler, future_predict, "future_predict")
            else:
                future_predict = np.full((1, NUM_PRED_DAYS), actual_prices[-1])
            
            # Create future dates for plotting
            last_date = df_smoothed.index[-1] if USE_SMOOTHING else df.index[-1]
            future_dates = pd.date_range(start=last_date, periods=len(future_predict) + NUM_PRED_DAYS, inclusive='right')
            
            # **CORRECTED PLOT SECTION - MAINTAINS ALIGNMENT FROM newcgpt1.py**
            print("\nCreating prediction plots...")
            plt.figure(figsize=(16, 8))
            plt.plot(df.index, actual_prices, label='Actual Stock Price', color='blue')

            # Map indices to original df - PROPER ALIGNMENT MAINTAINED
            if MODE == 'single':
                # Use indices from the first sequence length
                train_indices_plot = train_indices
                test_indices_plot = test_indices + (train_size - time_step)
            else:
                train_indices_plot = train_indices
                test_indices_plot = test_indices + (train_size - time_step)

            # Plot train predictions at correct positions
            if len(train_predict.shape) > 1:
                plt.plot(df.index[train_indices_plot], train_predict[:, 0], label='Train Predict (Day 1)', color='red')
            else:
                plt.plot(df.index[train_indices_plot], train_predict, label='Train Predict (Day 1)', color='red')

            # Plot test predictions at correct positions
            if len(test_predict.shape) > 1:
                plt.plot(df.index[test_indices_plot], test_predict[:, 0], label='Test Predict (Day 1)', color='orange')
            else:
                plt.plot(df.index[test_indices_plot], test_predict, label='Test Predict (Day 1)', color='orange')

            # Plot future predictions
            for i in range(min(NUM_PRED_DAYS, len(future_predict[0]))):
                if i < len(future_dates):
                    plt.plot(future_dates[i:len(future_predict)+i], future_predict[:, i],
                            label=f'Future Predictions (Day {i+1})', 
                            color=plt.cm.rainbow(i/NUM_PRED_DAYS))

            # Add split lines
            plt.axvline(x=df.index[train_size], color='gray', linestyle='--', label='Train/Test Split')
            plt.axvline(x=last_date, color='gray', linestyle=':', label='Test/Future Split')

            plt.title(f'Enhanced Stock Price Prediction for {symbol} ({stock_name})\nNaN Protected with Proper Alignment')
            plt.xlabel('Time')
            plt.ylabel('Stock Price')
            plt.legend(loc='center left', bbox_to_anchor=(0, 0.5))
            plt.grid(True)
            plt.gcf().autofmt_xdate()
            plt.savefig(os.path.join(symbol_results_dir, f'{symbol}_enhanced_prediction.pdf'))
            plt.close()
            print(f"Enhanced prediction plot saved to {symbol}_enhanced_prediction.pdf")
            
            # Save future predictions to text file
            with open(os.path.join(symbol_results_dir, f'{symbol}_enhanced_forecast.txt'), 'w') as f:
                f.write(f"Stock: {symbol} ({stock_name})\n")
                f.write(f"Enhanced predictions with NaN protection and proper alignment\n")
                f.write(f"Model Architecture: {ARCH}, Mode: {MODE}\n")
                f.write(f"Sequence Lengths: {MULTI_SEQ_LENGTHS}\n")
                f.write(f"Prediction Horizon: {NUM_PRED_DAYS} days\n")
                f.write(f"NaN Protection: Enabled\n")
                f.write("-" * 60 + "\n")
                f.write("Date: " + ", ".join([f"Day {i+1} Prediction" for i in range(NUM_PRED_DAYS)]) + "\n")
                for i in range(len(future_predict)):
                    if i < len(future_dates):
                        current_date = future_dates[i]
                        predictions = ", ".join([f"{price:.4f}" for price in future_predict[i]])
                        f.write(f"{current_date.date()}: {predictions}\n")
            
            print(f"Enhanced future predictions saved to {symbol}_enhanced_forecast.txt")
            print(f"\n[SUCCESS] Enhanced processing for {symbol} completed successfully")
            successful_symbols.append(symbol)
            
        except Exception as e:
            print(f"\n[ERROR] Failed to process {symbol}: {str(e)}")
            import traceback
            traceback.print_exc()
            failed_symbols.append(symbol)
    
    # Save names cache
    save_names_cache(names_cache)
    
    # Print summary
    print("\n" + "="*70)
    print("ENHANCED PROCESSING SUMMARY - NaN PROTECTED")
    print("="*70)
    print(f"Total Symbols: {len(SYMBOLS)}")
    print(f"Successful: {len(successful_symbols)}")
    print(f"Failed: {len(failed_symbols)}")
    
    if successful_symbols:
        print(f"\nSuccessfully processed: {', '.join(successful_symbols)}")
        print("Features: NaN protection + Proper alignment maintained")
    
    if failed_symbols:
        print(f"\nFailed to process: {', '.join(failed_symbols)}")
    
    print(f"\nResults saved to: {RESULTS_DIR}/")
    print("Enhanced with: NaN protection, Safe transforms, Robust error handling")
    print("="*70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced Stock Price Prediction - NaN Protected Version")
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
    parser.add_argument("--results_dir", type=str, default="CGPT3_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), 
                        help="Directory to store results")
    parser.add_argument("--force_download", action="store_true", help="Force download fresh data (ignore cache)")
    
    args = parser.parse_args()
    main(args)
