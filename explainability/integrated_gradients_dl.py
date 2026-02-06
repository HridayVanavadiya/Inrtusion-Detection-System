
import sys
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# Ensure we can import from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import train_nids

# Set matplotlib backend to Agg
matplotlib.use('Agg')

def get_feature_names(filepath, expected_count):
    """
    Reads dataset to get feature names, aligning with train_nids logic.
    """
    df_chunk = pd.read_csv(filepath, nrows=5)
    df_chunk.columns = df_chunk.columns.str.strip()
    
    # Columns normally dropped
    drop_cols = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp', '__source_file', 'Label']
    
    # Drop valid existing columns
    existing_drop = [c for c in drop_cols if c in df_chunk.columns]
    df_chunk = df_chunk.drop(columns=existing_drop)
    
    # Keep only numeric to match select_dtypes(include=[np.number])
    df_chunk = df_chunk.select_dtypes(include=[np.number])
    
    names = df_chunk.columns.tolist()
    
    if len(names) != expected_count:
        print(f"Warning: Feature names count {len(names)} != Expected {expected_count}")
        # If mismatch, just return generic names
        return [f"Feature {i}" for i in range(expected_count)]
        
    return names

def get_integrated_gradients(model, baseline, input_sample, target_class_idx, steps=50):
    """
    Computes Integrated Gradients for a single sample.
    Args:
        model: Trained Keras model.
        baseline: Baseline input (tensor).
        input_sample: Input sample to explain (tensor).
        target_class_idx: The class index to explain.
        steps: Number of interpolation steps.
    Returns:
        integrated_gradients: Attribution scores for features.
    """
    # 1. Generate interpolated inputs
    alphas = tf.linspace(start=0.0, stop=1.0, num=steps+1) # (steps+1,)
    
    # Reshape alphas for broadcasting
    # Input shape is (1, 1, features)
    alphas_reshaped = tf.cast(alphas, tf.float32)[:, tf.newaxis, tf.newaxis, tf.newaxis]
    
    # baseline and input must be cast to float32
    baseline = tf.cast(baseline, tf.float32)
    input_sample = tf.cast(input_sample, tf.float32)
    
    # Interpolate: x_i = baseline + alpha * (input - baseline)
    delta = input_sample - baseline
    interpolated_path = baseline + alphas_reshaped * delta
    # interpolated_path shape: (steps+1, 1, 1, features) -> Remove extra dim for batching
    interpolated_path = tf.squeeze(interpolated_path, axis=1) # (steps+1, 1, features)

    # 2. Compute gradients for each interpolated input
    with tf.GradientTape() as tape:
        tape.watch(interpolated_path)
        predictions = model(interpolated_path)
        # Get score for target class
        target_score = predictions[:, target_class_idx]
        
    gradients = tape.gradient(target_score, interpolated_path) 
    # gradients shape: (steps+1, 1, features)

    # 3. Approximate the integral using Trapezoidal rule or simply average (Riemann sum)
    # Using Riemann sum for simplicity: avg_grad * (input - baseline)
    avg_gradients = tf.reduce_mean(gradients, axis=0) # (1, features)
    
    # 4. Compute IG
    integrated_gradients = (input_sample - baseline) * avg_gradients
    
    return integrated_gradients.numpy()

def plot_attributions(attributions, feature_names, title, filename):
    """
    Plots feature attributions.
    """
    # Flatten if necessary
    attributions = attributions.flatten()
    
    # Sort by absolute value to show most important features
    indices = np.argsort(np.abs(attributions))[-20:] # Top 20 features
    
    plt.figure(figsize=(10, 8))
    plt.barh(np.arange(len(indices)), attributions[indices], align='center')
    plt.yticks(np.arange(len(indices)), np.array(feature_names)[indices])
    plt.xlabel("Attribution Value")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved {filename}")

def main():
    print("Starting Integrated Gradients for Deep Learning Models...")
    
    os.makedirs('outputs', exist_ok=True)
    
    # 1. Load Data
    X_train, X_test, y_train, y_test, class_names = train_nids.load_and_preprocess_data(train_nids.DATASET_PATH)
    
    num_features = X_train.shape[1]
    feature_names = get_feature_names(train_nids.DATASET_PATH, num_features)
    
    # Pick a sample to explain (e.g., first test sample)
    sample_idx = 0
    input_sample_raw = X_test[sample_idx] 
    target_class = y_test[sample_idx]
    
    print(f"\nExplaining Test Sample at index {sample_idx}")
    print(f"True Class: {class_names[target_class]}")
    
    # 2. LSTM
    print("\nTraining LSTM for IG...")
    lstm_model = train_nids.train_eval_lstm(X_train, X_test, y_train, y_test, class_names)
    
    print("Computing IG for LSTM...")
    # Reshape for LSTM: (1, 1, features)
    baseline_lstm = tf.zeros(shape=(1, 1, num_features))
    input_sample_lstm = input_sample_raw.reshape((1, 1, num_features))
    
    ig_lstm = get_integrated_gradients(
        lstm_model, 
        baseline_lstm, 
        input_sample_lstm, 
        target_class_idx=target_class
    )
    
    plot_attributions(
        ig_lstm, 
        feature_names, 
        f"LSTM Feature Attribution (True: {class_names[target_class]})", 
        "outputs/ig_lstm.png"
    )

    # 3. Transformer
    print("\nTraining Transformer for IG...")
    transformer_model = train_nids.train_eval_transformer(X_train, X_test, y_train, y_test, class_names)
    
    print("Computing IG for Transformer...")
    # Reshape for Transformer: (1, 1, features) - Same as LSTM in this codebase
    baseline_transformer = tf.zeros(shape=(1, 1, num_features))
    input_sample_transformer = input_sample_raw.reshape((1, 1, num_features))
    
    ig_transformer = get_integrated_gradients(
        transformer_model, 
        baseline_transformer, 
        input_sample_transformer, 
        target_class_idx=target_class
    )
    
    plot_attributions(
        ig_transformer, 
        feature_names, 
        f"Transformer Feature Attribution (True: {class_names[target_class]})", 
        "outputs/ig_transformer.png"
    )
    
    print("Integrated Gradients analysis complete.")

if __name__ == "__main__":
    main()
