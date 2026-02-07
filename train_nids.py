import pandas as pd
import numpy as np
import tensorflow as pd_tf  # Using tensorflow for Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
import time

# --- Configuration ---
DATASET_PATH = r"C:\Users\hrida\OneDrive\Desktop\major project\Dataset\Own Dataset\NIDS_FINAL_DATASET.csv"
RANDOM_SEED = 42
TEST_SIZE = 0.2

def load_and_preprocess_data(filepath):
    print("Loading dataset...")
    df = pd.read_csv(filepath)
    
    # Strip whitespace from column names just in case
    df.columns = df.columns.str.strip()
    
    print(f"Original shape: {df.shape}")

    # Drop identifiers and non-numeric columns unrelated to flow features
    # Based on file check: Flow ID, Src IP, Src Port, Dst IP, Dst Port, Protocol, Timestamp, __source_file
    # Note: Ports and Protocol CAN be categorical features, but often handled as numeric or dropped in pure flow stats. 
    # The prompt asks to "Drop non-numeric and identifier columns". 
    # IPs and Flow ID and Timestamp are definitely identifiers/non-numeric in this context.
    # __source_file is artifact of merging.
    
    columns_to_drop = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp', '__source_file']
    # Removing columns if they exist
    for col in columns_to_drop:
        if col in df.columns:
            df = df.drop(columns=[col])
            
    # Handle possible non-numeric values in other columns or drop them
    # "Label" is target.
    
    # Separate Features and Target
    y = df['Label']
    X = df.drop(columns=['Label'])
    
    # Drop any remaining non-numeric columns in X (like if 'Src Port' was parsed as string/object)
    X = X.select_dtypes(include=[np.number])
    
    print(f"Features shape after dropping identifiers: {X.shape}")

    # Handling Missing Values
    # Check for infinite values as well, common in flow datasets
    X = X.replace([np.inf, -np.inf], np.nan)
    if X.isnull().values.any():
        print("Handling missing/infinite values (imputing with mean)...")
        X = X.fillna(X.mean())
    
    # Encoding Label
    print("Encoding labels...")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_names = le.classes_
    print(f"Classes: {class_names}")
    
    # Train-Test Split (Stratified)
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y_encoded
    )
    
    # Scaling
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, class_names

def train_eval_rf(X_train, X_test, y_train, y_test, class_names):
    print("\n--- Random Forest ---")
    rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    print_metrics("Random Forest", y_test, y_pred, class_names)
    return rf

def train_eval_xgb(X_train, X_test, y_train, y_test, class_names):
    print("\n--- XGBoost ---")
    # xgboost expects classes to be 0 to N-1
    xgb = XGBClassifier(
        objective='multi:softprob', 
        eval_metric='mlogloss', 
        use_label_encoder=False,
        random_state=RANDOM_SEED
    )
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    
    print_metrics("XGBoost", y_test, y_pred, class_names)
    return xgb

def create_lstm_model(input_shape, num_classes):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(64),
        layers.Dense(32, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_eval_lstm(X_train, X_test, y_train, y_test, class_names):
    print("\n--- LSTM ---")
    # Reshape for LSTM: [samples, timesteps, features]
    # Treating the whole feature vector as 1 timestep
    X_train_reshaped = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_reshaped = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    
    model = create_lstm_model((X_train_reshaped.shape[1], X_train_reshaped.shape[2]), len(class_names))
    # model.summary()
    
    model.fit(X_train_reshaped, y_train, epochs=15, batch_size=32, validation_split=0.1, verbose=0)
    
    y_pred_prob = model.predict(X_test_reshaped, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    print_metrics("LSTM", y_test, y_pred, class_names)
    return model

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res

def create_transformer_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    
    # We treat features as a sequence of length 1 for simplicity given tabular data,
    # OR we could project features into a sequence. 
    # Let's stick to the reshaped (1, features) approach similar to LSTM to treat it as a sequence.
    
    # Transformer Encoder parameters
    head_size = 64
    num_heads = 4
    ff_dim = 64
    dropout = 0.1
    
    x = transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout)
    
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

def train_eval_transformer(X_train, X_test, y_train, y_test, class_names):
    print("\n--- Transformer ---")
    # Reshape same as LSTM: [samples, timesteps, features]
    # To use Transformer properly on tabular data without extensive embedding, 
    # we treat the feature vector as the embedding of a single token (sequence length 1).
    X_train_reshaped = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_reshaped = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    
    model = create_transformer_model((X_train_reshaped.shape[1], X_train_reshaped.shape[2]), len(class_names))
    # model.summary()
    
    model.fit(X_train_reshaped, y_train, epochs=15, batch_size=32, validation_split=0.1, verbose=0)
    
    y_pred_prob = model.predict(X_test_reshaped, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    print_metrics("Transformer", y_test, y_pred, class_names)
    return model

def print_metrics(model_name, y_true, y_pred, class_names):
    print(f"\nResults for {model_name}:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

import sys

def main():
    # Redirect stdout to a file
    original_stdout = sys.stdout
    with open('nids_results.txt', 'w') as f:
        sys.stdout = f
        try:
            X_train, X_test, y_train, y_test, class_names = load_and_preprocess_data(DATASET_PATH)
            
            train_eval_rf(X_train, X_test, y_train, y_test, class_names)
            train_eval_xgb(X_train, X_test, y_train, y_test, class_names)
            train_eval_lstm(X_train, X_test, y_train, y_test, class_names)
            train_eval_transformer(X_train, X_test, y_train, y_test, class_names)
            
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            sys.stdout = original_stdout # Restore stdout just in case
    
    # Print to console as well to confirm completion
    print("Training complete. Results saved to nids_results.txt")

if __name__ == "__main__":
    main()
