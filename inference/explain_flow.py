
import pandas as pd
import numpy as np
import os
import sys
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Add parent directory to path to import train_nids and explainability
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import train_nids
from explainability.attack_explainer import explain_attack

# --- Configuration ---
# Use the dataset found in the current environment
WORKSPACE_DATASET = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Dataset', 'NIDS_FINAL_DATASET.csv'))
SINGLE_FLOW_INPUT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'single_flow.csv'))

def get_data_and_scaler(filepath):
    """
    Replicates train_nids.load_and_preprocess_data but returns the scaler and label encoder.
    This is necessary because the original script doesn't save/return the scaler object.
    """
    print(f"Loading dataset for training/scaling from {filepath}...")
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    
    columns_to_drop = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp', '__source_file']
    for col in columns_to_drop:
        if col in df.columns:
            df = df.drop(columns=[col])
            
    y = df['Label']
    X = df.drop(columns=['Label'])
    X = X.select_dtypes(include=[np.number])
    
    X = X.replace([np.inf, -np.inf], np.nan)
    if X.isnull().values.any():
        X = X.fillna(X.mean())
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_names = le.classes_
    
    # We need the full training set to fit the scaler exactly as training did
    # train_nids uses a 0.2 split with RANDOM_SEED 42
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) # for consistency check if needed
    
    # Get feature names that survived preprocessing
    feature_names = X.columns.tolist()
    
    return X_train_scaled, X_test_scaled, y_train, y_test, class_names, scaler, feature_names

def preprocess_single_flow(filepath, scaler, feature_names):
    """
    Correctly preprocesses a single flow input to match training features.
    """
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    
    # identifier columns to drop
    columns_to_drop = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp', '__source_file', 'Label']
    existing_drop = [c for c in columns_to_drop if c in df.columns]
    df_clean = df.drop(columns=existing_drop)
    
    # Match the numeric-only and exact feature order from training
    # Instead of select_dtypes, we reorder based on 'feature_names' from training
    # and handle missing columns if any
    X_single = df_clean.reindex(columns=feature_names)
    
    # Handle missing/infinite values (impute with 0 or mean if we had it, but 0 is safe for single flow if training was clean)
    # Re-impute with a "safe" value if NaN in single flow
    X_single = X_single.replace([np.inf, -np.inf], np.nan)
    X_single = X_single.fillna(0) # Basic imputation for single flow
    
    # Scale
    X_single_scaled = scaler.transform(X_single)
    
    return X_single_scaled

def main():
    # 1. Setup Data and Scaler
    # Use workspace dataset path if possible
    dataset_path = WORKSPACE_DATASET
    if not os.path.exists(dataset_path):
        # Fallback to train_nids.DATASET_PATH if workspace one isn't found
        dataset_path = train_nids.DATASET_PATH
        
    X_train, X_test, y_train, y_test, class_names, scaler, feature_names = get_data_and_scaler(dataset_path)
    
    # 2. Get Trained Model
    # Per requirement (d), use Random Forest model
    print("\nTraining Random Forest model for inference...")
    model = train_nids.train_eval_rf(X_train, X_test, y_train, y_test, class_names)
    
    # 3. Accept Single Flow Input
    if not os.path.exists(SINGLE_FLOW_INPUT):
        print(f"Error: Single flow file not found at {SINGLE_FLOW_INPUT}")
        return
        
    print(f"Processing single flow from {SINGLE_FLOW_INPUT}...")
    X_single_scaled = preprocess_single_flow(SINGLE_FLOW_INPUT, scaler, feature_names)
    
    # 4. Predict
    prediction_idx = model.predict(X_single_scaled)[0]
    predicted_class = class_names[prediction_idx]
    
    # 5. Explain and Output
    explanation = explain_attack(predicted_class)
    
    print("\n" + "="*40)
    print(f"Detected Traffic Type: {predicted_class}")
    print("="*40)
    
    print("\nDescription:")
    print(explanation['description'])
    
    print("\nWhat is happening:")
    print(explanation['happening'])
    
    print("\nWhat to do:")
    print(explanation['to_do'])
    
    print("\nWhat NOT to do:")
    print(explanation['not_to_do'])
    print("="*40)

if __name__ == "__main__":
    main()
