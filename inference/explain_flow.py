
import pandas as pd
import numpy as np
import os
import sys
import joblib
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score

# Add parent directory to path to import train_nids and explainability
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import train_nids
from explainability.attack_explainer import explain_attack
from explainability.risk_assessor import assess_risk
from explainability.anomaly_detector import train_anomaly_detector, compute_anomaly_score

# --- Configuration ---
# Use the dataset found in the current environment
WORKSPACE_DATASET = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Dataset', 'NIDS_FINAL_DATASET.csv'))
SINGLE_FLOW_INPUT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'probe_flow.csv'))

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

def load_artifacts(artifacts_dir):
    """Load all necessary artifacts for inference."""
    try:
        print(f"Loading artifacts from {artifacts_dir}...")
        scaler = joblib.load(os.path.join(artifacts_dir, 'scaler.pkl'))
        le = joblib.load(os.path.join(artifacts_dir, 'label_encoder.pkl'))
        rf_model = joblib.load(os.path.join(artifacts_dir, 'rf_model.pkl'))
        anomaly_detector = joblib.load(os.path.join(artifacts_dir, 'anomaly_detector.pkl'))
        feature_names = joblib.load(os.path.join(artifacts_dir, 'feature_names.pkl'))
        return scaler, le, rf_model, anomaly_detector, feature_names
    except FileNotFoundError as e:
        print(f"Error loading artifacts: {e}")
        print("Please run train_nids.py first to generate artifacts.")
        sys.exit(1)

def preprocess_single_flow(filepath, scaler, feature_names):
    """
    Correctly preprocesses a single flow input to match training features.
    Includes robust validation and handling for missing features.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Input file not found: {filepath}")
        
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        raise ValueError(f"Failed to read CSV file: {e}")
        
    if df.empty:
        raise ValueError("Input CSV file is empty.")
        
    df.columns = df.columns.str.strip()
    
    # identifier columns to drop (just for cleanup, though reindex handles selection)
    columns_to_drop = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp', '__source_file', 'Label']
    existing_drop = [c for c in columns_to_drop if c in df.columns]
    df_clean = df.drop(columns=existing_drop)
    
    # Validate features
    missing_features = [col for col in feature_names if col not in df_clean.columns]
    if missing_features:
        print(f"[WARNING] Input is missing {len(missing_features)} features expected by the model.")
        limit = min(len(missing_features), 5)
        missing_sample = [missing_features[i] for i in range(limit)]
        print(f"Missing: {missing_sample}...")
        # We will fill them with 0s via reindex
        
    # Reindex to ensure exact feature order and presence
    # fill_value=0 assumes missing features (like flags or counters) are likely 0 in partial data
    X_single = df_clean.reindex(columns=feature_names, fill_value=0)
    
    # Handle infinite values
    X_single = X_single.replace([np.inf, -np.inf], np.nan)
    
    # Handle missing values (NaN)
    if X_single.isnull().values.any():
        # print("Imputing missing values with 0...")
        X_single = X_single.fillna(0)
    
    # Scale features using the loaded scaler
    try:
        X_single_scaled = scaler.transform(X_single)
    except Exception as e:
        raise RuntimeError(f"Scaling failed: {e}")
    
    return X_single_scaled

def main():
    try:
        # 1. Setup paths
        dataset_path = WORKSPACE_DATASET # Only used for fallback if needed, but we rely on artifacts now
        artifacts_dir = "artifacts"
        if not os.path.exists(artifacts_dir):
            # Try looking in parent directory just in case
            artifacts_dir = os.path.join('..', 'artifacts')
            
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs'))
        os.makedirs(output_dir, exist_ok=True)
        
        # 2. Load Artifacts
        scaler, le, rf_model, anomaly_detector, feature_names = load_artifacts(artifacts_dir)
        class_names = le.classes_
        
        # Try to load calibrated model if it exists
        calibrated_model_path = os.path.join(artifacts_dir, 'rf_model_calibrated.pkl')
        calibrated_model = None
        if os.path.exists(calibrated_model_path):
            print(f"Loading calibrated model from {calibrated_model_path}...")
            calibrated_model = joblib.load(calibrated_model_path)
        
        # Determine which model to use for final decision
        if calibrated_model:
            main_model = calibrated_model
            model_type = "Calibrated RF"
        else:
            print("Calibrated model not found. Using raw RF model (confidence scores may be uncalibrated).")
            main_model = rf_model
            model_type = "Raw RF"
        
        # 3. Accept Single Flow Input
        if not os.path.exists(SINGLE_FLOW_INPUT):
            print(f"Error: Single flow file not found at {SINGLE_FLOW_INPUT}")
            return
            
        print(f"\nProcessing single flow from {SINGLE_FLOW_INPUT}...")
        X_single_scaled = preprocess_single_flow(SINGLE_FLOW_INPUT, scaler, feature_names)
        
        # 4. Predict & Confirm Diagnostics
        # Get raw confidence (from original RF)
        raw_probs = rf_model.predict_proba(X_single_scaled)[0]
        raw_confidence = raw_probs.max() * 100
        
        # Get calibrated confidence and class probabilities
        if calibrated_model:
            calibrated_probs = calibrated_model.predict_proba(X_single_scaled)[0]
            confidence_score = calibrated_probs.max() * 100
            final_probs = calibrated_probs
        else:
            confidence_score = raw_confidence
            final_probs = raw_probs
            
        prediction_idx = final_probs.argmax()
        predicted_class = class_names[prediction_idx]
        
        # 5. Confidence Level
        if confidence_score >= 85.0:
            confidence_level = "High"
        elif confidence_score >= 70.0:
            confidence_level = "Medium"
        else:
            confidence_level = "Low"
            
        # 6. Anomaly Detection
        anomaly_result = compute_anomaly_score(anomaly_detector, X_single_scaled)
        is_anomaly = anomaly_result["is_anomaly"]
        anomaly_score = anomaly_result["anomaly_score"]
        anomaly_status = "Anomalous" if is_anomaly else "Normal"
        
        # 7. Final Decision Logic
        if confidence_score < 70.0 and is_anomaly:
            final_decision = "Unknown / Suspicious Traffic"
        elif confidence_score < 70.0:
            final_decision = "Low Confidence Prediction"
        elif is_anomaly:
            final_decision = "Possible Novel Behaviour"
        else:
            final_decision = predicted_class
            
        # 8. Risk Assessment
        risk_assessment = assess_risk(predicted_class, confidence_score)
        
        # 9. Attack Explanation
        explanation = explain_attack(predicted_class)
        
        # 10. Generate Output
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        output_lines = []
        output_lines.append("=" * 46)
        output_lines.append(" NIDS ALERT ")
        output_lines.append("=" * 46)
        output_lines.append(f"Detected Traffic Type : {predicted_class}")
        output_lines.append("")
        output_lines.append("--- Risk Assessment ---")
        output_lines.append("")
        output_lines.append(f"Model Confidence      : {confidence_score:.2f}% ({confidence_level})")
        output_lines.append(f"Risk Level            : {risk_assessment['risk_level']}")
        output_lines.append(f"Priority              : {risk_assessment['priority']}")
        output_lines.append("")
        output_lines.append("--- Confidence Diagnostics ---")
        output_lines.append("")
        output_lines.append("Class Probabilities (Calibrated):")
        
        # Add class probabilities
        for i, class_name in enumerate(class_names):
            prob = final_probs[i] * 100
            output_lines.append(f"  {class_name:<20}: {prob:.1f}%")
        
        output_lines.append("")
        output_lines.append(f"Raw Confidence        : {raw_confidence:.1f}%")
        output_lines.append(f"Calibrated Confidence : {confidence_score:.1f}%")
        output_lines.append(f"Confidence Level      : {confidence_level}")
        output_lines.append("")
        output_lines.append(f"Anomaly Status        : {anomaly_status} (Score: {anomaly_score:.2f})")
        output_lines.append(f"Final Decision        : {final_decision}")
        output_lines.append(f"Timestamp             : {timestamp}")
        output_lines.append("=" * 46)
        
        # Additional Details
        output_lines.append("\n--- Action Plan ---")
        output_lines.append(f"Description: {explanation['description']}")
        output_lines.append("What is happening:")
        output_lines.append(f"{explanation['happening']}")
        output_lines.append(f"Recommended: {risk_assessment['recommended_action']}")
        output_lines.append(f"Do: {explanation['to_do']}")
        output_lines.append(f"Don't: {explanation['not_to_do']}")
        output_lines.append("-" * 50 + "\n")
        
        output_text = "\n".join(output_lines)
        
        # Console Output
        print("\n" + output_text)
        
        # 11. Logging
        log_file = os.path.join(output_dir, 'nids_alert.txt')
        with open(log_file, 'w') as f: 
            f.write("\n" + output_text + "\n")

        print(f"\nAlert logged to: {log_file}")
        
    except Exception as e:
        print(f"[ERROR] Inference failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
