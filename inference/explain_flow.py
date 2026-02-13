
import pandas as pd
import numpy as np
import os
import sys
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
    
    # 2. Train Anomaly Detector (Isolation Forest on Normal Traffic)
    anomaly_model = train_anomaly_detector(X_train, y_train, class_names)
    
    # 3. Train Random Forest with increased stability (n_estimators=500)
    print("\nTraining Random Forest model (n_estimators=500) for inference...")
    rf_raw = RandomForestClassifier(n_estimators=500, random_state=42)
    rf_raw.fit(X_train, y_train)
    
    # 4. Calibrate probabilities using isotonic regression (5-fold CV)
    print("Calibrating probabilities with CalibratedClassifierCV (isotonic)...")
    calibrated_model = CalibratedClassifierCV(rf_raw, method='isotonic', cv=5)
    calibrated_model.fit(X_train, y_train)
    
    # Validate: compare raw vs calibrated accuracy on test set
    raw_preds = rf_raw.predict(X_test)
    cal_preds = calibrated_model.predict(X_test)
    raw_accuracy = accuracy_score(y_test, raw_preds)
    cal_accuracy = accuracy_score(y_test, cal_preds)
    print(f"\nRaw RF Accuracy:        {raw_accuracy:.4f}")
    print(f"Calibrated RF Accuracy: {cal_accuracy:.4f}")
    
    # 5. Accept Single Flow Input
    if not os.path.exists(SINGLE_FLOW_INPUT):
        print(f"Error: Single flow file not found at {SINGLE_FLOW_INPUT}")
        return
        
    print(f"\nProcessing single flow from {SINGLE_FLOW_INPUT}...")
    X_single_scaled = preprocess_single_flow(SINGLE_FLOW_INPUT, scaler, feature_names)
    
    # 6. Predict with dual probability capture (raw vs calibrated)
    raw_probs = rf_raw.predict_proba(X_single_scaled)[0]
    raw_confidence = raw_probs.max() * 100
    
    cal_probs = calibrated_model.predict_proba(X_single_scaled)[0]
    prediction_idx = cal_probs.argmax()
    predicted_class = class_names[prediction_idx]
    confidence_score = cal_probs.max() * 100  # Calibrated confidence
    
    # 7. Confidence level interpretation
    if confidence_score >= 85.0:
        confidence_level = "High"
    elif confidence_score >= 70.0:
        confidence_level = "Medium"
    else:
        confidence_level = "Low"
    
    # 8. Anomaly Detection (post-prediction novelty detection layer)
    anomaly_result = compute_anomaly_score(anomaly_model, X_single_scaled)
    is_anomaly = anomaly_result["is_anomaly"]
    anomaly_score = anomaly_result["anomaly_score"]
    anomaly_status = "Anomalous" if is_anomaly else "Normal"
    

    # 9. Novelty Detection Decision Logic
    # Combines classifier confidence with anomaly detection for robust decision
    if confidence_score < 70.0 and is_anomaly:
        final_decision = "Unknown / Suspicious Traffic"
    elif confidence_score < 70.0:
        final_decision = "Low Confidence Prediction"
    elif is_anomaly:
        final_decision = "Possible Novel Behaviour"
    else:
        final_decision = predicted_class  # Normal predicted class
    
    # 10. Assess Risk (post-prediction decision-support layer)
    risk_assessment = assess_risk(predicted_class, confidence_score)
    
    # 11. Explain Attack (existing logic)
    explanation = explain_attack(predicted_class)
    
    # 12. Output
    output_lines = []
    output_lines.append("=" * 50)
    output_lines.append("  NETWORK INTRUSION DETECTION ALERT  ")
    output_lines.append("=" * 50)
    output_lines.append(f"\nDetected Traffic Type: {predicted_class}")
    
    output_lines.append("\n--- Attack Explanation ---")
    output_lines.append(f"\nDescription:\n{explanation['description']}")
    output_lines.append(f"\nWhat is happening:\n{explanation['happening']}")
    output_lines.append(f"\nWhat to do:\n{explanation['to_do']}")
    output_lines.append(f"\nWhat NOT to do:\n{explanation['not_to_do']}")
    
    output_lines.append("\n--- Risk Assessment ---")
    output_lines.append(f"\nModel Confidence: {risk_assessment['confidence']:.1f}%")
    output_lines.append(f"Risk Level: {risk_assessment['risk_level']}")
    output_lines.append(f"Priority: {risk_assessment['priority']}")
    output_lines.append(f"Recommended Action: {risk_assessment['recommended_action']}")
    
    output_lines.append("\n--- Confidence Diagnostics ---")
    output_lines.append("\nClass Probabilities (Calibrated):")
    for i, cls_name in enumerate(class_names):
        output_lines.append(f"  {cls_name}: {cal_probs[i] * 100:.1f}%")
    output_lines.append(f"\nRaw Confidence (before calibration): {raw_confidence:.1f}%")
    output_lines.append(f"Calibrated Confidence: {confidence_score:.1f}%")
    output_lines.append(f"Confidence Level: {confidence_level}")
    
    output_lines.append("\n--- Calibration Validation ---")
    output_lines.append(f"\nRaw RF Test Accuracy:        {raw_accuracy:.4f}")
    output_lines.append(f"Calibrated RF Test Accuracy: {cal_accuracy:.4f}")
    
    output_lines.append("\n--- Novelty Detection ---")
    output_lines.append(f"\nModel Confidence: {confidence_score:.1f}%")
    output_lines.append(f"Anomaly Status: {anomaly_status}")
    output_lines.append(f"Anomaly Score: {anomaly_score:.4f}")
    output_lines.append(f"Final Decision: {final_decision}")
    output_lines.append("=" * 50)
    
    # Print to console
    output_text = "\n".join(output_lines)
    print("\n" + output_text)
    
    # Save to file
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs'))
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'alert_output.txt')
    with open(output_file, 'w') as f:
        f.write(output_text)
    print(f"\nOutput saved to: {output_file}")

if __name__ == "__main__":
    main()
