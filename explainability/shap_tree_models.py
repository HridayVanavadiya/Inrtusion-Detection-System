
import sys
import os
import shap
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Ensure we can import from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import train_nids

# Set matplotlib backend to Agg to avoid display issues
matplotlib.use('Agg')

def get_feature_names(filepath):
    """
    Reads the dataset header and returns feature names after dropping identifiers.
    This logic mimics the drop logic in train_nids.py to ensure alignment.
    """
    df = pd.read_csv(filepath, nrows=1)
    df.columns = df.columns.str.strip()
    
    # Columns to drop as per train_nids.py + Label
    columns_to_drop = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp', '__source_file', 'Label']
    
    # Filter matching columns
    feature_names = [c for c in df.columns if c not in columns_to_drop]
    
    # Check if train_nids drops any other non-numeric? 
    # train_nids.py: X = X.select_dtypes(include=[np.number])
    # To be safe, we should strictly follow the columns that survive preprocessing.
    # However, since we can't easily reproduce the exact select_dtypes without loading the whole file or matching logic perfectly,
    # we will trust that the dropped columns list covers the non-numeric identifiers.
    # The 'Src Port' etc might be numeric or object. train_nids comments say:
    # "Ports and Protocol CAN be categorical... but often handled as numeric"
    # "Drop non-numeric and identifier columns"
    # Let's hope the numeric filter doesn't remove legitimate features we expect names for.
    # A safer way: rely on the fact that train_nids returns X_train_scaled which has shape (N, F).
    # We need F names. 
    # Let's try to do a quick load + drop on a small chunk to get dtypes if needed, 
    # but for now simple drop is likely sufficient as per instructions.
    
    return feature_names

def main():
    print("Starting SHAP explanation for Tree Models...")
    
    # Ensure outputs directory exists
    os.makedirs('outputs', exist_ok=True)
    
    # 1. Load Data
    X_train, X_test, y_train, y_test, class_names = train_nids.load_and_preprocess_data(train_nids.DATASET_PATH)
    
    # Get feature names for plotting
    feature_names = get_feature_names(train_nids.DATASET_PATH)
    
    # Check consistency
    if len(feature_names) != X_train.shape[1]:
        print(f"Warning: Feature names count ({len(feature_names)}) does not match X_train columns ({X_train.shape[1]}).")
        print("Attempting to align by loading first row and applying numeric filter...")
        # Fallback alignment
        df_chunk = pd.read_csv(train_nids.DATASET_PATH, nrows=5)
        df_chunk.columns = df_chunk.columns.str.strip()
        drop_cols = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp', '__source_file', 'Label']
        df_chunk = df_chunk.drop(columns=[c for c in drop_cols if c in df_chunk.columns])
        df_chunk = df_chunk.select_dtypes(include=[np.number])
        feature_names = df_chunk.columns.tolist()
        print(f"New feature names count: {len(feature_names)}")

    # 2. Random Forest
    print("\nTraining Random Forest for SHAP...")
    rf_model = train_nids.train_eval_rf(X_train, X_test, y_train, y_test, class_names)
    
    print("Generating SHAP summary for Random Forest...")
    # TreeExplainer is efficient for trees
    # For large datasets, we might want to subsample X_train for the background dataset if using KernelExplainer,
    # but TreeExplainer handles the model structure. 
    # However, passing data to expected_value or shap_values might be heavy. 
    # Using a summary plot usually requires SHAP values for a set of samples (e.g. Test set).
    # Let's use a subset of test data to be faster if X_test is huge.
    
    sample_size = 100  # Use 100 samples for speed and clarity in plots
    if X_test.shape[0] > sample_size:
        X_sample = X_test[:sample_size]
    else:
        X_sample = X_test
        
    explainer_rf = shap.TreeExplainer(rf_model)
    # Check if rf_model is multiclass. If so, shap_values is a list of arrays (one per class).
    shap_values_rf = explainer_rf.shap_values(X_sample)
    
    plt.figure()
    # For multiclass, we can plot for a specific class or all. 
    # Summary plot handles multiclass by default (showing stacked bars) if plot_type="bar" 
    # or we can plot for class 0, or just generic summary.
    # The user requested "Global SHAP summary plots". 
    # Usually `shap.summary_plot(shap_values, X_sample)` does a good job.
    shap.summary_plot(shap_values_rf, X_sample, feature_names=feature_names, show=False)
    plt.savefig('outputs/shap_rf_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved outputs/shap_rf_summary.png")

    # 3. XGBoost
    print("\nTraining XGBoost for SHAP...")
    xgb_model = train_nids.train_eval_xgb(X_train, X_test, y_train, y_test, class_names)
    
    print("Generating SHAP summary for XGBoost...")
    # TreeExplainer fails with current XGBoost/SHAP versions on multiclass base_score (vector).
    # Switch to KernelExplainer as a robust fallback.
    
    # Use a small background dataset for KernelExplainer
    # taking 50 samples or kmeans centroids
    background_data = shap.kmeans(X_train, 10)
    
    # KernelExplainer needs a prediction function. 
    # For global summary, we can use predict_proba or predict directly if we want output alignment.
    # TreeExplainer produces margin/log-odds by default. KernelExplainer on predict_proba produces probability attribution.
    # To keep consistent with "explainability", probability space is fine or even better.
    explainer_xgb = shap.KernelExplainer(xgb_model.predict_proba, background_data)
    
    # Compute SHAP values for the sample
    # nsamples defines the number of model evaluations. Keep it reasonable.
    shap_values_xgb = explainer_xgb.shap_values(X_sample, nsamples=100)
    
    plt.figure()
    # shap_values_xgb for KernelExplainer on multiclass is a list of arrays (one per class).
    shap.summary_plot(shap_values_xgb, X_sample, feature_names=feature_names, show=False)
    plt.savefig('outputs/shap_xgb_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved outputs/shap_xgb_summary.png")

    
    print("SHAP analysis complete.")

if __name__ == "__main__":
    main()
