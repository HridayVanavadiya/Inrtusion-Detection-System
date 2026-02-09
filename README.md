# Intrusion Detection System
## AI-Based Network Intrusion Detection System (NIDS)
This project implements an AI-based Network Intrusion Detection System (NIDS) that detects, classifies, and explains malicious network activities using flow-based traffic features and machine learning. The system is designed to analyze network flows extracted from packet captures, making it suitable for modern high-speed networks.

### Project Overview
Traditional signature-based intrusion detection systems are limited in detecting novel and evolving cyber threats. To overcome this, the proposed system leverages machine learning and deep learning models trained on flow-level network traffic features to automatically identify and classify intrusions with high accuracy.
A custom dataset was generated in a controlled laboratory environment using real attack tools. Network traffic was captured and converted into flow-level statistics using CICFlowMeter, enabling scalable and efficient intrusion detection.
In addition to detection, the system integrates explainable and actionable security intelligence, allowing users to understand why a traffic flow is classified as malicious and what actions should be taken.

### Experimental Setup
* Attacker Machine: Kali Linux
* Target System: Windows Server
* Traffic Capture: Wireshark
* Feature Extraction: CICFlowMeter (flow-based features)

### Dataset Description
* Self-generated dataset containing ~16,000+ network flows
* 79 numerical flow-based features after preprocessing

#### Traffic Classes
1. Normal Traffic – Legitimate network activity under normal usage
2. Probe Attack – Network reconnaissance using multi-phase Nmap scans
3. DoS Attack – TCP SYN flood, UDP flood, and ICMP flood attacks
4. Brute Force Attack – Repeated authentication-like connection attempts

#### Preprocessing Steps
1. Removal of identifier fields (IP addresses, timestamps, ports)
2. Handling missing and infinite values
3. Label encoding
4. Feature scaling

### Machine Learning Models
* The following models were implemented and evaluated:
1. Random Forest
2. XGBoost
3. LSTM (Long Short-Term Memory)
4. Transformer (Attention-Based Neural Network)
* Tree-based models achieved the highest performance due to the tabular nature of flow-based features, while deep learning models demonstrated competitive accuracy and added architectural novelty.

### Results Summary
* Random Forest Accuracy: ~99.66%
* XGBoost Accuracy: ~99.66%
* LSTM Accuracy: ~99.57%
* Transformer Accuracy: ~99.3–99.5%
* Confusion matrices and validation metrics indicate balanced classification with no significant overfitting, supported by consistent training and validation performance.

### Explainability & Actionable Intelligence
* To avoid a black-box system, the project integrates multi-level explainability:

#### Feature-level explainability
1. SHAP for Random Forest and XGBoost
2. Integrated Gradients for LSTM and Transformer
3. These methods explain which flow features influence model predictions.

#### Decision-level explainability
1. The system translates predicted attack classes into human-readable explanations, describing:
2. What type of attack occurred
3. What is happening in the network
4. What actions should be taken
5. What should be avoided

### Single-Flow Inference
* The system supports single-flow inference, where an individual network flow (CSV row extracted via CICFlowMeter) can be provided as input. The model predicts the attack type and generates a complete explanation and mitigation guidance, making the system closer to a real-world NIDS deployment.

### Key Highlights
* Custom, real-world inspired dataset generation
* Flow-based intrusion detection suitable for high-speed networks
* Comparative study of classical ML and deep learning models
* Integrated explainable AI (SHAP & Integrated Gradients)
* Actionable, human-readable attack explanations
* Realistic single-flow inference pipeline

### Future Work
* Evaluation on cross-dataset and real-world traffic environments
* Extension to additional attack categories and zero-day detection
* Real-time deployment using streaming network flows
* Integration with SIEM or SOC monitoring tools
