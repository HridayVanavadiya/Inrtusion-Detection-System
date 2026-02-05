# Inrtusion Detection System

🔐 AI-Based Network Intrusion Detection System (NIDS)
This project presents an AI-based Network Intrusion Detection System (NIDS) designed to detect and classify malicious network activities using flow-based features and machine learning techniques. The system focuses on identifying multiple categories of cyberattacks by analyzing network traffic flows extracted from packet captures, making it suitable for modern, high-speed networks.

📌 Project Overview
Traditional signature-based intrusion detection systems struggle to detect novel and evolving cyber threats. To address this limitation, this project leverages machine learning and deep learning models trained on flow-level network traffic features to automatically detect and classify intrusions with high accuracy.
A custom dataset was created in a controlled laboratory environment by generating both benign and malicious traffic using real attack tools. Network traffic was captured and processed into flow-level statistics using CICFlowMeter, enabling efficient and scalable intrusion detection.

🧪 Experimental Setup
Attacker Machine: Kali Linux

Target System: Windows Server

Traffic Capture: Wireshark

Feature Extraction: CICFlowMeter (flow-based features)

📊 Dataset Description
The dataset used in this project is self-generated and consists of approximately 16,000+ network flows with 79 numerical flow-based features after preprocessing.
Traffic Classes

Normal Traffic: Legitimate network activity generated under normal usage conditions

Probe Attack: Network reconnaissance attacks generated using multi-phase Nmap scans

DoS Attack: Denial-of-Service attacks including TCP SYN flood, UDP flood, and ICMP flood

Brute Force Attack: Repeated authentication-like connection attempts simulating credential attacks

The dataset is preprocessed by removing identifier fields, handling missing and infinite values, encoding labels, and applying feature scaling.

🤖 Machine Learning Models
The following models were implemented and evaluated:
Random Forest
XGBoost
LSTM (Long Short-Term Memory)
Transformer (Attention-Based Neural Network)
Tree-based models achieved the highest performance due to the tabular nature of flow-based data, while deep learning models demonstrated competitive accuracy and added architectural novelty.

📈 Results Summary
Random Forest Accuracy: ~99.66%
XGBoost Accuracy: ~99.66%
LSTM Accuracy: ~99.57%
Transformer Accuracy: ~99.3–99.5%
Confusion matrices and validation metrics indicate balanced classification with no significant overfitting, supported by consistent training and validation performance.

🧠 Key Highlights
✔ Custom, real-world inspired dataset generation
✔ Flow-based intrusion detection suitable for high-speed networks
✔ Comparative study of classical ML and deep learning models
✔ Strong experimental results with overfitting analysis
✔ Designed for academic research and educational purposes

🔮 Future Work
Integration of Explainable AI (SHAP) for model interpretability
Evaluation in more diverse and real-world network environments
Extension to multi-class and zero-day attack detection
Real-time deployment using streaming network data
