"""
Risk Assessment Module for NIDS

This module provides post-prediction risk scoring and alert prioritization.
It operates as a decision-support layer that does NOT affect model accuracy.

Author: NIDS Project
"""


def assess_risk(predicted_attack_type: str, confidence_score: float) -> dict:
    """
    Assess risk level, priority, and recommended action based on attack type and confidence.
    
    Args:
        predicted_attack_type: The attack class predicted by the model
        confidence_score: Model confidence as a percentage (0-100)
    
    Returns:
        Dictionary containing:
            - risk_level: Low / Medium / High / Critical
            - priority: Info / Warning / Alert / Emergency
            - recommended_action: Human-readable response guidance
            - confidence: The input confidence score
    """
    
    # Define base risk levels per attack type
    RISK_LEVELS = ["Low", "Medium", "High", "Critical"]
    
    BASE_RISK_MAP = {
        "normal": "Low",
        "probe": "Medium",
        "brute force": "High",
        "dos": "Critical",
        # Additional attack types mapped to appropriate risk levels
        "web attack": "High",
        "bot": "Medium",
        "infiltration": "Critical",
    }
    
    # Map risk level to priority
    PRIORITY_MAP = {
        "Low": "Info",
        "Medium": "Warning",
        "High": "Alert",
        "Critical": "Emergency"
    }
    
    # Map risk level to recommended actions
    ACTION_MAP = {
        "Low": "Continue standard monitoring. No immediate action required.",
        "Medium": "Increase monitoring frequency. Review logs within 24 hours.",
        "High": "Investigate immediately. Consider temporary IP blocking and notify security team.",
        "Critical": "Initiate incident response protocol. Block source IP and escalate to SOC immediately."
    }
    
    # Normalize attack type for matching
    attack_type_lower = predicted_attack_type.lower()
    
    # Determine base risk level
    base_risk = "Medium"  # Default for unknown attack types
    for key, risk in BASE_RISK_MAP.items():
        if key in attack_type_lower:
            base_risk = risk
            break
    
    # Get current risk index
    risk_index = RISK_LEVELS.index(base_risk)
    
    # Downgrade risk by one level if confidence < 70%
    if confidence_score < 70.0 and risk_index > 0:
        risk_index -= 1
    
    # Final risk level
    final_risk = RISK_LEVELS[risk_index]
    
    return {
        "confidence": confidence_score,
        "risk_level": final_risk,
        "priority": PRIORITY_MAP[final_risk],
        "recommended_action": ACTION_MAP[final_risk]
    }
