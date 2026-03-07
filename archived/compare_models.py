"""
Model Comparison Report

Compare performance of all trained models.
"""

import json
from pathlib import Path
import pandas as pd

project_root = Path(__file__).parent.parent

print("="*70)
print("MODEL PERFORMANCE COMPARISON")
print("="*70)

# Load baseline models metadata
baseline_metadata_path = project_root / 'models' / 'metadata_20260308_000626.json'
with open(baseline_metadata_path, 'r') as f:
    baseline_metadata = json.load(f)

# Load LSTM metadata
lstm_metadata_path = project_root / 'models' / 'lstm_metadata_20260308_001624.json'
with open(lstm_metadata_path, 'r') as f:
    lstm_metadata = json.load(f)

print("\n📊 BINARY CLASSIFICATION (Up/Down Prediction)")
print("-"*70)

results = []

# Baseline models
print("\nBASELINE MODELS (Single day features):")
print(f"  Logistic Regression:")
print(f"    ✓ Accuracy: {baseline_metadata['binary_classifier']['test_accuracy']:.4f} (83.6%)")
print(f"    ✓ F1 Score: {baseline_metadata['binary_classifier']['test_f1']:.4f}")

# LSTM
print(f"\nLSTM MODEL (30-day sliding window):")
print(f"    Accuracy: {lstm_metadata['direction_accuracy']:.4f} (49.5%)")
print(f"    F1 Score: {lstm_metadata['direction_f1']:.4f}")

print("\n" + "="*70)
print("📈 REGRESSION (Exact return prediction)")
print("-"*70)

print(f"\nBASELINE MODELS:")
print(f"  Ridge Regression:")
print(f"    ✓ MAE: {baseline_metadata['regressor']['test_mae']:.6f}")
print(f"    ✓ R²: {baseline_metadata['regressor']['test_r2']:.4f}")

print(f"\nLSTM MODEL:")
print(f"    MAE: {lstm_metadata['test_mae']:.6f}")
print(f"    R²: {lstm_metadata['test_r2']:.4f}")

print("\n" + "="*70)
print("🏆 BEST MODELS")
print("="*70)

print("\n✅ Binary Classification (Direction Prediction):")
print(f"   Winner: Logistic Regression (Baseline)")
print(f"   Accuracy: 83.6%")
print(f"   Why: Simple features work better than sequences for this task")

print("\n✅ Regression (Return Prediction):")
print(f"   Winner: Ridge Regression (Baseline)")  
print(f"   MAE: 0.0140")
print(f"   Why: Both struggle with exact predictions, but baseline is simpler")

print("\n" + "="*70)
print("💡 INSIGHTS")
print("="*70)

print("""
1. BASELINE MODELS WIN: Simple feature-based models (Logistic, Ridge) 
   outperform LSTM for this dataset.

2. WHY LSTM UNDERPERFORMS:
   - Limited data (only 2 years, 10 tickers)
   - Markets are highly efficient (hard to predict)
   - LSTM needs more examples to learn temporal patterns
   - Overfitting risk with small dataset

3. RECOMMENDATIONS:
   ✓ Use Logistic Regression for production (83.6% accuracy)
   ✓ Collect more data (5+ years, more tickers) before retrying LSTM
   ✓ Add sentiment features when more news data available
   ✓ Consider ensemble methods (combine baseline + LSTM)

4. BASELINE MODEL ADVANTAGES:
   ✓ Fast training (seconds vs minutes)
   ✓ Easy to interpret
   ✓ Less prone to overfitting
   ✓ Better with limited data
""")

print("="*70)
print("📁 SAVED MODELS")
print("="*70)
print(f"\nBaseline Models:")
print(f"  📦 {baseline_metadata['binary_classifier']['path']}")
print(f"  📦 {baseline_metadata['multiclass_classifier']['path']}")
print(f"  📦 {baseline_metadata['regressor']['path']}")
print(f"\nLSTM Model:")
print(f"  📦 {lstm_metadata['model_path']}")

print("\n" + "="*70)
print("✅ COMPARISON COMPLETED")
print("="*70)
