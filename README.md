# 🧠 Quantum-Enhanced Depression Risk Prediction in AIDS Patients

**Final Year Project — ECE Department | Anna University**

🔗 **Live Website:** https://monicavediyappan.github.io/Depression-risk-project/

---

## Overview

A quantum-classical hybrid AI system predicting depression risk in AIDS 
patients — improving on Routh & Singh (IEEE Access, 2025) across every 
dimension.

## Key Results

| Metric | Base Paper | Our Project |
|--------|-----------|-------------|
| AUC-ROC | 0.91 | **0.98** (+7.7%) |
| Accuracy | 88% | **91.25%** |
| Dataset | 100 patients | **10,000 patients** |
| Features | 5 | **20** |
| XAI Methods | SHAP only | **SHAP + LIME + DiCE** |

## Innovations

1. 100x larger dataset — 10,000 patients with 20 clinical features
2. Quantum-Classical Hybrid — PennyLane VQC + Qiskit QSVC
3. Stacking Ensemble — RF + XGBoost + LightGBM + SVM + LogReg
4. Triple XAI — SHAP + LIME + DiCE explainability
5. Fairness Analysis — Fairlearn across gender, age, CD4 groups
6. Cross-Dataset Validation — 2,000 independent patient test
7. Auto Clinical Report — generated per patient

## Tech Stack

- **Quantum:** PennyLane, Qiskit
- **ML:** scikit-learn, XGBoost, LightGBM
- **XAI:** SHAP, LIME, DiCE
- **Fairness:** Fairlearn
- **Platform:** Google Colab
- **Website:** GitHub Pages

## Base Paper

Routh & Singh, IEEE Access 2025  
DOI: 10.1109/ACCESS.2025.3639110

## Author

**Monica V** — Final Year ECE Student, Anna University


