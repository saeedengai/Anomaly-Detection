# Anomaly Detection over CATS (RanSynCoder)

Detect anomalous behaviour across **multi-channel CATS sensor streams** in near-real-time—**minimizing false alarms** while catching true faults. The pipeline is built end-to-end: streaming ingestion → reproducible EDA → robust preprocessing → a synchronized-sinusoid autoencoder (**RanSynCoder**, TensorFlow) → **threshold calibration on a nominal holdout** → evaluation (**PR-AUC / F1**) → export + inference hooks.

The primary goal of this project is to develop an anomaly detection system and predictive model for multivariate time series data, focusing on identifying instability in signals, classifying the type of instability, detecting the root cause (i.e., the first signal to show instability), and analyzing the pattern of instability propagating from one channel to another. The system will utilize advanced algorithms to analyze the data, detect anomalies, and suggest possible root causes, with a focus on employing explainable AI and predictive modelling. The scope includes downloading and studying the dataset, researching existing models, developing new algorithms, and implementing an end-to-end pipeline for signal prediction, anomaly detection, classification, and reporting.

> **Data Notice:** The real CATS data is **not public** and is **not included** in this repository.

---

## TL;DR

- **Problem:** Highly imbalanced sensor data with **drift across runs** → static thresholds trigger noisy alerts.  
- **Approach:** Train an autoencoder (**RanSynCoder**) to model normal patterns; calibrate detection thresholds using a **nominal holdout** (not fixed contamination).  
- **Result:** Fewer spurious alerts and a better **precision/recall balance** under class imbalance.  
- **Outputs:** Metrics (**PR-AUC, F1**), confusion-matrix figures, saved models, and reproducible notebooks/scripts.

---

## Repository Structure


