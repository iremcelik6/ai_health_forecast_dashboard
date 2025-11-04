# 🧠 Model Card — AI Health Forecast Dashboard

This document describes the machine learning models used in the **AI Health Forecast Dashboard** project.

---

## 🧩 Model Overview

| Model | Type | Target | Input Features | Forecast Horizon | Framework |
|--------|------|--------|----------------|------------------|------------|
| `lstm_resp_forecaster.keras` | LSTM Neural Network | Respiratory Mortality | CO, NO₂, SO₂, O₃ + lag features | 3 years (2024–2028) | TensorFlow / Keras |
| `lstm_cardio_forecaster.keras` | LSTM Neural Network | Cardiovascular Mortality | CO, NO₂, SO₂, O₃ + lag features | 3 years (2024–2028) | TensorFlow / Keras |

---

## ⚙️ Training Configuration
- **Train/Validation/Test Split:** 70% / 20% / 10%  
- **Optimizer:** Adam (learning rate = 1e-3 → 5e-4 with scheduler)  
- **Loss Function:** Mean Squared Error (MSE)  
- **Regularization:** Dropout (0.2), EarlyStopping, ReduceLROnPlateau  
- **Epochs:** 40  
- **Batch Size:** 64  

---

## 🧮 Input Features
| Category | Features |
|-----------|-----------|
| Pollutants | CO, NO₂, SO₂, O₃ |
| Temporal | Year, lag-based averages (previous 8 years) |
| Targets | `resp_rate`, `cardio_rate` |

---

## 📈 Evaluation Metrics

| Metric | Respiratory (Val) | Respiratory (Test) | Cardiovascular (Val) | Cardiovascular (Test) |
|--------|------------------|--------------------|----------------------|------------------------|
| MAE | 0.1107 | 0.1450 | 0.3348 | 0.4427 |
| RMSE | 0.1377 | 0.1716 | 0.5130 | 0.6157 |
| R² | 0.9368 | 0.8614 | 0.6629 | 0.6777 |

> ✅ The respiratory mortality model achieved excellent generalization (R² ≈ 0.93 on validation).  
> ❤️ The cardiovascular model showed consistent accuracy with minor variance across states.

---

## 🧠 Model Interpretation
- The **LSTM** architecture effectively captured **temporal dependencies** between pollution and mortality trends.  
- The **What-If simulations** (+10% pollutant increase) revealed that:
  - **CO (Carbon Monoxide)** and **SO₂ (Sulfur Dioxide)** have the most significant impact on cardiovascular mortality.  
  - **O₃ (Ozone)** changes show smaller yet steady effects on respiratory health.  
- Models suggest a strong temporal link between pollutant exposure and delayed mortality effects.

---

## 🧾 Outputs
| File | Description |
|------|--------------|
| `outputs/predictions_state.csv` | Forecasted mortality rates (2024–2028) |
| `outputs/what_if_results.csv` | Predicted impact of +10% pollutant scenarios |
| `outputs/model_metrics.csv` | Evaluation metrics summary |
| `outputs/heatmap_pred_resp_rate.png` | State-level heatmap for respiratory mortality |
| `outputs/heatmap_pred_cardio_rate.png` | State-level heatmap for cardiovascular mortality |

---

## ⚠️ Limitations
- Limited pollutants (CO, NO₂, SO₂, O₃) — other toxins like PM2.5 or lead not included  
- Predictions assume static demographics and healthcare conditions  
- Regional anomalies (e.g., outliers like Alaska/Hawaii) may introduce bias  

---

## 📘 Ethical & Responsible AI Use
This model is designed for **educational and research purposes** only.  
It should **not** be used for clinical or policy-making decisions without expert validation.  
All data sources are public and anonymized.

---

## 📚 Citation
If you use this model or codebase, please cite:

> Çelik, İ. (2025). *AI Health Forecast Dashboard: Modeling the Impact of Air Pollution on Mortality Rates Using LSTM Neural Networks.*  
> Istanbul Aydın University, Department of Software Engineering.

---

## 🧾 Author
**İrem Çelik**  
🎓 Software Engineering — Istanbul Aydın University  
🌐 [github.com/iremcelik6](https://github.com/iremcelik6)  
📧 iremcelik@example.com  

---

