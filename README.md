# SIMTELE-RISK — Sistema Inteligente de Analítica Predictiva

**Proyecto Integrador - Mención: Inteligencia Artificial y Ciencia de Datos**
*Universidad Mayor de San Andrés (UMSA)*

## 📌 Descripción General
Este proyecto tiene como objetivo desarrollar una aplicación web interactiva para el sistema de transporte por cable (inspirado en Mi Teleférico). La plataforma resolverá cuatro casos de estudio mediante Métodos Numéricos y Machine Learning para simular condiciones de NORMALIDAD, ESTRÉS y MAL CONDICIONAMIENTO.

## 🎯 Los 4 Módulos
1. **Detención de Emergencia:** EDOs (RK2, RK4), Integración y Raíces.
2. **Deterioro Mecánico:** Raíces y Sistemas Lineales.
3. **Oscilaciones por Viento:** EDOs e Integración.
4. **Reconstrucción de Sensores:** Interpolación (Lagrange, Newton, Splines).

## 🛠️ Tecnologías Utilizadas
* **Lenguaje:** Python
* **Frontend:** Streamlit
* **Backend y Base de Datos:** FastAPI + SQLite (SQLAlchemy)
* **Matemáticas:** NumPy, SciPy
* **Machine Learning:** Scikit-Learn, XGBoost, Isolation Forest

## 📂 Estructura del Repositorio
```text
SIMTELE-RISK-IA-CD/
│
├── data/
├── notebooks/
├── src/
│   ├── numerical_engine/
│   ├── ml_models/
│   └── validation/
├── app/
│   └── dashboard.py
├── docs/
└── requirements.txt
