# Predicción de Enfermedades Cardíacas con Regresión Logística

## Resumen

Implementación de **regresión logística desde cero** para la predicción de enfermedades cardíacas. El proyecto incluye análisis exploratorio de datos, entrenamiento del modelo, visualización de límites de decisión y aplicación de regularización L2.

## Conjunto de Datos

**Fuente**: [Kaggle Heart Disease UCI](https://www.kaggle.com/datasets/neurocipher/heartdisease)

- **303 pacientes** con registros clínicos
- **Rango de edad**: 29-77 años
- **Colesterol**: 112-564 mg/dL
- **Tasa de presencia**: ~55% (equilibrado)
- **6 características seleccionadas**: Age, Cholesterol, Max HR, Number of Vessels, Exercise Angina, Thallium

## Resultados

| Métrica | Valor |
|---------|-------|
| **Accuracy (test)** | 85.19% |
| **Precision** | 80.00% |
| **Recall** | 88.89% |
| **F1-Score** | 0.8421 |
| **AUC-ROC** | 0.9179 |

**Regularización**: λ=0 (óptimo sin penalización)

## Detalles de Implementación

### Funciones Principales

```python
def sigmoid(z):
    """Función sigmoide"""
    return 1 / (1 + np.exp(-z))

def compute_cost(w, b, X, y):
    """Costo - Entropía cruzada binaria"""
    # J(w,b) = -1/m * Σ[y*log(f) + (1-y)*log(1-f)]

def compute_gradient(w, b, X, y):
    """Gradientes ∂J/∂w y ∂J/∂b"""
    pass

def gradient_descent(X, y, w_init, b_init, alpha, num_iters):
    """Optimización con descenso de gradiente"""
    pass
```

### Regularización L2

```python
def compute_cost_reg(w, b, X, y, lambda_):
    """Costo regularizado: J_reg = J + (λ/2m)||w||²"""
    pass

def compute_gradient_reg(w, b, X, y, lambda_):
    """Gradientes regularizados: ∂J_reg/∂w_j = ∂J/∂w_j + (λ/m)w_j"""
    pass
```



## Cómo Correr

### Prerrequisitos
```bash
pip install numpy pandas matplotlib jupyter scikit-learn
```

### Ejecutar el Cuaderno
```bash
jupyter notebook heart-disease-risk-prediction.ipynb
```

**Tiempo estimado de ejecución**: 2-3 minutos

---

## Estructura del Repositorio

```
heart-disease-risk-prediction/
├── heart-disease-risk-prediction.ipynb      (46 celdas ejecutables)
├── README.md                                (este archivo)
├── Heart_Disease_Prediction.csv             (dataset original)
│
├── Reportes Generados/
│   ├── DATA_PREPARATION_REPORT.md
│   ├── MODEL_EVALUATION_REPORT.md
│   ├── DECISION_BOUNDARIES_REPORT.md
│   └── REGULARIZATION_L2_REPORT.md
│
└── Modelos/
    ├── logistic_regression_model.pkl
    └── logistic_regression_model.json
```

---

## Referencias

- **Conjunto de datos**: [Kaggle Heart Disease UCI](https://www.kaggle.com/datasets/neurocipher/heartdisease)
- **Teoría**: Fundamentos de regresión logística y clasificación

---

## Licencia

Este proyecto tiene fines educativos y forma parte de un ejercicio de machine learning.

---

## Autor y Envío

**Autor**: María Paula Rodriguez Muñoz (Ejercicio de ML)

**Tarea**: Predicción de enfermedades cardíacas mediante regresión logística

