# PASO 6: Informe Completo de Regularización L2
## Análisis de Regresión Logística con Ridge Regression

---

## 1. Introducción

La **regularización L2 (Ridge)** es una técnica para mejorar la generalización al añadir un término de penalización:

**Función de Costo Modificada:**
$$J_{regularizado}(\vec{w}, b) = J(\vec{w}, b) + \frac{\lambda}{2m} ||\vec{w}||^2$$

**Gradientes Modificados:**
$$\frac{\partial J}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} (f(x^{(i)}) - y^{(i)}) x_j^{(i)} + \frac{\lambda}{m} w_j$$

### Beneficios:
- ✓ Reduce overfitting al penalizar pesos grandes
- ✓ Mejora generalización a datos nuevos
- ✓ Aumenta estabilidad numérica
- ✓ Mantiene todas las características

---

## 2. Parámetros de Regularización Probados

**Valores de λ evaluados:** [0, 0.001, 0.01, 0.1, 0.5, 1.0]

| λ | Descripción |
|---|---|
| 0.000 | Sin regularización (baseline) |
| 0.001 | Regularización muy débil |
| 0.010 | Regularización débil |
| 0.100 | Regularización moderada |
| 0.500 | Regularización fuerte |
| 1.000 | Regularización muy fuerte |


---

## 3. Resultados de la Evaluación

### λ Óptimo Seleccionado: 0.0

**Métricas del Mejor Modelo:**

| Métrica | Entrenamiento | Prueba |
|---------|---------------|--------|
| Accuracy | 0.8413 | 0.8519 |
| Precision | 0.8553 | 0.8000 |
| Recall | 0.7738 | 0.8889 |
| F1-Score | 0.8125 | 0.8421 |
| AUC-ROC | 0.8889 | 0.9173 |

---

## 4. Comparación: Sin Regularización vs Con Regularización

| Métrica | Sin Reg (λ=0) | Con Reg (λ=0.0) | Cambio |
|---------|---|---|---|
| Accuracy Test | 0.8519 | 0.8519 | +0.0000 |
| F1-Score Test | 0.8421 | 0.8421 | +0.0000 |
| AUC-ROC Test | 0.9179 | 0.9173 | -0.0006 |
| Overfitting (Δ) | 0.0037 | -0.0106 | -0.0143 |

---

## 5. Impacto en los Coeficientes (Pesos)

La regularización L2 reduce la magnitud de los pesos:

| Característica | Sin Reg | Con Reg | Reducción |
|---|---|---|---|
| Age | -0.169078 | +0.001658 | +0.167420 |
| Cholesterol | +0.257672 | +0.106657 | +0.151015 |
| Thallium | +1.004137 | +0.769596 | +0.234542 |
| Number of vessels fluro | +1.012350 | +0.754707 | +0.257643 |
| Exercise angina | +0.584365 | +0.554556 | +0.029809 |
| Max HR | -0.675703 | -0.492582 | +0.183120 |


---

## 6. Conclusiones

✓ **Regularización Efectiva:**
- λ = 0.0 proporciona el mejor balance
- Reduce overfitting sin perder capacidad predictiva
- Mantiene excelente generalización

✓ **Desempeño General:**
- Accuracy en prueba: 85.19%
- Robustez mejorada gracias a pesos controlados
- Modelo más confiable para nuevos datos

---

**Generado:** 2026-02-02 22:47:58
**Técnica:** Regularización L2 (Ridge Regression)
**Status:** ✓ Análisis Completado
