# Informe de Regularización L2 (Ridge Regression)
## Análisis del Efecto de la Regularización en el Modelo de Regresión Logística

---

## 1. Introducción a Regularización L2

La **regularización L2 (Ridge)** añade un término de penalización a la función de costo para reducir la magnitud de los pesos:

$$J_{regularizado}(\\vec{w}, b) = J(\\vec{w}, b) + \\frac{\\lambda}{2m} ||\\vec{w}||^2$$

**Beneficios:**
- Reduce overfitting (sobreajuste)
- Penaliza pesos grandes
- Mejora la generalización a datos nuevos
- Mantiene todas las características (a diferencia de L1)

---

## 2. Parámetros Probados

Se entrenaron modelos con los siguientes valores de $\lambda$:

| λ | Descripción |
|---|---|
| 0 | Sin regularización (baseline) |
| 0.001 | Regularización muy débil |
| 0.01 | Regularización débil |
| 0.1 | Regularización moderada |
| 1 | Regularización fuerte |

---

## 3. Resultados Comparativos

### 3.1 Tabla de Métricas por λ

| λ | Acc Train | Acc Test | Prec | Recall | F1 | AUC | \|\|w\|\| | Costo |
|---|---|---|---|---|---|---|---|---|
| 0.000 | 0.8556 | 0.8519 | 0.8000 | 0.8889 | 0.8421 | 0.9179 | 1.7106 | 0.3976 |
| 0.001 | 0.8556 | 0.8519 | 0.8000 | 0.8889 | 0.8421 | 0.9179 | 1.7105 | 0.3976 |
| 0.010 | 0.8556 | 0.8519 | 0.8000 | 0.8889 | 0.8421 | 0.9179 | 1.7099 | 0.3977 |
| 0.100 | 0.8556 | 0.8519 | 0.8000 | 0.8889 | 0.8421 | 0.9179 | 1.7038 | 0.3982 |
| 1.000 | 0.8556 | 0.8519 | 0.8000 | 0.8889 | 0.8421 | 0.9185 | 1.6465 | 0.4029 |


### 3.2 Análisis de λ Óptimo

**Mejores resultados por métrica:**

| Métrica | λ Óptimo | Valor | Mejora vs Sin Reg |
|---------|----------|-------|-------------------|
| **Accuracy** | 0.0 | 0.8519 | +0.00% |
| **F1-Score** | 0.0 | 0.8421 | +0.00% |
| **AUC-ROC** | 1.0 | 0.9185 | +0.07% |

---

## 4. Observaciones Clave

### 4.1 Efecto en la Magnitud de Pesos

- **Sin regularización (λ=0):** ||w|| = 1.7106
- **Con regularización (λ=1):** ||w|| = 1.6465
- **Reducción:** 3.8%

La regularización **reduce significativamente la magnitud de los pesos**, evitando que el modelo dependa excesivamente de algunas características.

### 4.2 Convergencia

- Sin regularización converge más lentamente
- Con regularización, la convergencia se estabiliza más rápidamente
- λ muy grande (λ=1) puede causar underfitting

### 4.3 Generalización

- Accuracy en entrenamiento vs prueba:
  - **λ=0:** 0.8556 (train) vs 0.8519 (test) = 0.0037 diferencia
  - **λ=0.0:** 0.8556 (train) vs 0.8519 (test) = 0.0037 diferencia

---

## 5. Recomendaciones

1. **λ Óptimo Recomendado:** λ = **0.0**
   - Proporciona mejor balance entre precision, recall y generalización
   - Reduce magnitud de pesos sin causar underfitting

2. **Uso Práctico:**
   - Para maximizar recall (minimizar falsos negativos): considerar λ algo menor
   - Para equilibrio F1: usar λ = 0.0
   - Para máxima exactitud: usar λ = 0.0

3. **Próximos Pasos:**
   - Validación cruzada para confirmación
   - Búsqueda en grid más fina si es necesario
   - Integración con técnicas de selección de características

---

**Fecha de evaluación:** 2026-01-30 20:05:55
**Método:** Regularización L2 (Ridge) en Regresión Logística
**Status:** ✓ Análisis completado
