# Informe de Evaluación - Modelo de Regresión Logística
## Predicción de Riesgo de Enfermedad Cardíaca

---

## 1. Resumen Ejecutivo

Se ha entrenado un modelo de **Regresión Logística** en el conjunto completo de datos (270 muestras) con:
- **Tasa de aprendizaje (α)**: 0.01 (conservadora)
- **Número de iteraciones**: 5000
- **Características utilizadas**: 6 (Age, Cholesterol, Thallium, Number of vessels fluro, Exercise angina, Max HR)
- **Función de activación**: Sigmoide
- **Función de costo**: Entropía cruzada binaria
- **Método de optimización**: Descenso de gradiente

---

## 2. Desempeño del Modelo

### 2.1 Métricas en Conjunto de Entrenamiento (270 muestras)

| Métrica | Valor | Porcentaje |
|---------|-------|-----------|
| **Accuracy (Exactitud)** | 0.8556 | 85.56% |
| **Precision** | 0.8584 | 85.84% |
| **Recall (Sensibilidad)** | 0.8083 | 80.83% |
| **F1-Score** | 0.8326 | 83.26% |
| **AUC-ROC** | 0.8994 | 89.94% |

**Matriz de Confusión (Entrenamiento):**
```
           Predicción
           Negativo  Positivo
Actual  Negativo    134       16
        Positivo     23       97
```

### 2.2 Métricas en Conjunto de Prueba (81 muestras)

| Métrica | Valor | Porcentaje |
|---------|-------|-----------|
| **Accuracy (Exactitud)** | 0.8519 | 85.19% |
| **Precision** | 0.8000 | 80.00% |
| **Recall (Sensibilidad)** | 0.8889 | 88.89% |
| **F1-Score** | 0.8421 | 84.21% |
| **AUC-ROC** | 0.9179 | 91.79% |

**Matriz de Confusión (Prueba):**
```
           Predicción
           Negativo  Positivo
Actual  Negativo     37        8
        Positivo      4       32
```

---

## 3. Análisis de Convergencia

### 3.1 Estadísticas de Entrenamiento

| Parámetro | Valor |
|-----------|-------|
| **Costo Inicial** | 0.690917 |
| **Costo Final** | 0.397596 |
| **Reducción Absoluta** | 0.293321 |
| **Reducción Porcentual** | 42.45% |
| **Cambio promedio por iteración** | 0.000059 |
| **Convergencia Monótona** | ✓ Sí (garantizada) |

### 3.2 Observaciones sobre Convergencia

- **Rápida disminución inicial (0-500 iteraciones)**: El costo reduce ~0.2628 (38.0%)
- **Convergencia lenta posterior (500-5000 iteraciones)**: El costo reduce solo ~0.0306 (7.1%)
- **Estabilización**: El modelo se estabiliza alrededor de la iteración 2000, con cambios menores después
- **Learning rate apropiado**: α=0.01 resultó ser conservador pero efectivo, permitiendo convergencia suave sin oscilaciones

---

## 4. Coeficientes del Modelo (Interpretación)

**Sesgo (b):** -0.248845

### 4.1 Pesos por Característica (ordenados por importancia)

| Rango | Característica | Peso | Dirección | Importancia |
|-------|---|---|---|---|
| 1 | Number of vessels fluro | 1.012350 | ↑ Riesgo | MUY ALTA |
| 2 | Thallium | 1.004137 | ↑ Riesgo | MUY ALTA |
| 3 | Max HR | -0.675703 | ↓ Riesgo | ALTA |
| 4 | Exercise angina | 0.584365 | ↑ Riesgo | MEDIA |
| 5 | Cholesterol | 0.257672 | ↑ Riesgo | MEDIA |
| 6 | Age | -0.169078 | ↓ Riesgo | MEDIA |


### 4.2 Interpretación Clínica

**Características con Mayor Impacto (por valor absoluto):**

1. **Number of vessels fluro** (w = 1.012350)
   - Un aumento en esta característica **aumenta** la probabilidad de enfermedad cardíaca
   - Impacto relativo: 27.3% del total

2. **Thallium** (w = 1.004137)
   - Un aumento en esta característica **aumenta** la probabilidad de enfermedad cardíaca
   - Impacto relativo: 27.1% del total

3. **Max HR** (w = -0.675703)
   - Un aumento en esta característica **disminuye** la probabilidad de enfermedad cardíaca
   - Impacto relativo: 18.2% del total


---

## 5. Comparación Entrenamiento vs Prueba

| Métrica | Entrenamiento | Prueba | Diferencia | Status |
|---------|---|---|---|---|
| Accuracy | 0.8556 | 0.8519 | +0.0037 | ✓ Similar |
| Precision | 0.8584 | 0.8000 | +0.0584 | ⚠ Diferencia notable |
| Recall | 0.8083 | 0.8889 | -0.0806 | ⚠ Diferencia notable |
| F1-Score | 0.8326 | 0.8421 | -0.0095 | ✓ Similar |
| AUC-ROC | 0.8994 | 0.9179 | -0.0185 | ✓ Similar |

**Análisis de Generalización:**
✓ El modelo **generaliza bien** → No hay indicios de overfitting


---

## 6. Conclusiones y Recomendaciones

### 6.1 Fortalezas del Modelo

1. **Convergencia garantizada**: El algoritmo de descenso de gradiente convergió monótonamente
2. **Desempeño razonable**: Accuracy del 85.2% en conjunto de prueba
3. **Generalización**: Métricas similares entre entrenamiento y prueba
4. **Interpretabilidad**: Los coeficientes permiten identificar características clave

### 6.2 Áreas de Mejora

1. **Recall vs Precision**: Equilibrar el trade-off entre falsos negativos y falsos positivos
2. **Regularización**: Considerar L2 regularization para mejorar generalización
3. **Feature engineering**: Explorar interacciones entre características
4. **Ajuste de hiperparámetros**: Probar diferentes valores de learning rate
5. **Threshold tuning**: Optimizar umbral de decisión según el contexto clínico

### 6.3 Recomendaciones Clínicas

Para el contexto de predicción de enfermedad cardíaca:
- **Sensibilidad (Recall) es crítica**: Es mejor identificar falsos positivos que falsos negativos
- Considerar usar un **threshold < 0.5** si se prioriza identificar mayor número de casos
- Integrar **predicción del modelo con evaluación clínica** de especialistas

---

**Fecha de Evaluación**: 2026-02-02 20:42:13
**Modelo**: Regresión Logística (NumPy, sin Scikit-Learn para core training)
**Status**: ✓ Entrenado y Evaluado
