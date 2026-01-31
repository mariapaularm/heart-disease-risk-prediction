# Análisis de Límites de Decisión

## Observaciones Generales

1. **Separabilidad de Clases**: Los límites de decisión muestran cómo el modelo logístico
   puede capturar relaciones lineales en espacios 2D de características.

2. **No-linealidad**: Aunque la regresión logística es un modelo lineal (en el espacio
   de características), los límites de decisión pueden aparecer curvos debido a la
   transformación sigmoide aplicada.

3. **Importancia Relativa**: La distancia del límite de decisión al origen indica
   la relevancia relativa de cada característica en la predicción.

## Pares de Características Analizados

| Par | Status | Observación |
|-----|--------|-------------|
| Age vs Cholesterol | Analizado | - |
| Max HR vs Thallium | Analizado | - |
| Exercise angina vs Vessels | Analizado | - |

## Recomendaciones

- Para mejorar la separabilidad, considerar:
  - Transformaciones no-lineales de características
  - Modelos más complejos (e.g., redes neuronales)
  - Ingeniería de características (interacciones)
  - Regularización L2 para evitar overfitting
