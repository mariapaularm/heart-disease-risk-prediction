# Informe de Preparación de Datos - Predicción de Riesgo de Enfermedad Cardíaca

## 1. Origen de los Datos

- **Fuente**: Kaggle - Heart Disease Dataset (UCI Machine Learning Repository)
- **URL**: https://www.kaggle.com/datasets/neurocipher/heartdisease
- **Descarga**: Realizada y almacenada localmente como `Heart_Disease_Prediction.csv`

## 2. Información del Dataset Original

| Métrica | Valor |
|---------|-------|
| **Total de muestras** | 270 registros de pacientes |
| **Total de características** | 13 características clínicas |
| **Característica objetivo** | Heart Disease (binaria: 0 = sin enfermedad, 1 = presencia) |
| **Muestras sin enfermedad (Clase 0)** | 150 (55.6%) |
| **Muestras con enfermedad (Clase 1)** | 120 (44.4%) |
| **Balance de clases** | Relativamente equilibrado |
| **Datos faltantes** | ✓ Ninguno |
| **Valores atípicos detectados** | Revisados mediante método IQR |

## 3. Características Seleccionadas

Se seleccionaron **6 características numéricas** para el modelado:

- Age
- Cholesterol
- Thallium
- Number of vessels fluro
- Exercise angina
- Max HR

**Criterio de selección**: Relevancia clínica e importancia en predicción de enfermedad cardíaca

## 4. Preprocesamiento de Datos

### 4.1 Análisis de Datos Faltantes
- ✓ **Resultado**: No se encontraron datos faltantes en el dataset
- **Acción tomada**: Ninguna (dataset limpio)

### 4.2 Manejo de Valores Atípicos
- Método: Rango Intercuartílico (IQR)
- Umbral: 1.5 × IQR
- **Decisión**: Se mantuvieron todos los outliers (potencialmente significativos clínicamente)

### 4.3 División Estratificada Train-Test (70/30)
| Conjunto | Muestras | Clase 0 | Clase 1 | % Clase 0 | % Clase 1 |
|----------|----------|---------|---------|-----------|-----------|
| **Entrenamiento (70%)** | 189 | 105 | 84 | 55.6% | 44.4% |
| **Prueba (30%)** | 81 | 45 | 36 | 55.6% | 44.4% |

**Método**: `train_test_split` con `stratify=y` para mantener proporción de clases
**Semilla aleatoria**: 42 (reproducibilidad)

### 4.4 Normalización de Características (Z-Score Normalization)
- **Método**: StandardScaler (sklearn)
- **Transformación**: $x_{norm} = \frac{x - \mu}{\sigma}$
- **Ajuste**: Realizado SOLO con datos de entrenamiento (prevenir data leakage)
- **Aplicación**: Transformación idéntica aplicada a conjunto de prueba

Estadísticas después de normalización:
- **Media (esperada)**: 0.0 para todas las características
- **Desviación Estándar (esperada)**: 1.0 para todas las características

## 5. Datos Listos para Modelado

| Variable | Shape | Descripción |
|----------|-------|-------------|
| `X_train_normalized` | (189, 6) | Características de entrenamiento normalizadas |
| `X_test_normalized` | (81, 6) | Características de prueba normalizadas |
| `y_train` | (189,) | Etiquetas de entrenamiento |
| `y_test` | (81,) | Etiquetas de prueba |

## 6. Próximos Pasos

1. ✓ Carga y exploración de datos (completado)
2. ✓ Preparación y normalización (completado)
3. → Implementación de regresión logística
4. → Entrenamiento con descenso de gradiente
5. → Visualización de límites de decisión
6. → Regularización L2
7. → Evaluación de métricas
8. → Análisis con Amazon SageMaker

---
**Fecha de generación**: 2026-01-30 19:56:40
**Estado**: Datos preparados y listos para modelado
