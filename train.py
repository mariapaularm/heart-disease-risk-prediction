import json
import numpy as np
import pandas as pd
import os
import joblib

# Crear directorio de salida si no existe
os.makedirs('model_output', exist_ok=True)

print("=" * 80)
print("ENTRENAMIENTO DEL MODELO - HEART DISEASE PREDICTION")
print("=" * 80)

# ============================================================================
# FUNCIONES DE REGRESIÓN LOGÍSTICA (DESDE CERO - SOLO NUMPY)
# ============================================================================

def sigmoid(z):
    """Función sigmoid: σ(z) = 1 / (1 + e^-z)"""
    return 1 / (1 + np.exp(-z))

def compute_cost_reg(w, b, X, y, lambda_):
    """
    Calcula el costo regularizado (L2).
    J_reg = J + (λ/2m)||w||²
    """
    m = X.shape[0]
    z = np.dot(X, w) + b
    a = sigmoid(z)
    
    # Costo sin regularización
    J = -(1/m) * np.sum(y * np.log(a + 1e-15) + (1-y) * np.log(1-a + 1e-15))
    
    # Término de regularización L2
    reg_term = (lambda_ / (2*m)) * np.sum(w**2)
    
    return J + reg_term

def compute_gradient_reg(w, b, X, y, lambda_):
    """
    Calcula los gradientes regularizados (L2).
    ∂J_reg/∂w_j = ∂J/∂w_j + (λ/m)w_j
    """
    m = X.shape[0]
    z = np.dot(X, w) + b
    a = sigmoid(z)
    
    # Gradientes sin regularización
    dw = (1/m) * np.dot(X.T, (a - y))
    db = (1/m) * np.sum(a - y)
    
    # Agregar regularización L2 al gradiente de w
    dw += (lambda_/m) * w
    
    return dw, db

def train_model(X, y, learning_rate=0.01, iterations=10000, lambda_=0.0):
    """
    Entrena la regresión logística usando gradient descent.
    """
    m, n = X.shape
    w = np.zeros(n)
    b = 0
    cost_history = []
    
    for i in range(iterations):
        dw, db = compute_gradient_reg(w, b, X, y, lambda_)
        w -= learning_rate * dw
        b -= learning_rate * db
        
        cost = compute_cost_reg(w, b, X, y, lambda_)
        cost_history.append(cost)
        
        if (i + 1) % 1000 == 0:
            print(f"    Iteración {i+1}/{iterations} - Costo: {cost:.6f}")
    
    return w, b, cost_history

def predict(X, w, b):
    """Hace predicciones."""
    z = np.dot(X, w) + b
    return sigmoid(z)

def accuracy(X, y, w, b, threshold=0.5):
    """Calcula accuracy."""
    predictions = predict(X, w, b)
    y_pred = (predictions >= threshold).astype(int)
    return np.mean(y_pred == y)

# ============================================================================
# 1. CARGAR DATOS
# ============================================================================
print("\n[1] Cargando datos...")
df = pd.read_csv('Heart_Disease_Prediction.csv')
print(f"    ✓ Dataset cargado: {df.shape[0]} muestras, {df.shape[1]} características")

# Separar features y target
X = df.drop('Heart Disease', axis=1)
y = df['Heart Disease']

# Convertir 'Presence'/'Absence' a 1/0
y = (y == 'Presence').astype(int)

feature_names = X.columns.tolist()

print(f"    ✓ Features: {feature_names}")
print(f"    ✓ Target distribution: {y.value_counts().to_dict()}")

# ============================================================================
# 2. NORMALIZAR (StandardScaler solo para preprocesamiento)
# ============================================================================
print("\n[2] Normalizando features...")

# Crear scaler manualmente con NumPy
X_mean = np.mean(X.values, axis=0)
X_std = np.std(X.values, axis=0)
X_scaled = (X.values - X_mean) / X_std

print(f"    ✓ Media: {X_mean[:3]}... (primeras 3)")
print(f"    ✓ Desv. Est.: {X_std[:3]}... (primeras 3)")

# Guardar scaler para inference
scaler_dict = {'mean': X_mean.tolist(), 'std': X_std.tolist()}

# ============================================================================
# 3. ENTRENAR MODELO (SIN SKLEARN - SOLO NUMPY)
# ============================================================================
print("\n[3] Entrenando modelo (Logistic Regression con L2 - NumPy puro)...")
learning_rate = 0.01
iterations = 10000
lambda_ = 0.0  # L2 regularization

w_trained, b_trained, cost_history = train_model(
    X_scaled, y.values, 
    learning_rate=learning_rate, 
    iterations=iterations,
    lambda_=lambda_
)
print(f"    ✓ Modelo entrenado exitosamente")

# ============================================================================
# 4. EVALUAR
# ============================================================================
train_acc = accuracy(X_scaled, y.values, w_trained, b_trained)
print(f"\n[4] Evaluación:")
print(f"    ✓ Accuracy (Train): {train_acc:.4f}")
print(f"    ✓ Coeficientes (w): {w_trained}")
print(f"    ✓ Bias (b): {b_trained:.6f}")

# ============================================================================
# 5. GUARDAR PARÁMETROS EN NUMPY
# ============================================================================
print("\n[5] Guardando parámetros en formato NumPy...")
model_params = np.concatenate([w_trained, [b_trained]])
numpy_path = 'model_output/better_modelo.npy'
np.save(numpy_path, model_params)
print(f"    ✓ Parámetros NumPy guardados: {numpy_path}")
print(f"    ✓ Dimensión: {model_params.shape} ({len(w_trained)} pesos + 1 bias)")

# ============================================================================
# 6. GUARDAR SCALER Y MODELO EN JOBLIB (para compatibility)
# ============================================================================
print("\n[6] Guardando artefactos...")
scaler_path = 'model_output/scaler.joblib'
model_dict = {'w': w_trained, 'b': b_trained, 'type': 'LogisticRegression_NumPy'}
model_path = 'model_output/model.joblib'

joblib.dump(scaler_dict, scaler_path)
joblib.dump(model_dict, model_path)
print(f"    ✓ Scaler guardado: {scaler_path}")
print(f"    ✓ Modelo guardado: {model_path}")

# ============================================================================
# 7. GUARDAR METADATOS
# ============================================================================
print("\n[7] Guardando metadatos...")
metadata = {
    'features': feature_names,
    'n_features': len(feature_names),
    'model_type': 'LogisticRegression_NumPy',
    'implementation': 'Desde cero (sin sklearn)',
    'lambda': float(lambda_),
    'learning_rate': learning_rate,
    'iterations': iterations,
    'train_accuracy': float(train_acc),
    'scaler_mean': scaler_dict['mean'],
    'scaler_std': scaler_dict['std'],
    'coefficients': w_trained.tolist(),
    'intercept': float(b_trained)
}
metadata_path = 'model_output/metadata.json'
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"    ✓ Metadatos guardados: {metadata_path}")

print("\n" + "=" * 80)
print("✓ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
print("=" * 80)
print("\nArchivos generados en 'model_output/':")
print("  - model.joblib (parámetros w, b)")
print("  - scaler.joblib (media y desv. est.)")
print("  - better_modelo.npy (parámetros en NumPy)")
print("  - metadata.json (metadatos del modelo)")

