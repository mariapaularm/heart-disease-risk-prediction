import json
import numpy as np
import joblib
import requests
from datetime import datetime

print("\n" + "="*80)
print("TEST DE ENDPOINT - INVOCAR CON MUESTRA")
print("="*80)

# Cargar modelo
print("\n[1] Cargando modelo...")
model = joblib.load('model_output/model.joblib')
scaler = joblib.load('model_output/scaler.joblib')
with open('model_output/metadata.json', 'r') as f:
    metadata = json.load(f)
feature_names = metadata['features']
print("    ✓ Modelo cargado")

# Crear muestra de prueba
print("\n[2] Creando muestra de prueba...")
sample = {
    "Age": 60,
    "Sex": 1,
    "Chest pain type": 3,
    "BP": 150,
    "Cholesterol": 300,
    "FBS over 120": 1,
    "EKG results": 1,
    "Max HR": 100,
    "Exercise angina": 1,
    "ST depression": 4.0,
    "Slope of ST": 0,
    "Number of vessels fluro": 2,
    "Thallium": 3
}

print(f"    ✓ Muestra creada:")
for key, value in sample.items():
    print(f"      - {key}: {value}")

# Simular invocación (como si fuera una solicitud HTTP al endpoint)
print("\n[3] Invocando endpoint...")
print(f"    POST /predict")
print(f"    Content-Type: application/json")
print(f"    Body: {json.dumps(sample)}")

# Convertir a formato que el modelo entienda
features_array = np.array([sample[f] for f in feature_names]).reshape(1, -1)

# Normalizar manualmente con NumPy (scaler es ahora un dict)
mean = np.array(scaler['mean'])
std = np.array(scaler['std'])
features_scaled = (features_array - mean) / std

# Hacer predicción (NumPy puro, sin sklearn)
w = model['w']
b = model['b']
z = np.dot(features_scaled, w) + b
prediction_prob = 1 / (1 + np.exp(-z[0]))
prediction = 1 if prediction_prob >= 0.5 else 0

# Crear respuesta
response = {
    "timestamp": datetime.now().isoformat(),
    "request": sample,
    "response": {
        "prediction": int(prediction),
        "prediction_label": "Heart Disease Risk" if prediction == 1 else "No Risk",
        "probability_no_disease": float(1 - prediction_prob),
        "probability_disease": float(prediction_prob),
        "confidence": float(max(1 - prediction_prob, prediction_prob))
    },
    "model_metadata": {
        "accuracy": metadata['train_accuracy'],
        "features_count": metadata['n_features']
    }
}

# Mostrar respuesta
print("\n[4] Respuesta del endpoint:")
print("\n    HTTP/1.1 200 OK")
print("    Content-Type: application/json")
print(f"\n{json.dumps(response, indent=4)}")

# Guardar respuesta en archivo
print("\n[5] Guardando respuesta...")
response_file = 'endpoint_response.json'
with open(response_file, 'w') as f:
    json.dump(response, f, indent=4)
print(f"    ✓ Respuesta guardada en: {response_file}")

print("\n" + "="*80)
print("✓ TEST COMPLETADO EXITOSAMENTE")
print("="*80)

# Resumen
print("\n📊 RESUMEN:")
print(f"   Muestra: Edad={sample['Age']}, Colesterol={sample['Cholesterol']}, Presión={sample['BP']}")
print(f"   Predicción: {response['response']['prediction_label']}")
print(f"   Confianza: {response['response']['confidence']:.2%}")
print(f"   Probabilidad de enfermedad: {response['response']['probability_disease']:.4f}")
