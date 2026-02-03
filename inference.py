import json
import numpy as np
import joblib

# Variables globales (se cargan una sola vez)
model = None
scaler = None
feature_names = None

def model_fn(model_dir):
    """
    Carga el modelo al iniciar el endpoint en SageMaker.
    SageMaker llama esta función automáticamente.
    """
    global model, scaler, feature_names
    
    print(f"[SageMaker] Cargando modelo desde: {model_dir}")
    
    # Cargar modelo y scaler
    model = joblib.load(f'{model_dir}/model.joblib')
    scaler = joblib.load(f'{model_dir}/scaler.joblib')
    
    with open(f'{model_dir}/metadata.json', 'r') as f:
        metadata = json.load(f)
    
    feature_names = metadata['features']
    print(f"[SageMaker] ✓ Modelo cargado. Features esperados: {len(feature_names)}")
    
    return model


def input_fn(request_body, request_content_type='application/json'):
    """
    Procesa la entrada del cliente (datos del paciente).
    
    Entrada esperada (JSON):
    {
        "Age": 50,
        "Sex": 1,
        "Cp": 2,
        "Trestbps": 130,
        "Chol": 250,
        "Fbs": 0,
        "Restecg": 1,
        "Thalach": 150,
        "Exang": 0,
        "Oldpeak": 2.5,
        "Slope": 2
    }
    """
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        print(f"[Inference] Input recibido: {input_data}")
        return input_data
    else:
        raise ValueError(f"Tipo de contenido no soportado: {request_content_type}")


def normalize_features(features, scaler_dict):
    """
    Normaliza los features usando media y desv. est. (NumPy puro).
    """
    mean = np.array(scaler_dict['mean'])
    std = np.array(scaler_dict['std'])
    return (features - mean) / std


def predict_fn(input_data, model):
    """
    Realiza la predicción usando el modelo cargado.
    """
    global scaler, feature_names
    
    print(f"[Inference] Procesando predicción...")
    
    # Convertir dict a array en el orden correcto de features
    features = np.array([input_data[f] for f in feature_names]).reshape(1, -1)
    
    print(f"[Inference] Features originales: {features}")
    
    # Normalizar usando el scaler (que ahora es un diccionario)
    features_scaled = normalize_features(features, scaler)
    
    print(f"[Inference] Features normalizados: {features_scaled}")
    
    # Predicción manual con NumPy (sin sklearn)
    w = model['w']
    b = model['b']
    
    # z = w*x + b
    z = np.dot(features_scaled, w) + b
    
    # Sigmoid
    prediction_prob = 1 / (1 + np.exp(-z[0]))
    prediction = 1 if prediction_prob >= 0.5 else 0
    
    result = {
        'prediction': int(prediction),
        'prediction_label': 'Heart Disease Risk' if prediction == 1 else 'No Risk',
        'probability_no_disease': float(1 - prediction_prob),
        'probability_disease': float(prediction_prob),
        'confidence': float(max(1 - prediction_prob, prediction_prob))
    }
    
    print(f"[Inference] Predicción: {result['prediction_label']} (confianza: {result['confidence']:.2%})")
    
    return result


def output_fn(prediction, response_content_type='application/json'):
    """
    Formatea la salida en JSON.
    """
    if response_content_type == 'application/json':
        return json.dumps(prediction)
    else:
        raise ValueError(f"Tipo de contenido no soportado: {response_content_type}")


# ============================================================================
# EJEMPLO DE USO LOCAL (para testing antes de desplegar en SageMaker)
# ============================================================================
if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("TESTING LOCAL DE INFERENCIA")
    print("=" * 80)
    
    # Simular carga del modelo (asume que 'model_output' existe localmente)
    print("\n[Test] Cargando modelo...")
    model = model_fn('model_output')
    
    # Test 1: Paciente CON riesgo
    print("\n[Test 1] Predicción - Paciente CON RIESGO:")
    test_data_1 = {
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
    
    input_data_1 = input_fn(json.dumps(test_data_1))
    result_1 = predict_fn(input_data_1, model)
    output_1 = output_fn(result_1)
    print(f"Resultado:\n{output_1}")
    
    # Test 2: Paciente SIN riesgo
    print("\n[Test 2] Predicción - Paciente SIN RIESGO:")
    test_data_2 = {
        "Age": 30,
        "Sex": 0,
        "Chest pain type": 1,
        "BP": 110,
        "Cholesterol": 180,
        "FBS over 120": 0,
        "EKG results": 0,
        "Max HR": 180,
        "Exercise angina": 0,
        "ST depression": 0.0,
        "Slope of ST": 2,
        "Number of vessels fluro": 0,
        "Thallium": 1
    }
    
    input_data_2 = input_fn(json.dumps(test_data_2))
    result_2 = predict_fn(input_data_2, model)
    output_2 = output_fn(result_2)
    print(f"Resultado:\n{output_2}")
    
    print("\n" + "=" * 80)
    print("✓ TESTING COMPLETADO")
    print("=" * 80)
