#!/usr/bin/env python3
"""
Script para consumir features desde Feast y hacer predicciones usando modelos registrados en MLflow.
Este script demuestra cómo integrar Feast Feature Store con MLflow para hacer predicciones en producción.
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from feast import FeatureStore
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def initialize_feature_store():
    """
    Inicializar el FeatureStore de Feast
    """
    print("Inicializando FeatureStore...")
    try:
        # Inicializar FeatureStore desde el directorio feature_repo
        store = FeatureStore(repo_path="./feature_repo")
        print("FeatureStore inicializado correctamente")
        return store
    except Exception as e:
        print(f"Error al inicializar FeatureStore: {e}")
        return None

def initialize_mlflow():
    """
    Inicializar MLflow y cargar el mejor modelo
    """
    print("Inicializando MLflow...")
    try:
        # Configurar tracking URI
        mlflow.set_tracking_uri("/home/hsilv/mlops/Lab03-CC3105/src/mlruns")
        
        # Buscar el mejor modelo basado en R² score
        experiment_id = "770921580831769508"  # california-housing-prediction
        runs = mlflow.search_runs(experiment_ids=[experiment_id])
        
        if runs.empty:
            print("No se encontraron runs en MLflow")
            return None, None
        
        # Encontrar el mejor modelo por R² score
        best_run = runs.loc[runs['metrics.r2_test'].idxmax()]
        best_run_id = best_run['run_id']
        best_model_name = best_run['tags.mlflow.runName']
        best_r2_score = best_run['metrics.r2_test']
        
        print(f"Mejor modelo encontrado: {best_model_name}")
        print(f"Run ID: {best_run_id}, R² Score: {best_r2_score:.4f}")
        
        # Cargar el mejor modelo desde Model Registry
        model = mlflow.sklearn.load_model("models:/california_housing_models_random_forest/1")
        print(f"Modelo cargado desde Model Registry: {type(model).__name__}")
        
        return model, best_run_id
        
    except Exception as e:
        print(f"Error al inicializar MLflow: {e}")
        return None, None

def retrieve_features_from_feast(store, entity_ids, feature_names=None):
    """
    Recuperar features desde Feast para un conjunto de entidades
    
    Args:
        store: FeatureStore de Feast
        entity_ids: Lista de house_id para los que queremos features
        feature_names: Lista de features específicas (opcional)
    
    Returns:
        DataFrame con las features recuperadas
    """
    print(f"Recuperando features para {len(entity_ids)} entidades...")
    
    try:
        # Crear DataFrame de entidades con timestamp
        entity_df = pd.DataFrame({
            'house_id': entity_ids,
            'event_timestamp': [datetime.now()] * len(entity_ids)
        })
        
        # Definir features a recuperar (todas las features del dataset)
        if feature_names is None:
            feature_names = [
                'california_housing_features:median_income',
                'california_housing_features:house_age', 
                'california_housing_features:average_rooms',
                'california_housing_features:average_bedrooms',
                'california_housing_features:population',
                'california_housing_features:average_occupants',
                'california_housing_features:latitude',
                'california_housing_features:longitude'
            ]
        
        # Recuperar features históricas para entrenamiento/batch scoring
        historical_features = store.get_historical_features(
            entity_df=entity_df,
            features=feature_names
        ).to_df()
        
        # También recuperar features online para demostración
        online_features = store.get_online_features(
            features=feature_names,
            entity_rows=entity_df.to_dict('records')
        ).to_df()
        
        print(f"Features recuperadas - Históricas: {historical_features.shape}, Online: {online_features.shape}")
        
        return historical_features, online_features
        
    except Exception as e:
        print(f"Error al recuperar features: {e}")
        return None, None

def make_predictions(model, features_df):
    """
    Hacer predicciones usando el modelo de MLflow
    
    Args:
        model: Modelo entrenado de MLflow
        features_df: DataFrame con las features
    
    Returns:
        Array con las predicciones
    """
    print("Haciendo predicciones...")
    
    try:
        # Preparar features para predicción
        # Remover columnas que no son features (house_id, event_timestamp, etc.)
        feature_columns = [col for col in features_df.columns 
                          if col not in ['house_id', 'event_timestamp', 'created']]
        
        # Crear DataFrame con nombres de columnas correctos
        X = features_df[feature_columns].copy()
        
        # Crear features derivadas como en el entrenamiento original
        X['rooms_per_household'] = X['average_rooms'] / X['average_occupants']
        X['population_per_household'] = X['population'] / X['average_bedrooms']  
        X['bedrooms_per_room'] = X['average_bedrooms'] / X['average_rooms']
        
        # Mapear nombres de columnas para que coincidan con el modelo entrenado
        column_mapping = {
            'median_income': 'median_income',
            'house_age': 'house_age',
            'average_rooms': 'average_rooms',
            'average_bedrooms': 'AveBedrms',  # El modelo espera el nombre original
            'population': 'Population',       # El modelo espera el nombre original
            'average_occupants': 'average_occupants',
            'latitude': 'latitude',
            'longitude': 'longitude',
            'rooms_per_household': 'rooms_per_household',
            'population_per_household': 'population_per_household',
            'bedrooms_per_room': 'bedrooms_per_room'
        }
        
        X = X.rename(columns=column_mapping)
        
        # Asegurar que las columnas estén en el orden correcto que espera el modelo
        expected_features = [
            'median_income', 'house_age', 'average_rooms', 'AveBedrms', 'Population',
            'average_occupants', 'latitude', 'longitude', 'rooms_per_household',
            'population_per_household', 'bedrooms_per_room'
        ]
        
        # Reordenar columnas para que coincidan exactamente con el modelo
        X = X[expected_features]
        
        # Hacer predicciones
        predictions = model.predict(X)
        
        print(f"Predicciones completadas: {len(predictions)} predicciones")
        print(f"Rango: ${predictions.min():.2f} - ${predictions.max():.2f}, Promedio: ${predictions.mean():.2f}")
        
        return predictions, feature_columns
        
    except Exception as e:
        print(f"Error al hacer predicciones: {e}")
        return None, None

def create_prediction_results(features_df, predictions, feature_columns):
    """
    Crear DataFrame con los resultados de las predicciones
    """
    print("Creando resultados de predicciones...")
    
    try:
        # Crear DataFrame de resultados
        results_df = features_df[['house_id']].copy()
        
        # Agregar features usadas
        for col in feature_columns:
            results_df[col] = features_df[col]
        
        # Agregar predicciones
        results_df['predicted_house_value'] = predictions
        results_df['predicted_house_value_usd'] = predictions * 100000  # Convertir a USD
        
        # Agregar timestamp de predicción
        results_df['prediction_timestamp'] = datetime.now()
        
        return results_df
        
    except Exception as e:
        print(f"Error al crear resultados: {e}")
        return None

def main():
    """
    Función principal que ejecuta todo el pipeline de consumo de features y predicciones
    """
    print("Iniciando pipeline de consumo de features y predicciones")
    print("=" * 60)
    
    # 1. Inicializar FeatureStore
    store = initialize_feature_store()
    if store is None:
        return
    
    # 2. Inicializar MLflow y cargar mejor modelo
    model, best_run_id = initialize_mlflow()
    if model is None:
        return
    
    print("\n" + "=" * 60)
    
    # 3. Definir entidades para predicción (ejemplo: primeras 10 casas)
    entity_ids = list(range(10))  # house_id de 0 a 9
    print(f"Entidades seleccionadas para predicción: {entity_ids}")
    
    # 4. Recuperar features online desde Feast
    historical_features, online_features = retrieve_features_from_feast(store, entity_ids)
    if online_features is None:
        return
    
    print("\n" + "=" * 60)
    
    # 5. Hacer predicciones usando el modelo de MLflow (usar features online)
    predictions, feature_columns = make_predictions(model, online_features)
    if predictions is None:
        return
    
    # 6. Crear resultados finales
    results_df = create_prediction_results(online_features, predictions, feature_columns)
    if results_df is None:
        return
    
    print("\n" + "=" * 60)
    print("RESULTADOS DE PREDICCIONES")
    print("=" * 60)
    
    # Mostrar resultados
    print("\nMuestra de predicciones:")
    print(results_df[['house_id', 'median_income', 'house_age', 'average_rooms', 
                     'predicted_house_value_usd']].head())
    
    print(f"\nEstadísticas de predicciones:")
    print(f"Promedio: ${results_df['predicted_house_value_usd'].mean():,.2f}")
    print(f"Mediana: ${results_df['predicted_house_value_usd'].median():,.2f}")
    print(f"Mínimo: ${results_df['predicted_house_value_usd'].min():,.2f}")
    print(f"Máximo: ${results_df['predicted_house_value_usd'].max():,.2f}")
    
    # Guardar resultados
    output_file = "./prediction_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nResultados guardados en: {output_file}")
    
    print("\n" + "=" * 60)
    print("Pipeline completado exitosamente!")
    print("=" * 60)
    
    return results_df

if __name__ == "__main__":
    # Ejecutar el pipeline principal
    results = main()
