import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import joblib
from preprocessing import preprocess_data

def train_model(X_train, y_train):
    print("--- Training Model ---")
    
    # 1. Inicializar el modelo (Random Forest) [cite: 31]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # 2. Aplicar Cross Validation [cite: 34, 135]
    scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"Cross-validation scores: {scores}")
    print(f"Mean accuracy: {scores.mean():.4f}")
    
    # 3. Ajustar el modelo con los datos de entrenamiento 
    model.fit(X_train, y_train)
    
    # 4. Guardar el modelo entrenado
    joblib.dump(model, 'src/cancer_model.pkl')
    print("Model saved as src/cancer_model.pkl")
    
    return model

if __name__ == "__main__":
    # Obtener datos preprocesados del script anterior
    path = "data/breast-cancer-wisconsin.data.csv"
    X_train, X_test, y_train, y_test = preprocess_data(path)
    
    # Entrenar
    train_model(X_train, y_train)