import mlflow
import mlflow.sklearn
import joblib
from preprocessing import preprocess_data
from model_training import train_model
from evaluation import evaluate_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def run_mlops_pipeline():
    # 1. Configurar el experimento
    mlflow.set_experiment("Cancer_Detection_Challenge")
    
    with mlflow.start_run():
        # Cargar y procesar datos
        path = "data/breast-cancer-wisconsin.data.csv"
        X_train, X_test, y_train, y_test = preprocess_data(path)
        
        # 2. Entrenar y Registrar Parámetros
        model = train_model(X_train, y_train)
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", 100)
        
        # 3. Predicciones y Métricas
        y_pred = model.predict(X_test)
        
        # Log metrics (Requerimiento Stage 3)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        
        # 4. Registrar el Modelo (Model Signature)
        mlflow.sklearn.log_model(model, "cancer_model")
        
        # 5. Registrar Artefactos (Gráficas)
        # Asegúrate de haber corrido evaluation.py para generar los archivos
        mlflow.log_artifact("notebooks/confusion_matrix.png")
        mlflow.log_artifact("notebooks/roc_curve.png")
        
        print("--- MLOps Run Completed Successfully ---")

if __name__ == "__main__":
    run_mlops_pipeline()