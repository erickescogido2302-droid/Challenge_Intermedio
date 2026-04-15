import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report, confusion_matrix, 
                             ConfusionMatrixDisplay, roc_curve, auc)
from preprocessing import preprocess_data

def evaluate_model(model, X_test, y_test):
    print("--- Evaluating Model ---")
    
    # 1. Realizar predicciones
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # 2. Generar Reporte de Métricas (Precision, Recall, F1)
    report = classification_report(y_test, y_pred)
    print("\nClassification Report:")
    print(report)
    
    # 3. Matriz de Confusión
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benign', 'Malignant'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.savefig('notebooks/confusion_matrix.png') # Guardar como artefacto
    plt.show()
    
    # 4. Curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig('notebooks/roc_curve.png') # Guardar como artefacto
    plt.show()

if __name__ == "__main__":
    # Cargar datos y modelo
    path = "data/breast-cancer-wisconsin.data.csv"
    _, X_test, _, y_test = preprocess_data(path)
    model = joblib.load('src/cancer_model.pkl')
    
    evaluate_model(model, X_test, y_test)