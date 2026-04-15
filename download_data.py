import pandas as pd
import os

# URL oficial del dataset Breast Cancer Wisconsin (Diagnostic)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"

# Nombres de las columnas (según la documentación de UCI)
column_names = [
    'id', 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 
    'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave_points_mean', 
    'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 
    'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 
    'concave_points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 
    'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 
    'compactness_worst', 'concavity_worst', 'concave_points_worst', 
    'symmetry_worst', 'fractal_dimension_worst'
]

def download():
    print("Descargando dataset...")
    df = pd.read_csv(url, header=None, names=column_names)
    
    # Crear carpeta data si no existe
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Guardar como CSV en la ruta que pide tu reto
    output_path = "data/breast-cancer-wisconsin.data.csv"
    df.to_csv(output_path, index=False)
    print(f"Archivo guardado exitosamente en: {output_path}")

if __name__ == "__main__":
    download()