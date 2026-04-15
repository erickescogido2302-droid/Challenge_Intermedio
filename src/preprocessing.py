import pandas as pd

def stage_1_analysis(file_path):
    # Cargar los datos
    df = pd.read_csv(file_path)
    
    # 1. Data Info 
    print("--- DATA INFO ---")
    print(df.info())
    
    # 2. Data Describe 
    print("\n--- DATA DESCRIBE ---")
    print(df.describe())
    
    # 3. Data Values and Counts [cite: 27]
    print("\n--- DIAGNOSIS COUNTS (M = Malignant, B = Benign) ---")
    print(df['diagnosis'].value_counts())
    
    return df

if __name__ == "__main__":
    path = "data/breast-cancer-wisconsin.data.csv"
    stage_1_analysis(path)