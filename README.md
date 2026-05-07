# Breast Cancer Classification with MLOps Pipeline

[cite_start]Este repositorio contiene un sistema automatizado de extremo a extremo para la clasificación de tumores (Benigno/Maligno) utilizando el dataset *Breast Cancer Wisconsin*[cite: 148]. [cite_start]El enfoque principal fue la integración de principios de **MLOps** para garantizar que el modelo sea reproducible, auditable y esté listo para entornos de producción[cite: 149, 172].

## 📋 Resumen Ejecutivo
[cite_start]El objetivo central es automatizar el ciclo de vida del aprendizaje automático, integrando **MLflow** para el seguimiento de experimentos, registro de métricas y gestión de artefactos[cite: 149].

* [cite_start]**Tipo de Modelo:** Clasificador Random Forest[cite: 153].
* [cite_start]**Rendimiento:** Precisión media del **95.60%** validada mediante K-Fold Cross-Validation ($K=5$)[cite: 157, 158].
* [cite_start]**Impacto:** Alta sensibilidad para minimizar los falsos negativos, un requisito crítico en diagnósticos médicos[cite: 161].

---

## 🏗️ Arquitectura del Pipeline
[cite_start]El sistema está modularizado en cuatro componentes principales para asegurar una separación clara de responsabilidades[cite: 150, 151]:

1.  [cite_start]**`preprocessing.py`**: Limpieza de datos, imputación de valores faltantes y escalado de características[cite: 152].
2.  [cite_start]**`model_training.py`**: Entrenamiento del clasificador con validación cruzada[cite: 153].
3.  [cite_start]**`evaluation.py`**: Generación de métricas de rendimiento y visualizaciones técnicas (Matriz de Confusión y Curva ROC)[cite: 154].
4.  [cite_start]**`mlops_pipeline.py`**: El orquestador que gestiona el ciclo de vida en MLflow y registra cada ejecución[cite: 155].


---

## 🚀 Integración de MLOps
[cite_start]Se utilizó **MLflow** con un backend de SQLite local para persistir todos los datos experimentales[cite: 164]:

* [cite_start]**Registro de Métricas**: Seguimiento automático de Accuracy, Precision, Recall y F1-Score[cite: 165].
* [cite_start]**Persistencia de Artefactos**: Almacenamiento directo del modelo serializado (`.pkl`) y gráficos de evaluación (`.png`)[cite: 166, 182].
* [cite_start]**Interfaz de Usuario (UI)**: Visualización de resultados a través de un servidor local (Puerto 5050) para facilitar la comparación de versiones[cite: 167, 201].

---

## 📈 Resultados Visuales
[cite_start]El modelo demuestra una excelente capacidad discriminativa[cite: 162]:

* [cite_start]**Matriz de Confusión**: Refleja la alta precisión en ambas clases[cite: 161, 193].
* [cite_start]**Curva ROC**: Con un Área Bajo la Curva (AUC) de **1.00**, confirmando un rendimiento óptimo[cite: 162, 233].

---

## 🛠️ Entorno y Optimización
[cite_start]Para evitar problemas de bloqueo de archivos comunes en entornos sincronizados (como OneDrive), el proyecto se desplegó en un directorio local dedicado (`C:\Proyectos\Challenge`)[cite: 169]. [cite_start]Esta optimización resolvió problemas de latencia durante el registro asíncrono de artefactos en la base de datos[cite: 170].

[cite_start]**Autor:** Erick de Jesús Escogido Escobedo [cite: 138]
[cite_start]**Programa:** Maestría en Ciencia de los Datos (MCD) - CUCEA [cite: 142, 145]


4\. View results: Run mlflow ui in your terminal and open http://localhost:5000





