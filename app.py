import streamlit as st
import joblib
import pandas as pd

# Título
st.title("Predicción de Género con Machine Learning")

# Inputs del usuario
ingresos = st.number_input("Ingresos Mensuales", min_value=0, value=5000)
frecuencia = st.number_input("Frecuencia de Compra", min_value=0, value=3)
ult_compra = st.number_input("Última compra (días)", min_value=0, value=10)
modelo_seleccionado = st.selectbox("Seleccione el modelo", ["Regresión Logística", "Random Forest"])

# Crear DataFrame con columnas en el mismo orden que el entrenamiento
datos_usuario = pd.DataFrame([[ingresos, frecuencia, ult_compra]], 
                             columns=["Ingresos_Mensuales", "Frecuencia_Compra", "Ultima_Compra_dias"])

# Cargar modelos
modelo_log = joblib.load("modelo_logistico.pkl")
modelo_rf = joblib.load("modelo_randomforest.pkl")

# Selección del modelo
modelo = modelo_log if modelo_seleccionado == "Regresión Logística" else modelo_rf

# Botón para predecir
if st.button("Predecir"):
    prediccion = modelo.predict(datos_usuario)
    genero = "Masculino" if prediccion[0] == 1 else "Femenino"
    st.success(f"Predicción: {genero}")
