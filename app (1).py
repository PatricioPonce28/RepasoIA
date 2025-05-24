import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.title("Predicción de Género con Machine Learning")

# Inputs usuario
ingresos = st.number_input("Ingresos Mensuales", min_value=0, value=5000)
frecuencia = st.number_input("Frecuencia de Compra", min_value=0, value=3)
ult_compra = st.number_input("Última compra (días)", min_value=0, value=10)
ciudad = st.selectbox("Ciudad", ["Ambato", "Cuenca", "Guayaquil", "Quito"])

modelo_seleccionado = st.selectbox(
    "Seleccione el modelo",
    ["Regresión Logística", "Random Forest", "KNN", "Regresión Lineal"]
)

# Cargar modelos y transformadores
modelo_log = joblib.load("modelo_logistico.pkl")
modelo_rf = joblib.load("modelo_randomforest.pkl")
modelo_knn = joblib.load("modelo_knn.pkl")
modelo_regresion = joblib.load("modelo_regresion.pkl")

scaler = joblib.load("scaler.pkl")
encoder_ciudad = joblib.load("encoder_ciudad.pkl")
encoder_genero = joblib.load("encoder_genero.pkl")

# Codificar la ciudad con el encoder
ciudad_encoded = encoder_ciudad.transform([[ciudad]])
ciudad_df = pd.DataFrame(ciudad_encoded, columns=encoder_ciudad.get_feature_names_out(["Ciudad"]))

# Construir DataFrame con columnas para modelos que las usan
datos_usuario = pd.DataFrame({
    "Ingresos_Mensuales": [ingresos],
    "Frecuencia_Compra": [frecuencia],
    "Ultima_Compra_dias": [ult_compra]
})

datos_final = pd.concat([datos_usuario, ciudad_df], axis=1)

# Selección del modelo y preprocesamiento
if modelo_seleccionado == "Regresión Logística":
    modelo = modelo_log
    usar_scaler = False
    X = datos_final
elif modelo_seleccionado == "Random Forest":
    modelo = modelo_rf
    usar_scaler = False
    X = datos_final
elif modelo_seleccionado == "KNN":
    modelo = modelo_knn
    usar_scaler = True
    X = datos_final
else:  # Regresión Lineal
    modelo = modelo_regresion
    usar_scaler = False
    # La regresión se entrenó solo con ingresos y frecuencia, sin ciudad ni ult_compra
    X = datos_usuario[["Ingresos_Mensuales", "Frecuencia_Compra"]]

# Botón para predecir
if st.button("Predecir"):
    if usar_scaler:
        datos_escalados = scaler.transform(X)
        prediccion = modelo.predict(datos_escalados)
    else:
        prediccion = modelo.predict(X)

    # Para regresión lineal, la predicción es un número continuo, redondear y convertir a entero
    if modelo_seleccionado == "Regresión Lineal":
        prediccion = np.round(prediccion).astype(int)
    
    genero = encoder_genero.inverse_transform(prediccion)[0]
    st.success(f"Predicción: {genero}")
