import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Cargar el modelo y el escalador
scaler = joblib.load("scaler.pkl")
model = joblib.load("knn_model.joblib")

# Configuración de la app
st.title("Predicción de enfermedad del corazón")
st.subheader("Realizado por Leydy Macareo")
st.image("https://cardiosalud.org/wp-content/uploads/2019/04/como-funciona-el-corazon.jpg")

# Instrucciones
st.markdown("""
### Instrucciones de uso:
1. Ajuste los valores de Edad y Colesterol usando los controles deslizantes.
2. Presione el botón para realizar la predicción.
3. Observe el resultado y siga las recomendaciones.
""")

# Controles de entrada
edad = st.slider("edad", 20, 77, 40)
colesterol = st.slider("colesterol", 100, 600, 200)

# Predicción
if st.button("Predecir"):
    # Crear DataFrame con los datos
    input_data = pd.DataFrame([[edad, colesterol]], columns=["edad", "colesterol"])
    
    # Escalar los datos
    input_scaled = scaler.transform(input_data)
    
    # Realizar predicción
    prediction = model.predict(input_scaled)[0]
    
    # Mostrar resultado
    if prediction == 0:
        st.success("✅ No tiene probabilidad de sufrir del corazón")
    else:
        st.error("⚠️ Cuídese, tiene riesgo de enfermedad del corazón")

# Pie de página
st.markdown("---")
st.markdown("UNAB 2025 ®")