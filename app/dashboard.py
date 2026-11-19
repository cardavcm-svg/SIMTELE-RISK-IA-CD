import streamlit as st

# Configuración básica de la página
st.set_page_config(page_title="SIMTELE-RISK", page_icon="🚡", layout="wide")

# Título principal
st.title("🚡 SIMTELE-RISK: Desafío Integrador Final")
st.markdown("### Plataforma de simulación y análisis numérico (Mención IA y Ciencia de Datos)")
st.write("Bienvenido al sistema. Selecciona uno de los cuatro módulos a continuación:")

st.divider()

# Crear dos columnas para las tarjetas de los módulos
col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.subheader("1. Detención de Emergencia")
        st.write("Simulación del frenado de cabinas usando EDOs (RK2, RK4) y Raíces.")
        st.button("Ir al Módulo 1", key="btn1")

    with st.container(border=True):
        st.subheader("3. Oscilaciones por Viento")
        st.write("Análisis de viento y amortiguamiento usando EDOs e Integración.")
        st.button("Ir al Módulo 3", key="btn3")

with col2:
    with st.container(border=True):
        st.subheader("2. Deterioro Mecánico")
        st.write("Mantenimiento predictivo usando Raíces y Sistemas Lineales.")
        st.button("Ir al Módulo 2", key="btn2")

    with st.container(border=True):
        st.subheader("4. Reconstrucción de Sensores")
        st.write("Reconstrucción de datos faltantes usando Interpolación.")
        st.button("Ir al Módulo 4", key="btn4")

st.divider()
st.info("🔧 **Nota de desarrollo:** Este es el prototipo de la interfaz (Frontend). En la Semana 2 se conectará con el motor matemático y la base de datos.")
