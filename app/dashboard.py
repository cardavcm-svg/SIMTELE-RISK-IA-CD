import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import sqlite3
import pandas as pd
from datetime import datetime

# --- CONFIGURACIÓN DE RUTAS ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.numerical_engine.edos import rk2_heun, rk4

# --- BASE DE DATOS (SQLite) ---
def init_db():
    conn = sqlite3.connect('simtele_risk.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS simulaciones
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  fecha TEXT, 
                  escenario TEXT, 
                  v0 REAL, tr REAL, a REAL, d_disp REAL, 
                  d_analitica REAL, d_numerica REAL, error_rel REAL)''')
    conn.commit()
    conn.close()

def guardar_simulacion(escenario, v0, tr, a, d_disp, d_ana, d_num, err):
    conn = sqlite3.connect('simtele_risk.db')
    c = conn.cursor()
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute('''INSERT INTO simulaciones (fecha, escenario, v0, tr, a, d_disp, d_analitica, d_numerica, error_rel)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''', (fecha, escenario, v0, tr, a, d_disp, d_ana, d_num, err))
    conn.commit()
    conn.close()

# Inicializar BD al arrancar
init_db()

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="SIMTELE-RISK", page_icon="🚡", layout="wide")

# --- MENÚ LATERAL ---
st.sidebar.title("🚡 SIMTELE-RISK")
menu = st.sidebar.radio("Navegación", ["Inicio", "Módulo 1: Detención", "Módulo 2: Deterioro", "Módulo 3: Oscilaciones", "Módulo 4: Sensores", "Historial"])

# ==========================================
# PÁGINA DE INICIO
# ==========================================
if menu == "Inicio":
    st.title("Bienvenido a SIMTELE-RISK")
    st.markdown("### Sistema Inteligente de Analítica Predictiva")
    st.write("Selecciona un módulo en la barra lateral para comenzar.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("**Módulo 1: Detención de Emergencia**\n\nEDOs (RK2, RK4)")
        st.info("**Módulo 3: Oscilaciones por Viento**\n\nIntegración (Trapecio, Simpson)")
    with col2:
        st.warning("**Módulo 2: Deterioro Mecánico**\n\nRaíces (Bisección, Newton)")
        st.success("**Módulo 4: Reconstrucción de Sensores**\n\nInterpolación (Lagrange, Splines)")

# ==========================================
# MÓDULO 1: DETENCIÓN
# ==========================================
elif menu == "Módulo 1: Detención":
    st.title("Módulo 1: Detención de Emergencia")
    
    escenario = st.sidebar.selectbox("Escenario", ["NORMAL", "ESTRÉS", "MAL CONDICIONADO"])
    
    # Parámetros por defecto
    v0_def, tr_def, a_def, d_def = (5.0, 1.0, 1.0, 30.0) if escenario == "NORMAL" else ((10.0, 2.0, 0.8, 85.0) if escenario == "ESTRÉS" else (12.0, 3.0, 0.5, 50.0))
    
    v0 = st.sidebar.number_input("Velocidad inicial (m/s)", value=v0_def)
    tr = st.sidebar.number_input("Tiempo de reacción (s)", value=tr_def)
    a = st.sidebar.number_input("Desaceleración (m/s²)", value=a_def)
    d_disp = st.sidebar.number_input("Distancia disponible (m)", value=d_def)
    
    if st.sidebar.button("Ejecutar Simulación"):
        # Cálculos
        d_ana = (v0 * tr) + (v0**2 / (2*a))
        t_ana = tr + (v0/a)
        
        def f_x(t, x, v): return v
        def f_v(t, x, v): return -a if t > tr else 0.0
        
        t_vals, x_vals, v_vals = rk4(f_x, f_v, 0, 0, v0, t_ana + 2.0, 0.01)
        idx = np.where(v_vals <= 0)[0]
        d_num = x_vals[idx[0]] if len(idx) > 0 else x_vals[-1]
        
        err_rel = abs(d_ana - d_num) / d_ana * 100
        
        # Mostrar Resultados
        st.header(f"Resultados: {escenario}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Distancia Analítica", f"{d_ana:.4f} m")
        c2.metric("Distancia Numérica", f"{d_num:.4f} m")
        c3.metric("Error Relativo", f"{err_rel:.4f} %")
        
        margen = d_disp - d_num
        if margen > 0: st.success(f"Margen: {margen:.2f} m (Seguro)")
        else: st.error(f"Margen: {margen:.2f} m (Crítico)")
        
        # Guardar en BD
        guardar_simulacion(escenario, v0, tr, a, d_disp, d_ana, d_num, err_rel)
        st.write("*(Simulación guardada en el historial)*")
        
        # Gráficos
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(t_vals, v_vals, 'b-'); ax1.set_title('Velocidad'); ax1.grid(True)
        ax2.plot(t_vals, x_vals, 'm-'); ax2.axhline(y=d_disp, color='r', linestyle='--'); ax2.set_title('Posición'); ax2.grid(True)
        st.pyplot(fig)

# ==========================================
# MÓDULOS 2, 3 Y 4 (Estructura para Semana 2)
# ==========================================
elif menu == "Módulo 2: Deterioro":
    st.title("Módulo 2: Deterioro Mecánico")
    st.write("🔧 *En construcción para la Semana 3.*")
    st.latex(r"D(t) = D_0 + \alpha t + \beta t^2")
    st.info("Aquí se implementarán los métodos de Raíces: Bisección, Newton-Raphson y Secante.")

elif menu == "Módulo 3: Oscilaciones":
    st.title("Módulo 3: Oscilaciones por Viento")
    st.write("🌬️ *En construcción para la Semana 3.*")
    st.latex(r"m \frac{d^2x}{dt^2} + c \frac{dx}{dt} + kx = F(t)")
    st.info("Aquí se implementará la Integración Numérica: Trapecio, Simpson 1/3 y Romberg.")

elif menu == "Módulo 4: Sensores":
    st.title("Módulo 4: Reconstrucción de Sensores")
    st.write("📡 *En construcción para la Semana 3.*")
    st.info("Aquí se implementará la Interpolación: Lagrange, Newton y Splines Cúbicos.")

# ==========================================
# HISTORIAL
# ==========================================
elif menu == "Historial":
    st.title("Historial de Simulaciones")
    st.write("Registro de todas las ejecuciones guardadas en la base de datos.")
    
    conn = sqlite3.connect('simtele_risk.db')
    df = pd.read_sql_query("SELECT * FROM simulaciones", conn)
    conn.close()
    
    if len(df) > 0:
        st.dataframe(df)
    else:
        st.warning("No hay simulaciones guardadas todavía. Ve al Módulo 1 y ejecuta una.")
