import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Esto permite que Python encuentre la carpeta src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.numerical_engine.edos import rk2_heun, rk4

st.set_page_config(page_title="SIMTELE-RISK", page_icon="🚡", layout="wide")

st.title("🚡 SIMTELE-RISK: Módulo 1 - Detención de Emergencia")
st.markdown("Simulación del frenado de cabinas usando Ecuaciones Diferenciales Ordinarias (RK2 y RK4).")

# --- BARRA LATERAL: Parámetros ---
st.sidebar.header("Parámetros de Entrada")
escenario = st.sidebar.selectbox("Selecciona el Escenario:", ["NORMAL", "ESTRÉS", "MAL CONDICIONADO"])

# Valores por defecto según el escenario
if escenario == "NORMAL":
    v0_def, tr_def, a_def, d_def = 5.0, 1.0, 1.0, 30.0
elif escenario == "ESTRÉS":
    v0_def, tr_def, a_def, d_def = 10.0, 2.0, 0.8, 85.0
else:
    v0_def, tr_def, a_def, d_def = 12.0, 3.0, 0.5, 50.0

v0 = st.sidebar.number_input("Velocidad inicial (m/s)", value=v0_def, step=0.5)
tr = st.sidebar.number_input("Tiempo de reacción (s)", value=tr_def, step=0.1)
a = st.sidebar.number_input("Desaceleración (m/s²)", value=a_def, step=0.1)
d_disp = st.sidebar.number_input("Distancia disponible (m)", value=d_def, step=1.0)

# --- CÁLCULOS ---
if st.sidebar.button("Ejecutar Simulación", type="primary"):
    
    # 1. Solución Analítica (Ground Truth)
    x_reaccion = v0 * tr
    d_frenado = (v0**2) / (2 * a)
    d_total_analitica = x_reaccion + d_frenado
    t_total_analitico = tr + (v0 / a)
    
    # 2. Solución Numérica (RK4)
    # Definimos las funciones de derivada
    def f_x(t, x, v):
        return v
    
    def f_v(t, x, v):
        return -a if t > tr else 0.0
    
    # Usamos un paso h=0.01 para alta precisión
    h = 0.01
    tf = t_total_analitico + 2.0 # Simulamos un poco más de tiempo para ver la curva
    t_vals, x_vals, v_vals = rk4(f_x, f_v, 0, 0, v0, tf, h)
    
    # Buscamos el punto donde se detiene numéricamente
    idx_detencion = np.where(v_vals <= 0)[0]
    if len(idx_detencion) > 0:
        t_det_num = t_vals[idx_detencion[0]]
        d_det_num = x_vals[idx_detencion[0]]
    else:
        t_det_num = t_vals[-1]
        d_det_num = x_vals[-1]
        
    # --- MOSTRAR RESULTADOS ---
    st.header(f"Resultados: Escenario {escenario}")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Distancia Total Analítica", f"{d_total_analitica:.4f} m")
    col2.metric("Distancia Total Numérica (RK4)", f"{d_det_num:.4f} m")
    
    # Error
    error_abs = abs(d_total_analitica - d_det_num)
    error_rel = (error_abs / d_total_analitica) * 100 if d_total_analitica != 0 else 0
    col3.metric("Error Relativo", f"{error_rel:.4f} %")
    
    # Margen de Seguridad
    margen = d_disp - d_det_num
    if margen > 0:
        st.success(f"✅ **Margen de Seguridad:** {margen:.2f} m (Operación Segura)")
    else:
        st.error(f"🚨 **ALERTA CRÍTICA:** El margen de seguridad es {margen:.2f} m (Colisión inminente)")
        
    # --- GRÁFICOS ---
    st.subheader("Gráficos de Simulación")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Gráfico 1: Velocidad vs Tiempo
    ax1.plot(t_vals, v_vals, 'b-', label='Velocidad (RK4)')
    ax1.axvline(x=tr, color='r', linestyle='--', label='Fin de Reacción')
    ax1.axvline(x=t_det_num, color='g', linestyle='--', label='Detención')
    ax1.set_xlabel('Tiempo (s)')
    ax1.set_ylabel('Velocidad (m/s)')
    ax1.set_title('Perfil de Velocidad')
    ax1.legend()
    ax1.grid(True)
    
    # Gráfico 2: Posición vs Tiempo
    ax2.plot(t_vals, x_vals, 'm-', label='Posición (RK4)')
    ax2.axhline(y=d_disp, color='r', linestyle='--', label='Límite Disponible')
    ax2.set_xlabel('Tiempo (s)')
    ax2.set_ylabel('Posición (m)')
    ax2.set_title('Perfil de Posición')
    ax2.legend()
    ax2.grid(True)
    
    st.pyplot(fig)

else:
    st.info("👈 Configura los parámetros en la barra lateral y haz clic en 'Ejecutar Simulación'.")
