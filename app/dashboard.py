import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import sqlite3
import pandas as pd
from datetime import datetime
import joblib

# --- CONFIGURACIÓN DE RUTAS ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.numerical_engine.edos import rk2_heun, rk4
from src.numerical_engine.otros_metodos import biseccion, newton_raphson, secante, trapecio, simpson_13, romberg, lagrange
from src.numerical_engine.sistemas_lineales import jacobi, gauss_seidel, lu_doolittle, gradiente_conjugado

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

init_db()

st.set_page_config(page_title="SIMTELE-RISK", page_icon="🚡", layout="wide")

# --- MENÚ LATERAL ---
if 'menu' not in st.session_state:
    st.session_state.menu = "Inicio"

# Quitamos el Módulo 5 de las opciones
opciones_menu = ["Inicio", "Módulo 1: Detención", "Módulo 2: Deterioro", "Módulo 3: Oscilaciones", "Módulo 4: Sensores", "🤖 Predicción IA", "Historial"]
menu = st.sidebar.radio("Navegación", opciones_menu, index=opciones_menu.index(st.session_state.menu))

st.session_state.menu = menu

# ==========================================
# INICIO: PANEL DE CONTROL GLOBAL
# ==========================================
if menu == "Inicio":
    # --- ESTILOS CSS (Los mismos de antes) ---
    st.markdown("""
        <style>
        .stApp { background-color: #0E1117; }
        div[data-testid="metric-container"] { background-color: #1E2129; border-left: 5px solid #00A86B; padding: 15px; border-radius: 5px; box-shadow: 2px 2px 10px rgba(0,0,0,0.5); }
        div[data-testid="metric-container"] > div > div > div > div { color: #FFFFFF; }
        h1 { color: #FFFFFF; text-shadow: 0px 0px 10px rgba(255,255,255,0.2); }
        .module-card { background-color: #1E2129; padding: 20px; border-radius: 10px; border: 1px solid #333; transition: transform 0.2s; text-align: center; cursor: pointer; margin-bottom: 15px; }
        .module-card:hover { border: 1px solid #00A86B; transform: scale(1.02); }
        .module-icon { font-size: 50px; margin-bottom: 10px; }
        .module-title { font-weight: bold; font-size: 18px; color: #FFF; margin-bottom: 5px;}
        .module-desc { font-size: 14px; color: #AAA; }
        </style>
    """, unsafe_allow_html=True)

    st.title("🚡 Centro de Comandos: SIMTELE-RISK")
    st.markdown("<h4 style='color: #00A86B; margin-top: -15px;'>SISTEMA OPERATIVO | MONITOREO EN TIEMPO REAL</h4>", unsafe_allow_html=True)
    st.divider()

    st.subheader("📊 Telemetría General")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label="Estado del Sistema", value="OPERATIVO", delta="Normal", delta_color="normal")
    col2.metric(label="Cabinas en Línea", value="24/24", delta="Estable", delta_color="off")
    col3.metric(label="Velocidad del Viento", value="12 km/h", delta="Moderado", delta_color="inverse")
    col4.metric(label="Alertas Activas", value="0", delta="Sin riesgos", delta_color="normal")
    st.divider()

    st.subheader("🖥️ Consolas de Diagnóstico")
    st.write("Haz clic en una tarjeta para acceder al módulo correspondiente.")

    col_a, col_b = st.columns(2)
    with col_a:
        # Tarjeta Módulo 1
        st.markdown('<div class="module-card"><div class="module-icon">🛑</div><div class="module-title">Módulo 1: Detención</div><div class="module-desc">Evaluación de frenado y márgenes de seguridad.</div></div>', unsafe_allow_html=True)
        if st.button("Abrir Consola de Detención", use_container_width=True):
            st.session_state.menu = "Módulo 1: Detención"; st.rerun()

        # Tarjeta Módulo 3
        st.markdown('<div class="module-card"><div class="module-icon">🌬️</div><div class="module-title">Módulo 3: Oscilaciones</div><div class="module-desc">Análisis de viento y desplazamiento acumulado.</div></div>', unsafe_allow_html=True)
        if st.button("Abrir Consola de Viento", use_container_width=True):
            st.session_state.menu = "Módulo 3: Oscilaciones"; st.rerun()

    with col_b:
        # Tarjeta Módulo 2
        st.markdown('<div class="module-card"><div class="module-icon">⚙️</div><div class="module-title">Módulo 2: Deterioro y Calibración</div><div class="module-desc">Mantenimiento predictivo y sistemas lineales.</div></div>', unsafe_allow_html=True)
        if st.button("Abrir Consola de Deterioro", use_container_width=True):
            st.session_state.menu = "Módulo 2: Deterioro"; st.rerun()

        # Tarjeta Módulo 4
        st.markdown('<div class="module-card"><div class="module-icon">📡</div><div class="module-title">Módulo 4: Telemetría</div><div class="module-desc">Reconstrucción de datos con interpolación.</div></div>', unsafe_allow_html=True)
        if st.button("Abrir Consola de Telemetría", use_container_width=True):
            st.session_state.menu = "Módulo 4: Sensores"; st.rerun()

    st.divider()
    st.subheader("📜 Registro de Eventos Reciente")
    logs = pd.DataFrame({
        "Hora": ["12:00:01", "12:01:15", "12:05:30", "12:10:00"],
        "Evento": ["Sistema iniciado correctamente.", "Sincronización de sensores completada.", "Simulación de frenado (Módulo 1) ejecutada.", "Modelo de IA cargado en memoria."],
        "Estado": ["OK", "OK", "ÉXITO", "OK"]
    })
    st.dataframe(logs, use_container_width=True, hide_index=True)

# ==========================================
# MÓDULO 1: DETENCIÓN
# ==========================================
elif menu == "Módulo 1: Detención":
    # (El código de tu Módulo 1 se queda EXACTAMENTE igual)
    st.markdown("<h1 style='font-family: Courier New; color: #58a6ff;'>🛑 Consola 01 // Detención de Emergencia</h1>", unsafe_allow_html=True)
    with st.expander("📚 Ver Fundamento Numérico y Fórmulas Matemáticas", expanded=False):
        st.markdown("""
        **EDOs (Runge-Kutta 4):** Resuelve \(\\frac{dx}{dt} = v(t)\) y \(\\frac{dv}{dt} = -a(v,t)\).
        **Teoría de Errores:** \(E_r = \\frac{|D_{analitica} - D_{numerica}|}{D_{analitica}} \\times 100\%\).
        **Mal Condicionamiento:** Se demuestra con un paso de integración \(h\) grande (0.5) que amplifica el error de truncamiento.
        """)
    st.divider()
    escenario = st.sidebar.selectbox("Escenario Operativo", ["NORMAL", "ESTRÉS", "MAL CONDICIONADO"])
    v0_def, tr_def, a_def, d_def = (5.0, 1.0, 1.0, 30.0) if escenario=="NORMAL" else ((10.0, 2.0, 0.8, 85.0) if escenario=="ESTRÉS" else (12.0, 3.0, 0.5, 50.0))
    v0 = st.sidebar.number_input("Velocidad inicial (m/s)", value=v0_def)
    tr = st.sidebar.number_input("Tiempo de reacción (s)", value=tr_def)
    a = st.sidebar.number_input("Desaceleración (m/s²)", value=a_def)
    d_disp = st.sidebar.number_input("Distancia disponible (m)", value=d_def)
    ejecutar = st.sidebar.button("▶ GUARDAR EN HISTORIAL", type="primary", use_container_width=True)
    d_ana = (v0 * tr) + (v0**2 / (2*a)); t_ana = tr + (v0/a)
    def f_x(t, x, v): return v
    def f_v(t, x, v): return -a if t > tr else 0.0
    paso = 0.01 if escenario != "MAL CONDICIONADO" else 0.5
    t_vals, x_vals, v_vals = rk4(f_x, f_v, 0, 0, v0, t_ana + 2.0, paso)
    idx = np.where(v_vals <= 0)[0]
    d_num = x_vals[idx[0]] if len(idx) > 0 else x_vals[-1]
    t_num = t_vals[idx[0]] if len(idx) > 0 else t_vals[-1]
    err_rel = abs(d_ana - d_num) / d_ana * 100
    margen = d_disp - d_num
    if ejecutar: guardar_simulacion(escenario, v0, tr, a, d_disp, d_ana, d_num, err_rel); st.toast("✅ Guardado", icon="💾")
    if margen > 5.0: st.markdown(f'<div style="background-color:#238636; padding:20px; border-radius:8px; border-left:10px solid #2ea043; font-family:Courier; font-weight:bold; color:white; margin-bottom:20px;">🟢 SISTEMA SEGURO // Margen: {margen:.2f} m</div>', unsafe_allow_html=True)
    elif margen > 0: st.markdown(f'<div style="background-color:#9e6a03; padding:20px; border-radius:8px; border-left:10px solid #d29922; font-family:Courier; font-weight:bold; color:white; margin-bottom:20px;">🟡 PRECAUCIÓN // Margen: {margen:.2f} m</div>', unsafe_allow_html=True)
    else: st.markdown(f'<div style="background-color:#da3633; padding:20px; border-radius:8px; border-left:10px solid #f85149; font-family:Courier; font-weight:bold; color:white; margin-bottom:20px;">🔴 PELIGRO // Déficit: {abs(margen):.2f} m</div>', unsafe_allow_html=True)
    if escenario == "MAL CONDICIONADO": st.warning(f"⚠️ **Inestabilidad Numérica:** Paso h={paso} genera error severo.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Distancia Analítica", f"{d_ana:.2f} m"); c2.metric("Distancia RK4", f"{d_num:.2f} m"); c3.metric("Tiempo Detención", f"{t_num:.2f} s"); c4.metric("Error Relativo", f"{err_rel:.4f} %", delta_color="inverse")
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    ax1.plot(t_vals, v_vals, color='#58a6ff', linewidth=2); ax1.axvline(x=tr, color='#d29922', linestyle='--'); ax1.axvline(x=t_num, color='#2ea043', linestyle='--'); ax1.set_title('Perfil de Velocidad'); ax1.grid(True, color='#30363d', linestyle=':')
    ax2.plot(t_vals, x_vals, color='#d2a8ff', linewidth=2); ax2.axhline(y=d_disp, color='#f85149', linestyle='--'); ax2.set_title('Perfil de Posición'); ax2.grid(True, color='#30363d', linestyle=':')
    st.pyplot(fig)
    with st.expander("📋 Ver Tabla de Discretización (RK4)"):
        st.dataframe(pd.DataFrame({"Iter": range(1, 11), "t (s)": t_vals[:10], "x (m)": x_vals[:10], "v (m/s)": v_vals[:10]}), use_container_width=True, hide_index=True)

# ==========================================
# MÓDULO 2: DETERIORO Y CALIBRACIÓN (INTEGRADO)
# ==========================================
elif menu == "Módulo 2: Deterioro":
    st.markdown("<h1 style='font-family: Courier New; color: #58a6ff;'>⚙️ Consola 02 // Deterioro y Calibración</h1>", unsafe_allow_html=True)
    
    # Selector de herramienta
    herramienta = st.radio("Selecciona la Herramienta:", ["Predicción de Mantenimiento (Raíces)", "Calibración de Sensores (Sistemas Lineales)"], horizontal=True)
    st.divider()

    # --- PARTE A: RAÍCES ---
    if herramienta == "Predicción de Mantenimiento (Raíces)":
        with st.expander("📚 Ver Fundamento Numérico (Raíces)", expanded=False):
            st.markdown("""
            **Métodos:** Bisección, Newton-Raphson, Secante.
            **Función:** \(f(t) = D_0 + \\alpha t + \\beta t^2 - D_{umbral} = 0\).
            **Mal condicionamiento:** Raíz múltiple (derivada cercana a cero).
            """)
        escenario = st.sidebar.selectbox("Escenario (Raíces)", ["NORMAL", "ESTRÉS", "MAL CONDICIONADO"])
        D0, alpha, beta, umbral = (10.0, 0.5, 0.05, 50.0) if escenario=="NORMAL" else ((10.0, 0.8, 0.1, 50.0) if escenario=="ESTRÉS" else (10.0, 0.0, 0.0001, 50.0))
        D0 = st.sidebar.number_input("D_0", value=D0); alpha = st.sidebar.number_input("alpha", value=alpha)
        beta = st.sidebar.number_input("beta", value=beta); umbral = st.sidebar.number_input("Umbral", value=umbral)
        metodo = st.sidebar.selectbox("Método", ["Bisección", "Newton-Raphson", "Secante"])
        ejecutar = st.sidebar.button("▶ GUARDAR MANTENIMIENTO", type="primary", use_container_width=True)
        
        f = lambda t: (D0 + alpha*t + beta*t**2) - umbral; df = lambda t: alpha + 2*beta*t
        if metodo == "Bisección": raiz, it, hist = biseccion(f, 0, 100, tol=1e-5)
        elif metodo == "Newton-Raphson": raiz, it, hist = newton_raphson(f, df, 50, tol=1e-5)
        else: raiz, it, hist = secante(f, 0, 100, tol=1e-5)
        
        if ejecutar: guardar_simulacion(f"M2 Raíces - {escenario}", D0, alpha, beta, umbral, raiz, raiz, 0.0); st.toast("✅ Guardado", icon="💾")
        
        if isinstance(hist, str): st.error(f"🔴 FALLO: {hist}")
        else:
            if raiz < 20.0: st.markdown(f'<div style="background-color:#da3633; padding:20px; border-radius:8px; border-left:10px solid #f85149; font-family:Courier; font-weight:bold; color:white; margin-bottom:20px;">🔴 ALERTA: Inspección en {raiz:.2f} días.</div>', unsafe_allow_html=True)
            elif raiz < 60.0: st.markdown(f'<div style="background-color:#9e6a03; padding:20px; border-radius:8px; border-left:10px solid #d29922; font-family:Courier; font-weight:bold; color:white; margin-bottom:20px;">🟡 PRECAUCIÓN: Inspección en {raiz:.2f} días.</div>', unsafe_allow_html=True)
            else: st.markdown(f'<div style="background-color:#238636; padding:20px; border-radius:8px; border-left:10px solid #2ea043; font-family:Courier; font-weight:bold; color:white; margin-bottom:20px;">🟢 ÓPTIMO: Margen de {raiz:.2f} días.</div>', unsafe_allow_html=True)
            if escenario == "MAL CONDICIONADO": st.warning("⚠️ Raíz múltiple: derivada cercana a cero, convergencia lenta.")
            c1, c2, c3 = st.columns(3)
            c1.metric("Tiempo al Umbral", f"{raiz:.4f} días"); c2.metric("Iteraciones", it); c3.metric("Método", metodo)
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(10, 4))
            t_plot = np.linspace(0, max(100, raiz + 20), 200)
            ax.plot(t_plot, f(t_plot), color='#58a6ff', linewidth=2); ax.axhline(y=0, color='#f85149', linestyle='--'); ax.scatter([raiz], [0], color='#2ea043', s=100, zorder=5); ax.grid(True, color='#30363d', linestyle=':')
            st.pyplot(fig)
            with st.expander("📋 Ver Tabla de Iteraciones"):
                st.dataframe(pd.DataFrame(hist, columns=["Iter", "t", "f(t)", "Error"]), use_container_width=True, hide_index=True)

    # --- PARTE B: SISTEMAS LINEALES ---
    else:
        with st.expander("📚 Ver Fundamento Numérico (Sistemas Lineales)", expanded=False):
            st.markdown("""
            **Métodos:** Jacobi, Gauss-Seidel, LU, Gradiente Conjugado.
            **Sistema:** \(Ax = b\) (matriz SPD de 4x4).
            **Mal condicionamiento:** Matriz casi singular.
            """)
        st.markdown("### Calibración de Sensores (\(Ax = b\))")
        A = np.array([[4.0, 1.0, 0.0, 0.0], [1.0, 5.0, 1.0, 0.0], [0.0, 1.0, 6.0, 1.0], [0.0, 0.0, 1.0, 7.0]])
        b = np.dot(A, np.array([1.0, 2.0, 3.0, 4.0]))
        escenario_sl = st.sidebar.selectbox("Escenario (Sistemas)", ["NORMAL", "ESTRÉS", "MAL CONDICIONADO"])
        if escenario_sl == "MAL CONDICIONADO": A[3,3] = A[3,2] + 0.00001; st.error("🚨 Matriz casi singular (Mal Condicionamiento).")
        col_mat1, col_mat2 = st.columns(2)
        col_mat1.dataframe(pd.DataFrame(A, columns=[f"S{i+1}" for i in range(4)]), use_container_width=True)
        col_mat2.dataframe(pd.DataFrame(b, columns=["Lecturas"]), use_container_width=True)
        ejecutar_sl = st.sidebar.button("▶ GUARDAR CALIBRACIÓN", type="primary", use_container_width=True)
        x_jacobi, it_jacobi = jacobi(A, b); x_gs, it_gs = gauss_seidel(A, b)
        x_lu = lu_doolittle(A, b); x_gc, it_gc = gradiente_conjugado(A, b)
        if ejecutar_sl: guardar_simulacion(f"M2 SistLineales - {escenario_sl}", 4, 4, 4, 4, it_jacobi, it_gs, 0.0); st.toast("✅ Guardado", icon="💾")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Iter. Jacobi", it_jacobi); c2.metric("Iter. Gauss-Seidel", it_gs); c3.metric("Iter. G. Conjugado", it_gc); c4.metric("Residuo LU", f"{np.linalg.norm(np.dot(A, x_lu) - b):.2e}")
        st.write(f"**Solución Jacobi:** {np.round(x_jacobi, 4)}")
        st.write(f"**Solución Gauss-Seidel:** {np.round(x_gs, 4)}")
        st.write(f"**Solución LU:** {np.round(x_lu, 4)}")
        st.write(f"**Solución Gradiente Conjugado:** {np.round(x_gc, 4)}")

# ==========================================
# MÓDULO 3: OSCILACIONES
# ==========================================
elif menu == "Módulo 3: Oscilaciones":
    # (Tu Módulo 3 se queda EXACTAMENTE igual)
    st.markdown("<h1 style='font-family: Courier New; color: #58a6ff;'>🌬️ Consola 03 // Oscilaciones por Viento</h1>", unsafe_allow_html=True)
    with st.expander("📚 Ver Fundamento Numérico y Fórmulas Matemáticas", expanded=False):
        st.markdown("""
        **Integración Numérica:** Trapecio, Simpson 1/3, Romberg.
        **Función:** \(\\int_{0}^{T} |A \\cdot \\sin(w \\cdot t)| dt\).
        **Mal Condicionamiento:** Onda cuadrada (discontinua) que afecta la precisión de Simpson y Romberg.
        """)
    st.divider()
    escenario = st.sidebar.selectbox("Escenario", ["NORMAL", "ESTRÉS", "MAL CONDICIONADO"])
    A, w, T = (2.0, 1.5, 10.0) if escenario=="NORMAL" else ((5.0, 3.0, 20.0) if escenario=="ESTRÉS" else (2.0, 1.5, 10.0))
    A = st.sidebar.number_input("Amplitud", value=A); w = st.sidebar.number_input("Frecuencia", value=w)
    T = st.sidebar.number_input("Tiempo", value=T); n = st.sidebar.slider("Subintervalos", 10, 1000, 100)
    ejecutar = st.sidebar.button("▶ GUARDAR EN HISTORIAL", type="primary", use_container_width=True)
    f_viento = (lambda t: np.sign(np.sin(w*t))*A) if escenario=="MAL CONDICIONADO" else (lambda t: abs(A*np.sin(w*t)))
    val_trap = trapecio(f_viento, 0, T, n); val_simp = simpson_13(f_viento, 0, T, n); val_romb, _ = romberg(f_viento, 0, T, 5)
    if ejecutar: guardar_simulacion(f"M3 - {escenario}", A, w, T, n, val_trap, val_romb, 0.0); st.toast("✅ Guardado", icon="💾")
    if val_romb > 40.0: st.markdown(f'<div style="background-color:#da3633; padding:20px; border-radius:8px; border-left:10px solid #f85149; font-family:Courier; font-weight:bold; color:white; margin-bottom:20px;">🔴 ALERTA: Balanceo crítico ({val_romb:.2f} m).</div>', unsafe_allow_html=True)
    elif val_romb > 20.0: st.markdown(f'<div style="background-color:#9e6a03; padding:20px; border-radius:8px; border-left:10px solid #d29922; font-family:Courier; font-weight:bold; color:white; margin-bottom:20px;">🟡 PRECAUCIÓN: Oscilación moderada ({val_romb:.2f} m).</div>', unsafe_allow_html=True)
    else: st.markdown(f'<div style="background-color:#238636; padding:20px; border-radius:8px; border-left:10px solid #2ea043; font-family:Courier; font-weight:bold; color:white; margin-bottom:20px;">🟢 SEGURO: Desplazamiento controlado ({val_romb:.2f} m).</div>', unsafe_allow_html=True)
    if escenario == "MAL CONDICIONADO": st.warning("⚠️ Romberg pierde precisión ante discontinuidades.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Trapecio", f"{val_trap:.5f} m"); c2.metric("Simpson", f"{val_simp:.5f} m"); c3.metric("Romberg", f"{val_romb:.5f} m")
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 4))
    t_plot = np.linspace(0, T, 500); y_plot = f_viento(t_plot)
    ax.plot(t_plot, y_plot, color='#58a6ff', linewidth=2); ax.fill_between(t_plot, y_plot, color='#58a6ff', alpha=0.2); ax.grid(True, color='#30363d', linestyle=':')
    st.pyplot(fig)
    with st.expander("📋 Ver Tabla de Errores"):
        ref = val_romb if escenario != "MAL CONDICIONADO" else val_trap
        st.dataframe(pd.DataFrame({"Método": ["Trapecio", "Simpson", "Romberg"], "Error Abs": [abs(val_trap-ref), abs(val_simp-ref), abs(val_romb-ref)]}), use_container_width=True, hide_index=True)

# ==========================================
# MÓDULO 4: SENSORES
# ==========================================
elif menu == "Módulo 4: Sensores":
    # (Tu Módulo 4 se queda EXACTAMENTE igual)
    st.markdown("<h1 style='font-family: Courier New; color: #58a6ff;'>📡 Consola 04 // Reconstrucción de Sensores</h1>", unsafe_allow_html=True)
    with st.expander("📚 Ver Fundamento Numérico y Fórmulas Matemáticas", expanded=False):
        st.markdown("""
        **Interpolación:** Lagrange.
        **Función:** \(P(x) = \\sum_{i=0}^{n} y_i \\prod_{j \\neq i} \\frac{x - x_j}{x_i - x_j}\).
        **Mal Condicionamiento:** Nodos duplicados (división por cero).
        """)
    st.divider()
    escenario = st.sidebar.selectbox("Escenario", ["NORMAL", "ESTRÉS", "MAL CONDICIONADO"])
    if escenario == "NORMAL": x_p = np.array([0, 1, 2, 4, 5]); y_p = np.array([0, 0.84, 0.91, -0.75, -0.95]); x_real = 3.0
    elif escenario == "ESTRÉS": x_p = np.array([0, 0.1, 0.2, 4.8, 5.0]); y_p = np.array([0, 0.1, 0.2, -0.99, -0.95]); x_real = 3.0
    else: x_p = np.array([0, 1, 1, 4, 5]); y_p = np.array([0, 0.84, 0.91, -0.75, -0.95]); x_real = 3.0
    y_real = np.sin(x_real)
    x_faltante = st.sidebar.slider("Sensor perdido (x)", 0.0, 5.0, float(x_real))
    ejecutar = st.sidebar.button("▶ GUARDAR EN HISTORIAL", type="primary", use_container_width=True)
    if len(np.unique(x_p)) != len(x_p): y_est = None; error = None
    else: y_est = lagrange(x_p, y_p, x_faltante); error = abs(y_real - y_est) / abs(y_real) * 100 if y_real != 0 else 0
    if ejecutar:
        if y_est is not None: guardar_simulacion(f"M4 - {escenario}", x_faltante, 0, 0, 0, y_est, y_est, error); st.toast("✅ Guardado", icon="💾")
        else: st.toast("❌ Fallo de simulación", icon="⚠️")
    if y_est is None: st.markdown('<div style="background-color:#da3633; padding:20px; border-radius:8px; border-left:10px solid #f85149; font-family:Courier; font-weight:bold; color:white; margin-bottom:20px;">🔴 FALLO: Nodos duplicados.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="background-color:#238636; padding:20px; border-radius:8px; border-left:10px solid #2ea043; font-family:Courier; font-weight:bold; color:white; margin-bottom:20px;">🟢 RECONSTRUCCIÓN EXITOSA en x={x_faltante:.2f}</div>', unsafe_allow_html=True)
        if escenario == "ESTRÉS": st.warning("⚠️ Oscilación de Runge detectada.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Valor Real", f"{y_real:.4f}"); c2.metric("Valor Lagrange", f"{y_est:.4f}"); c3.metric("Error", f"{error:.2f} %")
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(12, 4))
        x_plot = np.linspace(0, 5, 200); y_plot = [lagrange(x_p, y_p, xi) for xi in x_plot]
        ax.plot(x_plot, y_plot, color='#58a6ff', linewidth=2); ax.scatter(x_p, y_p, color='#f85149', s=80, zorder=5); ax.scatter([x_faltante], [y_est], color='#2ea043', s=150, marker='*', zorder=6); ax.grid(True, color='#30363d', linestyle=':')
        st.pyplot(fig)
        with st.expander("📋 Ver Tabla de Nodos"):
            st.dataframe(pd.DataFrame({"Sensor": [f"S{i+1}" for i in range(len(x_p))], "x": x_p, "y": y_p}), use_container_width=True, hide_index=True)

# ==========================================
# PREDICCIÓN IA
# ==========================================
elif menu == "🤖 Predicción IA":
    st.title("Consola de Predicción con Machine Learning")
    try:
        ruta_modelo = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'numerical_engine', 'ml_models', 'modelo_rf.pkl'))
        modelo = joblib.load(ruta_modelo)
        v0_ia = st.sidebar.number_input("Velocidad inicial", value=7.5); tr_ia = st.sidebar.number_input("Tiempo de reacción", value=1.5)
        a_ia = st.sidebar.number_input("Desaceleración", value=1.2); d_disp_ia = st.sidebar.number_input("Distancia disponible", value=50.0)
        if st.sidebar.button("Predecir"):
            entrada = np.array([[v0_ia, tr_ia, a_ia, d_disp_ia]]); prediccion = modelo.predict(entrada)[0]
            st.header("Resultados de la IA")
            if prediccion > d_disp_ia: st.error(f"🔴 PELIGRO: La IA predice colisión. Distancia: {prediccion:.2f} m.")
            else: st.success(f"🟢 SEGURO: La IA predice detención segura. Distancia: {prediccion:.2f} m.")
            importancias = modelo.feature_importances_; variables = ['Velocidad', 'T. Reacción', 'Desaceleración', 'Distancia']
            fig, ax = plt.subplots(figsize=(8, 3)); ax.barh(variables, importancias, color='teal'); ax.set_title('Importancia de las variables'); st.pyplot(fig)
    except Exception as e: st.warning("Modelo de IA no encontrado.")

# ==========================================
# HISTORIAL
# ==========================================
elif menu == "Historial":
    st.title("Historial de Operaciones")
    st.write("Registro de simulaciones guardadas en la base de datos del Centro de Comandos.")
    conn = sqlite3.connect('simtele_risk.db')
    df = pd.read_sql_query("SELECT * FROM simulaciones", conn)
    conn.close()
    if len(df) > 0: st.dataframe(df, use_container_width=True)
    else: st.warning("No hay simulaciones registradas.")
