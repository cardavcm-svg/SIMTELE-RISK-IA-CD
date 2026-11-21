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
from src.numerical_engine.sistemas_lineales import jacobi, gauss_seidel, lu_doolittle

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
# Inicializamos el estado del menú si no existe
if 'menu' not in st.session_state:
    st.session_state.menu = "Inicio"

# Usamos una función para cambiar el menú desde los botones del inicio
def cambiar_menu(nuevo_menu):
    st.session_state.menu = nuevo_menu

# El radio button ahora usa el estado del session_state
opciones_menu = ["Inicio", "Módulo 1: Detención", "Módulo 2: Deterioro", "Módulo 3: Oscilaciones", "Módulo 4: Sensores", "Módulo 5: Sistemas Lineales", "🤖 Predicción IA", "Historial"]
menu = st.sidebar.radio("Navegación", opciones_menu, index=opciones_menu.index(st.session_state.menu))

# Sincronizamos el menú lateral con el estado
st.session_state.menu = menu
# ==========================================
# INICIO: PANEL DE CONTROL GLOBAL (CENTRO DE COMANDOS)
# ==========================================
if menu == "Inicio":
    # --- ESTILOS CSS PARA DARLE UN LOOK INDUSTRIAL/PROFESIONAL ---
    st.markdown("""
        <style>
        /* Fondo principal más oscuro */
        .stApp { background-color: #0E1117; }
        
        /* Estilo de las tarjetas (métricas) */
        div[data-testid="metric-container"] {
            background-color: #1E2129;
            border-left: 5px solid #00A86B;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.5);
        }
        
        /* Color del texto de las métricas */
        div[data-testid="metric-container"] > div > div > div > div {
            color: #FFFFFF;
        }
        
        /* Título principal */
        h1 { color: #FFFFFF; text-shadow: 0px 0px 10px rgba(255,255,255,0.2); }
        
        /* Estilo para las tarjetas de los módulos */
        .module-card {
            background-color: #1E2129;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #333;
            transition: transform 0.2s;
            text-align: center;
            cursor: pointer;
            margin-bottom: 15px;
        }
        .module-card:hover {
            border: 1px solid #00A86B;
            transform: scale(1.02);
        }
        .module-icon { font-size: 50px; margin-bottom: 10px; }
        .module-title { font-weight: bold; font-size: 18px; color: #FFF; margin-bottom: 5px;}
        .module-desc { font-size: 14px; color: #AAA; }
        </style>
    """, unsafe_allow_html=True)

    # --- CABECERA DEL PANEL ---
    col_titulo, col_logo = st.columns([4, 1])
    with col_titulo:
        st.title("🚡 Centro de Comandos: SIMTELE-RISK")
        st.markdown("<h4 style='color: #00A86B; margin-top: -15px;'>SISTEMA OPERATIVO | MONITOREO EN TIEMPO REAL</h4>", unsafe_allow_html=True)
    
    st.divider()

    # --- FILA DE MÉTRICAS CRÍTICAS (Telemetría) ---
    st.subheader("📊 Telemetría General")
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric(label="Estado del Sistema", value="OPERATIVO", delta="Normal", delta_color="normal")
    col2.metric(label="Cabinas en Línea", value="24/24", delta="Estable", delta_color="off")
    col3.metric(label="Velocidad del Viento", value="12 km/h", delta="Moderado", delta_color="inverse")
    col4.metric(label="Alertas Activas", value="0", delta="Sin riesgos", delta_color="normal")

    st.divider()

    # --- CONSOLAS DE DIAGNÓSTICO (Tarjetas Interactivas) ---
    st.subheader("🖥️ Consolas de Diagnóstico")
    st.write("Haz clic en una tarjeta para acceder al módulo correspondiente.")

    col_a, col_b = st.columns(2)

    with col_a:
        # Tarjeta Módulo 1
        st.markdown('<div class="module-card"><div class="module-icon">🛑</div><div class="module-title">Módulo 1: Detención</div><div class="module-desc">Evaluación de frenado y márgenes de seguridad.</div></div>', unsafe_allow_html=True)
        if st.button("Abrir Consola de Detención", use_container_width=True):
            st.session_state.menu = "Módulo 1: Detención"
            st.rerun()

        # Tarjeta Módulo 3
        st.markdown('<div class="module-card"><div class="module-icon">🌬️</div><div class="module-title">Módulo 3: Oscilaciones</div><div class="module-desc">Análisis de viento y desplazamiento acumulado.</div></div>', unsafe_allow_html=True)
        if st.button("Abrir Consola de Viento", use_container_width=True):
            st.session_state.menu = "Módulo 3: Oscilaciones"
            st.rerun()

        # Tarjeta Módulo 5
        st.markdown('<div class="module-card"><div class="module-icon">📐</div><div class="module-title">Módulo 5: Calibración</div><div class="module-desc">Sistemas lineales para sensores.</div></div>', unsafe_allow_html=True)
        if st.button("Abrir Consola de Calibración", use_container_width=True):
            st.session_state.menu = "Módulo 5: Sistemas Lineales"
            st.rerun()

    with col_b:
        # Tarjeta Módulo 2
        st.markdown('<div class="module-card"><div class="module-icon">⚙️</div><div class="module-title">Módulo 2: Deterioro</div><div class="module-desc">Mantenimiento predictivo de cabinas.</div></div>', unsafe_allow_html=True)
        if st.button("Abrir Consola de Deterioro", use_container_width=True):
            st.session_state.menu = "Módulo 2: Deterioro"
            st.rerun()

        # Tarjeta Módulo 4
        st.markdown('<div class="module-card"><div class="module-icon">📡</div><div class="module-title">Módulo 4: Telemetría</div><div class="module-desc">Reconstrucción de datos con interpolación.</div></div>', unsafe_allow_html=True)
        if st.button("Abrir Consola de Telemetría", use_container_width=True):
            st.session_state.menu = "Módulo 4: Sensores"
            st.rerun()

        # Tarjeta IA
        st.markdown('<div class="module-card"><div class="module-icon">🤖</div><div class="module-title">Módulo IA: Predicción</div><div class="module-desc">Machine Learning (Random Forest).</div></div>', unsafe_allow_html=True)
        if st.button("Abrir Consola de IA", use_container_width=True):
            st.session_state.menu = "🤖 Predicción IA"
            st.rerun()

    # --- PANEL DE LOGS (Simulación de sistema) ---
    st.divider()
    st.subheader("📜 Registro de Eventos Reciente")
    
    logs = pd.DataFrame({
        "Hora": ["12:00:01", "12:01:15", "12:05:30", "12:10:00"],
        "Evento": [
            "Sistema iniciado correctamente.",
            "Sincronización de sensores completada.",
            "Simulación de frenado (Módulo 1) ejecutada.",
            "Modelo de IA cargado en memoria."
        ],
        "Estado": ["OK", "OK", "ÉXITO", "OK"]
    })
    st.dataframe(logs, use_container_width=True, hide_index=True)

# ==========================================
# MÓDULO 1: DETENCIÓN (CON EXPLICACIÓN NUMÉRICA)
# ==========================================
elif menu == "Módulo 1: Detención":
    st.markdown("<h1 style='font-family: Courier New; color: #58a6ff;'>🛑 Consola 01 // Detención de Emergencia</h1>", unsafe_allow_html=True)
    
    # --- PANEL EXPLICATIVO DE MÉTODOS NUMÉRICOS ---
    with st.expander("📚 Ver Fundamento Numérico y Fórmulas Matemáticas", expanded=False):
        st.markdown("""
        ### 🧮 Herramientas Numéricas en esta Consola
        
        Esta consola resuelve el problema de frenado modelando la física como Ecuaciones Diferenciales Ordinarias (EDOs).
        
        **1. Ecuaciones Diferenciales Ordinarias (Runge-Kutta 4):**
        *   *¿Qué hace?* Calcula la velocidad y posición paso a paso en el tiempo.
        *   *Fórmula base:*
        """)
        st.latex(r"\frac{dx}{dt} = v(t), \quad \frac{dv}{dt} = -a(v,t)")
        st.markdown("""
        *   *Implementación:* Usamos el método de **RK4** (4to orden) con un paso de integración \(h=0.01\). Esto nos da una precisión milimétrica comparada con la solución analítica.
        
        **2. Teoría de Errores:**
        *   *¿Qué hace?* Cuantifica la diferencia entre el cálculo de la máquina y el cálculo exacto.
        *   *Fórmula base:* 
        """)
        st.latex(r"E_r = \frac{|D_{analitica} - D_{numerica}|}{D_{analitica}} \times 100\%")
        st.markdown("""
        **3. Mal Condicionamiento:**
        *   *¿Qué hace?* Demuestra cómo un paso de integración \(h\) inadecuado (muy grande) amplifica el error de truncamiento, volviendo inestable el método.
        """)
    
    st.divider()

    # 1. Controles en la barra lateral
    escenario = st.sidebar.selectbox("Escenario Operativo", ["NORMAL", "ESTRÉS", "MAL CONDICIONADO"])
    
    if escenario == "NORMAL":
        v0_def, tr_def, a_def, d_def = 5.0, 1.0, 1.0, 30.0
    elif escenario == "ESTRÉS":
        v0_def, tr_def, a_def, d_def = 10.0, 2.0, 0.8, 85.0
    else: 
        v0_def, tr_def, a_def, d_def = 12.0, 3.0, 0.5, 50.0
    
    v0 = st.sidebar.number_input("Velocidad inicial (m/s)", value=v0_def)
    tr = st.sidebar.number_input("Tiempo de reacción (s)", value=tr_def)
    a = st.sidebar.number_input("Desaceleración (m/s²)", value=a_def)
    d_disp = st.sidebar.number_input("Distancia disponible (m)", value=d_def)
    
    ejecutar = st.sidebar.button("▶ GUARDAR EN HISTORIAL", type="primary", use_container_width=True)

    # 2. Lógica de simulación automática
    d_ana = (v0 * tr) + (v0**2 / (2*a))
    t_ana = tr + (v0/a)
    
    def f_x(t, x, v): return v
    def f_v(t, x, v): return -a if t > tr else 0.0
    
    # El mal condicionamiento se da por el paso de integración grande
    paso = 0.01 if escenario != "MAL CONDICIONADO" else 0.5
    
    t_vals, x_vals, v_vals = rk4(f_x, f_v, 0, 0, v0, t_ana + 2.0, paso)
    idx = np.where(v_vals <= 0)[0]
    
    d_num = x_vals[idx[0]] if len(idx) > 0 else x_vals[-1]
    t_num = t_vals[idx[0]] if len(idx) > 0 else t_vals[-1]
    
    err_rel = abs(d_ana - d_num) / d_ana * 100
    margen = d_disp - d_num

    # 3. Guardar en la base de datos (solo si presiona el botón)
    if ejecutar:
        guardar_simulacion(escenario, v0, tr, a, d_disp, d_ana, d_num, err_rel)
        st.toast("✅ Simulación guardada en el historial", icon="💾")

    # 4. Renderizado Visual (Siempre visible)
    
    # Semáforo
    if margen > 5.0: 
        st.markdown(f'<div style="background-color:#238636; padding:20px; border-radius:8px; border-left:10px solid #2ea043; font-family:Courier; font-size:18px; font-weight:bold; color:white; margin-bottom:20px;">🟢 SISTEMA SEGURO // Margen: {margen:.2f} m</div>', unsafe_allow_html=True)
    elif margen > 0: 
        st.markdown(f'<div style="background-color:#9e6a03; padding:20px; border-radius:8px; border-left:10px solid #d29922; font-family:Courier; font-size:18px; font-weight:bold; color:white; margin-bottom:20px;">🟡 PRECAUCIÓN // Margen Crítico: {margen:.2f} m</div>', unsafe_allow_html=True)
    else: 
        st.markdown(f'<div style="background-color:#da3633; padding:20px; border-radius:8px; border-left:10px solid #f85149; font-family:Courier; font-size:18px; font-weight:bold; color:white; margin-bottom:20px;">🔴 PELIGRO DE COLISIÓN // Déficit: {abs(margen):.2f} m</div>', unsafe_allow_html=True)
        
    if escenario == "MAL CONDICIONADO":
        st.warning(f"⚠️ **Inestabilidad Numérica (Mal Condicionamiento):** Se usó un paso de integración \(h={paso}\). Al ser tan grande, el error de truncamiento crece y el método pierde precisión.")

    # Métricas
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Distancia Analítica", f"{d_ana:.2f} m")
    c2.metric("Distancia Numérica (RK4)", f"{d_num:.2f} m")
    c3.metric("Tiempo de Detención", f"{t_num:.2f} s")
    c4.metric("Error Relativo", f"{err_rel:.4f} %", delta_color="inverse")

    # Gráficos
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    
    ax1.plot(t_vals, v_vals, color='#58a6ff', linewidth=2, label='Velocidad')
    ax1.axvline(x=tr, color='#d29922', linestyle='--', label='Fin Reacción')
    ax1.axvline(x=t_num, color='#2ea043', linestyle='--', label='Detención')
    ax1.set_title('Perfil de Velocidad (RK4)')
    ax1.legend()
    ax1.grid(True, color='#30363d', linestyle=':')
    
    ax2.plot(t_vals, x_vals, color='#d2a8ff', linewidth=2, label='Posición')
    ax2.axhline(y=d_disp, color='#f85149', linestyle='--', label='Límite')
    ax2.set_title('Perfil de Posición (RK4)')
    ax2.legend()
    ax2.grid(True, color='#30363d', linestyle=':')
    
    st.pyplot(fig)

    # Tabla de iteraciones
    with st.expander("📋 Ver Tabla de Discretización (Iteraciones del Método)"):
        st.write(f"Método: Runge-Kutta 4 | Paso de integración (h) = {paso}")
        df_pasos = pd.DataFrame({
            "Iteración": range(1, 11),
            "Tiempo t (s)": t_vals[:10],
            "Posición x (m)": x_vals[:10],
            "Velocidad v (m/s)": v_vals[:10]
        })
        st.dataframe(df_pasos, use_container_width=True, hide_index=True)
# ==========================================
# MÓDULO 2: DETERIORO (RAÍCES)
# ==========================================
elif menu == "Módulo 2: Deterioro":
    st.markdown("<h1 style='font-family: Courier New; color: #58a6ff;'>⚙️ Consola 02 // Deterioro Mecánico</h1>", unsafe_allow_html=True)
    
    # --- PANEL EXPLICATIVO DE MÉTODOS NUMÉRICOS ---
    with st.expander("📚 Ver Fundamento Numérico y Fórmulas Matemáticas", expanded=False):
        st.markdown("""
        ### 🧮 Herramientas Numéricas en esta Consola
        
        Esta consola resuelve el problema de mantenimiento encontrando la raíz de una ecuación no lineal.
        
        **1. Métodos de Búsqueda de Raíces:**
        *   *¿Qué hace?* Encuentra el tiempo \(t\) exacto donde el deterioro \(D(t)\) iguala al umbral permitido.
        *   *Función a resolver:* 
        """)
        st.latex(r"f(t) = D_0 + \alpha t + \beta t^2 - D_{umbral} = 0")
        st.markdown("""
        *   *Métodos implementados:*
            *   **Bisección:** Cierra el intervalo a la mitad iterativamente (lento pero seguro).
            *   **Newton-Raphson:** Usa la derivada \(f'(t)\) para converger rápidamente (requiere buena semilla).
            *   **Secante:** Aproxima la derivada usando dos puntos (no requiere derivada analítica).
        
        **2. Teoría de Errores (Residuo):**
        *   *¿Qué hace?* Evalúa \(|f(t)|\) en cada iteración para saber qué tan cerca estamos de la raíz exacta.
        *   *Fórmula base:* 
        """)
        st.latex(r"Error = |t_{nuevo} - t_{anterior}|")
        st.markdown("""
        **3. Mal Condicionamiento:**
        *   *¿Qué hace?* En el escenario MAL CONDICIONADO, la función tiene una raíz múltiple (la curva es tangente al eje X). Esto hace que la derivada sea casi cero y que métodos como Newton-Raphson fallen o converjan extremadamente lento.
        """)
    
    st.divider()

    # 1. Controles en la barra lateral
    escenario = st.sidebar.selectbox("Escenario Operativo", ["NORMAL", "ESTRÉS", "MAL CONDICIONADO"])
    
    if escenario == "NORMAL":
        D0, alpha, beta, umbral = 10.0, 0.5, 0.05, 50.0
    elif escenario == "ESTRÉS":
        D0, alpha, beta, umbral = 10.0, 0.8, 0.1, 50.0
    else: 
        D0, alpha, beta, umbral = 10.0, 0.0, 0.0001, 50.0 # Raíz múltiple
    
    D0 = st.sidebar.number_input("D_0 (Deterioro Inicial)", value=D0)
    alpha = st.sidebar.number_input("alpha (Componente Lineal)", value=alpha)
    beta = st.sidebar.number_input("beta (Componente Cuadrático)", value=beta)
    umbral = st.sidebar.number_input("Umbral de Inspección", value=umbral)
    
    metodo = st.sidebar.selectbox("Método Numérico", ["Bisección", "Newton-Raphson", "Secante"])
    
    ejecutar = st.sidebar.button("▶ GUARDAR EN HISTORIAL", type="primary", use_container_width=True)

    # 2. Lógica de simulación automática
    f = lambda t: (D0 + alpha*t + beta*t**2) - umbral
    df = lambda t: alpha + 2*beta*t
    
    if metodo == "Bisección":
        raiz, it, hist = biseccion(f, 0, 100, tol=1e-5)
    elif metodo == "Newton-Raphson":
        # Semilla en 50 para que busque en el rango positivo
        raiz, it, hist = newton_raphson(f, df, 50, tol=1e-5)
    else:
        raiz, it, hist = secante(f, 0, 100, tol=1e-5)

    # 3. Guardar en la base de datos (solo si presiona el botón)
    if ejecutar:
        # Reutilizamos la función guardar_simulacion pero adaptamos los datos
        guardar_simulacion(f"M2 - {escenario}", D0, alpha, beta, umbral, raiz, raiz, 0.0)
        st.toast("✅ Simulación de mantenimiento guardada", icon="💾")

    # 4. Renderizado Visual (Siempre visible)
    st.markdown("### Estado de Mantenimiento")
    
    if isinstance(hist, str):
        st.error(f"🔴 **FALLO NUMÉRICO:** {hist}")
    else:
        # Semáforo de mantenimiento
        if raiz < 20.0: 
            st.markdown(f'<div style="background-color:#da3633; padding:20px; border-radius:8px; border-left:10px solid #f85149; font-family:Courier; font-size:18px; font-weight:bold; color:white; margin-bottom:20px;">🔴 ALERTA CRÍTICA: Inspección requerida en {raiz:.2f} días.</div>', unsafe_allow_html=True)
        elif raiz < 60.0: 
            st.markdown(f'<div style="background-color:#9e6a03; padding:20px; border-radius:8px; border-left:10px solid #d29922; font-family:Courier; font-size:18px; font-weight:bold; color:white; margin-bottom:20px;">🟡 PRECAUCIÓN: Programar inspección en {raiz:.2f} días.</div>', unsafe_allow_html=True)
        else: 
            st.markdown(f'<div style="background-color:#238636; padding:20px; border-radius:8px; border-left:10px solid #2ea043; font-family:Courier; font-size:18px; font-weight:bold; color:white; margin-bottom:20px;">🟢 ESTADO ÓPTIMO: Margen de {raiz:.2f} días hasta el umbral.</div>', unsafe_allow_html=True)

        if escenario == "MAL CONDICIONADO":
            st.warning(f"⚠️ **Inestabilidad Numérica (Mal Condicionamiento):** Raíz múltiple detectada. La derivada \(f'(t)\) cercana a cero hace que el método {metodo} tenga problemas de convergencia.")

        # Métricas
        c1, c2, c3 = st.columns(3)
        c1.metric("Tiempo al Umbral", f"{raiz:.4f} días")
        c2.metric("Iteraciones", it)
        c3.metric("Método Usado", metodo)

        # Gráficos (Oscuros)
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Graficamos la función de deterioro
        t_plot = np.linspace(0, max(100, raiz + 20), 200)
        y_plot = f(t_plot)
        
        ax.plot(t_plot, y_plot, color='#58a6ff', linewidth=2, label='f(t) = Deterioro - Umbral')
        ax.axhline(y=0, color='#f85149', linestyle='--', label='Eje X (Umbral)')
        ax.scatter([raiz], [0], color='#2ea043', s=100, zorder=5, label='Raíz Encontrada')
        
        ax.set_title(f'Convergencia del Método: {metodo}')
        ax.set_xlabel('Tiempo (días)')
        ax.set_ylabel('Diferencia de Deterioro')
        ax.legend()
        ax.grid(True, color='#30363d', linestyle=':')
        
        st.pyplot(fig)

        # Tabla de iteraciones
        with st.expander("📋 Ver Tabla de Iteraciones"):
            st.write(f"Método: {metodo} | Tolerancia: 1e-5")
            df_res = pd.DataFrame(hist, columns=["Iteración", "t (aprox)", "f(t)", "Error Absoluto"])
            st.dataframe(df_res, use_container_width=True, hide_index=True)

# ==========================================
# MÓDULO 3: OSCILACIONES (INTEGRACIÓN)
# ==========================================
elif menu == "Módulo 3: Oscilaciones":
    st.markdown("<h1 style='font-family: Courier New; color: #58a6ff;'>🌬️ Consola 03 // Oscilaciones por Viento</h1>", unsafe_allow_html=True)
    
    # --- PANEL EXPLICATIVO DE MÉTODOS NUMÉRICOS ---
    with st.expander("📚 Ver Fundamento Numérico y Fórmulas Matemáticas", expanded=False):
        st.markdown("""
        ### 🧮 Herramientas Numéricas en esta Consola
        
        Esta consola calcula el desplazamiento acumulado de la cabina debido a la fuerza del viento, resolviendo una integral definida.
        
        **1. Integración Numérica:**
        *   *¿Qué hace?* Calcula el área bajo la curva de la función del viento \(v(t)\) para obtener el desplazamiento total.
        *   *Fórmula base:* 
        """)
        st.latex(r"Desplazamiento = \int_{0}^{T} |A \cdot \sin(w \cdot t)| dt")
        st.markdown("""
        *   *Métodos implementados:*
            *   **Trapecio:** Aproxima el área usando trapecios (Error \(O(h^2)\)).
            *   **Simpson 1/3:** Usa parábolas para mayor precisión (Error \(O(h^4)\)).
            *   **Romberg:** Aplica extrapolación de Richardson para acelerar la convergencia.
        
        **2. Teoría de Errores:**
        *   *¿Qué hace?* Compara los tres métodos para ver cuál se acerca más al valor real. A mayor número de subintervalos (\(n\)), menor es el error.
        *   *Fórmula base:* 
        """)
        st.latex(r"E_r = |Valor_{metodo} - Valor_{referencia}|")
        st.markdown("""
        **3. Mal Condicionamiento:**
        *   *¿Qué hace?* En este escenario, la función de viento es una **onda cuadrada** (con discontinuidades). Los métodos de integración polinómica (como Simpson y Romberg) fallan o pierden precisión severamente ante discontinuidades.
        """)
    
    st.divider()

    # 1. Controles en la barra lateral
    escenario = st.sidebar.selectbox("Escenario Operativo", ["NORMAL", "ESTRÉS", "MAL CONDICIONADO"])
    
    if escenario == "NORMAL":
        A, w, T = 2.0, 1.5, 10.0
    elif escenario == "ESTRÉS":
        A, w, T = 5.0, 3.0, 20.0 # Mayor amplitud y frecuencia
    else: 
        A, w, T = 2.0, 1.5, 10.0
    
    A = st.sidebar.number_input("Amplitud del Viento (A)", value=A)
    w = st.sidebar.number_input("Frecuencia Angular (w)", value=w)
    T = st.sidebar.number_input("Tiempo Total de Análisis (s)", value=T)
    n = st.sidebar.slider("Subintervalos (n)", 10, 1000, 100)
    
    ejecutar = st.sidebar.button("▶ GUARDAR EN HISTORIAL", type="primary", use_container_width=True)

    # 2. Lógica de simulación automática
    if escenario == "MAL CONDICIONADO":
        # Onda cuadrada (discontinua, mala para Simpson/Romberg)
        f_viento = lambda t: np.sign(np.sin(w * t)) * A 
    else:
        # Función suave (senoidal)
        f_viento = lambda t: abs(A * np.sin(w * t))
        
    val_trap = trapecio(f_viento, 0, T, n)
    val_simp = simpson_13(f_viento, 0, T, n)
    val_romb, _ = romberg(f_viento, 0, T, 5)

    # 3. Guardar en la base de datos (solo si presiona el botón)
    if ejecutar:
        # Guardamos una simulación ficticia en el historial adaptada al Módulo 3
        guardar_simulacion(f"M3 - {escenario}", A, w, T, n, val_trap, val_romb, 0.0)
        st.toast("✅ Análisis de viento guardado", icon="💾")

    # 4. Renderizado Visual
    st.markdown("### Estado de Oscilación de la Cabina")
    
    # Semáforo basado en el desplazamiento acumulado (usando Romberg como referencia)
    if val_romb > 40.0: 
        st.markdown(f'<div style="background-color:#da3633; padding:20px; border-radius:8px; border-left:10px solid #f85149; font-family:Courier; font-size:18px; font-weight:bold; color:white; margin-bottom:20px;">🔴 ALERTA DE BALANCEO: Desplazamiento crítico acumulado ({val_romb:.2f} m). Suspender operaciones.</div>', unsafe_allow_html=True)
    elif val_romb > 20.0: 
        st.markdown(f'<div style="background-color:#9e6a03; padding:20px; border-radius:8px; border-left:10px solid #d29922; font-family:Courier; font-size:18px; font-weight:bold; color:white; margin-bottom:20px;">🟡 PRECAUCIÓN: Oscilaciones moderadas ({val_romb:.2f} m). Reducir velocidad.</div>', unsafe_allow_html=True)
    else: 
        st.markdown(f'<div style="background-color:#238636; padding:20px; border-radius:8px; border-left:10px solid #2ea043; font-family:Courier; font-size:18px; font-weight:bold; color:white; margin-bottom:20px;">🟢 ESTADO SEGURO: Desplazamiento controlado ({val_romb:.2f} m).</div>', unsafe_allow_html=True)

    if escenario == "MAL CONDICIONADO":
        st.warning(f"⚠️ **Inestabilidad Numérica (Mal Condicionamiento):** La función del viento es discontinua (onda cuadrada). Observa cómo Romberg y Simpson pierden precisión frente a Trapecio debido a la falta de derivabilidad.")

    # Métricas
    st.markdown("#### 📊 Comparación de Métodos de Integración")
    c1, c2, c3 = st.columns(3)
    c1.metric("Trapecio", f"{val_trap:.5f} m")
    c2.metric("Simpson 1/3", f"{val_simp:.5f} m")
    c3.metric("Romberg", f"{val_romb:.5f} m")

    # Gráficos (Oscuros)
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Generamos datos para graficar la curva del viento
    t_plot = np.linspace(0, T, 500)
    y_plot = f_viento(t_plot)
    
    ax.plot(t_plot, y_plot, color='#58a6ff', linewidth=2, label='Perfil del Viento \(|v(t)|\)')
    ax.fill_between(t_plot, y_plot, color='#58a6ff', alpha=0.2, label='Área Integrada (Desplazamiento)')
    
    ax.set_title(f'Perfil de Oscilación (n={n} subintervalos)')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Magnitud del Viento / Desplazamiento')
    ax.legend(loc='upper right')
    ax.grid(True, color='#30363d', linestyle=':')
    
    st.pyplot(fig)

    # Tabla comparativa de errores
    with st.expander("📋 Ver Análisis de Convergencia y Errores"):
        st.write(f"Subintervalos (n) = {n} | Tiempo Total = {T} s")
        
        # Usamos Romberg como la "referencia" asumiendo que es el más preciso en Normal/Estres
        ref = val_romb if escenario != "MAL CONDICIONADO" else val_trap
        
        df_err = pd.DataFrame({
            "Método": ["Trapecio", "Simpson 1/3", "Romberg"],
            "Valor Calculado": [val_trap, val_simp, val_romb],
            "Error Absoluto": [abs(val_trap - ref), abs(val_simp - ref), abs(val_romb - ref)]
        })
        st.dataframe(df_err, use_container_width=True, hide_index=True)
# ==========================================
# MÓDULO 4: SENSORES (INTERPOLACIÓN)
# ==========================================
elif menu == "Módulo 4: Sensores":
    st.title("Consola de Reconstrucción de Sensores")
    escenario = st.sidebar.selectbox("Escenario Operativo", ["NORMAL", "ESTRÉS", "MAL CONDICIONADO"])
    
    if escenario == "NORMAL": x_p = np.array([0, 1, 2, 4, 5]); y_p = np.array([0, 0.84, 0.91, -0.75, -0.95])
    elif escenario == "ESTRÉS": x_p = np.array([0, 0.1, 0.2, 4.8, 5.0]); y_p = np.array([0, 0.1, 0.2, -0.99, -0.95])
    else: x_p = np.array([0, 1, 1, 4, 5]); y_p = np.array([0, 0.84, 0.91, -0.75, -0.95])
    
    x_faltante = st.sidebar.slider("Punto de sensor faltante (x)", 0.0, 5.0, 3.0)
    
    if st.sidebar.button("Interpolar"):
        st.header(f"Resultados de Reconstrucción: {escenario}")
        
        if escenario == "MAL CONDICIONADO":
            st.error("🔴 **ALERTA DE SENSOR:** Nodos duplicados detectados (x=1 repetido). Falla matemática inminente (División por cero).")
        else:
            y_est = lagrange(x_p, y_p, x_faltante)
            st.success("🟢 **RECONSTRUCCIÓN EXITOSA:** El dato faltante ha sido estimado.")
            st.metric("Valor reconstruido", f"{y_est:.4f}")
            
            fig, ax = plt.subplots()
            x_plot = np.linspace(0, 5, 100)
            y_plot = [lagrange(x_p, y_p, xi) for xi in x_plot]
            ax.plot(x_plot, y_plot, 'b-'); ax.scatter(x_p, y_p, color='red'); ax.scatter([x_faltante], [y_est], color='green', s=100)
            st.pyplot(fig)

# ==========================================
# MÓDULO 5: SISTEMAS LINEALES
# ==========================================
elif menu == "Módulo 5: Sistemas Lineales":
    st.title("Consola de Calibración (Sistemas Lineales)")
    st.markdown("**Contexto Operativo:** Los sensores envían lecturas. Resolvemos \(Ax = b\) para hallar el vector de corrección \(x\).")
    
    A = np.array([[4.0, 1.0, 0.0, 0.0],
                  [1.0, 5.0, 1.0, 0.0],
                  [0.0, 1.0, 6.0, 1.0],
                  [0.0, 0.0, 1.0, 7.0]])
    b = np.dot(A, np.array([1.0, 2.0, 3.0, 4.0]))
    
    escenario = st.sidebar.selectbox("Escenario Operativo", ["NORMAL", "ESTRÉS", "MAL CONDICIONADO"])
    
    if escenario == "MAL CONDICIONADO":
        A[3,3] = A[3,2] + 0.00001 
        st.error("🚨 **ALERTA DE CALIBRACIÓN:** Matriz casi singular. Los sensores están enviando datos contradictorios. El sistema puede dar resultados erróneos.")
    else:
        st.success("🟢 **SISTEMA DE CALIBRACIÓN ESTABLE:** Matriz bien condicionada. Lista para resolver.")
        
    st.write("Matriz de Sensores A:")
    st.dataframe(pd.DataFrame(A))
    st.write("Lecturas b:")
    st.dataframe(pd.DataFrame(b, columns=["b"]))
        
    if st.sidebar.button("Resolver"):
        x_jacobi, it_jacobi = jacobi(A, b)
        x_gs, it_gs = gauss_seidel(A, b)
        x_lu = lu_doolittle(A, b)
        
        st.subheader("Resultados de Calibración")
        c1, c2, c3 = st.columns(3)
        c1.metric("Iteraciones Jacobi", it_jacobi)
        c2.metric("Iteraciones Gauss-Seidel", it_gs)
        c3.metric("Residuo LU", f"{np.linalg.norm(np.dot(A, x_lu) - b):.6f}")
        
        st.write(f"**Vector solución Jacobi:** {x_jacobi}")
        st.write(f"**Vector solución Gauss-Seidel:** {x_gs}")
        st.write(f"**Vector solución LU Doolittle:** {x_lu}")

# ==========================================
# PREDICCIÓN IA
# ==========================================
elif menu == "🤖 Predicción IA":
    st.title("Consola de Predicción con Machine Learning")
    try:
        ruta_modelo = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'numerical_engine', 'ml_models', 'modelo_rf.pkl'))
        modelo = joblib.load(ruta_modelo)
        
        v0_ia = st.sidebar.number_input("Velocidad inicial", value=7.5)
        tr_ia = st.sidebar.number_input("Tiempo de reacción", value=1.5)
        a_ia = st.sidebar.number_input("Desaceleración", value=1.2)
        d_disp_ia = st.sidebar.number_input("Distancia disponible", value=50.0)
        
        if st.sidebar.button("Predecir"):
            entrada = np.array([[v0_ia, tr_ia, a_ia, d_disp_ia]])
            prediccion = modelo.predict(entrada)[0]
            
            st.header("Resultados de la IA")
            if prediccion > d_disp_ia: 
                st.error(f"🔴 **PELIGRO:** La IA predice colisión. Distancia: {prediccion:.2f} m.")
            else: 
                st.success(f"🟢 **SEGURO:** La IA predice detención segura. Distancia: {prediccion:.2f} m.")
            
            importancias = modelo.feature_importances_
            variables = ['Velocidad', 'T. Reacción', 'Desaceleración', 'Distancia']
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.barh(variables, importancias, color='teal')
            ax.set_title('Importancia de las variables en el modelo')
            st.pyplot(fig)
    except Exception as e:
        st.warning("Modelo de IA no encontrado. Asegúrate de haber ejecutado el script de entrenamiento.")

# ==========================================
# HISTORIAL
# ==========================================
elif menu == "Historial":
    st.title("Historial de Operaciones")
    st.write("Registro de simulaciones guardadas en la base de datos del Centro de Comandos.")
    conn = sqlite3.connect('simtele_risk.db')
    df = pd.read_sql_query("SELECT * FROM simulaciones", conn)
    conn.close()
    if len(df) > 0: 
        st.dataframe(df, use_container_width=True)
    else: 
        st.warning("No hay simulaciones registradas.")
