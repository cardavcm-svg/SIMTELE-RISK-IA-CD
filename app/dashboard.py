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
st.sidebar.title("🚡 SIMTELE-RISK")
menu = st.sidebar.radio("Navegación", ["Inicio", "Módulo 1: Detención", "Módulo 2: Deterioro", "Módulo 3: Oscilaciones", "Módulo 4: Sensores", "Módulo 5: Sistemas Lineales", "🤖 Predicción IA", "Historial"])

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
# MÓDULO 1: DETENCIÓN
# ==========================================
elif menu == "Módulo 1: Detención":
    st.title("Consola de Detención de Emergencia")
    escenario = st.sidebar.selectbox("Escenario Operativo", ["NORMAL", "ESTRÉS", "MAL CONDICIONADO"])
    
    if escenario == "NORMAL": v0_def, tr_def, a_def, d_def = 5.0, 1.0, 1.0, 30.0
    elif escenario == "ESTRÉS": v0_def, tr_def, a_def, d_def = 10.0, 2.0, 0.8, 85.0
    else: v0_def, tr_def, a_def, d_def = 12.0, 3.0, 0.5, 50.0
    
    v0 = st.sidebar.number_input("Velocidad inicial (m/s)", value=v0_def)
    tr = st.sidebar.number_input("Tiempo de reacción (s)", value=tr_def)
    a = st.sidebar.number_input("Desaceleración (m/s²)", value=a_def)
    d_disp = st.sidebar.number_input("Distancia disponible (m)", value=d_def)
    
    if st.sidebar.button("Ejecutar Simulación"):
        d_ana = (v0 * tr) + (v0**2 / (2*a))
        t_ana = tr + (v0/a)
        def f_x(t, x, v): return v
        def f_v(t, x, v): return -a if t > tr else 0.0
        
        paso = 0.01 if escenario != "MAL CONDICIONADO" else 0.5
        t_vals, x_vals, v_vals = rk4(f_x, f_v, 0, 0, v0, t_ana + 2.0, paso)
        idx = np.where(v_vals <= 0)[0]
        d_num = x_vals[idx[0]] if len(idx) > 0 else x_vals[-1]
        err_rel = abs(d_ana - d_num) / d_ana * 100
        
        st.header(f"Resultados de Telemetría: {escenario}")
        margen = d_disp - d_num
        
        if margen > 5.0: st.success(f"🟢 **ESTADO: SEGURO** | Margen de Frenado: {margen:.2f} m")
        elif margen > 0: st.warning(f"🟡 **ESTADO: PRECAUCIÓN** | Margen Crítico: {margen:.2f} m")
        else: st.error(f"🔴 **ESTADO: PELIGRO DE COLISIÓN** | Déficit: {abs(margen):.2f} m")
            
        if escenario == "MAL CONDICIONADO":
            st.warning("⚠️ Alerta de Inestabilidad: El paso de integración h=0.5 es demasiado grande, generando un error numérico severo (Mal Condicionamiento).")
            
        c1, c2, c3 = st.columns(3)
        c1.metric("Distancia Teórica", f"{d_ana:.4f} m")
        c2.metric("Distancia Calculada", f"{d_num:.4f} m")
        c3.metric("Error de Cálculo", f"{err_rel:.4f} %")
        
        guardar_simulacion(escenario, v0, tr, a, d_disp, d_ana, d_num, err_rel)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(t_vals, v_vals, 'b-'); ax1.set_title('Perfil de Velocidad'); ax1.grid(True)
        ax2.plot(t_vals, x_vals, 'm-'); ax2.axhline(y=d_disp, color='r', linestyle='--'); ax2.set_title('Perfil de Posición'); ax2.grid(True)
        st.pyplot(fig)

# ==========================================
# MÓDULO 2: DETERIORO (RAÍCES)
# ==========================================
elif menu == "Módulo 2: Deterioro":
    st.title("Consola de Deterioro Mecánico")
    escenario = st.sidebar.selectbox("Escenario Operativo", ["NORMAL", "ESTRÉS", "MAL CONDICIONADO"])
    
    if escenario == "NORMAL": D0, alpha, beta, umbral = 10.0, 0.5, 0.05, 50.0
    elif escenario == "ESTRÉS": D0, alpha, beta, umbral = 10.0, 0.8, 0.1, 50.0
    else: D0, alpha, beta, umbral = 10.0, 0.0, 0.0001, 50.0 
    
    D0 = st.sidebar.number_input("D_0 (Inicial)", value=D0)
    alpha = st.sidebar.number_input("alpha (Lineal)", value=alpha)
    beta = st.sidebar.number_input("beta (Cuadrático)", value=beta)
    umbral = st.sidebar.number_input("Umbral de Inspección", value=umbral)
    metodo = st.sidebar.selectbox("Método Numérico", ["Bisección", "Newton-Raphson", "Secante"])
    
    if st.sidebar.button("Calcular Raíz"):
        f = lambda t: (D0 + alpha*t + beta*t**2) - umbral
        df = lambda t: alpha + 2*beta*t
        
        if metodo == "Bisección": raiz, it, hist = biseccion(f, 0, 100, tol=1e-5)
        elif metodo == "Newton-Raphson": raiz, it, hist = newton_raphson(f, df, 50, tol=1e-5)
        else: raiz, it, hist = secante(f, 0, 100, tol=1e-5)
            
        if isinstance(hist, str):
            st.error(hist)
        else:
            st.header(f"Resultados de Mantenimiento: {escenario}")
            
            # Semáforo
            if raiz < 20.0: st.error(f"🔴 **ALERTA MANTENIMIENTO:** Umbral en {raiz:.2f} días. Inspección inmediata requerida.")
            elif raiz < 50.0: st.warning(f"🟡 **PRECAUCIÓN:** Umbral en {raiz:.2f} días. Programar inspección.")
            else: st.success(f"🟢 **ESTADO ÓPTIMO:** Umbral en {raiz:.2f} días. Sin riesgos inmediatos.")
            
            if escenario == "MAL CONDICIONADO":
                st.warning("⚠️ Alerta: Raíz múltiple detectada. La derivada cercana a cero causa que el método converja muy lentamente (Mal Condicionamiento).")
                
            st.metric("Iteraciones Realizadas", it)
            df_res = pd.DataFrame(hist, columns=["Iter", "t", "f(t)", "Error"])
            st.dataframe(df_res)

# ==========================================
# MÓDULO 3: OSCILACIONES (INTEGRACIÓN)
# ==========================================
elif menu == "Módulo 3: Oscilaciones":
    st.title("Consola de Oscilaciones por Viento")
    escenario = st.sidebar.selectbox("Escenario Operativo", ["NORMAL", "ESTRÉS", "MAL CONDICIONADO"])
    
    if escenario == "NORMAL": A, w, T = 2.0, 1.5, 10.0
    elif escenario == "ESTRÉS": A, w, T = 5.0, 3.0, 20.0
    else: A, w, T = 2.0, 1.5, 10.0
    
    A = st.sidebar.number_input("Amplitud del Viento", value=A)
    w = st.sidebar.number_input("Frecuencia Angular", value=w)
    T = st.sidebar.number_input("Tiempo Total", value=T)
    n = st.sidebar.slider("Subintervalos (n)", 10, 1000, 100)
    
    if st.sidebar.button("Calcular Integral"):
        if escenario == "MAL CONDICIONADO": f_viento = lambda t: np.sign(np.sin(w*t)) * A 
        else: f_viento = lambda t: abs(A * np.sin(w * t))
            
        val_trap = trapecio(f_viento, 0, T, n)
        val_simp = simpson_13(f_viento, 0, T, n)
        val_romb, _ = romberg(f_viento, 0, T, 5)
        
        st.header(f"Resultados de Análisis de Viento: {escenario}")
        
        # Semáforo basado en amplitud
        if val_trap > 40.0: st.error("🔴 **ALERTA:** Desplazamiento acumulado crítico. Peligro de balanceo severo.")
        elif val_trap > 20.0: st.warning("🟡 **PRECAUCIÓN:** Oscilaciones moderadas. Monitorear velocidad del viento.")
        else: st.success("🟢 **ESTADO SEGURO:** Desplazamiento dentro de los límites operativos.")
            
        c1, c2, c3 = st.columns(3)
        c1.metric("Trapecio", f"{val_trap:.5f}")
        c2.metric("Simpson 1/3", f"{val_simp:.5f}")
        c3.metric("Romberg", f"{val_romb:.5f}")
        
        if escenario == "MAL CONDICIONADO":
            st.warning("⚠️ Alerta: La función de viento es discontinua (onda cuadrada). Romberg pierde precisión severamente (Mal Condicionamiento).")

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
