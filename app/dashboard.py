import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import sqlite3
import pandas as pd
from datetime import datetime
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.numerical_engine.edos import rk2_heun, rk4
from src.numerical_engine.otros_metodos import biseccion, newton_raphson, secante, trapecio, simpson_13, romberg, lagrange
from src.numerical_engine.sistemas_lineales import jacobi, gauss_seidel, lu_doolittle

# --- BASE DE DATOS ---
def init_db():
    conn = sqlite3.connect('simtele_risk.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS simulaciones
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, fecha TEXT, escenario TEXT, v0 REAL, tr REAL, a REAL, d_disp REAL, d_analitica REAL, d_numerica REAL, error_rel REAL)''')
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
# INICIO
# ==========================================
if menu == "Inicio":
    st.title("Bienvenido a SIMTELE-RISK")
    st.write("Selecciona un módulo en la barra lateral.")
    col1, col2 = st.columns(2)
    with col1:
        st.info("**Módulo 1: Detención** (EDOs)")
        st.info("**Módulo 3: Oscilaciones** (Integración)")
        st.info("**Módulo 5: Sistemas Lineales** (Jacobi, LU)")
    with col2:
        st.warning("**Módulo 2: Deterioro** (Raíces)")
        st.success("**Módulo 4: Sensores** (Interpolación)")
        st.info("**Predicción IA** (Random Forest)")

# ==========================================
# MÓDULO 1 (Con escenario Mal Condicionado explícito)
# ==========================================
elif menu == "Módulo 1: Detención":
    st.title("Módulo 1: Detención de Emergencia")
    escenario = st.sidebar.selectbox("Escenario", ["NORMAL", "ESTRÉS", "MAL CONDICIONADO"])
    
    if escenario == "NORMAL":
        v0_def, tr_def, a_def, d_def = 5.0, 1.0, 1.0, 30.0
    elif escenario == "ESTRÉS":
        v0_def, tr_def, a_def, d_def = 10.0, 2.0, 0.8, 85.0
    else: # MAL CONDICIONADO (Matemáticamente inestable: paso de integración muy grande)
        v0_def, tr_def, a_def, d_def = 12.0, 3.0, 0.5, 50.0
    
    v0 = st.sidebar.number_input("Velocidad inicial (m/s)", value=v0_def)
    tr = st.sidebar.number_input("Tiempo de reacción (s)", value=tr_def)
    a = st.sidebar.number_input("Desaceleración (m/s²)", value=a_def)
    d_disp = st.sidebar.number_input("Distancia disponible (m)", value=d_def)
    
    if st.sidebar.button("Ejecutar Simulación"):
        d_ana = (v0 * tr) + (v0**2 / (2*a))
        t_ana = tr + (v0/a)
        def f_x(t, x, v): return v
        def f_v(t, x, v): return -a if t > tr else 0.0
        
        # Paso de integración dinámico: Normal/Estres -> 0.01, Mal Condicionado -> 0.5 (Fuerza el error)
        paso = 0.01 if escenario != "MAL CONDICIONADO" else 0.5
        
        t_vals, x_vals, v_vals = rk4(f_x, f_v, 0, 0, v0, t_ana + 2.0, paso)
        idx = np.where(v_vals <= 0)[0]
        d_num = x_vals[idx[0]] if len(idx) > 0 else x_vals[-1]
        err_rel = abs(d_ana - d_num) / d_ana * 100
        
        st.header(f"Resultados: {escenario}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Distancia Analítica", f"{d_ana:.4f} m")
        c2.metric("Distancia Numérica", f"{d_num:.4f} m")
        c3.metric("Error Relativo", f"{err_rel:.4f} %")
        
        if escenario == "MAL CONDICIONADO":
            st.warning("⚠️ El error es grande porque el paso de integración es muy grande (h=0.5). Esto es un ejemplo de MAL CONDICIONAMIENTO numérico.")
        
        margen = d_disp - d_num
        if margen > 0: st.success(f"Margen: {margen:.2f} m (Seguro)")
        else: st.error(f"Margen: {margen:.2f} m (Crítico)")
        
        guardar_simulacion(escenario, v0, tr, a, d_disp, d_ana, d_num, err_rel)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(t_vals, v_vals, 'b-'); ax1.set_title('Velocidad'); ax1.grid(True)
        ax2.plot(t_vals, x_vals, 'm-'); ax2.axhline(y=d_disp, color='r', linestyle='--'); ax2.set_title('Posición'); ax2.grid(True)
        st.pyplot(fig)

# ==========================================
# MÓDULO 2 (Raíces con 3 escenarios)
# ==========================================
elif menu == "Módulo 2: Deterioro":
    st.title("Módulo 2: Deterioro Mecánico (Raíces)")
    escenario = st.sidebar.selectbox("Escenario", ["NORMAL", "ESTRÉS", "MAL CONDICIONADO"])
    
    if escenario == "NORMAL":
        D0, alpha, beta, umbral = 10.0, 0.5, 0.05, 50.0
    elif escenario == "ESTRÉS":
        D0, alpha, beta, umbral = 10.0, 0.8, 0.1, 50.0
    else: # MAL CONDICIONADO: Raíz múltiple (derivada casi cero)
        D0, alpha, beta, umbral = 10.0, 0.0, 0.0001, 50.0 # Cambio para que la raíz sea cuadrática pura
    
    D0 = st.sidebar.number_input("D_0", value=D0)
    alpha = st.sidebar.number_input("alpha", value=alpha)
    beta = st.sidebar.number_input("beta", value=beta)
    umbral = st.sidebar.number_input("Umbral", value=umbral)
    
    metodo = st.sidebar.selectbox("Método", ["Bisección", "Newton-Raphson", "Secante"])
    
    if st.sidebar.button("Calcular Raíz"):
        f = lambda t: (D0 + alpha*t + beta*t**2) - umbral
        df = lambda t: alpha + 2*beta*t
        
        if metodo == "Bisección":
            raiz, it, hist = biseccion(f, 0, 100, tol=1e-5)
        elif metodo == "Newton-Raphson":
            raiz, it, hist = newton_raphson(f, df, 50, tol=1e-5)
        else:
            raiz, it, hist = secante(f, 0, 100, tol=1e-5)
            
        if isinstance(hist, str):
            st.error(hist)
        else:
            st.success(f"Raíz encontrada en t = {raiz:.4f}")
            df_res = pd.DataFrame(hist, columns=["Iter", "t", "f(t)", "Error"])
            st.dataframe(df_res)
            if escenario == "MAL CONDICIONADO":
                st.warning("⚠️ Mal condicionamiento: Observa cómo las iteraciones avanzan lentamente debido a la derivada pequeña (raíz múltiple).")

# ==========================================
# MÓDULO 3 (Integración con 3 escenarios y Romberg)
# ==========================================
elif menu == "Módulo 3: Oscilaciones":
    st.title("Módulo 3: Oscilaciones por Viento (Integración)")
    escenario = st.sidebar.selectbox("Escenario", ["NORMAL", "ESTRÉS", "MAL CONDICIONADO"])
    
    if escenario == "NORMAL":
        A, w, T = 2.0, 1.5, 10.0
    elif escenario == "ESTRÉS":
        A, w, T = 5.0, 3.0, 20.0
    else: # MAL CONDICIONADO: Función con discontinuidad (Mal para integración)
        A, w, T = 2.0, 1.5, 10.0
    
    A = st.sidebar.number_input("Amplitud", value=A)
    w = st.sidebar.number_input("Frecuencia", value=w)
    T = st.sidebar.number_input("Tiempo Total", value=T)
    n = st.sidebar.slider("Subintervalos (n)", 10, 1000, 100)
    
    if st.sidebar.button("Calcular Integral"):
        if escenario == "MAL CONDICIONADO":
            f_viento = lambda t: np.sign(np.sin(w*t)) * A # Onda cuadrada (discontinua, mala para Romberg)
        else:
            f_viento = lambda t: abs(A * np.sin(w * t))
            
        val_trap = trapecio(f_viento, 0, T, n)
        val_simp = simpson_13(f_viento, 0, T, n)
        val_romb, _ = romberg(f_viento, 0, T, 5) # Romberg
        
        st.metric("Trapecio", f"{val_trap:.5f}")
        st.metric("Simpson 1/3", f"{val_simp:.5f}")
        st.metric("Romberg", f"{val_romb:.5f}")
        
        if escenario == "MAL CONDICIONADO":
            st.warning("⚠️ Mal condicionamiento: La función es discontinua. Romberg fallará o dará un error grande.")

# ==========================================
# MÓDULO 4 (Sensores con 3 escenarios)
# ==========================================
elif menu == "Módulo 4: Sensores":
    st.title("Módulo 4: Reconstrucción de Sensores (Interpolación)")
    escenario = st.sidebar.selectbox("Escenario", ["NORMAL", "ESTRÉS", "MAL CONDICIONADO"])
    
    if escenario == "NORMAL":
        x_p = np.array([0, 1, 2, 4, 5]); y_p = np.array([0, 0.84, 0.91, -0.75, -0.95])
    elif escenario == "ESTRÉS":
        x_p = np.array([0, 0.1, 0.2, 4.8, 5.0]); y_p = np.array([0, 0.1, 0.2, -0.99, -0.95]) # Nodos muy pegados
    else: # MAL CONDICIONADO: Nodos iguales (División por cero)
        x_p = np.array([0, 1, 1, 4, 5]); y_p = np.array([0, 0.84, 0.91, -0.75, -0.95])
    
    x_faltante = st.sidebar.slider("Punto a interpolar (x)", 0.0, 5.0, 3.0)
    
    if st.sidebar.button("Interpolar"):
        if escenario == "MAL CONDICIONADO":
            st.error("🚨 MAL CONDICIONADO: Tienes nodos duplicados en X (x=1 repetido). La interpolación fallará por división por cero.")
        else:
            y_est = lagrange(x_p, y_p, x_faltante)
            st.success(f"Valor reconstruido: {y_est:.4f}")
            
            fig, ax = plt.subplots()
            x_plot = np.linspace(0, 5, 100)
            y_plot = [lagrange(x_p, y_p, xi) for xi in x_plot]
            ax.plot(x_plot, y_plot, 'b-'); ax.scatter(x_p, y_p, color='red'); ax.scatter([x_faltante], [y_est], color='green', s=100)
            st.pyplot(fig)

# ==========================================
# MÓDULO 5: SISTEMAS LINEALES (¡Nuevo!)
# ==========================================
elif menu == "Módulo 5: Sistemas Lineales":
    st.title("Módulo 5: Sistemas Lineales")
    st.write("Comparación de métodos para resolver \(Ax = b\). Usamos una matriz SPD (Simétrica y Definida Positiva) de 4x4.")
    
    # Matriz SPD estricta
    A = np.array([[4.0, 1.0, 0.0, 0.0],
                  [1.0, 5.0, 1.0, 0.0],
                  [0.0, 1.0, 6.0, 1.0],
                  [0.0, 0.0, 1.0, 7.0]])
    # Solución real x* = [1, 2, 3, 4]
    b = np.dot(A, np.array([1.0, 2.0, 3.0, 4.0]))
    
    st.write("Matriz A:")
    st.dataframe(pd.DataFrame(A))
    st.write("Vector b:")
    st.dataframe(pd.DataFrame(b, columns=["b"]))
    
    escenario = st.sidebar.selectbox("Escenario", ["NORMAL", "ESTRÉS", "MAL CONDICIONADO"])
    
    if escenario == "MAL CONDICIONADO":
        # Hacemos la matriz casi singular para elevar el número de condición
        A[3,3] = A[3,2] + 0.00001 
        st.warning("⚠️ Matriz mal condicionada: Número de condición elevado. Los métodos iterativos podrían no converger.")
        st.write("Matriz A (Mal Condicionada):")
        st.dataframe(pd.DataFrame(A))
        
    if st.sidebar.button("Resolver"):
        x_jacobi, it_jacobi = jacobi(A, b)
        x_gs, it_gs = gauss_seidel(A, b)
        x_lu = lu_doolittle(A, b)
        
        st.subheader("Resultados")
        st.write(f"**Jacobi** (Iteraciones: {it_jacobi}): {x_jacobi}")
        st.write(f"**Gauss-Seidel** (Iteraciones: {it_gs}): {x_gs}")
        st.write(f"**LU Doolittle**: {x_lu}")
        
        # Cálculo del residuo
        st.write(f"Residuo Jacobi: {np.linalg.norm(np.dot(A, x_jacobi) - b)}")
        st.write(f"Residuo Gauss-Seidel: {np.linalg.norm(np.dot(A, x_gs) - b)}")

# ==========================================
# PREDICCIÓN IA
# ==========================================
elif menu == "🤖 Predicción IA":
    st.title("🤖 Predicción con Machine Learning")
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
            st.success(f"Distancia predicha: {prediccion:.2f} m")
            if prediccion > d_disp_ia: st.error("Colisión predicha.")
            else: st.info("Seguro.")
            
            importancias = modelo.feature_importances_
            variables = ['v0', 'tr', 'a', 'd_disp']
            fig, ax = plt.subplots()
            ax.barh(variables, importancias, color='teal')
            st.pyplot(fig)
    except:
        st.warning("Modelo no encontrado.")

# ==========================================
# HISTORIAL
# ==========================================
elif menu == "Historial":
    st.title("Historial de Simulaciones")
    conn = sqlite3.connect('simtele_risk.db')
    df = pd.read_sql_query("SELECT * FROM simulaciones", conn)
    conn.close()
    if len(df) > 0: st.dataframe(df)
    else: st.warning("No hay simulaciones.")
