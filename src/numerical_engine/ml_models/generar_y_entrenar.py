import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import sys
import os

# Conectamos con el motor numérico
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from numerical_engine.edos import rk4

def generar_datos(n_simulaciones=5000):
    print(f"Generando {n_simulaciones} escenarios sintéticos...")
    datos = []
    
    for _ in range(n_simulaciones):
        # Generamos parámetros aleatorios dentro de rangos lógicos
        v0 = np.random.uniform(2.0, 15.0)      # Velocidad inicial
        tr = np.random.uniform(0.5, 3.0)       # Tiempo de reacción
        a = np.random.uniform(0.5, 3.0)        # Desaceleración
        d_disp = np.random.uniform(20.0, 120.0) # Distancia disponible
        
        # Resolvemos con RK4
        def f_x(t, x, v): return v
        def f_v(t, x, v): return -a if t > tr else 0.0
        
        t_ana = tr + (v0 / a)
        # Simulamos hasta un poco después del tiempo de detención analítico
        tf = t_ana + 1.0 
        
        try:
            t_vals, x_vals, v_vals = rk4(f_x, f_v, 0, 0, v0, tf, 0.01)
            idx = np.where(v_vals <= 0)[0]
            if len(idx) > 0:
                dist_total = x_vals[idx[0]]
            else:
                dist_total = x_vals[-1]
                
            # Calculamos si es seguro (1) o crítico (0)
            seguro = 1 if dist_total <= d_disp else 0
            
            datos.append([v0, tr, a, d_disp, dist_total, seguro])
        except Exception as e:
            pass

    df = pd.DataFrame(datos, columns=['v0', 'tr', 'a', 'd_disp', 'dist_total', 'seguro'])
    return df

# 1. Generar y guardar el Dataset
df = generar_datos(5000)
os.makedirs('../data', exist_ok=True)
df.to_csv('../data/dataset_simtele.csv', index=False)
print("Dataset guardado en data/dataset_simtele.csv")

# 2. Entrenar el modelo de IA (Random Forest)
print("Entrenando modelo de Random Forest...")
X = df[['v0', 'tr', 'a', 'd_disp']] # Variables de entrada
y = df['dist_total']                # Variable a predecir

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo_rf = RandomForestRegressor(n_estimators=100, random_state=42)
modelo_rf.fit(X_train, y_train)

# 3. Evaluar el modelo
y_pred = modelo_rf.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Error Absoluto Medio (MAE): {mae:.4f} m")
print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse:.4f} m")

# 4. Guardar el modelo entrenado
os.makedirs('ml_models', exist_ok=True)
joblib.dump(modelo_rf, 'ml_models/modelo_rf.pkl')
print("Modelo guardado en ml_models/modelo_rf.pkl")
