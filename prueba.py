from src.numerical_engine.edos import rk2_heun, rk4

# Datos del Escenario Normal
v0 = 5.0
a = 1.0
tr = 1.0

def f_x(t, x, v):
    return v

def f_v(t, x, v):
    # Si el tiempo es mayor al de reacción, desacelera. Si no, va constante.
    return -a if t > tr else 0.0

# Ejecutamos RK4 desde t=0 hasta t=10, con paso h=0.1
t, x, v = rk4(f_x, f_v, 0, 0, v0, 10, 0.001)

# Buscamos dónde se detiene
for i in range(len(t)):
    if v[i] <= 0:
        print(f"Tiempo de detención: {t[i]:.2f} s")
        print(f"Distancia total: {x[i]:.2f} m")
        break