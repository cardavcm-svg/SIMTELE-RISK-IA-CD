import numpy as np

# ==========================================
# MÓDULO 2: RAÍCES (Bisección y Newton)
# ==========================================
def biseccion(f, a, b, tol=1e-5, max_iter=100):
    if f(a) * f(b) >= 0:
        return None, 0, "El intervalo no contiene una raíz (f(a)*f(b) > 0)"
    
    iteraciones = []
    for i in range(max_iter):
        c = (a + b) / 2.0
        fc = f(c)
        error = abs(b - a) / 2.0
        iteraciones.append((i+1, c, fc, error))
        
        if error < tol or fc == 0:
            return c, i+1, iteraciones
        
        if f(a) * fc < 0:
            b = c
        else:
            a = c
    return c, max_iter, iteraciones

def newton_raphson(f, df, x0, tol=1e-5, max_iter=100):
    iteraciones = []
    x = x0
    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)
        if dfx == 0:
            return None, i, "Derivada cero. El método falla."
        
        x_new = x - fx / dfx
        error = abs(x_new - x)
        iteraciones.append((i+1, x_new, f(x_new), error))
        
        if error < tol:
            return x_new, i+1, iteraciones
        x = x_new
    return x, max_iter, iteraciones

# ==========================================
# MÓDULO 3: INTEGRACIÓN (Trapecio y Simpson)
# ==========================================
def trapecio(f, a, b, n):
    h = (b - a) / n
    suma = f(a) + f(b)
    for i in range(1, n):
        suma += 2 * f(a + i*h)
    return (h / 2) * suma

def simpson_13(f, a, b, n):
    if n % 2 != 0: n += 1 # Simpson requiere n par
    h = (b - a) / n
    suma = f(a) + f(b)
    for i in range(1, n):
        x = a + i*h
        if i % 2 == 0:
            suma += 2 * f(x)
        else:
            suma += 4 * f(x)
    return (h / 3) * suma

# ==========================================
# MÓDULO 4: INTERPOLACIÓN (Lagrange)
# ==========================================
def lagrange(x_puntos, y_puntos, x_eval):
    n = len(x_puntos)
    y_eval = 0.0
    for i in range(n):
        termino = y_puntos[i]
        for j in range(n):
            if i != j:
                termino *= (x_eval - x_puntos[j]) / (x_puntos[i] - x_puntos[j])
        y_eval += termino
    return y_eval
