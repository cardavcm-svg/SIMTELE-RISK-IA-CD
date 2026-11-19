import numpy as np

# ==========================================
# MÓDULO 2: RAÍCES
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
        if f(a) * fc < 0: b = c
        else: a = c
    return c, max_iter, iteraciones

def newton_raphson(f, df, x0, tol=1e-5, max_iter=100):
    iteraciones = []
    x = x0
    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)
        if dfx == 0:
            return None, i, "Derivada cero."
        x_new = x - fx / dfx
        error = abs(x_new - x)
        iteraciones.append((i+1, x_new, f(x_new), error))
        if error < tol:
            return x_new, i+1, iteraciones
        x = x_new
    return x, max_iter, iteraciones

def secante(f, x0, x1, tol=1e-5, max_iter=100):
    iteraciones = []
    for i in range(max_iter):
        fx0 = f(x0)
        fx1 = f(x1)
        if fx1 - fx0 == 0:
            return None, i, "División por cero."
        x_new = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        error = abs(x_new - x1)
        iteraciones.append((i+1, x_new, f(x_new), error))
        if error < tol:
            return x_new, i+1, iteraciones
        x0, x1 = x1, x_new
    return x1, max_iter, iteraciones

# ==========================================
# MÓDULO 3: INTEGRACIÓN
# ==========================================
def trapecio(f, a, b, n):
    h = (b - a) / n
    suma = f(a) + f(b)
    for i in range(1, n):
        suma += 2 * f(a + i*h)
    return (h / 2) * suma

def simpson_13(f, a, b, n):
    if n % 2 != 0: n += 1
    h = (b - a) / n
    suma = f(a) + f(b)
    for i in range(1, n):
        x = a + i*h
        if i % 2 == 0: suma += 2 * f(x)
        else: suma += 4 * f(x)
    return (h / 3) * suma

def romberg(f, a, b, n):
    # Romberg usando n niveles (n=4 o 5 es suficiente)
    R = np.zeros((n, n))
    h = b - a
    R[0, 0] = 0.5 * h * (f(a) + f(b))
    for i in range(1, n):
        h /= 2
        suma = 0
        for k in range(1, 2**(i-1) + 1):
            suma += f(a + (2*k - 1) * h)
        R[i, 0] = 0.5 * R[i-1, 0] + h * suma
        for j in range(1, i + 1):
            R[i, j] = R[i, j-1] + (R[i, j-1] - R[i-1, j-1]) / (4**j - 1)
    return R[n-1, n-1], R

# ==========================================
# MÓDULO 4: INTERPOLACIÓN
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
