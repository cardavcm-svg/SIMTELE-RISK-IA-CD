import numpy as np

def jacobi(A, b, tol=1e-6, max_iter=100):
    """Método de Jacobi para resolver Ax = b"""
    n = len(b)
    x = np.zeros(n)
    x_new = np.zeros(n)
    iteraciones = []
    
    for k in range(max_iter):
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if i != j)
            x_new[i] = (b[i] - s) / A[i][i]
        
        # Cálculo del error relativo (Teoría de errores)
        error = np.linalg.norm(x_new - x, ord=np.inf) / (np.linalg.norm(x_new, ord=np.inf) + 1e-12)
        iteraciones.append({"iteracion": k+1, "x": x_new.tolist(), "error_relativo": error})
        
        if error < tol:
            return {"solucion": x_new.tolist(), "iteraciones": iteraciones, "convergio": True, "iteraciones_totales": k+1}
            
        x = x_new.copy()
        
    return {"solucion": x.tolist(), "iteraciones": iteraciones, "convergio": False, "iteraciones_totales": max_iter}

def gauss_seidel(A, b, tol=1e-6, max_iter=100):
    """Método de Gauss-Seidel para resolver Ax = b"""
    n = len(b)
    x = np.zeros(n)
    iteraciones = []
    
    for k in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            s1 = sum(A[i][j] * x[j] for j in range(i))
            s2 = sum(A[i][j] * x_old[j] for j in range(i + 1, n))
            x[i] = (b[i] - s1 - s2) / A[i][i]
            
        error = np.linalg.norm(x - x_old, ord=np.inf) / (np.linalg.norm(x, ord=np.inf) + 1e-12)
        iteraciones.append({"iteracion": k+1, "x": x.tolist(), "error_relativo": error})
        
        if error < tol:
            return {"solucion": x.tolist(), "iteraciones": iteraciones, "convergio": True, "iteraciones_totales": k+1}
            
    return {"solucion": x.tolist(), "iteraciones": iteraciones, "convergio": False, "iteraciones_totales": max_iter}
