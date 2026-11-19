8import numpy as np

def rk2_heun(f_x, f_v, t0, x0, v0, tf, h):
    """
    Método de Runge-Kutta de 2do orden (Heun)
    f_x: función de la derivada de la posición (dx/dt)
    f_v: función de la derivada de la velocidad (dv/dt)
    """
    t_vals = np.arange(t0, tf + h, h)
    n = len(t_vals)
    x_vals = np.zeros(n)
    v_vals = np.zeros(n)
    
    x_vals[0] = x0
    v_vals[0] = v0
    
    for i in range(n - 1):
        t_i = t_vals[i]
        x_i = x_vals[i]
        v_i = v_vals[i]
        
        # Predictor (Euler)
        k1_x = f_x(t_i, x_i, v_i)
        k1_v = f_v(t_i, x_i, v_i)
        
        x_pred = x_i + h * k1_x
        v_pred = v_i + h * k1_v
        
        # Corrector (Heun)
        k2_x = f_x(t_i + h, x_pred, v_pred)
        k2_v = f_v(t_i + h, x_pred, v_pred)
        
        x_vals[i+1] = x_i + (h / 2) * (k1_x + k2_x)
        v_vals[i+1] = v_i + (h / 2) * (k1_v + k2_v)
        
    return t_vals, x_vals, v_vals

def rk4(f_x, f_v, t0, x0, v0, tf, h):
    """
    Método de Runge-Kutta de 4to orden
    """
    t_vals = np.arange(t0, tf + h, h)
    n = len(t_vals)
    x_vals = np.zeros(n)
    v_vals = np.zeros(n)
    
    x_vals[0] = x0
    v_vals[0] = v0
    
    for i in range(n - 1):
        t_i = t_vals[i]
        x_i = x_vals[i]
        v_i = v_vals[i]
        
        # k1
        k1_x = f_x(t_i, x_i, v_i)
        k1_v = f_v(t_i, x_i, v_i)
        
        # k2
        k2_x = f_x(t_i + h/2, x_i + h/2 * k1_x, v_i + h/2 * k1_v)
        k2_v = f_v(t_i + h/2, x_i + h/2 * k1_x, v_i + h/2 * k1_v)
        
        # k3
        k3_x = f_x(t_i + h/2, x_i + h/2 * k2_x, v_i + h/2 * k2_v)
        k3_v = f_v(t_i + h/2, x_i + h/2 * k2_x, v_i + h/2 * k2_v)
        
        # k4
        k4_x = f_x(t_i + h, x_i + h * k3_x, v_i + h * k3_v)
        k4_v = f_v(t_i + h, x_i + h * k3_x, v_i + h * k3_v)
        
        x_vals[i+1] = x_i + (h / 6) * (k1_x + 2*k2_x + 2*k3_x + k4_x)
        v_vals[i+1] = v_i + (h / 6) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
        
    return t_vals, x_vals, v_vals
