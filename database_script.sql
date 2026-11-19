-- Script de Creación de Base de Datos para SIMTELE-RISK
-- Autor: Carlos David Castañón Mendez

-- 1. Tabla de Simulaciones (Entrada)
CREATE TABLE IF NOT EXISTS Simulacion (
    id_simulacion INTEGER PRIMARY KEY AUTOINCREMENT,
    fecha DATETIME DEFAULT CURRENT_TIMESTAMP,
    escenario VARCHAR(50) NOT NULL, -- NORMAL, ESTRÉS, MAL CONDICIONADO
    v0 REAL NOT NULL,
    tr REAL NOT NULL,
    a REAL NOT NULL,
    d_disp REAL NOT NULL
);

-- 2. Tabla de Resultados Numéricos (EDOs)
CREATE TABLE IF NOT EXISTS Resultado_Numerico (
    id_resultado INTEGER PRIMARY KEY AUTOINCREMENT,
    id_simulacion INTEGER NOT NULL,
    d_analitica REAL NOT NULL,
    d_numerica REAL NOT NULL,
    error_rel REAL NOT NULL,
    FOREIGN KEY (id_simulacion) REFERENCES Simulacion(id_simulacion) ON DELETE CASCADE
);

-- 3. Tabla de Predicciones de IA
CREATE TABLE IF NOT EXISTS Prediccion_IA (
    id_prediccion INTEGER PRIMARY KEY AUTOINCREMENT,
    id_simulacion INTEGER NOT NULL,
    modelo_usado VARCHAR(50) NOT NULL, -- Random Forest
    distancia_predicha REAL NOT NULL,
    es_seguro BOOLEAN NOT NULL,
    FOREIGN KEY (id_simulacion) REFERENCES Simulacion(id_simulacion) ON DELETE CASCADE
);

-- Índices para optimizar búsquedas
CREATE INDEX idx_escenario ON Simulacion(escenario);
CREATE INDEX idx_fecha ON Simulacion(fecha);
