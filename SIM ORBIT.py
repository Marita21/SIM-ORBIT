# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 12:10:56 2025
@author: mbaigorria
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Constantes
G = 6.67430e-11  # m^3 / kg / s^2
M_tierra = 5.972e24  # kg
R_tierra = 6371e3  # m
mu = G * M_tierra  # parámetro gravitacional
omega_tierra = 7.2921159e-5  # rad/s

# Función para convertir elementos keplerianos a estado cartesiano
def kepler_a_estado(a, e, i, RAAN, arg_peri, nu):
    i = np.radians(i)
    RAAN = np.radians(RAAN)
    arg_peri = np.radians(arg_peri)
    nu = np.radians(nu)

    r = a * (1 - e**2) / (1 + e * np.cos(nu))
    x_pf = r * np.cos(nu)
    y_pf = r * np.sin(nu)
    z_pf = 0
    h = np.sqrt(mu * a * (1 - e**2))
    vx_pf = -mu / h * np.sin(nu)
    vy_pf = mu / h * (e + np.cos(nu))
    vz_pf = 0

    R1 = np.array([
        [np.cos(RAAN), -np.sin(RAAN), 0],
        [np.sin(RAAN), np.cos(RAAN), 0],
        [0, 0, 1]
    ])
    R2 = np.array([
        [1, 0, 0],
        [0, np.cos(i), -np.sin(i)],
        [0, np.sin(i), np.cos(i)]
    ])
    R3 = np.array([
        [np.cos(arg_peri), -np.sin(arg_peri), 0],
        [np.sin(arg_peri), np.cos(arg_peri), 0],
        [0, 0, 1]
    ])
    Q = R1 @ R2 @ R3
    r_vec = Q @ np.array([x_pf, y_pf, z_pf])
    v_vec = Q @ np.array([vx_pf, vy_pf, vz_pf])
    return r_vec, v_vec

# Función de derivadas
def derivadas(t, estado):
    x, y, z, vx, vy, vz = estado
    r = np.sqrt(x**2 + y**2 + z**2)
    ax = -mu * x / r**3
    ay = -mu * y / r**3
    az = -mu * z / r**3
    return [vx, vy, vz, ax, ay, az]

# ECI a latitud y longitud
def eci_a_latlon(x, y, z, t):
    lon = np.degrees(np.arctan2(y, x) - omega_tierra * t)
    lat = np.degrees(np.arcsin(z / np.sqrt(x**2 + y**2 + z**2)))
    lon = (lon + 180) % 360 - 180
    return lat, lon

# Parámetros orbitales
a = 6871e3
e = 0.0
i = 45.0
RAAN = 0.0
arg_peri = 0.0
nu = 0.0

# Estado inicial
r_vec, v_vec = kepler_a_estado(a, e, i, RAAN, arg_peri, nu)
estado_inicial = list(r_vec) + list(v_vec)

# Tiempo de simulación
tiempo_total = 5600  # segundos (una órbita)
step = 60  # segundos

# Integración
sol = solve_ivp(derivadas, [0, tiempo_total], estado_inicial, t_eval=np.arange(0, tiempo_total, step))

# Convertir a lat/lon y detectar saltos de ±180°
lats, lons = [], []
for i in range(len(sol.t)):
    lat, lon = eci_a_latlon(sol.y[0][i], sol.y[1][i], sol.y[2][i], sol.t[i])
    if lon > 180:
        lon -= 360
    elif lon < -180:
        lon += 360
    lats.append(lat)
    lons.append(lon)

# Insertar NaN para romper líneas con saltos
lons_plot, lats_plot = [lons[0]], [lats[0]]
for j in range(1, len(lons)):
    delta = abs(lons[j] - lons[j - 1])
    if delta > 180:
        lons_plot.append(np.nan)
        lats_plot.append(np.nan)
    lons_plot.append(lons[j])
    lats_plot.append(lats[j])

# Graficar mapa
fig = plt.figure(figsize=(12, 6))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_global()
ax.stock_img()
ax.coastlines()
ax.gridlines(draw_labels=True)

ax.plot(lons_plot, lats_plot, color='cyan', transform=ccrs.PlateCarree(), label='Ground Track')
ax.set_title('Proyección de la órbita sobre la Tierra (Ground Track)')
plt.legend()
plt.show()