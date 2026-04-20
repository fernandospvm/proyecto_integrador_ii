"""
Algoritmo Voraz (greedy)
Complejidad: O(n² · V)  donde n = clientes, V = vehículos
Este algoritmo construye rutas de forma iterativa asignando en cada paso el cliente no asignado más cercano al extremo actual de alguna ruta disponible, siempre que no se supere la capacidad máxima del vehículo.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from math import radians, sin, cos, sqrt, atan2

# ── Parámetros ────────────────────────────────────────────────────────────────
RUTA_CSV          = "amazon_delivery.csv"
CIUDAD_LAT        = 12.97       # Bangalore
CIUDAD_LON        = 77.59
RADIO_KM          = 20
N_PEDIDOS         = 80
CAPACIDAD         = 15
N_VEHICULOS       = 6
SEMILLA           = 42
# ─────────────────────────────────────────────────────────────────────────────


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def matriz_distancias(nodos):
    n = len(nodos)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = haversine(nodos[i][0], nodos[i][1], nodos[j][0], nodos[j][1])
            D[i][j] = D[j][i] = d
    return D


def distancia_ruta(ruta, D):
    return sum(D[ruta[i]][ruta[i+1]] for i in range(len(ruta)-1))


def cargar_datos():
    df = pd.read_csv(RUTA_CSV)
    df['Traffic'] = df['Traffic'].str.strip()
    df['Vehicle'] = df['Vehicle'].str.strip()
    df['Area']    = df['Area'].str.strip()
    df.dropna(subset=['Store_Latitude','Store_Longitude',
                      'Drop_Latitude','Drop_Longitude'], inplace=True)

    mask = (
        (abs(df['Store_Latitude']  - CIUDAD_LAT) < RADIO_KM / 111) &
        (abs(df['Store_Longitude'] - CIUDAD_LON) < RADIO_KM / 111)
    )
    df = df[mask].reset_index(drop=True)

    demanda_map = {'Electronics':3,'Clothing':1,'Sports':2,'Cosmetics':1,
                   'Toys':2,'Snacks':2,'Beverages':3,'Books':1}
    df['Demand'] = df['Category'].map(demanda_map).fillna(1).astype(int)
    df = df.sample(n=min(N_PEDIDOS, len(df)), random_state=SEMILLA).reset_index(drop=True)

    depot_lat = df['Store_Latitude'].mean()
    depot_lon = df['Store_Longitude'].mean()
    nodos   = [(depot_lat, depot_lon)] + list(zip(df['Drop_Latitude'], df['Drop_Longitude']))
    demandas = [0] + df['Demand'].tolist()
    return df, nodos, demandas


def algoritmo_voraz(nodos, demandas, D):
    n_clientes = len(nodos) - 1
    pendientes = list(range(1, n_clientes + 1))
    rutas      = [[0] for _ in range(N_VEHICULOS)]
    cargas     = [0] * N_VEHICULOS

    while pendientes:
        mejor_dist     = float('inf')
        mejor_ruta_idx = -1
        mejor_cliente  = -1

        for ri, ruta in enumerate(rutas):
            cap_restante = CAPACIDAD - cargas[ri]
            ultimo = ruta[-1]
            for c in pendientes:
                if demandas[c] <= cap_restante and D[ultimo][c] < mejor_dist:
                    mejor_dist     = D[ultimo][c]
                    mejor_ruta_idx = ri
                    mejor_cliente  = c

        if mejor_cliente == -1:
            break

        rutas[mejor_ruta_idx].append(mejor_cliente)
        cargas[mejor_ruta_idx] += demandas[mejor_cliente]
        pendientes.remove(mejor_cliente)

    return [r + [0] for r in rutas if len(r) > 1]


def visualizar(nodos, rutas, dist_total, t_computo):
    colores = ['#E63946','#2A9D8F','#E9C46A','#F4A261','#264653','#6A4C93']
    fig, ax = plt.subplots(figsize=(9, 7))

    depot = nodos[0]
    ax.scatter(depot[1], depot[0], s=200, marker='*', color='black', zorder=5, label='Depósito')

    for k, ruta in enumerate(rutas):
        color = colores[k % len(colores)]
        lats  = [nodos[n][0] for n in ruta]
        lons  = [nodos[n][1] for n in ruta]
        ax.plot(lons, lats, '-o', color=color, linewidth=1.8,
                markersize=5, alpha=0.85, label=f'Ruta {k+1}')

    ax.set_title(f'Algoritmo Voraz (Greedy)\n'
                 f'Distancia total: {dist_total:.2f} km  |  '
                 f'Tiempo de cómputo: {t_computo:.4f} s  |  '
                 f'Rutas: {len(rutas)}',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Longitud'); ax.set_ylabel('Latitud')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('resultados_algoritmos/resultado_algoritmo_voraz.png', dpi=150)
    plt.show()


def main():
    print("=" * 55)
    print("  ALGORITMO VORAZ (GREEDY)")
    print("=" * 55)

    print("\n Datos")
    df, nodos, demandas = cargar_datos()
    print(f"  Pedidos: {len(df)}  |  Nodos totales: {len(nodos)}")

    print("\n Matriz de distancias")
    D = matriz_distancias(nodos)

    print("\n Ejecución de algoritmo")
    t0    = time.time()
    rutas = algoritmo_voraz(nodos, demandas, D)
    t_cpu = time.time() - t0

    dist_total = sum(distancia_ruta(r, D) for r in rutas)

    print(f"\n  Distancia total : {dist_total:.2f} km")
    print(f"  Número de rutas de rutas     : {len(rutas)}")
    print(f"  Tiempo cómputo  : {t_cpu:.4f} s")
    print("\n  Detalle por ruta:")
    for k, r in enumerate(rutas):
        d = distancia_ruta(r, D)
        carga = sum(demandas[n] for n in r if n != 0)
        print(f"    Ruta {k+1}: {len(r)-2} clientes  |  {d:.2f} km  |  carga {carga}/{CAPACIDAD}")

    print("\n Gráfica de la ruta")
    visualizar(nodos, rutas, dist_total, t_cpu)


if __name__ == '__main__':
    main()