"""
Algoritmo Clarke-Wright + la mejora 2OPT

Este algoritmo aplica el operador de mejora 2-opt sobre cada ruta generada por Clarke-Wright. El 2-opt invierte segmentos de una ruta cuando ello reduce la distancia total, eliminando cruces de arcos. Se itera hasta que ningún intercambio produce mejora (convergencia al óptimo local).

Complejidad: Clarke-Wright O(n² log n) + 2-opt O(n²) por ruta por iteración
n= número de clientes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from math import radians, sin, cos, sqrt, atan2

# ── Parámetros ────────────────────────────────────────────────────────────────
RUTA_CSV          = "amazon_delivery.csv"
CIUDAD_LAT        = 12.97
CIUDAD_LON        = 77.59
RADIO_KM          = 20
N_PEDIDOS         = 80
CAPACIDAD         = 15
SEMILLA           = 42
MAX_ITER_2OPT     = 500
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
    nodos    = [(depot_lat, depot_lon)] + list(zip(df['Drop_Latitude'], df['Drop_Longitude']))
    demandas = [0] + df['Demand'].tolist()
    return df, nodos, demandas


def clarke_wright(nodos, demandas, D):
    n_clientes = len(nodos) - 1
    ahorros = []
    for i in range(1, n_clientes + 1):
        for j in range(i+1, n_clientes + 1):
            ahorros.append((D[0][i] + D[0][j] - D[i][j], i, j))
    ahorros.sort(reverse=True)

    rutas   = {i: [0, i, 0] for i in range(1, n_clientes + 1)}
    en_ruta = {i: i for i in range(1, n_clientes + 1)}

    for s, i, j in ahorros:
        ri, rj = en_ruta[i], en_ruta[j]
        if ri == rj:
            continue
        ruta_i, ruta_j = rutas[ri], rutas[rj]
        ultima_i  = ruta_i[-2] == i
        primera_j = ruta_j[1]  == j
        if not (ultima_i and primera_j):
            if ruta_i[1] == i and ruta_j[-2] == j:
                ruta_i = ruta_i[::-1]
                ruta_j = ruta_j[::-1]
            else:
                continue
        dem = (sum(demandas[n] for n in ruta_i if n != 0) +
               sum(demandas[n] for n in ruta_j if n != 0))
        if dem > CAPACIDAD:
            continue
        nueva = ruta_i[:-1] + ruta_j[1:]
        rutas[ri] = nueva
        del rutas[rj]
        for n in nueva:
            if n != 0:
                en_ruta[n] = ri

    return list(rutas.values())


def dos_opt(ruta, D):
    """Mejora una ruta mediante intercambios 2-opt hasta convergencia."""
    mejor    = ruta[:]
    mejorado = True
    iters    = 0
    while mejorado and iters < MAX_ITER_2OPT:
        mejorado = False
        iters   += 1
        for i in range(1, len(mejor) - 2):
            for j in range(i+1, len(mejor) - 1):
                nueva = mejor[:i] + mejor[i:j+1][::-1] + mejor[j+1:]
                if distancia_ruta(nueva, D) < distancia_ruta(mejor, D) - 1e-9:
                    mejor    = nueva
                    mejorado = True
    return mejor, iters


def visualizar(nodos, rutas_antes, rutas_despues, D, dist_antes, dist_despues, t_cw, t_total):
    colores = ['#E63946','#2A9D8F','#E9C46A','#F4A261','#264653','#6A4C93','#1982C4']
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, rutas, titulo in zip(
        axes,
        [rutas_antes, rutas_despues],
        [f'Clarke-Wright\n{dist_antes:.2f} km', f'Clarke-Wright + 2-opt\n{dist_despues:.2f} km']
    ):
        depot = nodos[0]
        ax.scatter(depot[1], depot[0], s=200, marker='*', color='black', zorder=5)
        for k, ruta in enumerate(rutas):
            color = colores[k % len(colores)]
            lats  = [nodos[n][0] for n in ruta]
            lons  = [nodos[n][1] for n in ruta]
            ax.plot(lons, lats, '-o', color=color, linewidth=1.8, markersize=5, alpha=0.85)
        ax.set_title(titulo, fontsize=11, fontweight='bold')
        ax.set_xlabel('Longitud'); ax.set_ylabel('Latitud')
        ax.grid(True, alpha=0.3)

    mejora = (dist_antes - dist_despues) / dist_antes * 100
    fig.suptitle(f'Mejora 2-opt: {mejora:.1f}%  |  '
                 f't(CW): {t_cw:.4f}s  |  t(total): {t_total:.4f}s',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('resultados_algoritmos/resultado_clarke_wright_2opt.png', dpi=150)
    plt.show()


def main():
    print("=" * 55)
    print("  ALGORITMO CLARKE-WRIGHT + 2-OPT")
    print("=" * 55)

    print("\n Cargando datos:")
    df, nodos, demandas = cargar_datos()
    print(f"  Pedidos: {len(df)}  |  Nodos totales: {len(nodos)}")

    print("\n Matriz de distancias:")
    D = matriz_distancias(nodos)

    print("\n Fase Clarke-Wright:")
    t0       = time.time()
    rutas_cw = clarke_wright(nodos, demandas, D)
    t_cw     = time.time() - t0
    dist_cw  = sum(distancia_ruta(r, D) for r in rutas_cw)
    print(f"  Distancia CW: {dist_cw:.2f} km  ({t_cw:.4f} s)")

    print("\nMejora 2-opt:")
    t0_2opt   = time.time()
    rutas_2opt = []
    total_iters = 0
    for r in rutas_cw:
        r_mejor, iters = dos_opt(r, D)
        rutas_2opt.append(r_mejor)
        total_iters += iters
    t_2opt = time.time() - t0_2opt
    t_total = t_cw + t_2opt

    dist_2opt = sum(distancia_ruta(r, D) for r in rutas_2opt)
    mejora    = (dist_cw - dist_2opt) / dist_cw * 100

    print(f"\n  Distancia CW         : {dist_cw:.2f} km")
    print(f"  Distancia CW + 2-opt : {dist_2opt:.2f} km")
    print(f"  Mejora 2-opt         : {mejora:.2f}%")
    print(f"  Iteraciones 2-opt    : {total_iters}")
    print(f"  Tiempo CW            : {t_cw:.4f} s")
    print(f"  Tiempo 2-opt         : {t_2opt:.4f} s")
    print(f"  Tiempo total         : {t_total:.4f} s")
    print(f"  Número de rutas          : {len(rutas_2opt)}")
    print("\n  Detalle por ruta (tras 2-opt):")
    for k, r in enumerate(rutas_2opt):
        d     = distancia_ruta(r, D)
        carga = sum(demandas[n] for n in r if n != 0)
        print(f"    Ruta {k+1}: {len(r)-2} clientes  |  {d:.2f} km  |  carga {carga}/{CAPACIDAD}")
    visualizar(nodos, rutas_cw, rutas_2opt, D, dist_cw, dist_2opt, t_cw, t_total)


if __name__ == '__main__':
    main()