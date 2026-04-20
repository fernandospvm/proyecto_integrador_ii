"""

Algoritmo de búsqueda de vecinidad variable (VNS)

Es un algoritmo metaheurístico que alterna entre una fase de perturbación y una fase de búsqueda local para escapar de óptimos locales. Combina:
    Perturbación: intercambio aleatorio de clientes entre rutas
    Búsqueda local: 2-opt intra-ruta + reubicación inter-ruta
    Criterio de aceptación: acepta cualquier mejora estricta
    Reinicio desde la mejor solución tras 10 iteraciones sin mejora

Parte de la solución voraz como punto de partida.

Complejidad: O(max_iter · n²)  en la práctica muy eficiente

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
N_VEHICULOS       = 6
SEMILLA           = 42
MAX_ITER_VNS      = 50
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


def evaluar(rutas, D):
    return sum(distancia_ruta(r, D) for r in rutas)


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


# ── Solución inicial: Voraz ───────────────────────────────────────────────────

def voraz(nodos, demandas, D):
    n_clientes = len(nodos) - 1
    pendientes = list(range(1, n_clientes + 1))
    rutas      = [[0] for _ in range(N_VEHICULOS)]
    cargas     = [0] * N_VEHICULOS

    while pendientes:
        mejor_dist, mejor_ri, mejor_c = float('inf'), -1, -1
        for ri, ruta in enumerate(rutas):
            cap = CAPACIDAD - cargas[ri]
            for c in pendientes:
                if demandas[c] <= cap and D[ruta[-1]][c] < mejor_dist:
                    mejor_dist, mejor_ri, mejor_c = D[ruta[-1]][c], ri, c
        if mejor_c == -1:
            break
        rutas[mejor_ri].append(mejor_c)
        cargas[mejor_ri] += demandas[mejor_c]
        pendientes.remove(mejor_c)

    return [r + [0] for r in rutas if len(r) > 1]


# ── Búsqueda local: 2-opt ─────────────────────────────────────────────────────

def dos_opt(ruta, D):
    mejor = ruta[:]
    mejorado = True
    iters = 0
    while mejorado and iters < MAX_ITER_2OPT:
        mejorado = False
        iters += 1
        for i in range(1, len(mejor) - 2):
            for j in range(i+1, len(mejor) - 1):
                nueva = mejor[:i] + mejor[i:j+1][::-1] + mejor[j+1:]
                if distancia_ruta(nueva, D) < distancia_ruta(mejor, D) - 1e-9:
                    mejor = nueva
                    mejorado = True
    return mejor


def aplicar_2opt(rutas, D):
    return [dos_opt(r, D) for r in rutas]


# ── Búsqueda local: reubicación inter-ruta ────────────────────────────────────

def reubicacion(rutas, demandas, D):
    mejor_dist  = evaluar(rutas, D)
    mejor_rutas = [r[:] for r in rutas]
    mejorado    = True

    while mejorado:
        mejorado = False
        for ri in range(len(mejor_rutas)):
            if len(mejor_rutas[ri]) <= 2:
                continue
            for pos_c in range(1, len(mejor_rutas[ri]) - 1):
                cliente = mejor_rutas[ri][pos_c]
                for rj in range(len(mejor_rutas)):
                    if ri == rj:
                        continue
                    dem_rj = sum(demandas[n] for n in mejor_rutas[rj] if n != 0)
                    if dem_rj + demandas[cliente] > CAPACIDAD:
                        continue
                    for pos_ins in range(1, len(mejor_rutas[rj])):
                        nuevas = [r[:] for r in mejor_rutas]
                        nuevas[ri] = nuevas[ri][:pos_c] + nuevas[ri][pos_c+1:]
                        nuevas[rj] = nuevas[rj][:pos_ins] + [cliente] + nuevas[rj][pos_ins:]
                        nuevas = [r for r in nuevas if len(r) > 2]
                        d = evaluar(nuevas, D)
                        if d < mejor_dist - 1e-9:
                            mejor_dist  = d
                            mejor_rutas = nuevas
                            mejorado    = True
                            break
                    if mejorado: break
                if mejorado: break
            if mejorado: break
    return mejor_rutas


# ── VNS principal ─────────────────────────────────────────────────────────────

def vns(rutas_iniciales, demandas, D):
    np.random.seed(SEMILLA)
    historial = []

    # Búsqueda local desde la solución inicial
    rutas_actuales = aplicar_2opt(rutas_iniciales, D)
    rutas_actuales = reubicacion(rutas_actuales, demandas, D)
    mejor_rutas    = [r[:] for r in rutas_actuales]
    mejor_dist     = evaluar(mejor_rutas, D)
    historial.append(mejor_dist)

    sin_mejora = 0

    for it in range(MAX_ITER_VNS):
        nuevas = [r[:] for r in rutas_actuales]

        # Perturbación: intercambio de dos clientes entre rutas distintas
        rutas_validas = [i for i, r in enumerate(nuevas) if len(r) > 3]
        if len(rutas_validas) >= 2:
            ri, rj = np.random.choice(rutas_validas, 2, replace=False)
            pi = np.random.randint(1, len(nuevas[ri]) - 1)
            pj = np.random.randint(1, len(nuevas[rj]) - 1)
            ci, cj = nuevas[ri][pi], nuevas[rj][pj]
            dem_ri = sum(demandas[n] for n in nuevas[ri] if n != 0)
            dem_rj = sum(demandas[n] for n in nuevas[rj] if n != 0)
            if (dem_ri - demandas[ci] + demandas[cj] <= CAPACIDAD and
                    dem_rj - demandas[cj] + demandas[ci] <= CAPACIDAD):
                nuevas[ri][pi], nuevas[rj][pj] = cj, ci

        # Búsqueda local
        nuevas = aplicar_2opt(nuevas, D)
        nuevas = reubicacion(nuevas, demandas, D)
        d_nueva = evaluar(nuevas, D)

        historial.append(d_nueva)

        if d_nueva < mejor_dist - 1e-9:
            mejor_dist     = d_nueva
            mejor_rutas    = [r[:] for r in nuevas]
            rutas_actuales = nuevas
            sin_mejora     = 0
        else:
            sin_mejora += 1
            if sin_mejora > 10:
                rutas_actuales = [r[:] for r in mejor_rutas]
                sin_mejora = 0

    return mejor_rutas, historial


# ── Visualización ─────────────────────────────────────────────────────────────

def visualizar(nodos, rutas_iniciales, rutas_vns, D, demandas, historial, dist_ini, dist_vns, t_ini, t_vns):
    colores = ['#E63946','#2A9D8F','#E9C46A','#F4A261','#264653','#6A4C93','#1982C4']
    fig = plt.figure(figsize=(16, 7))
    gs  = fig.add_gridspec(1, 3)

    # Rutas iniciales (voraz)
    ax1 = fig.add_subplot(gs[0])
    depot = nodos[0]
    ax1.scatter(depot[1], depot[0], s=200, marker='*', color='black', zorder=5)
    for k, ruta in enumerate(rutas_iniciales):
        c = colores[k % len(colores)]
        ax1.plot([nodos[n][1] for n in ruta], [nodos[n][0] for n in ruta],
                 '-o', color=c, linewidth=1.8, markersize=5, alpha=0.85)
    ax1.set_title(f'Solución inicial (Voraz)\n{dist_ini:.2f} km', fontsize=10, fontweight='bold')
    ax1.set_xlabel('Longitud'); ax1.set_ylabel('Latitud')
    ax1.grid(True, alpha=0.3)

    # Rutas VNS
    ax2 = fig.add_subplot(gs[1])
    ax2.scatter(depot[1], depot[0], s=200, marker='*', color='black', zorder=5)
    for k, ruta in enumerate(rutas_vns):
        c = colores[k % len(colores)]
        ax2.plot([nodos[n][1] for n in ruta], [nodos[n][0] for n in ruta],
                 '-o', color=c, linewidth=1.8, markersize=5, alpha=0.85, label=f'Ruta {k+1}')
    mejora = (dist_ini - dist_vns) / dist_ini * 100
    ax2.set_title(f'Solución VNS\n{dist_vns:.2f} km  (mejora {mejora:.1f}%)', fontsize=10, fontweight='bold')
    ax2.set_xlabel('Longitud')
    ax2.legend(fontsize=7, loc='best')
    ax2.grid(True, alpha=0.3)

    # Historial de convergencia
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(historial, color='#E63946', linewidth=1.5)
    ax3.axhline(min(historial), color='#2A9D8F', linestyle='--', linewidth=1,
                label=f'Mejor: {min(historial):.2f} km')
    ax3.set_title('Convergencia del VNS', fontsize=10, fontweight='bold')
    ax3.set_xlabel('Iteración'); ax3.set_ylabel('Distancia total (km)')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    fig.suptitle(f'VNS  |  t(voraz): {t_ini:.4f}s  |  t(VNS): {t_vns:.4f}s  |  '
                 f't(total): {t_ini+t_vns:.4f}s',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig('resultados_algoritmos/resultado_vns.png', dpi=150)
    plt.show()


def main():
    print("=" * 55)
    print("  ALGORITMO VNS (VARIABLE NEIGHBORHOOD SEARCH)")
    print("=" * 55)

    print("\n Datos:")
    df, nodos, demandas = cargar_datos()
    print(f"  Pedidos: {len(df)}  |  Nodos totales: {len(nodos)}")

    print("\n Matriz de distancias:")
    D = matriz_distancias(nodos)

    print("\n Solución inicial (Algoritmo Voraz):")
    t0           = time.time()
    rutas_ini    = voraz(nodos, demandas, D)
    t_ini        = time.time() - t0
    dist_ini     = evaluar(rutas_ini, D)
    print(f"  Distancia inicial: {dist_ini:.2f} km  ({t_ini:.4f} s)")

    print(f"\n VNS ({MAX_ITER_VNS} iteraciones)")
    t0          = time.time()
    rutas_vns, historial = vns(rutas_ini, demandas, D)
    t_vns       = time.time() - t0
    dist_vns    = evaluar(rutas_vns, D)
    mejora      = (dist_ini - dist_vns) / dist_ini * 100

    print(f"\n  Distancia inicial  : {dist_ini:.2f} km")
    print(f"  Distancia VNS      : {dist_vns:.2f} km")
    print(f"  Mejora             : {mejora:.2f}%")
    print(f"  Número de rutas        : {len(rutas_vns)}")
    print(f"  Tiempo voraz       : {t_ini:.4f} s")
    print(f"  Tiempo VNS         : {t_vns:.4f} s")
    print(f"  Tiempo total       : {t_ini+t_vns:.4f} s")
    print("\n  Detalle por ruta (solución VNS):")
    for k, r in enumerate(rutas_vns):
        d     = distancia_ruta(r, D)
        carga = sum(demandas[n] for n in r if n != 0)
        print(f"    Ruta {k+1}: {len(r)-2} clientes  |  {d:.2f} km  |  carga {carga}/{CAPACIDAD}")
    visualizar(nodos, rutas_ini, rutas_vns, D, demandas,
               historial, dist_ini, dist_vns, t_ini, t_vns)


if __name__ == '__main__':
    main()