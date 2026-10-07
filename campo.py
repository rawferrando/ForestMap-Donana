"""
campo.py — Inventario de campo (verdad terreno) de la parcela: lectura, conversión a UTM y comparación con el recuento LiDAR.

El inventario de 2011 (Ojillo_2011.shp / datos_ojillo_11.xlsx) trae WGS84 (LAT/LON) y una ficha por sabina:
IDENT, altura (m), daño, 'Cambio11' (Muerto/Adulto/Juvenil)…
"""
import numpy as np
import pandas as pd


def latlon_a_utm(lat, lon, huso=29):
    """WGS84/ETRS89 (grados) → UTM (m), serie de Krüger. ≤ 1 mm de error frente a PROJ."""
    a, f = 6378137.0, 1 / 298.257222101
    k0, e0 = 0.9996, 500000.0
    lat, lon = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    lon0 = np.radians(6 * huso - 183)
    n = f / (2 - f)
    A = a / (1 + n) * (1 + n ** 2 / 4 + n ** 4 / 64)
    al = [n / 2 - 2 * n ** 2 / 3 + 5 * n ** 3 / 16, 13 * n ** 2 / 48 - 3 * n ** 3 / 5, 61 * n ** 3 / 240]
    e = np.sqrt(f * (2 - f))
    t = np.sinh(np.arctanh(np.sin(lat)) - e * np.arctanh(e * np.sin(lat)))
    xi = np.arctan2(t, np.cos(lon - lon0))
    eta = np.arctanh(np.sin(lon - lon0) / np.sqrt(1 + t ** 2))
    x = eta + sum(al[j] * np.cos(2 * (j + 1) * xi) * np.sinh(2 * (j + 1) * eta) for j in range(3))
    y = xi + sum(al[j] * np.sin(2 * (j + 1) * xi) * np.cosh(2 * (j + 1) * eta) for j in range(3))
    return e0 + k0 * A * x, k0 * A * y


def leer_inventario(xlsx, desplaza=(0.0, 0.0)):
    """DataFrame con IDENT, x, y (UTM 29N), altura_m, vivo, cambio. `xlsx` = ruta o archivo subido."""
    import openpyxl
    wb = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
    filas = list(wb.active.iter_rows(values_only=True))
    df = pd.DataFrame(filas[1:], columns=filas[0])
    x, y = latlon_a_utm(df['LAT'].astype(float), df['LON'].astype(float))
    out = pd.DataFrame(dict(ident=df['IDENT'], x=x + desplaza[0], y=y + desplaza[1],
                            altura_m=pd.to_numeric(df['Altura'], errors='coerce').fillna(0.0),
                            cambio=df['Cambio11'].fillna(''), dano_11=df['DAÑO_11'], anio_ficha=df['Año_fichaj']))
    out['vivo'] = (out['altura_m'] > 0) & (out['cambio'] != 'Muerto')
    return out


def resumen_campo(inv, roi, hmin=1.3, hpino=5.0):
    d = inv[roi.puntos(inv.x.values, inv.y.values)]
    v = d[d.vivo]
    return dict(n_fichas=len(d), n_muertos=int((~d.vivo).sum()), n_vivos=len(v), n_vivos_hmin=int((v.altura_m >= hmin).sum()),
                n_menores=int((v.altura_m < hmin).sum()), h_media=float(v.altura_m[v.altura_m >= hmin].mean()),
                n_mayores_5=int((v.altura_m >= hpino).sum()))


# ---------------------------------------------------------------- comparación con el recuento LiDAR
def estimar_desplazamiento(tabla, inv, radio=1.2, rango=12.0, paso=0.5):
    """Desplazamiento (dx, dy) del inventario de campo respecto al LiDAR: el que maximiza los ápices con una ficha a ≤ radio."""
    from scipy.spatial import cKDTree
    arbol = cKDTree(np.c_[tabla['x'], tabla['y']])
    F = np.c_[inv['x'], inv['y']]
    mejor = (-1, 0.0, 0.0)
    g = np.arange(-rango, rango + 1e-9, paso)
    for dx in g:
        for dy in g:
            d, _ = arbol.query(F + [dx, dy])
            n = int((d < radio).sum())
            if n > mejor[0]:
                mejor = (n, dx, dy)
    T = np.array(mejor[1:])
    for _ in range(8):                                            # afinado por mediana de residuos
        d, i = arbol.query(F + T)
        ok = d < 2.5
        if ok.sum() < 5:
            break
        T = T + np.median(np.c_[tabla['x'], tabla['y']][i[ok]] - (F[ok] + T), axis=0)
    return float(T[0]), float(T[1]), int(mejor[0])


def comparar_con_campo(tabla, labels, malla, res, inv, roi, desplaza, hmin=1.3, radio=2.0):
    """Compara los individuos LiDAR con el inventario de campo (vivos ≥ hmin dentro de la parcela).

    Devuelve dict con cifras y la tabla de campo con su estado:
      'emparejado' (ápice LiDAR a ≤ radio), 'bajo copa (tapado)' (cae en una copa ya emparejada con otro),
      'bajo copa sin ápice' (en copa sin ficha), 'sin copa LiDAR' (CHM < hmin: matojo/pequeño no visible).
    """
    from scipy.optimize import linear_sum_assignment
    c = inv[roi.puntos(inv['x'].values + desplaza[0], inv['y'].values + desplaza[1]) & inv['vivo'] & (inv['altura_m'] >= hmin)].copy()
    c['xl'], c['yl'] = c['x'] + desplaza[0], c['y'] + desplaza[1]
    n, m = len(tabla), len(c)
    dist = np.hypot(tabla['x'].values[:, None] - c['xl'].values[None, :], tabla['y'].values[:, None] - c['yl'].values[None, :]) \
        if n and m else np.zeros((n, m))
    cost = np.where(dist <= radio, dist, 1e6)
    emp = np.full(m, -1)
    if n and m:
        r, k = linear_sum_assignment(cost)
        for i, j in zip(r, k):
            if cost[i, j] < 1e6:
                emp[j] = i
    x0, y1, nx, ny = malla
    col = np.floor((c['xl'].values - x0) / res).astype(int).clip(0, labels.shape[1] - 1)
    fil = np.floor((y1 - c['yl'].values) / res).astype(int).clip(0, labels.shape[0] - 1)
    lab = labels[fil, col]
    estado = np.where(emp >= 0, "emparejado", np.where(lab > 0, "bajo copa (tapado)", "sin copa LiDAR"))
    c['estado'] = estado
    c['id_copa'] = np.where(lab > 0, lab, 0)
    c['id_lidar'] = np.where(emp >= 0, tabla['id'].values[np.clip(emp, 0, max(n - 1, 0))] if n else 0, 0)
    copas_con_ficha = np.unique(lab[lab > 0])
    out = dict(n_campo=m, n_lidar=n, n_emparejados=int((emp >= 0).sum()),
               n_tapados=int((estado == "bajo copa (tapado)").sum()), n_sin_copa=int((estado == "sin copa LiDAR").sum()),
               n_copas_con_ficha=int(len(copas_con_ficha)),
               n_copas_sin_ficha=int(n - len(np.intersect1d(copas_con_ficha, tabla['id'].values))),
               desplaza_x=float(desplaza[0]), desplaza_y=float(desplaza[1]), radio_m=radio,
               ratio=n / m if m else np.nan, h_campo_media=float(c['altura_m'].mean()) if m else np.nan)
    return out, c
