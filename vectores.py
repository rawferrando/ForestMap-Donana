"""
vectores.py — Vectorización de rásters de etiquetas y exportación, sin dependencias.

* `poligonizar`  convierte un ráster de etiquetas en polígonos exactos (bordes de
  píxel, 4-conectividad) con huecos, igual que `gdal_polygonize`, solo con numpy.
* `escribir_geojson`  guarda cualquier lista de entidades en GeoJSON (con CRS para QGIS).
* `a_geodataframe`  (opcional) convierte las entidades a GeoPandas para GPKG/SHP.

Una "entidad" es un dict: {"tipo": "Polygon"|"Point"|"LineString"|"MultiPolygon",
"coords": ..., "props": {...}}.  Los polígonos son ([exterior, hueco1, ...]).
"""
import json

import numpy as np


# ------------------------------------------------------------------ geometría
def area_anillo(a):
    """Área con signo (shoelace) de un anillo (n,2)."""
    x, y = a[:, 0], a[:, 1]
    return 0.5 * float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))


def _punto_en_anillo(px, py, anillo):
    x, y = anillo[:, 0], anillo[:, 1]
    dentro = False
    for i in range(len(x) - 1):
        if (y[i] > py) != (y[i + 1] > py):
            xi = x[i] + (py - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i])
            if px < xi:
                dentro = not dentro
    return dentro


def _quitar_colineales(pts):
    """Elimina vértices intermedios sobre segmentos rectos (anillo sin repetir el primero)."""
    p = np.asarray(pts)
    prev = np.roll(p, 1, axis=0)
    sig = np.roll(p, -1, axis=0)
    cruz = (p[:, 0] - prev[:, 0]) * (sig[:, 1] - p[:, 1]) - (p[:, 1] - prev[:, 1]) * (sig[:, 0] - p[:, 0])
    return p[cruz != 0]


# --------------------------------------------------------------- poligonizar
def poligonizar(etiquetas, x0, y1, res, simplificar=0.0):
    """Polígonos exactos por etiqueta.

    Parámetros
    ----------
    etiquetas : (ny, nx) enteros; 0 = fondo.
    x0, y1    : esquina superior izquierda de la malla (coordenadas de mapa).
    res       : tamaño de píxel.
    simplificar : tolerancia Douglas-Peucker en metros (0 = sin simplificar).

    Devuelve  {etiqueta: [(exterior(n,2), [hueco(n,2), ...]), ...]} con anillos
    cerrados, exterior antihorario y huecos horarios (coordenadas de mapa).
    """
    L = np.pad(np.asarray(etiquetas).astype(np.int64), 1)
    ny, nx = L.shape[0] - 2, L.shape[1] - 2
    W = nx + 1

    ini, fin, cod = [], [], []

    def agrega(mask, ci, cj, ei, ej, codigo):
        j, i = np.nonzero(mask)
        if len(j) == 0:
            return
        ini.append((j + cj) * W + (i + ci))
        fin.append((j + ej) * W + (i + ei))
        cod.append(codigo[mask])

    # aristas horizontales (línea j entre las filas j-1 y j; columna c)
    abajo, arriba = L[1:, 1:-1], L[:-1, 1:-1]
    dif = abajo != arriba
    agrega(dif & (abajo != 0), 0, 0, 1, 0, abajo)            # borde superior del píxel de abajo
    agrega(dif & (arriba != 0), 1, 0, 0, 0, arriba)          # borde inferior del píxel de arriba
    # aristas verticales (línea i entre las columnas i-1 e i; fila r)
    der, izq = L[1:-1, 1:], L[1:-1, :-1]
    dif = der != izq
    agrega(dif & (der != 0), 0, 1, 0, 0, der)                # borde izquierdo del píxel de la derecha
    agrega(dif & (izq != 0), 0, 0, 0, 1, izq)                # borde derecho del píxel de la izquierda
    if not ini:
        return {}

    ini = np.concatenate(ini)
    fin = np.concatenate(fin)
    cod = np.concatenate(cod)
    orden = np.argsort(cod, kind="stable")
    ini, fin, cod = ini[orden], fin[orden], cod[orden]
    cortes = np.r_[0, np.flatnonzero(np.diff(cod)) + 1, len(cod)]

    resultado = {}
    for a, b in zip(cortes[:-1], cortes[1:]):
        codigo = int(cod[a])
        salidas = {}
        for s, e in zip(ini[a:b].tolist(), fin[a:b].tolist()):
            salidas.setdefault(s, []).append(e)
        anillos = []
        while salidas:
            unicos = [k for k, v in salidas.items() if len(v) == 1]
            cur = unicos[0] if unicos else next(iter(salidas))
            inicio = cur
            ring = [cur]
            dir_prev = None
            while True:
                cands = salidas[cur]
                if len(cands) == 1:
                    nxt = cands[0]
                else:                                           # vértice de silla: giro a la derecha
                    ci, cj = cur % W, cur // W
                    pref = None
                    for pasos in ("derecha", "recto", "izquierda"):
                        for c in cands:
                            d = (c % W - ci, c // W - cj)
                            if dir_prev is None:
                                alvo = d
                            elif pasos == "derecha":
                                alvo = (-dir_prev[1], dir_prev[0])
                            elif pasos == "recto":
                                alvo = dir_prev
                            else:
                                alvo = (dir_prev[1], -dir_prev[0])
                            if d == alvo:
                                pref = c
                                break
                        if pref is not None:
                            break
                    nxt = pref if pref is not None else cands[0]
                salidas[cur].remove(nxt)
                if not salidas[cur]:
                    del salidas[cur]
                dir_prev = (nxt % W - cur % W, nxt // W - cur // W)
                cur = nxt
                if cur == inicio:
                    break
                ring.append(cur)
            anillos.append(ring)

        # a coordenadas de retícula y limpieza
        rets = []
        for ring in anillos:
            arr = np.column_stack([np.array(ring) % W, np.array(ring) // W]).astype(float)
            arr = _quitar_colineales(arr)
            if len(arr) >= 3:
                rets.append(arr)
        exteriores = [r for r in rets if area_anillo(np.vstack([r, r[:1]])) > 0]
        huecos = [r for r in rets if area_anillo(np.vstack([r, r[:1]])) < 0]

        def a_mapa(r):
            r = np.vstack([r, r[:1]])[::-1]                       # cerrar y cambiar sentido
            xy = np.column_stack([x0 + r[:, 0] * res, y1 - r[:, 1] * res])
            if simplificar and len(xy) > 5:
                from skimage.measure import approximate_polygon
                s = approximate_polygon(xy, tolerance=simplificar)
                if len(s) >= 4:
                    xy = s
            return xy

        ext_m = [a_mapa(r) for r in exteriores]
        hue_m = [a_mapa(r) for r in huecos]
        asignados = [[] for _ in ext_m]
        for h, hr in zip(hue_m, huecos):
            # punto dentro de la región, junto a la primera arista del hueco
            p0, p1 = hr[0], hr[1 % len(hr)]
            derecha = np.array([-(p1[1] - p0[1]), p1[0] - p0[0]], float)
            n = np.hypot(*derecha) or 1.0
            q = (p0 + p1) / 2 + 0.25 * derecha / n
            qx, qy = x0 + q[0] * res, y1 - q[1] * res
            mejor, a_mejor = None, np.inf
            for k, e in enumerate(ext_m):
                if _punto_en_anillo(qx, qy, e):
                    ar = abs(area_anillo(e))
                    if ar < a_mejor:
                        mejor, a_mejor = k, ar
            if mejor is not None:
                asignados[mejor].append(h)
        resultado[codigo] = [(e, asignados[k]) for k, e in enumerate(ext_m)]
    return resultado


def area_poligono(exterior, huecos=()):
    return abs(area_anillo(exterior)) - sum(abs(area_anillo(h)) for h in huecos)


# ------------------------------------------------------------------- GeoJSON
def _py(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        v = float(v)
    if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))):
        return None
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, np.ndarray):
        return [_py(x) for x in v.tolist()]
    return v


def _geom(e):
    t, c = e["tipo"], e["coords"]
    if t == "Polygon":
        return {"type": "Polygon", "coordinates": [np.asarray(a).round(4).tolist() for a in c]}
    if t == "MultiPolygon":
        return {"type": "MultiPolygon",
                "coordinates": [[np.asarray(a).round(4).tolist() for a in p] for p in c]}
    if t == "Point":
        return {"type": "Point", "coordinates": [round(float(c[0]), 4), round(float(c[1]), 4)]}
    if t == "LineString":
        return {"type": "LineString", "coordinates": np.asarray(c).round(4).tolist()}
    raise ValueError(t)


def a_geojson(entidades, epsg=None):
    fc = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": _geom(e),
         "properties": {k: _py(v) for k, v in e.get("props", {}).items()}}
        for e in entidades]}
    if epsg:
        fc["crs"] = {"type": "name", "properties": {"name": f"urn:ogc:def:crs:EPSG::{int(epsg)}"}}
    return fc


def escribir_geojson(ruta, entidades, epsg=None):
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(a_geojson(entidades, epsg), f, ensure_ascii=False)
    return ruta


def a_geodataframe(entidades, epsg=None):
    """GeoDataFrame (requiere geopandas + shapely) para GPKG / Shapefile."""
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import LineString, MultiPolygon, Point, Polygon

    def g(e):
        t, c = e["tipo"], e["coords"]
        if t == "Polygon":
            return Polygon(c[0], c[1:])
        if t == "MultiPolygon":
            return MultiPolygon([Polygon(p[0], p[1:]) for p in c])
        if t == "Point":
            return Point(c[0], c[1])
        return LineString(c)

    props = pd.DataFrame([{k: _py(v) for k, v in e.get("props", {}).items()} for e in entidades])
    gdf = gpd.GeoDataFrame(props, geometry=[g(e) for e in entidades])
    if epsg:
        gdf = gdf.set_crs(epsg=int(epsg))
    return gdf


def guardar_vector(ruta, entidades, epsg=None, capa=None):
    """Guarda en GeoJSON (sin dependencias) o GPKG/SHP (con geopandas) según la extensión."""
    ext = ruta.lower().rsplit(".", 1)[-1]
    if ext == "geojson":
        return escribir_geojson(ruta, entidades, epsg)
    gdf = a_geodataframe(entidades, epsg)
    if len(gdf) == 0:
        return None
    if ext == "gpkg":
        gdf.to_file(ruta, layer=capa or "capa", driver="GPKG")
    elif ext == "shp":
        gdf.columns = [c[:10] if c != "geometry" else c for c in gdf.columns]   # límite DBF
        gdf.to_file(ruta, driver="ESRI Shapefile")
    else:
        raise ValueError(ext)
    return ruta
