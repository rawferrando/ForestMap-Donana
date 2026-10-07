"""
roi.py — Área de interés (parcela) como objeto independiente de la interfaz.

Una parcela puede ser toda la nube, un rectángulo, una parcela cuadrada de N ha o un
polígono real (p. ej. `parcela.gpkg`, que puede estar girado respecto a los ejes).
Todos los gráficos, estadísticas y exportaciones usan este mismo recorte.
"""
import numpy as np

import procesado as pr


def _pip(X, Y, anillo):
    """Punto-en-polígono (ray casting) para arrays de cualquier forma."""
    dentro = np.zeros(X.shape, bool)
    for (xa, ya), (xb, yb) in zip(anillo[:-1], anillo[1:]):
        if ya == yb:
            continue
        cruza = (ya > Y) != (yb > Y)
        xint = xa + (Y - ya) * (xb - xa) / (yb - ya)
        dentro ^= cruza & (X < xint)
    return dentro


class Roi:
    def __init__(self, bounds, anillos=None, nombre="Toda la nube"):
        self.bounds = tuple(float(v) for v in bounds)       # xmin, xmax, ymin, ymax
        self.anillos = [np.asarray(a, float) for a in anillos] if anillos else None
        self.nombre = nombre

    # ------------------------------------------------------------------ máscaras
    def puntos(self, x, y):
        b = self.bounds
        m = (x >= b[0]) & (x <= b[1]) & (y >= b[2]) & (y <= b[3])
        if self.anillos:
            d = np.zeros(len(x), bool)
            for a in self.anillos:
                d |= _pip(x, y, a)
            m &= d
        return m

    def mascara(self, malla, res, forma):
        """True = celda (por su centro) dentro de la parcela."""
        cx, cy = pr.centros(malla, res, forma)
        b = self.bounds
        m = ((cy >= b[2]) & (cy <= b[3]))[:, None] & ((cx >= b[0]) & (cx <= b[1]))[None, :]
        if self.anillos:
            X, Y = np.meshgrid(cx, cy)
            d = np.zeros(m.shape, bool)
            for a in self.anillos:
                d |= _pip(X, Y, a)
            m &= d
        return m

    def area_ha(self):
        if self.anillos:
            a = 0.0
            for r in self.anillos:
                a += 0.5 * abs(np.sum(r[:-1, 0] * r[1:, 1] - r[1:, 0] * r[:-1, 1]))
            return a / 1e4
        b = self.bounds
        return (b[1] - b[0]) * (b[3] - b[2]) / 1e4

    # ------------------------------------------------------------------ recorte
    def recortar(self, raster, malla, res):
        """(ráster con NaN fuera de la parcela y reducido a su ventana, xs, ys de esquinas)."""
        m = self.mascara(malla, res, raster.shape)
        xs, ys = pr.ejes(malla, res, raster.shape)
        if not m.any():
            return raster, xs, ys
        f, c = np.nonzero(m)
        r0, r1, c0, c1 = f.min(), f.max() + 1, c.min(), c.max() + 1
        out = np.where(m, raster, np.nan)[r0:r1, c0:c1]
        return out, xs[c0:c1], ys[r0:r1]

    def contornos(self):
        """Anillos para dibujar el contorno."""
        if self.anillos:
            return self.anillos
        b = self.bounds
        return [np.array([[b[0], b[2]], [b[1], b[2]], [b[1], b[3]], [b[0], b[3]], [b[0], b[2]]])]

    def descripcion(self):
        b = self.bounds
        return (f"{self.nombre}: X {b[0]:.1f}–{b[1]:.1f} m, Y {b[2]:.1f}–{b[3]:.1f} m, "
                f"{self.area_ha():.3f} ha")


# ---------------------------------------------------------------------- construcción
def extension(listas_xyz):
    mn = np.min([x[:, :2].min(axis=0) for x in listas_xyz], axis=0)
    mx = np.max([x[:, :2].max(axis=0) for x in listas_xyz], axis=0)
    return float(mn[0]), float(mx[0]), float(mn[1]), float(mx[1])


def _acotar(roi, ext):
    r = (max(roi[0], ext[0]), min(roi[1], ext[1]), max(roi[2], ext[2]), min(roi[3], ext[3]))
    return r if (r[1] > r[0] and r[3] > r[2]) else ext


def construir_roi(ext, modo, borde=10.0, coords=None, hectareas=1.0, dx=0.0, dy=0.0, poligono=None):
    """Crea una Roi.  modo ∈ {'nube', 'borde', 'coords', 'parcela', 'poligono'}."""
    xmin, xmax, ymin, ymax = ext
    if modo == "borde":
        return Roi(_acotar((xmin + borde, xmax - borde, ymin + borde, ymax - borde), ext),
                   nombre=f"Sin borde de {borde:g} m")
    if modo == "coords" and coords and None not in coords:
        return Roi(_acotar(tuple(coords), ext), nombre="Coordenadas manuales")
    if modo == "parcela":
        lado = float(np.sqrt(hectareas * 1e4))
        cx, cy = (xmin + xmax) / 2 + dx, (ymin + ymax) / 2 + dy
        return Roi(_acotar((cx - lado / 2, cx + lado / 2, cy - lado / 2, cy + lado / 2), ext),
                   nombre=f"Parcela cuadrada de {hectareas:g} ha")
    if modo == "poligono" and poligono is not None:
        return Roi(_acotar(poligono['bbox'], ext), anillos=poligono['anillos'],
                   nombre=poligono.get('nombre', "Polígono"))
    return Roi(ext, nombre="Toda la nube")


def _wkb_poligonos(b, o=0):
    """Parsea WKB (Polygon/MultiPolygon, 2D/Z) → lista de anillos exteriores. Devuelve (anillos, nuevo_offset)."""
    import struct
    en = "<" if b[o] == 1 else ">"
    tipo = struct.unpack(en + "I", b[o + 1:o + 5])[0]
    o += 5
    base = tipo % 1000
    dim = 3 if (tipo // 1000) in (1, 3) or tipo & 0x80000000 else 2
    if base == 3:
        nr = struct.unpack(en + "I", b[o:o + 4])[0]; o += 4
        anillos = []
        for _ in range(nr):
            n = struct.unpack(en + "I", b[o:o + 4])[0]; o += 4
            a = np.frombuffer(b, dtype=en + "f8", count=n * dim, offset=o).reshape(n, dim)[:, :2].copy()
            o += n * dim * 8
            anillos.append(a)
        return [anillos[0]] if anillos else [], o      # solo exterior
    if base in (6, 7):
        n = struct.unpack(en + "I", b[o:o + 4])[0]; o += 4
        out = []
        for _ in range(n):
            r, o = _wkb_poligonos(b, o)
            out += r
        return out, o
    raise ValueError(f"Geometría WKB no soportada ({tipo}).")


def _leer_gpkg_sqlite(ruta):
    """Lee polígonos de un GeoPackage solo con sqlite3 (sin geopandas)."""
    import sqlite3
    con = sqlite3.connect(ruta)
    try:
        tabla, col = con.execute("select table_name, column_name from gpkg_geometry_columns limit 1").fetchone()
        anillos = []
        for (blob,) in con.execute(f'select "{col}" from "{tabla}"'):
            if blob is None:
                continue
            blob = bytes(blob)
            flags = blob[3]
            env = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}.get((flags >> 1) & 7, 0)
            wkb = blob[8 + env:]
            r, _ = _wkb_poligonos(wkb)
            anillos += r
    finally:
        con.close()
    return anillos


def leer_poligono(origen, nombre_archivo, epsg=None):
    """Lee .gpkg / .geojson / .zip(shp) → dict(bbox, anillos, nombre). Usa geopandas si está; si no, .gpkg/.geojson con módulos estándar."""
    import os
    import tempfile
    suf = os.path.splitext(nombre_archivo)[1].lower()
    if hasattr(origen, "getvalue"):
        datos = origen.getvalue()
    elif isinstance(origen, (bytes, bytearray)):
        datos = bytes(origen)
    else:
        datos = open(origen, "rb").read()
    with tempfile.NamedTemporaryFile(suffix=suf, delete=False) as tmp:
        tmp.write(datos)
    try:
        try:
            import geopandas as gpd
            gdf = gpd.read_file(("zip://" + tmp.name) if suf == ".zip" else tmp.name)
            if gdf.crs is not None and epsg:
                gdf = gdf.to_crs(int(epsg))
            anillos = []
            for geom in gdf.geometry:
                if geom is None:
                    continue
                for pol in (geom.geoms if geom.geom_type == "MultiPolygon" else [geom]):
                    anillos.append(np.asarray(pol.exterior.coords)[:, :2])
        except ImportError:
            if suf == ".gpkg":
                anillos = _leer_gpkg_sqlite(tmp.name)
            elif suf in (".geojson", ".json"):
                import json
                gj = json.loads(datos.decode("utf-8"))
                feats = gj["features"] if "features" in gj else [gj]
                anillos = []
                for f in feats:
                    g = f.get("geometry", f)
                    pols = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
                    anillos += [np.asarray(p[0], float)[:, :2] for p in pols]
            else:
                raise ValueError("Sin geopandas solo se pueden leer .gpkg y .geojson.")
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
    if not anillos:
        raise ValueError("El archivo no contiene polígonos.")
    allp = np.vstack(anillos)
    return dict(bbox=(float(allp[:, 0].min()), float(allp[:, 0].max()), float(allp[:, 1].min()), float(allp[:, 1].max())),
                anillos=anillos, nombre=os.path.splitext(os.path.basename(nombre_archivo))[0])
