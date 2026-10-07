"""
procesado.py — Procesamiento LiDAR base para ForestMap Doñana V3.

Lectura y submuestreo de nubes LAS/LAZ, clasificación de suelo (CSF con
alternativa morfológica), rasterización, DEM, DSM, CHM y hillshade.

Dependencias mínimas: numpy, scipy.  Opcionales: laspy[lazrs] (leer .las/.laz),
cloth-simulation-filter (CSF).
"""
import numpy as np
from scipy import ndimage
from scipy.interpolate import griddata


# ------------------------------------------------------------------
# 1. Lectura y submuestreo
# ------------------------------------------------------------------
def _leer_las_numpy(fuente):
    """Lector LAS 1.0–1.4 (sin comprimir) en numpy puro. Devuelve (xyz, clasificacion)."""
    import struct
    if isinstance(fuente, (bytes, bytearray)):
        buf = np.frombuffer(fuente, np.uint8)
    elif hasattr(fuente, "getbuffer"):
        buf = np.frombuffer(fuente.getbuffer(), np.uint8)
    elif hasattr(fuente, "read"):
        buf = np.frombuffer(fuente.read(), np.uint8)
    else:
        buf = np.memmap(fuente, dtype=np.uint8, mode="r")
    h = bytes(buf[:375])
    if h[:4] != b"LASF":
        raise ValueError("No es un archivo LAS válido.")
    vmin = h[25]
    off = struct.unpack("<I", h[96:100])[0]
    fmt = h[104]
    if fmt & 0x80 or fmt > 10:
        raise ValueError("LAZ comprimido: instale laspy[lazrs] (pip install 'laspy[lazrs]').")
    rl = struct.unpack("<H", h[105:107])[0]
    n = struct.unpack("<I", h[107:111])[0]
    if vmin >= 4 and n == 0:
        n = struct.unpack("<Q", h[247:255])[0]
    sc = struct.unpack("<3d", h[131:155])
    of = struct.unpack("<3d", h[155:179])
    cls_off = 15 if fmt <= 5 else 16
    dt = np.dtype({"names": ["x", "y", "z", "c"], "formats": ["<i4", "<i4", "<i4", "u1"],
                   "offsets": [0, 4, 8, cls_off], "itemsize": rl})
    reg = np.ndarray(shape=(n,), dtype=dt, buffer=buf, offset=off)
    xyz = np.empty((n, 3), np.float64)
    for k, c in enumerate("xyz"):
        xyz[:, k] = reg[c] * sc[k] + of[k]
    clase = (reg["c"] & (31 if fmt <= 5 else 255)).copy()
    return xyz, clase


def leer_las(fichero, densidad_max=1000, semilla=42, con_clase=False):
    """Lee un .las/.laz (ruta o archivo subido) y devuelve un array Nx3 (x, y, z).

    Submuestrea de forma aleatoria hasta `densidad_max` puntos por m².
    Si laspy no está instalado se usa un lector propio (solo .las sin comprimir).
    con_clase=True devuelve también la clasificación original (2 = suelo).
    """
    try:
        import laspy
        if hasattr(fichero, "seek"):
            fichero.seek(0)
        las = laspy.read(fichero)
        xyz = np.column_stack([las.x, las.y, las.z]).astype(np.float64)
        clase = np.asarray(las.classification, np.uint8)
    except ImportError:
        if hasattr(fichero, "seek"):
            fichero.seek(0)
        xyz, clase = _leer_las_numpy(fichero)
    if densidad_max is None or len(xyz) == 0:
        return (xyz, clase) if con_clase else xyz
    idx = _indices_submuestreo(xyz, densidad_max, semilla)
    return (xyz[idx], clase[idx]) if con_clase else xyz[idx]


def _indices_submuestreo(xyz, densidad_max, semilla=42):
    celda_x = np.floor(xyz[:, 0] - xyz[:, 0].min()).astype(np.int64)
    celda_y = np.floor(xyz[:, 1] - xyz[:, 1].min()).astype(np.int64)
    clave = celda_x * 1_000_003 + celda_y
    orden_aleatorio = np.random.default_rng(semilla).permutation(len(xyz))
    clave_perm = clave[orden_aleatorio]
    idx = np.argsort(clave_perm, kind="stable")
    clave_ord = clave_perm[idx]
    inicio = np.r_[0, np.flatnonzero(np.diff(clave_ord)) + 1]
    tam = np.diff(np.r_[inicio, len(clave_ord)])
    rango = np.arange(len(clave_ord)) - np.repeat(inicio, tam)
    mantener = np.zeros(len(xyz), bool)
    mantener[orden_aleatorio[idx[rango < densidad_max]]] = True
    return np.flatnonzero(mantener)


def submuestrear(xyz, densidad_max, semilla=42):
    """Limita la densidad a `densidad_max` puntos/m² (celdas de 1 m, selección aleatoria)."""
    if densidad_max is None or len(xyz) == 0:
        return xyz
    return xyz[_indices_submuestreo(xyz, densidad_max, semilla)]


# ------------------------------------------------------------------
# 2. Clasificación de suelo
# ------------------------------------------------------------------
def clasificar_suelo(xyz, rigidez=3, umbral=0.5, resolucion=0.5):
    """Clasifica suelo con CSF (Zhang et al., 2016). Devuelve máscara booleana (True = suelo).

    Si la librería `cloth-simulation-filter` no está instalada se usa de forma
    automática `clasificar_suelo_morfologico`.
    """
    try:
        import CSF
    except ImportError:
        return clasificar_suelo_morfologico(xyz, umbral=max(umbral, 0.3))
    csf = CSF.CSF()
    csf.params.bSloopSmooth = False
    csf.params.cloth_resolution = resolucion
    csf.params.rigidness = int(rigidez)
    csf.params.class_threshold = umbral
    csf.setPointCloud(xyz.tolist())
    suelo, no_suelo = CSF.VecInt(), CSF.VecInt()
    csf.do_filtering(suelo, no_suelo)
    mascara = np.zeros(len(xyz), bool)
    mascara[np.array(suelo, dtype=int)] = True
    return mascara


def clasificar_suelo_morfologico(xyz, celda=2.0, umbral=0.35, iteraciones=4):
    """Alternativa sin dependencias para terreno llano/ondulado suave.

    Estima la superficie del terreno a partir del mínimo local de cada celda y
    la refina de forma iterativa usando solo los puntos ya aceptados como suelo.
    """
    x0, y0 = xyz[:, 0].min(), xyz[:, 1].min()
    nx = int((xyz[:, 0].max() - x0) / celda) + 1
    ny = int((xyz[:, 1].max() - y0) / celda) + 1
    col = ((xyz[:, 0] - x0) / celda).astype(int)
    fil = ((xyz[:, 1] - y0) / celda).astype(int)
    idx = fil * nx + col
    acepta = np.ones(len(xyz), bool)
    superficie = None
    for _ in range(iteraciones):
        zmin = np.full(ny * nx, np.inf)
        np.minimum.at(zmin, idx[acepta], xyz[acepta, 2])
        zmin = zmin.reshape(ny, nx)
        zmin[np.isinf(zmin)] = np.nan
        zmin = rellenar_nan(zmin)
        zmin = ndimage.median_filter(zmin, size=3, mode="nearest")
        superficie = ndimage.gaussian_filter(zmin, sigma=1.0, mode="nearest")
        z_sup = ndimage.map_coordinates(
            superficie,
            [(xyz[:, 1] - y0) / celda - 0.5, (xyz[:, 0] - x0) / celda - 0.5],
            order=1, mode="nearest")
        acepta = (xyz[:, 2] - z_sup) <= umbral
    return acepta & ((xyz[:, 2] - z_sup) >= -3 * umbral)


# ------------------------------------------------------------------
# 3. Rasterización
# ------------------------------------------------------------------
def definir_malla(xyz, res):
    """Malla (x0, y1, nx, ny): esquina superior izquierda y tamaño."""
    x0 = np.floor(xyz[:, 0].min())
    y1 = np.ceil(xyz[:, 1].max())
    nx = int(np.ceil((xyz[:, 0].max() - x0) / res)) + 1
    ny = int(np.ceil((y1 - xyz[:, 1].min()) / res)) + 1
    return x0, y1, nx, ny


def rasterizar(xyz, malla, res, funcion="min"):
    """Mínimo, máximo, cuenta o media de z por celda. NaN donde no hay puntos."""
    x0, y1, nx, ny = malla
    col = np.floor((xyz[:, 0] - x0) / res).astype(np.int64)
    fil = np.floor((y1 - xyz[:, 1]) / res).astype(np.int64)
    ok = (col >= 0) & (col < nx) & (fil >= 0) & (fil < ny)
    out = np.full(ny * nx, np.nan)
    if not ok.any():
        return out.reshape(ny, nx)
    idx = fil[ok] * nx + col[ok]
    z = xyz[ok, 2]
    orden = np.argsort(idx, kind="stable")
    idx, z = idx[orden], z[orden]
    inicio = np.r_[0, np.flatnonzero(np.diff(idx)) + 1]
    celdas = idx[inicio]
    if funcion == "min":
        v = np.minimum.reduceat(z, inicio)
    elif funcion == "max":
        v = np.maximum.reduceat(z, inicio)
    elif funcion == "count":
        v = np.diff(np.r_[inicio, len(z)]).astype(float)
    elif funcion == "mean":
        v = np.add.reduceat(z, inicio) / np.diff(np.r_[inicio, len(z)])
    else:
        raise ValueError(funcion)
    out[celdas] = v
    return out.reshape(ny, nx)


def rellenar_nan(r, max_puntos=250_000):
    """Interpola huecos (lineal) y rellena los bordes con el vecino más cercano."""
    validos = ~np.isnan(r)
    if validos.all():
        return r
    if not validos.any():
        return np.zeros_like(r)
    fil, col = np.indices(r.shape)
    pts = np.column_stack([fil[validos], col[validos]])
    vals = r[validos]
    if len(vals) > max_puntos:                       # el terreno es suave: basta una muestra
        sel = np.random.default_rng(0).choice(len(vals), max_puntos, replace=False)
        pts, vals = pts[sel], vals[sel]
    huecos = ~validos
    out = r.copy()
    interp = griddata(pts, vals, (fil[huecos], col[huecos]), method="linear")
    out[huecos] = interp
    nan = np.isnan(out)
    if nan.any():
        idx = ndimage.distance_transform_edt(nan, return_distances=False, return_indices=True)
        out[nan] = out[tuple(i[nan] for i in idx)]
    return out


def muestrear(raster, malla, res, x, y):
    """Valor interpolado (bilineal) de un ráster en las coordenadas (x, y)."""
    x0, y1, _, _ = malla
    col = (np.asarray(x) - x0) / res - 0.5
    fil = (y1 - np.asarray(y)) / res - 0.5
    return ndimage.map_coordinates(raster, [fil, col], order=1, mode="nearest")


def altura_sobre_suelo(xyz, dem, malla, res):
    """Altura normalizada (HAG) de cada punto respecto al DEM."""
    return xyz[:, 2] - muestrear(dem, malla, res, xyz[:, 0], xyz[:, 1])


# ------------------------------------------------------------------
# 4. DEM, DSM, CHM, hillshade
# ------------------------------------------------------------------
def generar_dem(xyz_suelo, malla, res, f_min=3, f_med=15):
    dem = rasterizar(xyz_suelo, malla, res, "min")
    dem = rellenar_nan(dem)
    dem = ndimage.minimum_filter(dem, size=int(f_min), mode="nearest")
    dem = ndimage.uniform_filter(dem, size=int(f_med), mode="nearest")
    return dem


def generar_dsm(xyz, malla, res):
    return rellenar_nan(rasterizar(xyz, malla, res, "max"))


def generar_chm(xyz_todos, dem_fino, malla, res, pit=0.025, dsm=None):
    """CHM = DSM − DEM, con relleno de depresiones aisladas (pit-filling)."""
    if dsm is None:
        dsm = generar_dsm(xyz_todos, malla, res)
    chm = np.clip(dsm - dem_fino, 0, None)
    kernel = np.ones((3, 3), bool)
    kernel[1, 1] = False
    min_vecinos = ndimage.minimum_filter(chm, footprint=kernel, mode="nearest")
    es_pit = (min_vecinos - chm) > pit
    chm[es_pit] = min_vecinos[es_pit]
    return chm


def hillshade(dem, res, azimut=315, altitud=45):
    az, alt = np.radians(360 - azimut + 90), np.radians(altitud)
    dy, dx = np.gradient(dem, res)
    pend = np.arctan(np.hypot(dx, dy))
    asp = np.arctan2(-dx, dy)
    hs = np.sin(alt) * np.cos(pend) + np.cos(alt) * np.sin(pend) * np.cos(az - asp)
    return np.clip(hs, 0, 1) * 255


# ------------------------------------------------------------------
# Utilidades
# ------------------------------------------------------------------
def ejes(malla, res, forma):
    """Coordenadas de la ESQUINA superior-izquierda de cada columna / fila."""
    x0, y1, _, _ = malla
    ny, nx = forma
    return x0 + np.arange(nx) * res, y1 - np.arange(ny) * res


def centros(malla, res, forma):
    """Coordenadas del CENTRO de cada columna / fila."""
    xs, ys = ejes(malla, res, forma)
    return xs + res / 2, ys - res / 2


def reducir(r, max_px=900):
    """Reduce el ráster (por bloques, conservando máximos) para mostrarlo rápido."""
    paso = max(1, int(np.ceil(max(r.shape) / max_px)))
    if paso == 1:
        return r, 1
    ny, nx = (r.shape[0] // paso) * paso, (r.shape[1] // paso) * paso
    bloque = r[:ny, :nx].reshape(ny // paso, paso, nx // paso, paso)
    with np.errstate(all="ignore"):
        return np.nanmax(bloque, axis=(1, 3)), paso


def reducir_ejes(r, xs, ys, max_px=900):
    """Como `reducir` pero devolviendo también ejes coherentes con el ráster reducido."""
    red, paso = reducir(r, max_px)
    return red, xs[::paso][:red.shape[1]], ys[::paso][:red.shape[0]]


def malla_comun(listas_xyz, res):
    """Malla que cubre varias nubes (necesaria para comparar épocas píxel a píxel)."""
    mins = np.min([x[:, :2].min(axis=0) for x in listas_xyz], axis=0)
    maxs = np.max([x[:, :2].max(axis=0) for x in listas_xyz], axis=0)
    falso = np.array([[mins[0], mins[1], 0.0], [maxs[0], maxs[1], 0.0]])
    return definir_malla(falso, res)
