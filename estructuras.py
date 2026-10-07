"""
estructuras.py — Antena / mástiles, vallado e instrumentos (sensores) en la nube de puntos.

No usa el CHM (un mástil de 10 cm desaparece en un ráster de 0,5 m): trabaja con los
puntos normalizados (altura sobre el suelo, HAG).

1. MÁSTILES / ANTENA
   Un mástil es el único objeto cuya cima sobresale de su entorno con una columna
   estrecha y continua de retornos y un anillo vacío alrededor. Una copa de pino también
   es más alta que sus vecinos, pero llena ese anillo, así que se descarta.
2. VALLADO E INSTRUMENTOS (alrededor de la antena o dentro de una zona dada)
   Mancha de ocupación en planta; la vegetación forma manchas gruesas (> ~1,2 m) y las
   alambradas, postes, cajas y paneles son elementos finos. Se clasifican por forma.

Todo es heurístico y parametrizable; el resultado es una TABLA EDITABLE (campo
`excluir`): el laboratorio decide qué se descuenta del recuento de árboles.
"""
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.measure import label as sk_label, regionprops
from skimage.morphology import skeletonize


@dataclass
class ParamsEstructuras:
    # --- mástiles
    h_semilla_m: float = 4.0          # la cima debe estar al menos a esta altura sobre el suelo
    sobresale_min_m: float = 1.5      # y sobresalir del anillo de su entorno
    celda_semilla_m: float = 0.5
    radio_nucleo_m: float = 0.6       # radio del "cilindro" del mástil
    radio_anillo_m: float = 2.5       # anillo que debe estar vacío
    max_anillo: float = 0.15          # fracción tolerada de puntos en el anillo
    puntos_nucleo_min: int = 3        # por rodaja de 0,5 m
    longitud_expuesta_min_m: float = 3.0
    # --- vallado e instrumentos
    radio_busqueda_m: float = 25.0    # alrededor de cada antena (si no se da otra zona)
    hag_min_m: float = 0.4
    celda_m: float = 0.25
    grosor_vegetacion_m: float = 1.2  # manchas más gruesas = vegetación, no estructura
    largo_vallado_min_m: float = 5.0
    ancho_max_lineal_m: float = 1.0
    tam_max_instrumento_m: float = 3.5
    area_max_instrumento_m2: float = 4.0
    cerca_antena_m: float = 8.0       # "dentro del recinto" si está a menos de esto
    radio_brazos_m: float = 5.0       # brazos / sensores colgados del mástil
    brazos_sobre_m: float = 1.5       # solo cuentan si están ≥ 1,5 m por encima del dosel del entorno
    buffer_mascara_m: float = 0.5

    def dict(self):
        return asdict(self)


class Estructuras:
    """Tabla + huellas (celdas ocupadas) + mensajes de registro."""

    def __init__(self, tabla, huellas, params, log):
        self.tabla, self.huellas, self.params, self.log = tabla, huellas, params, log

    def vacio(self):
        return len(self.tabla) == 0


COLUMNAS = ['id', 'tipo', 'x', 'y', 'h_m', 'cota_m', 'h_media_m', 'longitud_m', 'area_m2',
            'ancho_m', 'n_puntos', 'confianza', 'excluir', 'notas']


def _tabla(filas):
    df = pd.DataFrame(filas, columns=COLUMNAS)
    return df


# ------------------------------------------------------------------- mástiles
def _detectar_mastiles(x, y, z, hag, p, log):
    filas, huellas = [], {}
    g = p.celda_semilla_m
    x0, y0 = x.min(), y.min()
    nx, ny = int((x.max() - x0) / g) + 1, int((y.max() - y0) / g) + 1
    ci = ((x - x0) / g).astype(np.int64)
    ri = ((y - y0) / g).astype(np.int64)
    H = np.zeros(ny * nx)
    np.maximum.at(H, ri * nx + ci, hag)
    H = H.reshape(ny, nx)

    rad = int(np.ceil(p.radio_anillo_m / g))
    yy, xx = np.ogrid[-rad:rad + 1, -rad:rad + 1]
    d = np.hypot(yy, xx) * g
    anillo = (d >= 1.0) & (d <= p.radio_anillo_m)
    max_anillo = ndimage.maximum_filter(H, footprint=anillo, mode="constant", cval=0.0)
    semillas = (H >= p.h_semilla_m) & (H - max_anillo >= p.sobresale_min_m)
    sr, sc = np.nonzero(semillas)
    if len(sr) == 0:
        log.append("Mástiles: ninguna cima sobresale lo suficiente de su entorno.")
        return filas, huellas
    # semillas ordenadas de la más alta a la más baja
    orden = np.argsort(-H[sr, sc])
    sr, sc = sr[orden], sc[orden]
    sx, sy = x0 + (sc + 0.5) * g, y0 + (sr + 0.5) * g

    # solo se usan los puntos cercanos a alguna semilla
    cerca = ndimage.binary_dilation(semillas, iterations=rad + 1)
    sel = cerca[ri, ci] & (hag >= 0.3)
    px, py, pz, ph = x[sel], y[sel], z[sel], hag[sel]
    arbol = cKDTree(np.column_stack([px, py]))

    aceptados = []
    for k in range(len(sx)):
        if any(np.hypot(sx[k] - a[0], sy[k] - a[1]) < 2.0 for a in aceptados):
            continue
        idx = np.array(arbol.query_ball_point([sx[k], sy[k]], p.radio_anillo_m + 0.8))
        if len(idx) < p.puntos_nucleo_min:
            continue
        near = idx[np.hypot(px[idx] - sx[k], py[idx] - sy[k]) <= 0.8]
        if len(near) < p.puntos_nucleo_min:
            continue
        top = ph[near].max()
        # eje: mediana de los puntos del metro superior cerca de la semilla
        c0 = near[ph[near] >= top - 1.0]
        if len(c0) < p.puntos_nucleo_min:
            continue
        ex, ey = np.median(px[c0]), np.median(py[c0])
        rho = np.hypot(px[idx] - ex, py[idx] - ey)
        hh = ph[idx]
        h_top = hh[rho <= p.radio_nucleo_m].max()
        # rodajas de 0,5 m de arriba abajo
        n_rod = int(np.ceil(h_top / 0.5))
        gaps, ultima = 0, -1
        for s in range(n_rod):
            a, b = h_top - (s + 1) * 0.5, h_top - s * 0.5
            en = (hh > a) & (hh <= b + 1e-9)
            nucleo = int((en & (rho <= p.radio_nucleo_m)).sum())
            ani = int((en & (rho > p.radio_nucleo_m) & (rho <= p.radio_anillo_m)).sum())
            ok = nucleo >= p.puntos_nucleo_min and ani <= max(2, p.max_anillo * nucleo)
            if ok:
                ultima, gaps = s, 0
            else:
                gaps += 1
                if gaps >= 2:
                    break
        longitud = (ultima + 1) * 0.5
        if ultima < 0 or longitud < p.longitud_expuesta_min_m:
            continue
        nuc = rho <= p.radio_nucleo_m
        base_h = h_top - longitud
        en_tramo = nuc & (hh >= base_h)
        radio = float(np.median(rho[en_tramo]))
        aceptados.append((ex, ey))
        cota = float(pz[idx][nuc].max())
        conf = "alta" if longitud >= 6.0 else "media"
        filas.append(dict(
            tipo="Antena / mástil", x=ex, y=ey, h_m=float(h_top), cota_m=cota,
            h_media_m=float(hh[en_tramo].mean()), longitud_m=float(longitud), area_m2=np.pi * radio ** 2,
            ancho_m=2 * radio, n_puntos=int(en_tramo.sum()), confianza=conf, excluir=True,
            notas=f"Sobresale {longitud:.1f} m sobre su entorno; radio del fuste ≈ {radio * 100:.0f} cm."))
        t = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        r_h = max(0.5, 2 * radio)
        pts_h = [[ex, ey]]
        for f in (0.5, 1.0):
            pts_h.extend(np.column_stack([ex + r_h * f * np.cos(t), ey + r_h * f * np.sin(t)]).tolist())
        # brazos y sensores del mástil: puntos altos y aislados por encima del dosel vecino
        dd = np.hypot(x - ex, y - ey)
        anillo_v = (dd >= 3.0) & (dd <= 8.0)
        ref = float(np.percentile(hag[anillo_v], 95)) if anillo_v.sum() > 20 else 0.0
        brazo = (dd <= p.radio_brazos_m) & (hag >= max(ref + p.brazos_sobre_m, 3.0))
        if brazo.sum() > 5:
            g2 = np.unique(np.round(np.column_stack([x[brazo], y[brazo]]) / 0.25) * 0.25, axis=0)
            pts_h.extend(g2.tolist())
            filas[-1]['notas'] += f" Incluye brazos/sensores colgados (radio {p.radio_brazos_m:g} m)."
        huellas[len(filas)] = np.array(pts_h)
    return filas, huellas


# ------------------------------------------------------ vallado e instrumentos
def _detectar_componentes(x, y, hag, z, mastiles, p, region, log):
    filas, huellas = [], {}
    sel = hag >= p.hag_min_m
    if region is not None:
        sel &= region(x, y)
    elif mastiles:
        m = np.zeros(len(x), bool)
        for (mx, my) in mastiles:
            m |= np.hypot(x - mx, y - my) <= p.radio_busqueda_m
        sel &= m
    else:
        log.append("Vallado/instrumentos: no hay antena ni zona definida; paso omitido.")
        return filas, huellas
    for (mx, my) in mastiles:                               # quitar el propio fuste
        sel &= np.hypot(x - mx, y - my) > p.radio_nucleo_m + 0.3
    if sel.sum() < 50:
        log.append("Vallado/instrumentos: muy pocos puntos en la zona.")
        return filas, huellas
    x, y, hag, z = x[sel], y[sel], hag[sel], z[sel]
    g = p.celda_m
    x0, y0 = x.min() - 1, y.min() - 1
    nx, ny = int((x.max() - x0) / g) + 3, int((y.max() - y0) / g) + 3
    ci = ((x - x0) / g).astype(np.int64)
    ri = ((y - y0) / g).astype(np.int64)
    flat = ri * nx + ci
    cuenta = np.bincount(flat, minlength=nx * ny).reshape(ny, nx)
    hmax = np.zeros(ny * nx)
    np.maximum.at(hmax, flat, hag)
    hmax = hmax.reshape(ny, nx)
    suma_h = np.bincount(flat, weights=hag, minlength=nx * ny).reshape(ny, nx)
    zmax = np.full(ny * nx, -np.inf)
    np.maximum.at(zmax, flat, z)
    zmax = zmax.reshape(ny, nx)

    occ = ndimage.binary_closing(cuenta >= 1, structure=np.ones((3, 3)))
    r_px = max(1, int(round(p.grosor_vegetacion_m / 2 / g)))
    yy, xx = np.ogrid[-r_px:r_px + 1, -r_px:r_px + 1]
    disco = (yy ** 2 + xx ** 2) <= r_px ** 2
    grueso = ndimage.binary_opening(occ, structure=disco)
    fino = occ & ~ndimage.binary_dilation(grueso, structure=np.ones((3, 3)))
    cerca_grueso = ndimage.binary_dilation(grueso, structure=np.ones((3, 3)), iterations=3)
    lab = sk_label(fino, connectivity=2)
    props = regionprops(lab)
    for rp in props:
        n_px = rp.area
        area = n_px * g * g
        if area < 0.02:
            continue
        rr, cc = rp.coords[:, 0], rp.coords[:, 1]
        if cerca_grueso[rr, cc].mean() > 0.5:                # flequillo de una mancha de vegetación
            continue
        comp = lab[rp.bbox[0]:rp.bbox[2], rp.bbox[1]:rp.bbox[3]] == rp.label
        esq = skeletonize(comp).sum()
        longitud = esq * g
        ancho = area / max(longitud, g)
        extension = max(rp.bbox[2] - rp.bbox[0], rp.bbox[3] - rp.bbox[1]) * g
        h_max = float(hmax[rr, cc].max())
        h_med = float(suma_h[rr, cc].sum() / max(cuenta[rr, cc].sum(), 1))
        cota = float(zmax[rr, cc].max())
        npts = int(cuenta[rr, cc].sum())
        cx, cy = x0 + (cc.mean() + 0.5) * g, y0 + (rr.mean() + 0.5) * g
        cerca = min([np.hypot(cx - mx, cy - my) for mx, my in mastiles], default=np.inf)
        cerrado = rp.euler_number <= 0
        if longitud >= p.largo_vallado_min_m and ancho <= p.ancho_max_lineal_m:
            tipo = "Vallado / alambrada" + (" (perímetro cerrado)" if cerrado else "")
            conf = "alta" if cerrado else "media"
            notas = (f"Elemento lineal de ≈{longitud:.0f} m de longitud, {h_med:.1f} m de altura media"
                     + ("; forma un recinto cerrado." if cerrado else "."))
        elif extension <= p.tam_max_instrumento_m and area <= p.area_max_instrumento_m2:
            poste = ancho <= 0.5 and h_max >= 1.5 and extension <= 1.2
            tipo = "Poste / mástil de sensor" if poste else "Instrumento / equipo"
            dentro = cerca <= p.cerca_antena_m
            conf = "media" if dentro else "baja"
            notas = (f"Objeto compacto de {extension:.1f} m; "
                     + ("dentro del recinto de la antena." if dentro else
                        "fuera del recinto: podría ser un arbusto pequeño, revisar."))
        else:
            continue
        filas.append(dict(
            tipo=tipo, x=cx, y=cy, h_m=h_max, cota_m=cota, h_media_m=h_med, longitud_m=float(longitud),
            area_m2=float(area), ancho_m=float(ancho), n_puntos=npts, confianza=conf,
            excluir=conf in ("alta", "media"), notas=notas))
        huellas[len(filas)] = np.column_stack([x0 + (cc + 0.5) * g, y0 + (rr + 0.5) * g])
    return filas, huellas


# ------------------------------------------------------------------ principal
def detectar_estructuras(x, y, z, hag, params=None, region=None, buscar_componentes=True):
    """Detecta mástiles y, alrededor de ellos (o en `region`), vallado e instrumentos.

    x, y, z, hag : puntos NO suelo (arrays). hag = altura sobre el suelo.
    region       : función (x, y) -> bool opcional con la zona de búsqueda (p. ej. polígono del recinto).
    """
    p = params or ParamsEstructuras()
    log = []
    ok = hag >= 0.3
    x, y, z, hag = x[ok], y[ok], z[ok], hag[ok]
    if len(x) == 0:
        return Estructuras(_tabla([]), {}, p, ["Sin puntos por encima del suelo."])
    f_m, h_m = _detectar_mastiles(x, y, z, hag, p, log)
    mast_xy = [(f['x'], f['y']) for f in f_m]
    f_c, h_c = [], {}
    if buscar_componentes:
        f_c, h_c = _detectar_componentes(x, y, hag, z, mast_xy, p, region, log)
    filas = f_m + f_c
    huellas = {}
    for k, f in enumerate(f_m, 1):
        huellas[k] = h_m[k]
    for k, f in enumerate(f_c, 1):
        huellas[len(f_m) + k] = h_c[k]
    tabla = _tabla(filas)
    if len(tabla):
        tabla['excluir'] = tabla['confianza'] != 'baja'          # lo dudoso no se excluye: lo decide la persona
        orden = tabla.sort_values('h_m', ascending=False).index
        mapa = {old + 1: new + 1 for new, old in enumerate(orden)}
        tabla = tabla.loc[orden].reset_index(drop=True)
        tabla['id'] = np.arange(1, len(tabla) + 1)
        huellas = {mapa[k]: v for k, v in huellas.items()}
    log.append(f"Estructuras detectadas: {len(tabla)} "
               f"({int((tabla['tipo'].str.startswith('Antena')).sum()) if len(tabla) else 0} antena/mástil).")
    return Estructuras(tabla, huellas, p, log)


# ---------------------------------------------------------- máscaras y etiquetado
def mascara_raster(est, malla, res, forma, tabla=None, buffer_m=None):
    """Máscara booleana (True = estructura a excluir del recuento de árboles)."""
    tabla = est.tabla if tabla is None else tabla
    x0, y1, nx, ny = malla
    m = np.zeros(forma, bool)
    for _, f in tabla.iterrows():
        if not f.get('excluir', True):
            continue
        pts = est.huellas.get(int(f['id']))
        if pts is None:
            continue
        c = np.floor((pts[:, 0] - x0) / res).astype(int)
        r = np.floor((y1 - pts[:, 1]) / res).astype(int)
        ok = (c >= 0) & (c < forma[1]) & (r >= 0) & (r < forma[0])
        m[r[ok], c[ok]] = True
    b = est.params.buffer_mascara_m if buffer_m is None else buffer_m
    n = int(round(b / res))
    if n > 0 and m.any():
        m = ndimage.binary_dilation(m, structure=np.ones((3, 3)), iterations=n)
    return m


def etiquetar_puntos(x, y, est, distancia=0.35):
    """Id de estructura más cercana de cada punto (0 = ninguna). Para vistas de control."""
    out = np.zeros(len(x), np.int32)
    if est.vacio():
        return out
    pts, ids = [], []
    for k, v in est.huellas.items():
        pts.append(v)
        ids.append(np.full(len(v), k))
    pts, ids = np.vstack(pts), np.concatenate(ids)
    d, i = cKDTree(pts).query(np.column_stack([x, y]), distance_upper_bound=distancia)
    ok = np.isfinite(d)
    out[ok] = ids[i[ok]]
    return out
