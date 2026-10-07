"""
figuras.py — Figuras estáticas (matplotlib) para el informe. Estilo de mapa técnico.
"""
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

import procesado as pr

# paleta de capas: rojo (bajo) → naranja → amarillo → verdes → azul (alto), como en QGIS
_PALETA = ["#d7191c", "#f46d43", "#fdae61", "#fee08b", "#d9ef8b", "#a6d96a", "#66bd63", "#1a9850", "#3288bd"]
VERDE, GRIS, ROJO = "#1e4620", "#555555", "#c0392b"

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.labelsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130, "savefig.dpi": 130,
    "axes.grid": False, "legend.frameon": False,
})


def colores_capas(n):
    base = np.array([[int(c[i:i + 2], 16) for i in (1, 3, 5)] for c in _PALETA], float)
    x = np.linspace(0, 1, len(base))
    xs = np.linspace(0, 1, max(n, 2))
    rgb = np.column_stack([np.interp(xs, x, base[:, k]) for k in range(3)])
    return ["#%02x%02x%02x" % tuple(int(round(v)) for v in c) for c in rgb[:n]]


def a_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def _extent(xs, ys, res):
    return [xs[0], xs[-1] + res, ys[-1] - res, ys[0]]


def _ejes_mapa(ax, titulo=None):
    ax.set_aspect("equal")
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.tick_params(labelsize=7)
    ax.set_xlabel("X UTM (m)")
    ax.set_ylabel("Y UTM (m)")
    if titulo:
        ax.set_title(titulo, loc="left")
    for lab in ax.get_yticklabels():
        lab.set_rotation(90)
        lab.set_va("center")


def _barra_escala(ax, longitud=None):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ancho = x1 - x0
    if longitud is None:
        longitud = next(v for v in (5, 10, 20, 25, 50, 100, 200) if v >= ancho / 6)
    xi, yi = x0 + ancho * 0.04, y0 + (y1 - y0) * 0.04
    ax.plot([xi, xi + longitud], [yi, yi], color="black", lw=2.5, solid_capstyle="butt", zorder=9)
    ax.text(xi + longitud / 2, yi + (y1 - y0) * 0.012, f"{longitud:g} m", ha="center", va="bottom", fontsize=7, zorder=9)
    ax.annotate("N", xy=(x1 - ancho * 0.04, y1 - (y1 - y0) * 0.04), xytext=(x1 - ancho * 0.04, y1 - (y1 - y0) * 0.12),
                ha="center", fontsize=9, fontweight="bold", arrowprops=dict(arrowstyle="-|>", color="black"), zorder=9)


def _contorno_roi(ax, roi):
    for a in roi.contornos():
        ax.plot(a[:, 0], a[:, 1], color=ROJO, lw=1.2, ls="--", zorder=8)


def _recorte(roi, raster, malla, res):
    r, xs, ys = roi.recortar(raster, malla, res)
    return r, _extent(xs, ys, res)


# ---------------------------------------------------------------- mapas de capas
def mapa_capas(res_arb, modelos, roi, estructuras=None, titulo=None, etiquetar_altos=True, apices=True):
    cap, chm = res_arb['cap'], res_arb['chm_eff']
    malla, res = modelos['malla_chm'], modelos['res_chm']
    bordes, nb = cap['bordes'], len(cap['bordes']) - 1
    cols = colores_capas(nb)
    clase = np.where(cap['mascara'], cap['clase'], np.nan).astype(float)
    img, ext = _recorte(roi, clase, malla, res)
    fig, ax = plt.subplots(figsize=(7.4, 6.6))
    cmap = ListedColormap(cols)
    cmap.set_bad(color="#f4f4f0")
    norm = BoundaryNorm(np.arange(-0.5, nb + 0.5, 1), nb)
    im = ax.imshow(img, cmap=cmap, norm=norm, extent=ext, interpolation="nearest", zorder=1)
    # contornos de copa
    for e in res_arb['vec']['copas']:
        partes = [e['coords']] if e['tipo'] == "Polygon" else e['coords']
        for p in partes:
            for anillo in p:
                a = np.asarray(anillo)
                ax.plot(a[:, 0], a[:, 1], color="#222222", lw=0.45, zorder=4)
    tab = res_arb['tabla']
    if apices and len(tab):
        ax.plot(tab['x'], tab['y'], "+", color="black", ms=3.5, mew=0.7, zorder=5)
    # estructuras excluidas
    if estructuras is not None and not estructuras.vacio():
        for _, f in estructuras.tabla.iterrows():
            pts = estructuras.huellas.get(int(f['id']))
            if pts is None:
                continue
            ax.plot(pts[:, 0], pts[:, 1], ".", color="#ff00cc", ms=1.2, zorder=6, alpha=0.7)
            if f['tipo'].startswith("Antena"):
                ax.annotate(f"Antena\n{f['h_m']:.1f} m", (f['x'], f['y']), xytext=(18, 18), textcoords="offset points",
                            fontsize=7.5, fontweight="bold", color="#b0008f", zorder=10,
                            arrowprops=dict(arrowstyle="->", color="#b0008f"),
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#b0008f", lw=0.6))
    # individuos altos
    if etiquetar_altos and len(tab):
        altos = tab[tab['alto']].sort_values('h_m', ascending=False).head(12)
        for _, t in altos.iterrows():
            ax.annotate(f"#{int(t['id'])} · {t['h_m']:.1f} m", (t['x'], t['y']), xytext=(6, 6),
                        textcoords="offset points", fontsize=6.8, zorder=10,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#333333", lw=0.4, alpha=0.9))
    _contorno_roi(ax, roi)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    _ejes_mapa(ax, titulo or f"Capas de altura de copa — {res_arb['et']}")
    _barra_escala(ax)
    cb = fig.colorbar(im, ax=ax, ticks=np.arange(nb), shrink=0.78, pad=0.02)
    cb.ax.set_yticklabels([f"{bordes[b]:g}–{bordes[b + 1]:g}" for b in range(nb)], fontsize=6)
    cb.set_label("Altura de copa (m)", fontsize=8)
    ax.legend(handles=[Patch(fc="none", ec="#222222", lw=0.8, label="Copa individual"),
                       plt.Line2D([], [], marker="+", color="black", ls="", label="Ápice"),
                       plt.Line2D([], [], color=ROJO, ls="--", label="Parcela"),
                       plt.Line2D([], [], marker=".", color="#ff00cc", ls="", label="Infraestructura excluida")],
              loc="lower right", fontsize=6.5, framealpha=0.9, frameon=True)
    return a_png(fig)


def mapa_raster(raster, malla, res, roi, titulo, cmap="terrain", etiqueta="", vmin=None, vmax=None, divergente=False):
    img, ext = _recorte(roi, raster, malla, res)
    fig, ax = plt.subplots(figsize=(6.6, 5.8))
    if divergente:
        lim = float(np.nanpercentile(np.abs(img), 98)) if np.isfinite(img).any() else 1.0
        vmin, vmax = -lim, lim
        cmap = "RdBu"
    im = ax.imshow(img, cmap=cmap, extent=ext, vmin=vmin, vmax=vmax, interpolation="nearest")
    _contorno_roi(ax, roi)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    _ejes_mapa(ax, titulo)
    _barra_escala(ax)
    fig.colorbar(im, ax=ax, shrink=0.78, pad=0.02).set_label(etiqueta, fontsize=8)
    return a_png(fig)


def mapa_dem_hillshade(dem, hs, malla, res, roi, titulo):
    d, ext = _recorte(roi, dem, malla, res)
    h, _ = _recorte(roi, hs, malla, res)
    fig, ax = plt.subplots(figsize=(6.6, 5.8))
    ax.imshow(h, cmap="gray", extent=ext, interpolation="bilinear")
    im = ax.imshow(d, cmap="terrain", extent=ext, alpha=0.55, interpolation="bilinear")
    _contorno_roi(ax, roi)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    _ejes_mapa(ax, titulo)
    _barra_escala(ax)
    fig.colorbar(im, ax=ax, shrink=0.78, pad=0.02).set_label("Cota (m s.n.m.)", fontsize=8)
    return a_png(fig)


def mapa_densidad_suelo(xyz, suelo, malla, res_vis, roi, titulo):
    g = pr.rasterizar(xyz[suelo], malla, res_vis, "count")
    g = np.nan_to_num(g) / (res_vis ** 2)
    fig, ax = plt.subplots(figsize=(6.6, 5.8))
    img, ext = _recorte(roi, g, malla, res_vis)
    im = ax.imshow(img, cmap="viridis", extent=ext, vmin=0, vmax=np.nanpercentile(img, 98) if np.isfinite(img).any() else 1)
    _contorno_roi(ax, roi)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    _ejes_mapa(ax, titulo)
    fig.colorbar(im, ax=ax, shrink=0.78, pad=0.02).set_label("Puntos de suelo / m²", fontsize=8)
    return a_png(fig)


# ---------------------------------------------------------------- gráficos de datos
def hist_alturas(tabla, umbral_alto):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    h = tabla['h_m'].values
    bins = np.arange(0, np.ceil(h.max()) + 1, 0.5)
    ax.hist(h, bins=bins, color="#2e6930", edgecolor="white", lw=0.4)
    altos = h[h >= umbral_alto]
    if len(altos):
        ax.hist(altos, bins=bins, color="#c0392b", edgecolor="white", lw=0.4, label=f"≥ {umbral_alto:g} m ({len(altos)})")
        ax.legend()
    ax.axvline(np.median(h), color="black", ls=":", lw=1)
    ax.text(np.median(h), ax.get_ylim()[1] * 0.92, f" mediana {np.median(h):.1f} m", fontsize=7.5)
    ax.set_xlabel("Altura del individuo (m)")
    ax.set_ylabel("N.º de individuos")
    ax.set_title("Distribución de alturas", loc="left")
    return a_png(fig)


def dispersion_copas(tabla):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    sc = ax.scatter(tabla['h_m'], tabla['area_m2'], c=tabla['vol_m3'], s=14, cmap="YlGn", edgecolor="#333333", lw=0.3)
    ax.set_yscale("log")
    ax.set_xlabel("Altura (m)")
    ax.set_ylabel("Área de copa (m², escala log)")
    ax.set_title("Altura, tamaño de copa y volumen", loc="left")
    fig.colorbar(sc, ax=ax).set_label("Volumen de copa (m³)", fontsize=8)
    return a_png(fig)


def volumen_por_capa(glob, ancho):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 4.4), sharey=True, gridspec_kw=dict(wspace=0.08))
    y = np.arange(len(glob))
    cols = colores_capas(len(glob))
    a1.barh(y, glob['area_clase_m2'], height=0.85, color=cols, edgecolor="white", lw=0.3)
    a1.set_xlabel("Superficie de la capa (m²)")
    a1.set_title("Área por capa de altura", loc="left")
    a2.barh(y, glob['vol_estrato_m3'], height=0.85, color=cols, edgecolor="white", lw=0.3)
    a2.set_xlabel("Volumen del estrato (m³)")
    a2.set_title("Volumen por estrato", loc="left")
    a1.set_yticks(y[::max(1, len(y) // 14)])
    a1.set_yticklabels([f"{glob['h_min_m'].iloc[i]:g}–{glob['h_max_m'].iloc[i]:g} m" for i in y[::max(1, len(y) // 14)]], fontsize=7)
    return a_png(fig)


def clases_altura(g):
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    ax.bar(g['clase'], g['n'], color="#2e6930")
    for i, v in enumerate(g['n']):
        ax.text(i, v, str(int(v)), ha="center", va="bottom", fontsize=7.5)
    ax.set_ylabel("N.º de individuos")
    ax.set_title("Individuos por clase de altura", loc="left")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7.5)
    return a_png(fig)


# ---------------------------------------------------------------- infraestructura
COLOR_TIPO = {"Antena": "#b0008f", "Vallado": "#e67e22", "Poste": "#2980b9", "Instrumento": "#16a085"}


def _color_tipo(t):
    for k, v in COLOR_TIPO.items():
        if t.startswith(k):
            return v
    return "#7f8c8d"


def plano_estructuras(xyz, hag, suelo, est, radio=22.0, titulo="Infraestructura detectada"):
    t = est.tabla
    if len(t) == 0:
        return None
    ant = t[t['tipo'].str.startswith("Antena")]
    cx, cy = (ant.iloc[0]['x'], ant.iloc[0]['y']) if len(ant) else (t['x'].mean(), t['y'].mean())
    m = (np.hypot(xyz[:, 0] - cx, xyz[:, 1] - cy) <= radio) & ~suelo & (hag > 0.3)
    p, h = xyz[m], hag[m]
    if len(p) > 60000:
        s = np.random.default_rng(0).choice(len(p), 60000, replace=False)
        p, h = p[s], h[s]
    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    o = np.argsort(h)
    sc = ax.scatter(p[o, 0], p[o, 1], c=h[o], s=0.6, cmap="YlGn", vmin=0, vmax=max(6, np.percentile(h, 99)), zorder=1)
    for _, f in t.iterrows():
        pts = est.huellas.get(int(f['id']))
        if pts is not None:
            ax.plot(pts[:, 0], pts[:, 1], ".", ms=2.2, color=_color_tipo(f['tipo']), zorder=3)
    for _, f in t.iterrows():
        if f['tipo'].startswith("Vallado"):
            continue
        ax.annotate(f"{int(f['id'])}", (f['x'], f['y']), xytext=(5, 5), textcoords="offset points", fontsize=7,
                    fontweight="bold", color=_color_tipo(f['tipo']), zorder=5)
    ax.set_xlim(cx - radio, cx + radio); ax.set_ylim(cy - radio, cy + radio)
    _ejes_mapa(ax, titulo)
    _barra_escala(ax, 5)
    fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.02).set_label("Altura sobre el suelo (m)", fontsize=8)
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=c, label=k) for k, c in COLOR_TIPO.items()],
              loc="lower right", fontsize=7, frameon=True, framealpha=0.9)
    return a_png(fig)


def perfil_vertical(xyz, hag, suelo, p0, p1, ancho=3.0, est=None, titulo="Perfil vertical de la nube de puntos"):
    """Sección de la nube (como 'Elevation Profile' de QGIS) entre p0 y p1 con una banda de `ancho` m."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    L = np.hypot(*d)
    u = d / L
    v = np.array([-u[1], u[0]])
    rel = xyz[:, :2] - p0
    s, w = rel @ u, rel @ v
    m = (s >= 0) & (s <= L) & (abs(w) <= ancho / 2)
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    if m.sum() == 0:
        ax.text(0.5, 0.5, "Sin puntos en el perfil", ha="center")
        return a_png(fig)
    ss, zz, gg = s[m], xyz[m, 2], suelo[m]
    if m.sum() > 90000:
        k = np.random.default_rng(0).choice(m.sum(), 90000, replace=False)
        ss, zz, gg = ss[k], zz[k], gg[k]
    ax.scatter(ss[~gg], zz[~gg], s=0.5, color="#2e6930", alpha=0.6, lw=0, label="Vegetación / estructuras")
    ax.scatter(ss[gg], zz[gg], s=0.5, color="#8b5a2b", alpha=0.8, lw=0, label="Suelo")
    if est is not None and len(est.tabla):
        for _, f in est.tabla.iterrows():
            if f['tipo'].startswith("Antena"):
                sx = (np.array([f['x'], f['y']]) - p0) @ u
                wx = (np.array([f['x'], f['y']]) - p0) @ v
                if 0 <= sx <= L and abs(wx) <= ancho / 2 + 1:
                    ax.annotate(f"Antena {f['h_m']:.1f} m", (sx, f['cota_m']), xytext=(8, -4), textcoords="offset points",
                                fontsize=7.5, color="#b0008f", fontweight="bold")
    ax.set_xlabel("Distancia a lo largo del perfil (m)")
    ax.set_ylabel("Cota (m s.n.m.)")
    ax.set_title(titulo, loc="left")
    ax.legend(markerscale=12, loc="upper right")
    return a_png(fig)


# ---------------------------------------------------------------- multitemporal
def mapa_estados(comp, malla, res, roi, chm_ref, tit):
    fig, ax = plt.subplots(figsize=(6.8, 6.0))
    img, ext = _recorte(roi, np.where(chm_ref >= 0.5, 1.0, np.nan), malla, res)
    ax.imshow(img, cmap=ListedColormap(["#dfe8dc"]), extent=ext, interpolation="nearest")
    cfg = [("persiste", "#2e6930", "o", "Persiste"), ("mortalidad probable", "#c0392b", "X", "Mortalidad probable"),
           ("sin correspondencia (revisar)", "#e67e22", "s", "Sin correspondencia (revisar)")]
    for est, col, mk, lab in cfg:
        d = comp['df_a'][comp['df_a']['estado'] == est]
        if len(d):
            ax.scatter(d['x'], d['y'], s=18 if est != "persiste" else 8, c=col, marker=mk, label=f"{lab} ({len(d)})",
                       edgecolor="white", lw=0.3, zorder=5)
    dn = comp['df_b']
    if len(dn):
        ax.scatter(dn['x'], dn['y'], s=26, c="#2980b9", marker="^", label=f"Nuevo ({len(dn)})", edgecolor="white", lw=0.3, zorder=6)
    _contorno_roi(ax, roi)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    _ejes_mapa(ax, tit)
    _barra_escala(ax)
    ax.legend(loc="lower right", fontsize=7, frameon=True, framealpha=0.92)
    return a_png(fig)


def dispersion_h(comp):
    d = comp['df_a'][comp['df_a']['estado'] == 'persiste']
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.scatter(d['h_a'], d['h_b'], s=12, color="#2e6930", edgecolor="white", lw=0.3)
    lim = max(d['h_a'].max(), d['h_b'].max()) * 1.05 if len(d) else 1
    ax.plot([0, lim], [0, lim], color="black", lw=0.8, ls="--", label="1:1")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel(f"Altura {comp['a']} (m)"); ax.set_ylabel(f"Altura {comp['b']} (m)")
    ax.set_title("Crecimiento individual", loc="left")
    ax.legend()
    return a_png(fig)


def barras_demografia(comp):
    r = comp['resumen']
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    etiquetas = ["Persisten", "Mortalidad\nprobable", "Sin corresp.\n(revisar)", "Nuevos"]
    vals = [r['n_persisten'], r['n_mortalidad'], r['n_sin_corr'], r['n_nuevos']]
    cols = ["#2e6930", "#c0392b", "#e67e22", "#2980b9"]
    ax.bar(etiquetas, vals, color=cols)
    for i, v in enumerate(vals):
        ax.text(i, v, str(v), ha="center", va="bottom")
    ax.set_ylabel("N.º de individuos")
    ax.set_title(f"Dinámica {comp['a']} → {comp['b']}", loc="left")
    return a_png(fig)


# ---------------------------------------------------------------- vistas 3D de la nube (como CloudCompare / IsoTree)
def _proyectar(X, Y, Z, azim, elev, zexag):
    a, e = np.radians(azim), np.radians(elev)
    xr = X * np.cos(a) - Y * np.sin(a)
    yr = X * np.sin(a) + Y * np.cos(a)
    return xr, Z * zexag * np.cos(e) + yr * np.sin(e), yr


def _paleta_arboles(n, semilla=3):
    base = np.array(["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe",
                     "#008080", "#9a6324", "#aaffc3", "#808000", "#00bfff", "#ff7f50", "#7fff00", "#da70d6", "#20b2aa"])
    return base[np.random.default_rng(semilla).permutation(n) % len(base)]


def vista_nube_3d(xyz, hag, suelo, roi, labels=None, malla=None, res=None, azim=25, elev=28, zexag=1.6,
                  n_max=450000, titulo=None, estructuras=None, hmax=None):
    """Nube de puntos de la parcela en perspectiva ortogonal oblicua sobre fondo oscuro.

    labels=None → color por altura sobre el suelo; con `labels` → un color por individuo (segmentación).
    """
    dentro = roi.puntos(xyz[:, 0], xyz[:, 1])
    veg = dentro & ~suelo & (hag >= 0.3)
    sue = dentro & suelo
    rng = np.random.default_rng(1)

    def tomar(m, n):
        i = np.flatnonzero(m)
        return i if len(i) <= n else rng.choice(i, n, replace=False)
    iv, isu = tomar(veg, n_max), tomar(sue, n_max // 6)
    cx, cy = (roi.bounds[0] + roi.bounds[1]) / 2, (roi.bounds[2] + roi.bounds[3]) / 2
    z0 = float(np.percentile(xyz[sue, 2], 5)) if sue.any() else float(xyz[:, 2].min())
    fig, ax = plt.subplots(figsize=(8.2, 5.6), facecolor="#0a1a24")
    ax.set_facecolor("#0a1a24")
    # suelo
    xs, ys, ds = _proyectar(xyz[isu, 0] - cx, xyz[isu, 1] - cy, xyz[isu, 2] - z0, azim, elev, zexag)
    ax.scatter(xs, ys, s=0.25, c="#3b3a33", marker=".", linewidths=0, rasterized=True)
    # vegetación (de atrás hacia delante)
    xs, ys, ds = _proyectar(xyz[iv, 0] - cx, xyz[iv, 1] - cy, hag[iv], azim, elev, zexag)
    o = np.argsort(-ds)
    iv, xs, ys = iv[o], xs[o], ys[o]
    if labels is None:
        vmax = hmax or float(np.percentile(hag[iv], 99.7))
        sc = ax.scatter(xs, ys, s=0.7, c=np.clip(hag[iv], 0, vmax), cmap="turbo", vmin=0, vmax=vmax, marker=".",
                        linewidths=0, rasterized=True)
        cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.01)
        cb.set_label("Altura sobre el suelo (m)", color="white", fontsize=8)
        cb.ax.tick_params(colors="white", labelsize=7)
        cb.outline.set_edgecolor("#556")
    else:
        x0, y1, _, _ = malla
        col = np.floor((xyz[iv, 0] - x0) / res).astype(int).clip(0, labels.shape[1] - 1)
        fil = np.floor((y1 - xyz[iv, 1]) / res).astype(int).clip(0, labels.shape[0] - 1)
        lab = labels[fil, col]
        pal = _paleta_arboles(int(labels.max()) + 1)
        colores = np.where(lab[:, None] > 0, np.array([matplotlib.colors.to_rgb(c) for c in pal])[lab % len(pal)],
                           np.array(matplotlib.colors.to_rgb("#4a5a60"))[None, :])
        ax.scatter(xs, ys, s=0.7, c=colores, marker=".", linewidths=0, rasterized=True)
    if estructuras is not None and not estructuras.vacio():
        for _, f in estructuras.tabla.iterrows():
            if f['tipo'].startswith("Antena"):
                px, py, _ = _proyectar(np.array([f['x'] - cx]), np.array([f['y'] - cy]), np.array([f['h_m']]), azim, elev, zexag)
                ax.annotate(f"Antena {f['h_m']:.1f} m", (px[0], py[0]), xytext=(25, 15), textcoords="offset points",
                            color="white", fontsize=8, fontweight="bold", arrowprops=dict(arrowstyle="-", color="white"))
    ax.set_aspect("equal")
    ax.axis("off")
    if titulo:
        ax.set_title(titulo, loc="left", color="white", fontsize=10)
    ax.text(0.01, 0.01, f"{int(len(iv)):,} puntos de vegetación mostrados · parcela {roi.area_ha():.2f} ha".replace(",", "."),
            transform=ax.transAxes, color="#9ab", fontsize=7)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=140)
    plt.close(fig)
    return buf.getvalue()


# ---------------------------------------------------------------- especies y comparación con campo
COL_ESP = {"Sabina": "#2e6930", "Pino": "#c07a1c"}


def barras_especies(tabla, titulo="Individuos por clase de altura y especie"):
    bordes = [1.3, 2, 3, 4, 5, 6, 8, 12, 99]
    et = [f"{bordes[i]:g}–{bordes[i + 1]:g}" if bordes[i + 1] < 99 else f"≥ {bordes[i]:g}" for i in range(len(bordes) - 1)]
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    base = np.zeros(len(et))
    for sp in ("Sabina", "Pino"):
        v = tabla[tabla['especie'] == sp]['h_m'].values
        n = np.histogram(v, bins=bordes)[0]
        ax.bar(et, n, bottom=base, color=COL_ESP[sp], label=f"{sp} ({len(v)})", edgecolor="white", lw=0.4)
        base += n
    for i, t in enumerate(base):
        if t:
            ax.text(i, t, str(int(t)), ha="center", va="bottom", fontsize=7.5)
    ax.set_xlabel("Altura (m)")
    ax.set_ylabel("N.º de individuos")
    ax.set_title(titulo, loc="left")
    ax.legend()
    return a_png(fig)


def mapa_campo(res_arb, modelos, roi, campo_df, titulo="Inventario de campo (2011) sobre las copas LiDAR"):
    chm, labels = res_arb['chm_eff'], res_arb['labels']
    malla, res = modelos['malla_chm'], modelos['res_chm']
    img, ext = _recorte(roi, np.where(chm >= 1.3, chm, np.nan), malla, res)
    fig, ax = plt.subplots(figsize=(7.4, 6.6))
    im = ax.imshow(img, cmap="Greens", extent=ext, vmin=1, vmax=8, interpolation="nearest", alpha=0.85)
    for e in res_arb['vec']['copas']:
        partes = [e['coords']] if e['tipo'] == "Polygon" else e['coords']
        for p in partes:
            a = np.asarray(p[0])
            ax.plot(a[:, 0], a[:, 1], color="#333", lw=0.4, zorder=3)
    cfg = [("emparejado", "#1f77b4", "o", "Con ápice LiDAR (≤ radio)"),
           ("bajo copa (tapado)", "#e67e22", "s", "Bajo una copa ya contada (tapado)"),
           ("sin copa LiDAR", "#c0392b", "x", "Sin copa ≥ 1,3 m en el CHM")]
    for est, col, mk, lab in cfg:
        d = campo_df[campo_df['estado'] == est]
        if len(d):
            ax.scatter(d['xl'], d['yl'], s=14, c=col, marker=mk, label=f"{lab} ({len(d)})", edgecolor="white" if mk != "x" else None,
                       lw=0.3, zorder=6)
    t = res_arb['tabla']
    ax.plot(t['x'], t['y'], "k+", ms=3, mew=0.5, zorder=5)
    _contorno_roi(ax, roi)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    _ejes_mapa(ax, titulo)
    _barra_escala(ax)
    ax.legend(loc="lower right", fontsize=6.8, frameon=True, framealpha=0.92)
    fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02).set_label("CHM (m)", fontsize=8)
    return a_png(fig)


def barras_comparacion_campo(cmp, n_pino=None):
    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    et = ["Fichas de campo\n(vivas ≥ 1,3 m)", "Individuos\nLiDAR", "Emparejados\n(≤ radio)", "Tapados bajo\nuna copa"]
    v = [cmp['n_campo'], cmp['n_lidar'], cmp['n_emparejados'], cmp['n_tapados']]
    ax.bar(et, v, color=["#7f8c8d", "#2e6930", "#1f77b4", "#e67e22"])
    for i, t in enumerate(v):
        ax.text(i, t, str(t), ha="center", va="bottom")
    ax.set_title("Recuento LiDAR frente a inventario de campo", loc="left")
    ax.set_ylabel("N.º de individuos")
    plt.setp(ax.get_xticklabels(), fontsize=7.5)
    return a_png(fig)


def demografia_especies(comp):
    r = comp['resumen']
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    et = ["Persisten", "Mortalidad\nprobable", "Sin corresp.\n(revisar)", "Nuevos"]
    x = np.arange(4)
    for k, sp in enumerate(("Sabina", "Pino")):
        s = sp.lower()
        v = [r.get(f'{s}_persisten', 0), r.get(f'{s}_mortalidad', 0), r.get(f'{s}_sin_corr', 0), r.get(f'{s}_nuevos', 0)]
        ax.bar(x + (k - 0.5) * 0.38, v, 0.38, color=COL_ESP[sp], label=sp)
        for xi, t in zip(x + (k - 0.5) * 0.38, v):
            ax.text(xi, t, str(t), ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(et)
    ax.set_ylabel("N.º de individuos")
    ax.set_title(f"Demografía {comp['a']} → {comp['b']} por especie", loc="left")
    ax.legend()
    return a_png(fig)


def mapa_especies(res_arb, modelos, roi, titulo=None):
    """Copas coloreadas por especie (pino > h_pino_m, sabina el resto) sobre el CHM en grises."""
    chm = res_arb['chm_eff']
    malla, res = modelos['malla_chm'], modelos['res_chm']
    img, ext = _recorte(roi, np.where(chm >= 1.3, chm, np.nan), malla, res)
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    ax.imshow(img, cmap="Greys", extent=ext, vmin=0, vmax=10, interpolation="nearest", alpha=0.55)
    n = {"Sabina": 0, "Pino": 0}
    for e in res_arb['vec']['copas']:
        sp = e['props'].get('especie', 'Sabina')
        n[sp] = n.get(sp, 0) + 1
        partes = [e['coords']] if e['tipo'] == "Polygon" else e['coords']
        for p in partes:
            a = np.asarray(p[0])
            ax.fill(a[:, 0], a[:, 1], color=COL_ESP[sp], alpha=0.55 if sp == "Sabina" else 0.8, lw=0)
            ax.plot(a[:, 0], a[:, 1], color="#222", lw=0.35)
    _contorno_roi(ax, roi)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    _ejes_mapa(ax, titulo or f"Especies — {res_arb['et']}")
    _barra_escala(ax)
    ax.legend(handles=[Patch(fc=COL_ESP[k], label=f"{k} ({n.get(k, 0)})") for k in ("Sabina", "Pino")], loc="lower right", frameon=True)
    return a_png(fig)
