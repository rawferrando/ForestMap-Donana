"""graficos.py — Figuras interactivas Plotly para la interfaz (todo recortado a la parcela)."""
import numpy as np
import plotly.graph_objects as go

import figuras as fg


def _roi_traza(roi):
    xs, ys = [], []
    for a in roi.contornos():
        xs += list(a[:, 0]) + [None]
        ys += list(a[:, 1]) + [None]
    return go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="#c0392b", width=2, dash="dash"),
                      name="Parcela", hoverinfo="skip")


def _entidad_lineas(entidades):
    xs, ys = [], []
    for e in entidades:
        partes = [e['coords']] if e['tipo'] == "Polygon" else e['coords']
        for p in partes:
            for anillo in p:
                a = np.asarray(anillo)
                xs += list(a[:, 0]) + [None]
                ys += list(a[:, 1]) + [None]
    return xs, ys


def _layout(fig, titulo, ext, alto=640):
    fig.update_layout(title=titulo, height=alto, margin=dict(l=10, r=10, t=45, b=10), plot_bgcolor="#f4f4f0",
                      legend=dict(orientation="h", y=-0.08), template="plotly_white")
    fig.update_xaxes(range=[ext[0], ext[1]], title="X UTM (m)", showgrid=False, tickformat="d")
    fig.update_yaxes(range=[ext[2], ext[3]], title="Y UTM (m)", scaleanchor="x", showgrid=False, tickformat="d")
    return fig


def _ventana(roi, raster, malla, res):
    img, xs, ys = roi.recortar(raster.astype(float), malla, res)
    return img, xs + res / 2, ys - res / 2, (xs[0], xs[-1] + res, ys[-1] - res, ys[0])


def raster_plotly(raster, malla, res, roi, titulo, colorscale="Viridis", unidad="", divergente=False, zmin=None, zmax=None):
    img, cx, cy, ext = _ventana(roi, raster, malla, res)
    if divergente:
        lim = float(np.nanpercentile(np.abs(img), 98)) if np.isfinite(img).any() else 1.0
        zmin, zmax, colorscale = -lim, lim, "RdBu"
    fig = go.Figure(go.Heatmap(z=img, x=cx, y=cy, colorscale=colorscale, zmin=zmin, zmax=zmax,
                               colorbar=dict(title=unidad), hovertemplate="X %{x:.1f}<br>Y %{y:.1f}<br>%{z:.2f}<extra></extra>"))
    fig.add_trace(_roi_traza(roi))
    return _layout(fig, titulo, ext)


def _rejilla_clicks(cx, cy, n_objetivo):
    """Submuestreo de los centros de celda al que se enganchan los clicks.
    Devuelve (xx, yy, paso): el paso en celdas marca la precisión al clicar."""
    # El paso más fino que no pase del objetivo: redondear hacia arriba de golpe
    # empeora la precisión sin necesidad (2,03 -> 3 son 1,5 m en vez de 1 m).
    paso = max(1, int(np.sqrt(cx.size * cy.size / max(1, n_objetivo))))
    while cx[::paso].size * cy[::paso].size > n_objetivo:
        paso += 1
    xx, yy = np.meshgrid(cx[::paso], cy[::paso])
    return xx, yy, paso


def mapa_perfil_plotly(raster, malla, res, roi, p0=None, p1=None, ancho=3.0, titulo="CHM", unidad="m",
                       colorscale="Viridis", n_objetivo=20000):
    """CHM sobre el que se elige el perfil clicando, con la línea y la banda actuales dibujadas.

    Lleva encima una rejilla de puntos transparentes porque un go.Heatmap **no** emite
    eventos de selección que Streamlit recoja (comprobado el 2026-11-24): los clicks
    sobre el heatmap no llegan a Python y sobre un scatter sí. La rejilla es además la
    que fija la precisión al clicar: `res * paso` metros.
    """
    img, cx, cy, ext = _ventana(roi, raster, malla, res)
    fig = go.Figure(go.Heatmap(z=img, x=cx, y=cy, colorscale=colorscale, colorbar=dict(title=unidad),
                               hoverinfo="skip"))
    fig.add_trace(_roi_traza(roi))

    if p0 is not None and p1 is not None:
        a, b = np.asarray(p0, float), np.asarray(p1, float)
        d = b - a
        L = float(np.hypot(*d))
        if L > 1e-9:
            u = d / L
            v = np.array([-u[1], u[0]]) * (ancho / 2.0)
            esq = np.array([a + v, b + v, b - v, a - v, a + v])
            fig.add_trace(go.Scatter(x=esq[:, 0], y=esq[:, 1], mode="lines", fill="toself",
                                     line=dict(color="#ff7f0e", width=1), fillcolor="rgba(255,127,14,0.25)",
                                     name=f"Banda ({ancho:g} m)", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[a[0], b[0]], y=[a[1], b[1]], mode="lines+markers+text",
                                 line=dict(color="#ff7f0e", width=3),
                                 marker=dict(size=11, color=["#2ca02c", "#d62728"], line=dict(color="white", width=2)),
                                 text=["inicio", "fin"], textposition="top center",
                                 textfont=dict(size=11, color="#333"),
                                 name=f"Perfil ({L:.1f} m)", hoverinfo="skip"))

    xx, yy, paso = _rejilla_clicks(cx, cy, n_objetivo)
    z = img[::paso, ::paso]
    fig.add_trace(go.Scattergl(
        x=xx.ravel(), y=yy.ravel(), mode="markers",
        marker=dict(size=max(6, int(np.ceil(6 * paso))), opacity=0.0),
        customdata=np.where(np.isfinite(z), z, np.nan).ravel(),
        hovertemplate="X %{x:.1f}<br>Y %{y:.1f}<br>%{customdata:.2f} m<extra></extra>",
        name="clic", showlegend=False))

    fig = _layout(fig, titulo, ext)
    fig.update_layout(dragmode=False, clickmode="event+select")
    return fig, res * paso


def capas_plotly(res_arb, modelos, roi, estructuras=None, titulo=None, mostrar_copas=True, mostrar_apices=True):
    cap = res_arb['cap']
    malla, res = modelos['malla_chm'], modelos['res_chm']
    bordes = cap['bordes']
    nb = len(bordes) - 1
    cols = fg.colores_capas(nb)
    clase = np.where(cap['mascara'], cap['clase'], np.nan).astype(float)
    img, cx, cy, ext = _ventana(roi, clase, malla, res)
    # escala discreta
    esc = []
    for b in range(nb):
        esc += [[b / nb, cols[b]], [(b + 1) / nb, cols[b]]]
    etiquetas = [f"{bordes[b]:g}–{bordes[b + 1]:g}" for b in range(nb)]
    paso = max(1, nb // 12)
    fig = go.Figure(go.Heatmap(
        z=img, x=cx, y=cy, zmin=-0.5, zmax=nb - 0.5, colorscale=esc,
        colorbar=dict(title="Altura (m)", tickmode="array", tickvals=list(range(0, nb, paso)),
                      ticktext=[etiquetas[i] for i in range(0, nb, paso)]),
        customdata=np.where(np.isfinite(img), np.array(etiquetas, dtype=object)[np.nan_to_num(img).astype(int).clip(0, nb - 1)], ""),
        hovertemplate="X %{x:.1f}<br>Y %{y:.1f}<br>Capa %{customdata} m<extra></extra>"))
    tab = res_arb['tabla']
    if mostrar_copas and res_arb['vec']['copas']:
        xs, ys = _entidad_lineas(res_arb['vec']['copas'])
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="#222", width=0.8), name="Copas", hoverinfo="skip"))
    if mostrar_apices and len(tab):
        fig.add_trace(go.Scatter(
            x=tab['x'], y=tab['y'], mode="markers", name="Ápices", marker=dict(symbol="cross", size=5, color="black"),
            customdata=np.column_stack([tab['id'], tab['h_m'], tab['area_m2'], tab['vol_m3']]),
            hovertemplate="Árbol #%{customdata[0]:.0f}<br>H %{customdata[1]:.2f} m<br>Copa %{customdata[2]:.1f} m²<br>Vol %{customdata[3]:.1f} m³<extra></extra>"))
        altos = tab[tab['alto']]
        if len(altos):
            fig.add_trace(go.Scatter(x=altos['x'], y=altos['y'], mode="markers+text", name="Altos (auditoría)",
                                     text=[f"#{int(i)} · {h:.1f} m" for i, h in zip(altos['id'], altos['h_m'])],
                                     textposition="top right", textfont=dict(size=10),
                                     marker=dict(symbol="circle-open", size=14, color="#111", line=dict(width=2))))
    if estructuras is not None and not estructuras.vacio():
        t = estructuras.tabla
        for tipo_ini, col in (("Antena", "#b0008f"), ("Vallado", "#e67e22"), ("Poste", "#2980b9"), ("Instrumento", "#16a085")):
            s = t[t['tipo'].str.startswith(tipo_ini)]
            if len(s):
                fig.add_trace(go.Scatter(x=s['x'], y=s['y'], mode="markers+text", name=tipo_ini,
                                         text=[f"{tipo_ini} {h:.1f} m" if tipo_ini == "Antena" else "" for h in s['h_m']],
                                         textposition="top right", marker=dict(symbol="diamond", size=11, color=col, line=dict(width=1, color="white"))))
    fig.add_trace(_roi_traza(roi))
    return _layout(fig, titulo or f"Capas de altura de copa — {res_arb['et']}", ext, 720)


def histograma_alturas(tabla, umbral):
    h = tabla['h_m']
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=h[h < umbral], xbins=dict(size=0.5), marker_color="#2e6930", name="Individuos"))
    fig.add_trace(go.Histogram(x=h[h >= umbral], xbins=dict(size=0.5), marker_color="#c0392b", name=f"≥ {umbral:g} m"))
    fig.update_layout(barmode="overlay", height=340, template="plotly_white", xaxis_title="Altura (m)", yaxis_title="N.º de individuos",
                      margin=dict(l=10, r=10, t=30, b=10))
    return fig


def volumen_capas(glob):
    cols = fg.colores_capas(len(glob))
    y = [f"{a:g}–{b:g}" for a, b in zip(glob['h_min_m'], glob['h_max_m'])]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=y, x=glob['area_clase_m2'], orientation="h", marker_color=cols, name="Área (m²)"))
    fig.add_trace(go.Bar(y=y, x=glob['vol_estrato_m3'], orientation="h", marker_color=cols, name="Volumen (m³)", visible="legendonly"))
    fig.update_layout(height=520, template="plotly_white", xaxis_title="m² (clic en la leyenda para ver m³)",
                      yaxis_title="Capa (m)", margin=dict(l=10, r=10, t=30, b=10))
    return fig


def dispersion(tabla):
    fig = go.Figure(go.Scatter(x=tabla['h_m'], y=tabla['area_m2'], mode="markers",
                               marker=dict(size=7, color=tabla['vol_m3'], colorscale="YlGn", showscale=True, colorbar=dict(title="Vol m³"),
                                           line=dict(width=0.5, color="#333")),
                               customdata=tabla['id'], hovertemplate="#%{customdata}<br>H %{x:.2f} m<br>Copa %{y:.1f} m²<extra></extra>"))
    fig.update_layout(height=360, template="plotly_white", xaxis_title="Altura (m)", yaxis_title="Área de copa (m²)", yaxis_type="log",
                      margin=dict(l=10, r=10, t=30, b=10))
    return fig
