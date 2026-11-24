"""Figuras Plotly de la interfaz. Sobre todo el mapa con el que se elige el perfil clicando."""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

import figuras as fg
import fixture
import graficos as gr

F = fixture.construir()
mod, roi, datos = F['mod'], F['roi'], F['datos']
e = "2026"
ep = mod['epocas'][e]
m, r = mod['malla_chm'], mod['res_chm']
b = roi.bounds
ym = (b[2] + b[3]) / 2
p0, p1, ancho = (b[0], ym), (b[1], ym), 4.0

fig, prec = gr.mapa_perfil_plotly(ep['chm'], m, r, roi, p0, p1, ancho, n_objetivo=20000)
trazas = {t.name: t for t in fig.data}
print("trazas:", list(trazas))

# La rejilla de clicks es la razón de ser de la función: sin un scatter encima, los
# clicks sobre el heatmap no llegan a Python (comprobado en la app el 2026-11-24).
rej = trazas['clic']
assert rej.type == "scattergl", f"la rejilla debe ser scattergl, es {rej.type}"
assert rej.marker.opacity == 0, "la rejilla debe ser invisible"
n = rej.x.size
assert n <= 20000, f"rejilla de {n} puntos, por encima del objetivo"
assert n > 400, f"rejilla de solo {n} puntos: precisión insuficiente"
assert prec >= r, f"la precisión ({prec}) no puede ser mejor que la celda ({r})"
print(f"rejilla: {n} puntos | precisión ±{prec:g} m | celda {r:g} m")

# Los puntos clicables tienen que caer dentro de la parcela recortada, o se podrían
# elegir extremos fuera del CHM.
ext = (fig.layout.xaxis.range, fig.layout.yaxis.range)
assert ext[0][0] <= rej.x.min() and rej.x.max() <= ext[0][1], "rejilla fuera del eje X"
assert ext[1][0] <= rej.y.min() and rej.y.max() <= ext[1][1], "rejilla fuera del eje Y"

# Hover: debe seguir informando de la altura del CHM aunque el heatmap lo tenga apagado.
assert "customdata" in rej.hovertemplate, "la rejilla no muestra la altura al pasar por encima"
assert np.isfinite(np.asarray(rej.customdata, float)).any(), "customdata sin alturas válidas"

# Línea y banda dibujadas, que es el «ver el perfil que se está viendo».
linea = [t for t in fig.data if t.name and t.name.startswith("Perfil")][0]
assert tuple(linea.x) == (p0[0], p1[0]) and tuple(linea.y) == (p0[1], p1[1]), "la línea no une p0 y p1"
banda = [t for t in fig.data if t.name and t.name.startswith("Banda")][0]
assert banda.fill == "toself" and len(banda.x) == 5, "la banda debe ser un rectángulo cerrado"
anchos = np.hypot(np.diff(np.asarray(banda.x)[[0, 3]]), np.diff(np.asarray(banda.y)[[0, 3]]))
assert abs(anchos[0] - ancho) < 1e-6, f"la banda mide {anchos[0]:.3f} m de ancho, no {ancho}"

# Sin perfil definido no debe reventar ni dibujar línea.
fig2, _ = gr.mapa_perfil_plotly(ep['chm'], m, r, roi, None, None, ancho)
assert not [t for t in fig2.data if t.name and t.name.startswith("Perfil")], "dibuja perfil sin p0/p1"
# Inicio y fin coincidentes: la banda es degenerada, no debe dibujarse.
fig3, _ = gr.mapa_perfil_plotly(ep['chm'], m, r, roi, p0, p0, ancho)
assert not [t for t in fig3.data if t.name and t.name.startswith("Banda")], "banda con p0 == p1"

# Coste de un click: cada uno provoca un rerun que recalcula el perfil.
d = datos[e]
t = time.time()
png = fg.perfil_vertical(d['xyz'], ep['hag'], d['suelo'], p0, p1, ancho, F['estr'][e])
dt = time.time() - t
assert isinstance(png, (bytes, bytearray)) and len(png) > 1000, "perfil vacío"
print(f"perfil: {dt:.2f}s con {len(d['xyz']):,} puntos ({len(png)//1024} KB)")
print("TODO OK")
