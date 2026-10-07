import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import vectores as vc

def area_total(res):  # suma de áreas de polígonos por código
    return {k: sum(vc.area_poligono(e, h) for e, h in v) for k, v in res.items()}

# 1) un píxel
L = np.zeros((3, 3), int); L[1, 1] = 5
r = vc.poligonizar(L, 100.0, 200.0, 2.0)
assert list(r) == [5] and len(r[5]) == 1
e, h = r[5][0]
assert abs(vc.area_poligono(e, h) - 4.0) < 1e-9, vc.area_poligono(e, h)
assert vc.area_anillo(e) > 0, "exterior debe ser antihorario"
print("1 píxel OK", e.tolist())

# 2) anillo con hueco y una isla dentro del hueco
L = np.zeros((11, 11), int)
L[1:10, 1:10] = 1
L[3:8, 3:8] = 0
L[5, 5] = 1                                   # isla dentro del hueco
r = vc.poligonizar(L, 0.0, 11.0, 1.0)
a = area_total(r)
assert abs(a[1] - L.sum()) < 1e-9, (a, L.sum())
assert len(r[1]) == 2, len(r[1])               # anillo + isla
con_huecos = [len(h) for e, h in r[1]]
assert sorted(con_huecos) == [0, 1], con_huecos
print("hueco+isla OK, áreas", a)

# 3) píxeles en diagonal NO se unen (4-conectividad)
L = np.zeros((4, 4), int); L[1, 1] = 2; L[2, 2] = 2
r = vc.poligonizar(L, 0.0, 4.0, 1.0)
assert len(r[2]) == 2, len(r[2])
assert abs(area_total(r)[2] - 2.0) < 1e-9
print("diagonal OK")

# 4) varios códigos adyacentes: la suma de áreas = nº de píxeles
rng = np.random.default_rng(1)
L = rng.integers(0, 6, (60, 70))
r = vc.poligonizar(L, 721000.0, 4097300.0, 0.5)
a = area_total(r)
for k in range(1, 6):
    assert abs(a[k] - (L == k).sum() * 0.25) < 1e-6, (k, a[k], (L == k).sum() * 0.25)
print("aleatorio 60x70 OK (conserva el área exacta de cada etiqueta)")

# 5) blobs suaves + simplificación
yy, xx = np.mgrid[0:200, 0:200]
L = np.zeros((200, 200), int)
for k, (cy, cx, rr) in enumerate([(60, 60, 30), (120, 130, 40), (50, 150, 20)], 1):
    L[(yy - cy) ** 2 + (xx - cx) ** 2 < rr ** 2] = k
r0 = vc.poligonizar(L, 0.0, 200.0, 0.5)
r1 = vc.poligonizar(L, 0.0, 200.0, 0.5, simplificar=0.25)
n0 = sum(len(e) for v in r0.values() for e, h in v)
n1 = sum(len(e) for v in r1.values() for e, h in v)
a0, a1 = area_total(r0), area_total(r1)
print(f"vértices {n0} -> {n1} al simplificar; áreas {[round(a0[k],1) for k in a0]} vs {[round(a1[k],1) for k in a1]}")
assert n1 < n0 and all(abs(a0[k] - a1[k]) / a0[k] < 0.02 for k in a0)

# 6) GeoJSON
import json, tempfile
ents = [dict(tipo="Polygon", coords=[e] + h, props=dict(id=int(k), area=np.float64(vc.area_poligono(e, h)), n=np.int64(3)))
        for k, v in r0.items() for e, h in v]
ents.append(dict(tipo="Point", coords=(721000.5, 4097000.25), props=dict(h=np.float32(3.5), nan=float("nan"))))
p = os.path.join(tempfile.mkdtemp(), "t.geojson")
vc.escribir_geojson(p, ents, 25829)
d = json.load(open(p))
assert d["crs"]["properties"]["name"].endswith("25829") and len(d["features"]) == len(ents)
assert d["features"][-1]["properties"]["nan"] is None
print("GeoJSON OK,", len(d["features"]), "entidades")

# 7) rendimiento: 600x600 con 300 etiquetas
import time
L = np.zeros((600, 600), int)
for k in range(300):
    cy, cx, rr = rng.integers(20, 580, 2).tolist() + [int(rng.integers(8, 25))]
    yy, xx = np.ogrid[:600, :600]
    L[(yy - cy) ** 2 + (xx - cx) ** 2 < rr ** 2] = k + 1
t = time.time(); r = vc.poligonizar(L, 0.0, 600.0, 0.25); dt = time.time() - t
print(f"rendimiento 600x600, {len(r)} etiquetas: {dt:.1f}s")
print("OK vectores")
