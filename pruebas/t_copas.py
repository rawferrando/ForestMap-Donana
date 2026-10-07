import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
import sintetico as sy, procesado as pr, copas as cp

esc = sy.escenario(epocas=("2024",))
xyz, clase = esc['xyz']['2024'], esc['clase']['2024']
verdad = esc['arboles']['2024'].copy()
suelo = pr.clasificar_suelo_morfologico(xyz)
# zona de la antena/vallado: se excluye con la verdad (aquí solo se prueba la detección de copas)
cx, cy, s = esc['recinto']

def vol_verdad(a):
    if a['tipo'] == 'pino':
        zb = 0.5 * a['h']; return np.pi * a['R']**2 * (zb + (a['h'] - zb) / 1.6)
    return np.pi * a['R']**2 * a['h'] / 1.45

def evalua(res, metodo, **kw):
    malla = pr.definir_malla(xyz, res)
    dem = pr.generar_dem(xyz[suelo], malla, res, 3, 15)
    chm = pr.generar_chm(xyz, dem, malla, res, 0.025)
    X, Y = np.meshgrid(*pr.centros(malla, res, chm.shape))
    excl = (abs(X - cx) < s + 1.5) & (abs(Y - cy) < s + 1.5) & ~((abs(X-(cx+s*.62))<1.6)&(abs(Y-(cy+s*.62))<1.6))
    # sin excluir la sabina pequeña interior
    p = cp.ParamsArboles(metodo=metodo, **kw)
    t = time.time(); r = cp.detectar_copas(chm, malla, res, p, excluir=excl); dt = time.time() - t
    tab = r['tabla']
    tr = cKDTree(tab[['x', 'y']].values)
    v = verdad[~((abs(verdad.x - (cx)) < s) & (abs(verdad.y - cy) < s) & (verdad.h < 1.5) & False)]
    enc, nd = [], []
    for _, a in v.iterrows():
        idx = tr.query_ball_point([a['x'], a['y']], max(1.0, 0.8 * a['R']))
        enc.append(len(idx) > 0); nd.append(len(idx))
    v = v.assign(enc=enc, nd=nd)
    altos = v[v.h > 8]
    # volumen de árboles aislados
    sin_vec = []
    for _, a in v[v.tipo == 'pino'].iterrows():
        j = tr.query([a['x'], a['y']])[1]
        sin_vec.append(tab.iloc[j]['vol_m3'] / vol_verdad(a) - 1)
    precision = sum(1 for _, t_ in tab.iterrows() if ((np.hypot(v.x - t_['x'], v.y - t_['y']) <= np.maximum(1.0, 0.8 * v.R)).any())) / len(tab)
    print(f"res={res} {metodo:4s} {kw}: det={len(tab)} verdad={len(v)} recall={np.mean(enc):.2f} precisión={precision:.2f} "
          f"| altos {int(altos.enc.sum())}/{len(altos)} (dup>1: {(altos.nd>1).sum()}) | forzados={r['n_forzados']} "
          f"descartes={len(r['descartes'])} | pinos Δvol medio={np.mean(sin_vec)*100:+.1f}% | {dt:.1f}s")
    return r, chm, malla

for res in (0.5, 0.25):
    for metodo, kw in (("hmax", {}), ("lmf", {})):
        r, chm, malla = evalua(res, metodo, **kw)
    r, chm, malla = evalua(res, "hmax", hmin_m=0.3, prominencia_m=0.2)
print()
r, chm, malla = evalua(0.5, "hmax")
tab = r['tabla']
print(tab.head(8).round(2).to_string())
assert (r['labels'] > 0).sum() > 0
# capas
cap = cp.capas_altura(chm, r['labels'], 0.5, 0.5, 0.5, mascara=(chm >= 0.5))
print("capas:", len(cap['bordes']) - 1, "ancho", cap['ancho'])
g = cap['global_']
print(g.head(4).round(2).to_string())
tot_vol = g['vol_estrato_m3'].sum(); tot_cls = g['vol_clase_m3'].sum()
print(f"Σ volumen por estratos = {tot_vol:.1f} m³ ; Σ por clases = {tot_cls:.1f} m³ ; CHM·área = {chm[chm>=0.5].sum()*0.25:.1f}")
assert abs(tot_vol - tot_cls) / tot_cls < 1e-6
vt = cap['vol_capa'].sum(); print(f"Σ vol por árbol×capa = {vt:.1f} ; tabla vol_m3 = {tab.vol_m3.sum():.1f}")
assert abs(cap['vol_capa'].sum() - tab.vol_m3.sum()) / tab.vol_m3.sum() < 1e-6
# vectores
t = time.time()
vec = cp.construir_vectores(malla, 0.5, r['labels'], chm, tab, cap, simplificar=0.1)
print(f"vectores: copas={len(vec['copas'])} capas_arbol={len(vec['capas_arbol'])} capas_global={len(vec['capas_global'])} apices={len(vec['apices'])} ({time.time()-t:.1f}s)")
sv = sum(f['props']['vol_m3'] for f in vec['capas_arbol'])
print(f"Σ vol polígonos árbol×capa = {sv:.1f}")
assert abs(sv - tab.vol_m3.sum()) / tab.vol_m3.sum() < 0.01
print("OK copas")
