import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
import sintetico as sy, procesado as pr, estructuras as es, copas as cp, pipeline as pl
from roi import Roi, construir_roi, extension

esc = sy.escenario(epocas=("2024", "2026"))
datos = {}
t0 = time.time()
for e in ("2024", "2026"):
    datos[e] = dict(xyz=esc['xyz'][e], suelo=pr.clasificar_suelo_morfologico(esc['xyz'][e]))
mod = pl.generar_modelos(datos, res_dem=1.0, res_chm=0.5)
print(f"modelos: {time.time()-t0:.1f}s; épocas {list(mod['epocas'])}; dh pares {list(mod['dh'])}")

# ROI: cuadrado de ~0,99 ha girado 14° (como la parcela Ojillo), centrado en la nube
ext = extension([d['xyz'] for d in datos.values()])
cx, cy = (ext[0]+ext[1])/2, (ext[2]+ext[3])/2
L = 99.4/2; th = np.radians(14)
R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
esq = np.array([[-L,-L],[L,-L],[L,L],[-L,L],[-L,-L]]) @ R.T + [cx, cy]
roi = Roi((esq[:,0].min(), esq[:,0].max(), esq[:,1].min(), esq[:,1].max()), anillos=[esq], nombre="Parcela girada")
print(roi.descripcion())
assert abs(roi.area_ha() - 0.988) < 0.01

estr = {}
for e in datos:
    estr[e] = pl.detectar_estructuras_epoca(mod, datos, e)
    print(e, "estructuras:", len(estr[e].tabla), "| antena h =", estr[e].tabla[estr[e].tabla.tipo.str.startswith('Antena')].h_m.round(2).tolist())

arb = {}
for e in datos:
    t = time.time()
    arb[e] = pl.analizar_arboles(mod, datos, e, cp.ParamsArboles(), roi, est=estr[e])
    r = arb[e]['rodal']
    print(f"{e}: {r['n_arboles']} árboles en {r['area_roi_ha']:.3f} ha → {r['densidad_ha']:.0f}/ha | H media {r['h_media']:.2f} máx {r['h_max']:.2f} "
          f"| altos≥8: {r['n_altos']} | cobertura {r['cobertura_pct']:.1f}% | vol {r['volumen_total_m3']:.0f} m³ | forzados {r['n_forzados']} ({time.time()-t:.1f}s)")

# 1) Verdad: árboles dentro del ROI
for e in datos:
    v = esc['arboles'][e]
    dentro = roi.puntos(v.x.values, v.y.values)
    # excluir el de dentro del recinto (se queda como árbol), nada que quitar
    print(f"  verdad {e}: {int(dentro.sum())} árboles dentro; detectados {len(arb[e]['tabla'])}; altos verdad {int(((v.h>=8)&dentro).sum())}, detectados {arb[e]['rodal']['n_altos']}")
    assert arb[e]['rodal']['n_altos'] == int(((v.h >= 8) & dentro).sum()), "se ha perdido o creado algún individuo alto"
# la antena (16.5 m) no es árbol
for e in datos:
    assert arb[e]['tabla'].h_m.max() < 14.5, "la antena se ha colado como árbol"
    assert len(arb[e]['labels'][arb[e]['excl'] & (arb[e]['labels']>0)]) == 0

# 2) Clipping por ROI
lab = arb['2024']['labels']; m = arb['2024']['roi_mask']
assert (lab[~m] == 0).all()
print("labels fuera del ROI = 0 ✓")

# 3) Comparación
comp = pl.comparar_epocas(mod, arb, "2024", "2026", roi)
r = comp['resumen']
print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
v1, v2 = esc['arboles']['2024'], esc['arboles']['2026']
muertos = set(v1.id) - set(v2.id); nuevos = set(v2.id) - set(v1.id)
din = lambda v, ids: int(roi.puntos(v[v.id.isin(ids)].x.values, v[v.id.isin(ids)].y.values).sum())
print(f"verdad: mueren {din(v1, muertos)} dentro del ROI, nuevos {din(v2, nuevos)} dentro del ROI")
print(f"dh medio persistentes: {r['dh_medio']:.3f} m (verdad: crecimiento ≈ 6 % de la altura media = {v1.h.mean()*0.06:.2f} m)")
assert 0.05 < r['dh_medio'] < 0.4
print(comp['tabla_dh'].head(3).to_string())
print(comp['df_a'].estado.value_counts().to_string())
t = time.time()
print("OK pipeline")
