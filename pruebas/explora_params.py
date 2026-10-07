import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
import sintetico as sy, procesado as pr, copas as cp

esc = sy.escenario(epocas=("2024",))
xyz = esc['xyz']['2024']; verdad = esc['arboles']['2024']
suelo = pr.clasificar_suelo_morfologico(xyz)
cx, cy, s = esc['recinto']
res = 0.5
malla = pr.definir_malla(xyz, res)
dem = pr.generar_dem(xyz[suelo], malla, res, 3, 15)
chm = pr.generar_chm(xyz, dem, malla, res, 0.025)
X, Y = np.meshgrid(*pr.centros(malla, res, chm.shape))
excl = (abs(X - cx) < s + 1.5) & (abs(Y - cy) < s + 1.5) & ~((abs(X-(cx+s*.62))<1.6)&(abs(Y-(cy+s*.62))<1.6))

def run(**kw):
    p = cp.ParamsArboles(**kw); r = cp.detectar_copas(chm, malla, res, p, excluir=excl); tab = r['tabla']
    tr = cKDTree(tab[['x','y']].values)
    enc = [len(tr.query_ball_point([a.x, a.y], max(1.0, .8*a.R))) > 0 for a in verdad.itertuples()]
    return len(tab), np.mean(enc), verdad.assign(enc=enc)
for kw in [dict(), dict(sigma_m=0.3, prominencia_m=0.15), dict(sigma_m=0.2, prominencia_m=0.10),
           dict(sigma_m=0.3, prominencia_m=0.2, hmin_m=0.5), dict(metodo='lmf', sigma_m=0.3)]:
    n, rec, v = run(**kw)
    print(kw, n, f"recall={rec:.2f}")
n, rec, v = run(sigma_m=0.3, prominencia_m=0.15)
perd = v[~v.enc]
print("perdidos por clase de altura:\n", pd.cut(perd.h, [0,1,1.5,2,3,4,100]).value_counts().sort_index().to_string())
# solape con vecino más alto
tr = cKDTree(verdad[['x','y']].values)
sol = []
for a in perd.itertuples():
    idx = [i for i in tr.query_ball_point([a.x, a.y], 8) if verdad.iloc[i].id != a.id]
    m = max([ (a.R + verdad.iloc[i].R - np.hypot(a.x-verdad.iloc[i].x, a.y-verdad.iloc[i].y)) / (2*a.R) for i in idx], default=0)
    sol.append(m)
print("solape máx. con vecino (fracción del diámetro) de los perdidos: mediana", np.median(sol).round(2),
      "| de los hallados:", end=" ")
enc = v[v.enc]; sol2=[]
for a in enc.itertuples():
    idx = [i for i in tr.query_ball_point([a.x, a.y], 8) if verdad.iloc[i].id != a.id]
    sol2.append(max([(a.R + verdad.iloc[i].R - np.hypot(a.x-verdad.iloc[i].x, a.y-verdad.iloc[i].y)) / (2*a.R) for i in idx], default=0))
print(np.median(sol2).round(2))
