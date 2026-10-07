import sys, os, time, pickle, itertools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
import pipeline as pl, copas as cp, campo
SC = "/tmp/claude-0/-home-claude/dc26f558-be35-5e30-9947-b9a58c87f901/scratchpad/real/"
U = "/root/.claude/uploads/dc26f558-be35-5e30-9947-b9a58c87f901/"
D = pickle.load(open(SC + "real25.pkl", "rb")); datos, mod, roi = D['datos'], D['mod'], D['roi']
est = pickle.load(open(SC + "est25.pkl", "rb"))
inv = campo.leer_inventario(U + "6cb67bf1-datos_ojillo_11.xlsx")
inv = inv[roi.puntos(inv.x.values, inv.y.values) & inv.vivo & (inv.altura_m >= 1.3)]
print("campo vivos>=1.3 en parcela:", len(inv))
malla, res = mod['malla_chm'], mod['res_chm']
chm = np.nan_to_num(mod['epocas']['2025']['chm']); roi_m = roi.mascara(malla, res, chm.shape)
excl = pl.mascara_estructuras(mod, est)

def emparejar(df, inv, radio, dx=0, dy=0):
    a = np.c_[df.x, df.y]; b = np.c_[inv.x + dx, inv.y + dy]
    d = np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1])
    d[d > radio] = 1e3
    r, c = linear_sum_assignment(d)
    ok = d[r, c] < 1e3
    return int(ok.sum())

def correr(**kw):
    P = cp.ParamsArboles(**kw)
    r = cp.detectar_copas(chm, malla, res, P, excluir=excl, roi=roi_m)
    return r['tabla']

if __name__ == "__main__":
    base = dict(hmin_m=1.3, hcorte_m=1.3)
    t = correr(**base)
    print("base", len(t))
    # offset campo-LiDAR
    best = None
    for dx in np.arange(-4, 4.1, 1.0):
        for dy in np.arange(-4, 4.1, 1.0):
            m = emparejar(t, inv, 1.5, dx, dy)
            if best is None or m > best[0]: best = (m, dx, dy)
    print("mejor desplazamiento campo (m, dx, dy):", best, " sin desplazar:", emparejar(t, inv, 1.5))
