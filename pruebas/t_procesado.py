import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import sintetico as sy
import procesado as pr

esc = sy.escenario(epocas=("2024",))
xyz, clase = esc['xyz']['2024'], esc['clase']['2024']
verdad_suelo = clase == 2

# 1) suelo morfológico
t = time.time()
suelo = pr.clasificar_suelo_morfologico(xyz)
print(f"suelo morfológico: {time.time()-t:.1f}s  ", end="")
tp = (suelo & verdad_suelo).sum(); fp = (suelo & ~verdad_suelo).sum(); fn = (~suelo & verdad_suelo).sum()
hag_v = xyz[:, 2] - sy.terreno(xyz[:, 0], xyz[:, 1])
fp_alto = (suelo & ~verdad_suelo & (hag_v > 0.5)).sum()
print(f"recall={tp/(tp+fn):.3f} precisión={tp/(tp+fp):.3f}  falsos suelo={fp} (a >0.5 m del suelo real: {fp_alto})")
assert tp/(tp+fn) > 0.9 and fp_alto < 0.002 * len(xyz)

# 2) submuestreo
sub = pr.submuestrear(xyz, 30)
print("submuestreo 30 pts/m²:", len(xyz), "->", len(sub))
assert len(sub) < len(xyz)

# 3) DEM, CHM vs verdad
res = 0.5
malla = pr.definir_malla(xyz, res)
t = time.time()
dem = pr.generar_dem(xyz[suelo], malla, res, 3, 15)
chm = pr.generar_chm(xyz, dem, malla, res, 0.025)
print(f"DEM+CHM: {time.time()-t:.1f}s  shape={chm.shape}  hmax={chm.max():.2f}")
cx, cy = pr.centros(malla, res, dem.shape)
X, Y = np.meshgrid(cx, cy)
err = dem - sy.terreno(X, Y)
print(f"error DEM vs verdad: media={err.mean():+.3f} rmse={np.sqrt((err**2).mean()):.3f} p95|e|={np.percentile(abs(err),95):.3f}")
assert np.sqrt((err**2).mean()) < 0.35
# altura verdadera de pinos
v = esc['arboles']['2024']
for _, a in v[v.tipo=='pino'].head(3).iterrows():
    h = pr.muestrear(chm, malla, res, [a['x']], [a['y']])[0]
    print(f"  pino verdad h={a['h']:.2f}  CHM={h:.2f}")
# 4) HAG
hag = pr.altura_sobre_suelo(xyz, dem, malla, res)
print("HAG max:", hag.max().round(2), " p99.9:", np.percentile(hag, 99.9).round(2))
# 5) reducir
r, xs, ys = pr.reducir_ejes(chm, *pr.ejes(malla, res, chm.shape), max_px=100)
print("reducir:", chm.shape, "->", r.shape, len(xs), len(ys))
assert r.shape == (len(ys), len(xs))
print("OK procesado")
