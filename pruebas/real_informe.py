import sys, os, time, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pipeline as pl, copas as cp, campo, informe, exportar
SC = "/tmp/claude-0/-home-claude/dc26f558-be35-5e30-9947-b9a58c87f901/scratchpad/real/"
U = "/root/.claude/uploads/dc26f558-be35-5e30-9947-b9a58c87f901/"
D = pickle.load(open(SC + "real25.pkl", "rb")); datos, mod, roi = D['datos'], D['mod'], D['roi']
roi.nombre = "Parcela Ojillo (parcela.gpkg)"
est = pl.detectar_estructuras_epoca(mod, datos, "2025")
t = time.time(); A = pl.analizar_arboles(mod, datos, "2025", cp.ParamsArboles(**eval(os.environ.get("PARAMS","{}"))), roi, est=est); print("árboles", time.time() - t, len(A['tabla']))
inv = campo.leer_inventario(U + "6cb67bf1-datos_ojillo_11.xlsx")
dx, dy, n = campo.estimar_desplazamiento(A['tabla'], inv[inv.vivo & (inv.altura_m >= 1.3)])
cmp_, cdf = campo.comparar_con_campo(A['tabla'], A['labels'], mod['malla_chm'], mod['res_chm'], inv, roi, (dx, dy))
ficha = campo.resumen_campo(inv.assign(x=inv.x + dx, y=inv.y + dy), roi)
cam = dict(cmp=cmp_, df=cdf, anio=2011, desplaza=(dx, dy), ficha=ficha)
t = time.time()
h = informe.generar_informe(mod, datos, roi, {"2025": est}, {"2025": A}, None, meta=dict(titulo="Sabinar del Ojillo (parcela de seguimiento)", crs="EPSG:25829"), campo=cam)
print("informe", time.time() - t, len(h) // 1024, "KB")
open(sys.argv[1], "w", encoding="utf-8").write(h)
pickle.dump(dict(A=A, est=est, cam=cam), open(SC + "res_real.pkl", "wb"))
