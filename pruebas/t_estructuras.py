import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
import sintetico as sy, procesado as pr, estructuras as es

esc = sy.escenario(epocas=("2024",))
xyz, clase = esc['xyz']['2024'], esc['clase']['2024']
suelo = pr.clasificar_suelo_morfologico(xyz)
res = 0.5
malla = pr.definir_malla(xyz, res)
dem = pr.generar_dem(xyz[suelo], malla, res, 3, 15)
hag = pr.altura_sobre_suelo(xyz, dem, malla, res)
ns = ~suelo
t = time.time()
est = es.detectar_estructuras(xyz[ns, 0], xyz[ns, 1], xyz[ns, 2], hag[ns])
print(f"{time.time()-t:.1f}s")
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 70)
print(est.tabla[['id','tipo','x','y','h_m','longitud_m','area_m2','ancho_m','n_puntos','confianza','excluir']].round(2).to_string())
print("\n".join(est.log))
V = esc['verdad_est']
print(V.round(2).to_string())

tb = est.tabla
ant = tb[tb.tipo.str.startswith('Antena')]
assert len(ant) == 1, f"debía haber 1 antena y hay {len(ant)}"
a = ant.iloc[0]; va = V[V.tipo == 'antena'].iloc[0]
print(f"\nANTENA: x,y error = {np.hypot(a.x-va.x, a.y-va.y):.2f} m ; altura {a.h_m:.2f} vs verdad {va.h:.2f}")
assert np.hypot(a.x-va.x, a.y-va.y) < 0.3 and abs(a.h_m - va.h) < 0.3
# ningún pino se ha tomado por antena
pinos = esc['arboles']['2024']; pinos = pinos[pinos.tipo == 'pino']
for _, p_ in pinos.iterrows():
    assert not ((np.hypot(tb.x - p_.x, tb.y - p_.y) < 2) & tb.tipo.str.startswith('Antena')).any()
# vallado
v = tb[tb.tipo.str.startswith('Vallado')]
print("vallado:", v[['tipo','longitud_m','confianza']].round(1).values.tolist(), "(perímetro verdad: 56 m)")
assert len(v) >= 1
# instrumentos
ins = tb[tb.tipo.isin(['Instrumento / equipo', 'Poste / mástil de sensor'])]
print("instrumentos:", len(ins), "| verdad:", int((~V.tipo.isin(['antena','vallado'])).sum()))
# máscara
m = es.mascara_raster(est, malla, res, dem.shape)
print("máscara: celdas", int(m.sum()), "=", m.sum()*res*res, "m²")
# la sabina de dentro del recinto NO debe quedar enmascarada
cx, cy, s = esc['recinto']
sx, sy_ = cx + s*0.62, cy + s*0.62
c = int((sx - malla[0]) / res); r = int((malla[1] - sy_) / res)
print("sabina interior enmascarada:", bool(m[r, c]))
lab = es.etiquetar_puntos(xyz[ns,0], xyz[ns,1], est)
print("puntos etiquetados:", int((lab>0).sum()))
print("OK estructuras")
