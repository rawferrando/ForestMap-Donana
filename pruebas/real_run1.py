import sys, os, time, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pipeline as pl, copas as cp, campo, figuras as fg
SC = "/tmp/claude-0/-home-claude/dc26f558-be35-5e30-9947-b9a58c87f901/scratchpad/real/"
U = "/root/.claude/uploads/dc26f558-be35-5e30-9947-b9a58c87f901/"
D = pickle.load(open(SC + "real25.pkl", "rb")); datos, mod, roi = D['datos'], D['mod'], D['roi']
t = time.time(); est = pl.detectar_estructuras_epoca(mod, datos, "2025"); print("estructuras", time.time()-t)
print(est.tabla[['id','tipo','x','y','h_m','longitud_m','confianza']].to_string()); print(est.log)
pickle.dump(est, open(SC + "est25.pkl", "wb"))
P = cp.ParamsArboles()
t = time.time(); A = pl.analizar_arboles(mod, datos, "2025", P, roi, est=est); print("arboles", time.time()-t)
r = A['rodal']; print({k: (round(v,2) if isinstance(v,float) else v) for k,v in r.items() if k!='clases_altura'}); print(r['clases_altura'])
open(SC+"capas.png","wb").write(fg.mapa_capas(A, mod, roi, est))
