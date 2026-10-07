import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pickle
import procesado as pr, pipeline as pl, roi as rm, campo
SC = "/tmp/claude-0/-home-claude/dc26f558-be35-5e30-9947-b9a58c87f901/scratchpad/real/"
U = "/root/.claude/uploads/dc26f558-be35-5e30-9947-b9a58c87f901/"
xyz = np.load(SC + "xyz25.npy"); cl = np.load(SC + "cl25.npy")
p = rm.leer_poligono(U + "aab8c3dd-parcela.gpkg", "parcela.gpkg")
roi = rm.Roi(p['bbox'], anillos=p['anillos'], nombre="Ojillo")
b = roi.bounds; m = 12
sel = (xyz[:,0]>b[0]-m)&(xyz[:,0]<b[1]+m)&(xyz[:,1]>b[2]-m)&(xyz[:,1]<b[3]+m)
xyz, cl = xyz[sel], cl[sel]
print("puntos con margen:", len(xyz), "suelo orig:", (cl==2).mean())
suelo_orig = cl == 2
t = time.time(); suelo_m = pr.clasificar_suelo_morfologico(xyz); print("morf", time.time()-t, suelo_m.mean())
datos = {"2025": dict(xyz=xyz, suelo=suelo_orig)}
t = time.time(); mod = pl.generar_modelos(datos, 1.0, 0.5); print("modelos", time.time()-t)
ep = mod['epocas']['2025']; chm = np.nan_to_num(ep['chm'])
print("CHM pct:", np.percentile(chm, [50, 90, 99, 99.9]), chm.max())
pickle.dump(dict(datos=datos, mod=mod, roi=roi, suelo_m=suelo_m), open(SC + "real25.pkl", "wb"))
