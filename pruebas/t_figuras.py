import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import fixture, figuras as fg, procesado as pr
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
F = fixture.construir()
mod, roi, arb, estr, comp, datos = F['mod'], F['roi'], F['arb'], F['estr'], F['comp'], F['datos']
e = "2026"; A = arb[e]; ep = mod['epocas'][e]; m, r = mod['malla_chm'], mod['res_chm']
t = estr[e].tabla; ant = t[t.tipo.str.startswith('Antena')].iloc[0]
c = dict(
 capas=lambda: fg.mapa_capas(A, mod, roi, estr[e]),
 chm=lambda: fg.mapa_raster(ep['chm'], m, r, roi, "CHM", "viridis", "m"),
 dem=lambda: fg.mapa_dem_hillshade(ep['dem'], ep['hs'], mod['malla_dem'], mod['res_dem'], roi, "DEM"),
 suelo=lambda: fg.mapa_densidad_suelo(datos[e]['xyz'], datos[e]['suelo'], m, 1.0, roi, "suelo"),
 hist=lambda: fg.hist_alturas(A['tabla'], 8),
 disp=lambda: fg.dispersion_copas(A['tabla']),
 vcapa=lambda: fg.volumen_por_capa(A['cap']['global_'], 0.5),
 clases=lambda: fg.clases_altura(A['rodal']['clases_altura']),
 estruct=lambda: fg.plano_estructuras(datos[e]['xyz'], ep['hag'], datos[e]['suelo'], estr[e]),
 perfil=lambda: fg.perfil_vertical(datos[e]['xyz'], ep['hag'], datos[e]['suelo'], (ant.x-25, ant.y), (ant.x+25, ant.y), 4.0, estr[e]),
 estados=lambda: fg.mapa_estados(comp, m, r, roi, A['chm_eff'], "estados"),
 dispH=lambda: fg.dispersion_h(comp),
 demo=lambda: fg.barras_demografia(comp),
 dh=lambda: fg.mapa_raster(comp['dh'], m, r, roi, "dH", etiqueta="m", divergente=True),
)
import numpy as np
d = datos[e]; hag_pts = pr.muestrear(ep['hag'], d['xyz'][:,0], d['xyz'][:,1], m, r) if False else None
for k, f in c.items():
    try:
        b = f()
        if b: open(f"{out}/{k}.png", "wb").write(b); print(k, len(b))
    except Exception as ex:
        import traceback; traceback.print_exc(); print("FALLA", k)
