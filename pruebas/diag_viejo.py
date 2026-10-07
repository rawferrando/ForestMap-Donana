"""Diagnóstico: ¿el detector ANTERIOR pierde individuos altos? (usa el módulo viejo tal cual)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "_viejo"))
import numpy as np, pandas as pd
import sintetico as sy
import procesado_viejo as pv

esc = sy.escenario(epocas=("2024",))
xyz, clase = esc['xyz']['2024'], esc['clase']['2024']
verdad = esc['arboles']['2024']
suelo = clase == 2

for res, (ws, hmin) in [(0.5, (4, 3)), (0.5, (4, 1)), (0.5, (8, 1)), (0.25, (4, 1))]:
    malla = pv.definir_malla(xyz, res)
    dem = pv.generar_dem(xyz[suelo], malla, res, 3, 15)
    chm = pv.generar_chm(xyz, dem, malla, res, 0.025)
    picos = pv.detectar_arboles(chm, res, ws, hmin)
    xs, ys = pv.ejes(malla, res, chm.shape)
    det = pd.DataFrame({'x': xs[picos[:, 1]] + res/2, 'y': ys[picos[:, 0]] - res/2,
                        'h': chm[picos[:, 0], picos[:, 1]]})
    # emparejar con la verdad (<= 0.7 R)
    from scipy.spatial import cKDTree
    t = cKDTree(det[['x', 'y']].values)
    enc = []
    for _, a in verdad.iterrows():
        d, i = t.query([a['x'], a['y']])
        enc.append(d <= max(1.0, 0.7 * a['R']))
    verdad['enc'] = enc
    altos = verdad[verdad['h'] > 8]
    print(f"res={res} ws={ws} hmin={hmin}: detectados={len(det)} (verdad {len(verdad)}) | "
          f"recall total={np.mean(enc):.2f} | altos(>8m) hallados {int(altos['enc'].sum())}/{len(altos)} | "
          f"hmax CHM={np.nanmax(chm):.1f} | dets >8m: {(det['h']>8).sum()}")
    if len(altos) and not altos['enc'].all():
        print("   altos perdidos:\n", altos[~altos['enc']][['x','y','h','R']].round(1).to_string())
    # Duplicados en un mismo árbol alto (sobresegmentación)
    for _, a in altos.iterrows():
        m = np.hypot(det['x'] - a['x'], det['y'] - a['y']) < a['R']
        print(f"   pino h={a['h']:.1f} R={a['R']:.1f}: {int(m.sum())} ápices dentro de su copa (alt máx det {det[m]['h'].max() if m.any() else float('nan'):.1f})")
