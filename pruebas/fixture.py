"""Escenario sintético completo (2 épocas) con ROI girado; para pruebas de figuras, informe y exportación."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import sintetico as sy, procesado as pr, copas as cp, pipeline as pl
from roi import Roi, extension


def construir(epocas=("2024", "2026")):
    esc = sy.escenario(epocas=epocas)
    datos = {e: dict(xyz=esc['xyz'][e], suelo=pr.clasificar_suelo_morfologico(esc['xyz'][e])) for e in epocas}
    mod = pl.generar_modelos(datos, 1.0, 0.5)
    ext = extension([d['xyz'] for d in datos.values()])
    cx, cy = (ext[0] + ext[1]) / 2, (ext[2] + ext[3]) / 2
    L = 99.4 / 2; th = np.radians(14)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    esq = np.array([[-L, -L], [L, -L], [L, L], [-L, L], [-L, -L]]) @ R.T + [cx, cy]
    roi = Roi((esq[:, 0].min(), esq[:, 0].max(), esq[:, 1].min(), esq[:, 1].max()), anillos=[esq], nombre="Parcela Ojillo (sintética)")
    estr = {e: pl.detectar_estructuras_epoca(mod, datos, e) for e in epocas}
    arb = {e: pl.analizar_arboles(mod, datos, e, cp.ParamsArboles(), roi, est=estr[e]) for e in epocas}
    comp = pl.comparar_epocas(mod, arb, epocas[0], epocas[-1], roi)
    return dict(esc=esc, datos=datos, mod=mod, roi=roi, estr=estr, arb=arb, comp=comp)
