"""
sintetico.py — Nube LiDAR SINTÉTICA de un sabinar con antena, vallado y sensores.

SOLO PARA PRUEBAS Y DEMOSTRACIÓN: no son datos reales. Sirve para
  * verificar el código de segmentación / estructuras / informe,
  * enseñar el flujo de trabajo sin necesidad de cargar una nube real.

Devuelve también la "verdad de campo" (posición y altura de cada árbol y de
cada estructura) para poder medir aciertos y errores del algoritmo.
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ORIGEN = (721200.0, 4097200.0)


def terreno(x, y, origen=ORIGEN):
    """Superficie del terreno (m s.n.m.): casi llana, con ondulación suave."""
    dx, dy = x - origen[0], y - origen[1]
    return 17.0 + 0.012 * dx + 0.8 * np.sin(dy / 28.0) + 0.35 * np.cos(dx / 17.0)


# ------------------------------------------------------------------ árboles
def crear_arboles(semilla=7, lado=120.0, origen=ORIGEN, n_sabinas=200, n_pinos=6,
                  recinto=None):
    """Tabla de árboles (verdad). `recinto` = (cx, cy, semilado) zona sin vegetación."""
    rng = np.random.default_rng(semilla)
    filas, centros, radios = [], [], []

    def libre(x, y, R):
        if recinto is not None:
            cx, cy, s = recinto
            if abs(x - cx) < s + R + 0.5 and abs(y - cy) < s + R + 0.5:
                return False
        for (xx, yy), RR in zip(centros, radios):
            if np.hypot(x - xx, y - yy) < 0.55 * (R + RR):
                return False
        return True

    # pinos (individuos altos, > 8 m)
    intentos = 0
    while sum(f['tipo'] == 'pino' for f in filas) < n_pinos and intentos < 5000:
        intentos += 1
        x = origen[0] + rng.uniform(8, lado - 8)
        y = origen[1] + rng.uniform(8, lado - 8)
        H, R = rng.uniform(9.0, 14.0), rng.uniform(2.2, 3.2)
        if libre(x, y, R + 4):
            filas.append(dict(x=x, y=y, h=H, R=R, tipo='pino'))
            centros.append((x, y)); radios.append(R)

    # sabinas (matorral arbóreo bajo)
    intentos = 0
    while sum(f['tipo'] == 'sabina' for f in filas) < n_sabinas and intentos < 40000:
        intentos += 1
        x = origen[0] + rng.uniform(2, lado - 2)
        y = origen[1] + rng.uniform(2, lado - 2)
        H = float(np.clip(rng.lognormal(np.log(2.6), 0.45), 0.9, 6.5))
        R = float(np.clip(H * rng.uniform(0.8, 1.5), 0.7, 4.5))
        if libre(x, y, R):
            filas.append(dict(x=x, y=y, h=H, R=R, tipo='sabina'))
            centros.append((x, y)); radios.append(R)

    # una sabina pequeña DENTRO del recinto (caso de confusión con instrumentos)
    if recinto is not None:
        cx, cy, s = recinto
        filas.append(dict(x=cx + s * 0.62, y=cy + s * 0.62, h=1.4, R=1.1, tipo='sabina'))

    df = pd.DataFrame(filas)
    df.insert(0, 'id', np.arange(1, len(df) + 1))
    return df


def evolucionar(arboles, semilla=11, crecimiento=0.06, mueren=4, nuevos=3, origen=ORIGEN):
    """Siguiente época: crecimiento, algunas muertes y algunos reclutas."""
    rng = np.random.default_rng(semilla)
    df = arboles.copy()
    es_sab = (df['tipo'] == 'sabina').values
    cand = df.index[es_sab & (df['h'] < 3.0).values]
    quitar = rng.choice(cand, size=min(mueren, len(cand)), replace=False)
    df['murio'] = False
    df.loc[quitar, 'murio'] = True
    g = rng.normal(crecimiento, 0.02, len(df))
    df['h'] = df['h'] * (1 + g)
    df['R'] = df['R'] * (1 + 0.5 * g)
    out = df[~df['murio']].drop(columns='murio').copy()
    nid = int(arboles['id'].max()) + 1
    nuevos_l = []
    for _ in range(nuevos):
        for _ in range(200):
            x = origen[0] + rng.uniform(5, 115)
            y = origen[1] + rng.uniform(5, 115)
            d = np.hypot(out['x'] - x, out['y'] - y) - 0.8 * (out['R'] + 1.0)
            if (d > 0).all():
                break
        nuevos_l.append(dict(id=nid, x=x, y=y, h=rng.uniform(1.2, 1.8), R=rng.uniform(0.7, 1.0),
                             tipo='sabina'))
        nid += 1
    out = pd.concat([out, pd.DataFrame(nuevos_l)], ignore_index=True)
    return out


# ------------------------------------------------------------ puntos sueltos
def _arbol(rng, x, y, H, R, tipo, dens):
    n = max(int(dens * np.pi * R * R), 40)
    r = R * np.sqrt(rng.random(n))
    th = rng.random(n) * 2 * np.pi
    px, py = x + r * np.cos(th), y + r * np.sin(th)
    u = r / R
    if tipo == 'pino':
        zb = 0.5 * H
        zs = zb + (H - zb) * np.clip(1 - u ** 2, 0, 1) ** 0.6
    else:
        zb = 0.0
        zs = H * np.clip(1 - u ** 2, 0, 1) ** 0.45
    sup = rng.random(n) < 0.55
    z = np.where(sup, zs - np.abs(rng.normal(0, 0.06, n)), zb + (zs - zb) * rng.random(n))
    z = np.clip(z, 0.02, None)
    pts = [np.column_stack([px, py, z])]
    if tipo == 'pino':                                   # tronco fino
        m = int(18 * zb)
        zt = rng.uniform(0.2, zb, m)
        a = rng.random(m) * 2 * np.pi
        pts.append(np.column_stack([x + 0.14 * np.cos(a), y + 0.14 * np.sin(a), zt]))
    return np.vstack(pts)


def _cilindro(rng, x, y, h, radio, dens_m, z0=0.0):
    n = int(dens_m * (h - z0))
    z = rng.uniform(z0, h, n)
    a = rng.random(n) * 2 * np.pi
    return np.column_stack([x + radio * np.cos(a), y + radio * np.sin(a), z])


def _caja(rng, x, y, lx, ly, h, dens, z0=0.0):
    pts = []
    # tapa
    n = int(dens * lx * ly)
    pts.append(np.column_stack([x + rng.uniform(-lx / 2, lx / 2, n),
                                y + rng.uniform(-ly / 2, ly / 2, n), np.full(n, h)]))
    # paredes
    for (sx, sy, ax) in ((lx / 2, 0, 'x'), (-lx / 2, 0, 'x'), (0, ly / 2, 'y'), (0, -ly / 2, 'y')):
        L = ly if ax == 'x' else lx
        n = int(dens * L * (h - z0) * 0.6)
        t = rng.uniform(-L / 2, L / 2, n)
        zz = rng.uniform(z0, h, n)
        if ax == 'x':
            pts.append(np.column_stack([x + sx + 0 * t, y + t, zz]))
        else:
            pts.append(np.column_stack([x + t, y + sy + 0 * t, zz]))
    return np.vstack(pts)


def estructuras(rng, mast=(721262.0, 4097258.0), altura_antena=16.5, lado_vallado=14.0):
    """Antena + vallado + sensores. Devuelve (puntos HAG, tabla de verdad, recinto)."""
    mx, my = mast
    cx, cy = mx + 1.5, my - 1.0                      # el recinto no está centrado en la antena
    s = lado_vallado / 2
    pts, verdad = [], []

    pts.append(_cilindro(rng, mx, my, altura_antena, 0.12, 220))
    verdad.append(dict(tipo='antena', x=mx, y=my, h=altura_antena))

    # postes y alambrada
    esquinas = [(cx - s, cy - s), (cx + s, cy - s), (cx + s, cy + s), (cx - s, cy + s)]
    n_postes = 0
    for (xa, ya), (xb, yb) in zip(esquinas, esquinas[1:] + esquinas[:1]):
        L = np.hypot(xb - xa, yb - ya)
        k = int(round(L / 3.0))
        for i in range(k):
            t = i / k
            px, py = xa + (xb - xa) * t, ya + (yb - ya) * t
            pts.append(_cilindro(rng, px, py, 1.8, 0.04, 70))
            n_postes += 1
        for zz in (0.4, 0.9, 1.4, 1.8):             # hilos de la malla
            n = int(35 * L)
            t = rng.random(n)
            pts.append(np.column_stack([xa + (xb - xa) * t, ya + (yb - ya) * t,
                                        zz + rng.normal(0, 0.01, n)]))
    verdad.append(dict(tipo='vallado', x=cx, y=cy, h=1.8, n_postes=n_postes))

    # instrumentos
    pts.append(_caja(rng, mx + 2.2, my - 1.0, 0.6, 0.4, 0.8, 450))
    verdad.append(dict(tipo='caja_datalogger', x=mx + 2.2, y=my - 1.0, h=0.8))
    pts.append(_cilindro(rng, mx - 2.5, my + 2.0, 3.5, 0.035, 120))
    pts.append(_caja(rng, mx - 2.5, my + 2.0, 0.8, 0.05, 3.5, 300, z0=3.3))
    verdad.append(dict(tipo='mastil_sensor', x=mx - 2.5, y=my + 2.0, h=3.5))
    # panel solar inclinado sobre poste
    pts.append(_cilindro(rng, mx + 3.0, my + 3.0, 2.2, 0.04, 100))
    n = 700
    u, v = rng.uniform(-0.5, 0.5, n), rng.uniform(-0.35, 0.35, n)
    pts.append(np.column_stack([mx + 3.0 + u, my + 3.0 + v * np.cos(0.5),
                                2.2 + 0.25 + v * np.sin(0.5)]))
    verdad.append(dict(tipo='panel_solar', x=mx + 3.0, y=my + 3.0, h=2.6))
    pts.append(_cilindro(rng, mx + 1.0, my + 4.0, 1.0, 0.15, 400))
    verdad.append(dict(tipo='pluviometro', x=mx + 1.0, y=my + 4.0, h=1.0))

    return np.vstack(pts), pd.DataFrame(verdad), (cx, cy, s)


# --------------------------------------------------------------- nube completa
def nube(arboles, con_estructuras=True, semilla=3, lado=120.0, origen=ORIGEN,
         dens_suelo=18, dens_copa=55, mast=(721262.0, 4097258.0), altura_antena=16.5):
    """xyz (N,3, z absoluta), clase (2 suelo / 3 vegetación / 15 estructura), verdad estructuras."""
    rng = np.random.default_rng(semilla)

    # suelo (menos retornos bajo las copas)
    n = int(dens_suelo * lado * lado)
    gx = origen[0] + rng.uniform(0, lado, n)
    gy = origen[1] + rng.uniform(0, lado, n)
    keep = np.ones(n, bool)
    tree = cKDTree(np.column_stack([gx, gy]))
    for _, a in arboles.iterrows():
        idx = tree.query_ball_point([a['x'], a['y']], a['R'])
        if idx:
            idx = np.asarray(idx)
            keep[idx[rng.random(len(idx)) < 0.75]] = False
    gx, gy = gx[keep], gy[keep]
    gz = terreno(gx, gy, origen) + rng.normal(0, 0.015, len(gx))
    partes = [np.column_stack([gx, gy, gz])]
    clases = [np.full(len(gx), 2, np.int8)]

    # vegetación
    for _, a in arboles.iterrows():
        p = _arbol(rng, a['x'], a['y'], a['h'], a['R'], a['tipo'], dens_copa)
        p[:, 2] += terreno(p[:, 0], p[:, 1], origen)
        partes.append(p)
        clases.append(np.full(len(p), 3, np.int8))

    verdad_est = pd.DataFrame()
    recinto = None
    if con_estructuras:
        p, verdad_est, recinto = estructuras(rng, mast=mast, altura_antena=altura_antena)
        p[:, 2] += terreno(p[:, 0], p[:, 1], origen)
        partes.append(p)
        clases.append(np.full(len(p), 15, np.int8))

    xyz = np.vstack(partes)
    clase = np.concatenate(clases)
    # ruido instrumental pequeño
    xyz[:, 2] += rng.normal(0, 0.01, len(xyz))
    return xyz, clase, verdad_est, recinto


def escenario(semilla=7, epocas=("2024", "2026")):
    """Escenario completo: dos épocas + verdad. Devuelve dict listo para usar."""
    mast = (721262.0, 4097258.0)
    # recinto aproximado para vaciar de vegetación la zona de la antena
    rec = (mast[0] + 1.5, mast[1] - 1.0, 7.0)
    a1 = crear_arboles(semilla=semilla, recinto=rec)
    out = {'arboles': {}, 'xyz': {}, 'clase': {}, 'verdad_est': None, 'recinto': rec}
    actual = a1
    for i, e in enumerate(epocas):
        if i > 0:
            actual = evolucionar(actual, semilla=semilla + i)
        xyz, clase, vest, _ = nube(actual, semilla=semilla + 10 * i, mast=mast)
        out['arboles'][e] = actual
        out['xyz'][e] = xyz
        out['clase'][e] = clase
        if out['verdad_est'] is None:
            out['verdad_est'] = vest
    return out


if __name__ == "__main__":
    esc = escenario()
    for e, x in esc['xyz'].items():
        print(e, x.shape, "árboles:", len(esc['arboles'][e]),
              "altos >8 m:", int((esc['arboles'][e]['h'] > 8).sum()))
