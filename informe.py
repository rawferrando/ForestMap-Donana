"""
informe.py — Informe técnico autocontenido (HTML con figuras incrustadas, listo para imprimir a PDF).
"""
import base64
import datetime as dt
import html

import numpy as np
import pandas as pd

import figuras as fg

CSS = """
:root{--v:#2e6930;--g:#555;--bg:#fafaf7}
*{box-sizing:border-box}body{font-family:'Segoe UI',Helvetica,Arial,sans-serif;color:#222;background:#fff;margin:0;line-height:1.5}
.pag{max-width:960px;margin:0 auto;padding:28px 36px}
h1{font-size:30px;color:var(--v);margin:0}h2{color:var(--v);border-bottom:2px solid var(--v);padding-bottom:4px;margin-top:38px;font-size:21px}
h3{font-size:16px;margin:22px 0 6px}.sub{color:var(--g)}
.portada{padding:70px 0 40px;border-bottom:6px solid var(--v)}
.portada img{width:100%;border-radius:6px;margin-top:18px}.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:14px 0}
.kpi{background:var(--bg);border:1px solid #e1e1d8;border-left:4px solid var(--v);padding:10px 12px;border-radius:4px}
.kpi b{display:block;font-size:22px;color:var(--v)}.kpi span{font-size:12px;color:var(--g)}
table{border-collapse:collapse;width:100%;font-size:12.5px;margin:8px 0}th{background:var(--v);color:#fff;text-align:left;padding:5px 8px}
td{padding:4px 8px;border-bottom:1px solid #e5e5e0}tr:nth-child(even) td{background:var(--bg)}
figure{margin:14px 0;text-align:center}figure img{max-width:100%}figcaption{font-size:12px;color:var(--g);margin-top:4px;text-align:left}
.dos{display:grid;grid-template-columns:1fr 1fr;gap:12px}.aviso{background:#fff6e5;border-left:4px solid #e67e22;padding:8px 12px;margin:10px 0;font-size:13px}
.ok{background:#eaf5ea;border-left:4px solid var(--v);padding:8px 12px;margin:10px 0;font-size:13px}
.sint{background:#fdecea;border:2px solid #c0392b;color:#7b1d14;padding:10px 14px;font-weight:600;margin:14px 0}
code{background:#f0f0ea;padding:1px 4px;border-radius:3px}small{color:var(--g)}
@media print{.pag{padding:10mm}h2{page-break-after:avoid}figure,table,.kpis{page-break-inside:avoid}.salto{page-break-before:always}}
@media(max-width:700px){.dos{grid-template-columns:1fr}}
"""


def _img(png, pie):
    if png is None:
        return ""
    b = base64.b64encode(png).decode()
    return f'<figure><img src="data:image/png;base64,{b}"><figcaption>{html.escape(pie)}</figcaption></figure>'


def _tabla(df, cols=None, fmt=None, max_filas=None, rename=None):
    d = df if cols is None else df[cols]
    if max_filas:
        d = d.head(max_filas)
    fmt = fmt or {}
    rename = rename or {}
    h = "<table><tr>" + "".join(f"<th>{html.escape(rename.get(c, c))}</th>" for c in d.columns) + "</tr>"
    for _, f in d.iterrows():
        h += "<tr>"
        for c in d.columns:
            v = f[c]
            if c in fmt and pd.notna(v):
                v = fmt[c].format(v)
            elif isinstance(v, (float, np.floating)):
                v = f"{v:.2f}"
            h += f"<td>{html.escape(str(v))}</td>"
        h += "</tr>"
    return h + "</table>"


def _kpi(valor, etiqueta):
    return f'<div class="kpi"><b>{valor}</b><span>{html.escape(etiqueta)}</span></div>'


ALGORITMOS = [
    ("Lectura de la nube", "LAS 1.0–1.4 (laspy; lector propio de reserva)", "Submuestreo opcional por celdas de 1 m; clase 2 original opcional como suelo."),
    ("Clasificación de suelo", "CSF — Cloth Simulation Filter (Zhang et al. 2016); filtro morfológico progresivo de reserva (Zhang et al. 2003)",
     "Equivalente al PMF del script R (lidR::pmf)."),
    ("MDT (DEM)", "Mínimo por celda + interpolación lineal (SciPy) + filtro de mínimos y de media", "El script R usa interpolación MBA (B-splines multinivel)."),
    ("MDS y CHM", "MDS = máximo por celda; CHM = MDS − MDT con relleno de huecos («pit filling»)", "Equivalente a rasterize_canopy / pitfree (Khosravipour et al. 2014)."),
    ("Ápices de copa", "Transformada de h-máximos por prominencia (Soille &amp; Vincent) sobre CHM con suavizado gaussiano; alternativa: filtro de máximos locales de ventana variable, LMF (Popescu &amp; Wynne 2004)",
     "El script R usa LMF con ventana 1/2/4/6 m según la altura."),
    ("Delimitación de copas", "Cuencas hidrográficas con marcadores (watershed, Vincent &amp; Soille 1991), poda por altura relativa, 2.ª pasada y fusión de pinos grandes",
     "Equivalente a segment_trees (silva2016 / dalponte) del script R."),
    ("Pinos y sabinas", "Regla de altura: ápice ≥ umbral (5 m) = pino; resto = sabina", "El script R usa 6,5 m."),
    ("Vectorización y capas de altura", "Trazado exacto ráster→polígono por clase (4-conectividad, huecos e islas) por árbol × capa de 0,5 m", "Polígonos aptos para QGIS."),
    ("Volumen de copa", "Envolvente CHM: Σ(h·A_celda); volumen ocupado por vóxeles de 0,25 m", "El protocolo ICTS usa alpha-shapes 3D: ver limitaciones."),
    ("Infraestructura", "Detección de columnas expuestas (mástiles), ocupación planimétrica 0,25 m, esqueleto y número de Euler (vallado)", "Heurística; revisión manual."),
    ("Emparejamiento interanual", "Asignación óptima (algoritmo húngaro, Kuhn 1955) con ápices a ≤ 1,2 m (cKDTree, protocolo ICTS) o copas solapadas", "Estados: persiste, mortalidad probable, nuevo."),
    ("Validación con campo", "Estimación del desplazamiento del inventario + asignación húngara a ≤ 2 m", "Solo si se carga un inventario de campo."),
]


def generar_informe(modelos, datos, roi, estr, arb, comp=None, tablas_est=None, meta=None, sintetico=False,
                    et_principal=None, campo=None):
    """Devuelve el informe como cadena HTML.

    estr: {época: Estructuras};  arb: {época: resultado de analizar_arboles};  comp: resultado de comparar_epocas.
    tablas_est: {época: DataFrame editado} (con 'excluir').
    campo: dict opcional(cmp=cifras, df=tabla de campo con estado, anio=2011, desplaza=(dx,dy), ficha=cifras del inventario).
    """
    meta = meta or {}
    tablas_est = tablas_est or {}
    epocas = list(arb.keys())
    et = et_principal or epocas[-1]
    A = arb[et]
    p = A['params']
    r = A['rodal']
    tab = A['tabla']
    m, res = modelos['malla_chm'], modelos['res_chm']
    ep = modelos['epocas'][et]
    d = datos[et]
    fecha = dt.date.today().strftime("%d/%m/%Y")
    n_pino = int((tab['especie'] == 'Pino').sum()) if 'especie' in tab else 0
    n_sab = len(tab) - n_pino
    h = [f"<!doctype html><html lang='es'><head><meta charset='utf-8'><title>Informe ForestMap Doñana</title>"
         f"<meta name='viewport' content='width=device-width,initial-scale=1'><style>{CSS}</style></head><body><div class='pag'>"]

    # ---------------------------------------------------------- portada
    vista = fg.vista_nube_3d(d['xyz'], ep['hag'], d['suelo'], roi, titulo=f"Nube LiDAR {et} — altura sobre el suelo",
                             estructuras=estr.get(et))
    h.append("<div class='portada'><div class='sub'>Laboratorio de SIG y Teledetección · Estación Biológica de Doñana (LAST-EBD)</div>"
             f"<h1>Inventario forestal individual con LiDAR</h1><h3 class='sub'>{html.escape(meta.get('titulo', 'Sabinar de Juniperus phoenicea subsp. turbinata'))}</h3>"
             f"<p>Parcela: <b>{html.escape(roi.descripcion())}</b><br>Épocas: {', '.join(epocas)} · CRS: {html.escape(str(meta.get('crs', 'EPSG:25829')))}"
             f"<br>Generado el {fecha} con ForestMap Doñana V3"
             f"{'<br>Autor/a: ' + html.escape(meta['autor']) if meta.get('autor') else ''}</p>"
             f"<img src='data:image/png;base64,{base64.b64encode(vista).decode()}'></div>")
    if sintetico:
        h.append("<div class='sint'>⚠ INFORME DE DEMOSTRACIÓN: generado con una nube de puntos SINTÉTICA (no son datos reales de Doñana). "
                 "Sirve únicamente para comprobar el flujo de trabajo y el formato.</div>")

    # ---------------------------------------------------------- resumen
    ant = None
    te = tablas_est.get(et, estr[et].tabla if estr.get(et) is not None else None)
    if te is not None and len(te):
        a = te[te['tipo'].str.startswith('Antena')]
        ant = a.iloc[0] if len(a) else None
    h.append("<h2>1. Resumen ejecutivo</h2><div class='kpis'>" +
             _kpi(r['n_arboles'], f"individuos en {r['area_roi_ha']:.3f} ha") +
             _kpi(n_sab, f"sabinas (ápice &lt; {p['h_pino_m']:g} m)") +
             _kpi(n_pino, f"pinos (ápice ≥ {p['h_pino_m']:g} m)") +
             _kpi(f"{r['densidad_ha']:.0f}", "individuos / ha") +
             _kpi(f"{r['h_media']:.2f} m", "altura media") +
             _kpi(f"{r['h_max']:.1f} m", "altura máxima (árbol)") +
             _kpi(f"{r['cobertura_pct']:.1f} %", f"cobertura de copas (≥ {p['hcorte_m']:g} m)") +
             _kpi(f"{r['volumen_total_m3']:.0f} m³", "volumen de copas") +
             (_kpi(f"{ant['h_m']:.1f} m", "antena (excluida del recuento)") if ant is not None else "") + "</div>")
    txt = (f"En la época <b>{et}</b> se han delimitado <b>{r['n_arboles']}</b> copas individuales dentro de la parcela "
           f"(<b>{n_sab} sabinas y {n_pino} pinos</b>; {r['densidad_ha']:.0f} ind./ha). La altura media es {r['h_media']:.2f} m "
           f"(P95 {r['h_p95']:.1f} m). Todo lo que queda por debajo de {p['hcorte_m']:g} m se trata como suelo para no contaminar el recuento con matojos. ")
    if ant is not None:
        txt += f"La antena ({ant['h_m']:.1f} m) se ha identificado como estructura y excluida. "
    if A['n_forzados']:
        txt += f"{A['n_forzados']} copas se han añadido con la regla de completitud (sin ápice local claro) y están marcadas para revisión. "
    if campo:
        c = campo['cmp']
        txt += (f"Frente al inventario de campo de {campo.get('anio', '')} ({c['n_campo']} sabinas vivas ≥ 1,3 m), el LiDAR cuenta {c['n_lidar']} individuos: "
                f"{c['n_emparejados']} coinciden con una ficha y {c['n_tapados']} fichas quedan tapadas bajo copas mayores (contadas como una). ")
    if comp is not None:
        c = comp['resumen']
        txt += (f"Entre {comp['a']} y {comp['b']}: {c['n_persisten']} individuos persisten ({c['n_crecieron']} crecieron, {c['n_estables']} estables, "
                f"{c['n_perdieron']} perdieron altura), {c['n_mortalidad']} con mortalidad probable, {c['n_sin_corr']} sin correspondencia y {c['n_nuevos']} nuevos. ")
    h.append(f"<p>{txt}</p>")

    # ---------------------------------------------------------- algoritmos
    h.append("<h2>2. Algoritmos utilizados</h2><table><tr><th>Etapa</th><th>Algoritmo</th><th>Nota / equivalencia con el script R</th></tr>" +
             "".join(f"<tr><td><b>{a}</b></td><td>{b}</td><td>{c_}</td></tr>" for a, b, c_ in ALGORITMOS) + "</table>")

    # ---------------------------------------------------------- datos y calidad
    h.append("<h2>3. Datos y calidad</h2>")
    filas = []
    for e in epocas:
        x = datos[e]['xyz']; s = datos[e]['suelo']
        area = roi.area_ha() * 1e4
        inroi = roi.puntos(x[:, 0], x[:, 1])
        filas.append(dict(época=e, puntos_totales=f"{len(x):,}".replace(",", "."), dentro_parcela=f"{int(inroi.sum()):,}".replace(",", "."),
                          densidad_pts_m2=f"{inroi.sum() / area:.1f}", suelo_pct=f"{100 * s[inroi].mean():.1f}",
                          suelo_pts_m2=f"{(s & inroi).sum() / area:.2f}"))
    h.append(_tabla(pd.DataFrame(filas), rename={'puntos_totales': 'Puntos', 'dentro_parcela': 'En parcela',
                                                 'densidad_pts_m2': 'Pts/m²', 'suelo_pct': '% suelo', 'suelo_pts_m2': 'Suelo pts/m²'}))
    h.append(_img(fg.mapa_densidad_suelo(d['xyz'], d['suelo'], m, 1.0, roi, f"Densidad de puntos de suelo — {et}"),
                  "Densidad de puntos clasificados como suelo. Las zonas con poco suelo (copas densas) son las más inciertas en el MDT."))

    # ---------------------------------------------------------- modelos digitales
    h.append("<h2>4. Modelos digitales y nube de puntos</h2>")
    h.append(f"<p>DEM a {modelos['res_dem']:g} m y DSM/CHM a {res:g} m. CHM = DSM − DEM con relleno de huecos (pit {modelos['pit']}).</p>")
    h.append("<div class='dos'>" +
             _img(fg.mapa_dem_hillshade(ep['dem'], ep['hs'], modelos['malla_dem'], modelos['res_dem'], roi, f"DEM + sombreado — {et}"), "Modelo digital del terreno.") +
             _img(fg.mapa_raster(ep['chm'], m, res, roi, f"Modelo de altura de copas (CHM) — {et}", "viridis", "Altura (m)"), "CHM dentro de la parcela.") + "</div>")
    h.append(_img(vista, "Nube de puntos de la parcela coloreada por altura sobre el suelo."))
    h.append(_img(fg.vista_nube_3d(d['xyz'], ep['hag'], d['suelo'], roi, labels=A['labels'], malla=m, res=res,
                                   titulo="Segmentación de copas (un color por individuo)"),
                  "Cada color es un individuo delimitado; los puntos grises no pertenecen a ninguna copa (< 1,3 m o fuera de la parcela)."))

    # ---------------------------------------------------------- inventario
    h.append("<h2 class='salto'>5. Inventario de individuos</h2>")
    h.append(_img(fg.mapa_capas(A, modelos, roi, estr.get(et), f"Capas de altura de copa ({p['ancho_capa_m']:g} m) — {et}"),
                  "Capas de altura con las copas vectorizadas (contorno negro), ápices (+), individuos altos etiquetados y antena."))
    h.append("<div class='dos'>" + _img(fg.mapa_especies(A, modelos, roi), f"Pinos (ápice ≥ {p['h_pino_m']:g} m) y sabinas.") +
             _img(fg.barras_especies(tab), "Individuos por clase de altura y especie.") + "</div>")
    esp = tab.groupby('especie').agg(n=('id', 'count'), h_media=('h_m', 'mean'), h_max=('h_m', 'max'), area_copa_media=('area_m2', 'mean'),
                                      cobertura_m2=('area_m2', 'sum'), volumen_m3=('vol_m3', 'sum')).reset_index()
    h.append("<h3>Resumen por especie</h3>" + _tabla(esp, fmt={'h_media': '{:.2f}', 'h_max': '{:.1f}', 'area_copa_media': '{:.1f}',
                                                            'cobertura_m2': '{:.0f}', 'volumen_m3': '{:.0f}'},
                                                     rename={'especie': 'Especie', 'n': 'N.º', 'h_media': 'H media (m)', 'h_max': 'H máx (m)',
                                                             'area_copa_media': 'Copa media (m²)', 'cobertura_m2': 'Cobertura (m²)', 'volumen_m3': 'Volumen (m³)'}))
    h.append("<div class='dos'>" + _img(fg.hist_alturas(tab, p['umbral_alto_m']), "Los individuos ≥ umbral de auditoría se muestran en rojo.") +
             _img(fg.dispersion_copas(tab), "Relación entre altura, área de copa y volumen.") + "</div>")
    h.append("<h3>Auditoría de individuos altos</h3>")
    altos = tab[tab['alto']].sort_values('h_m', ascending=False)
    if len(altos):
        h.append(f"<div class='ok'>Se han conservado {len(altos)} individuos ≥ {p['umbral_alto_m']:g} m. No existe límite superior de altura.</div>")
        h.append(_tabla(altos, ['id', 'especie', 'x', 'y', 'h_m', 'area_m2', 'vol_m3', 'forzado'],
                        {'x': '{:.2f}', 'y': '{:.2f}', 'h_m': '{:.2f}', 'area_m2': '{:.1f}', 'vol_m3': '{:.1f}'},
                        rename={'h_m': 'H (m)', 'area_m2': 'Área (m²)', 'vol_m3': 'Vol (m³)'}))
    else:
        h.append(f"<p>No hay individuos ≥ {p['umbral_alto_m']:g} m en la parcela.</p>")
    desc = A['descartes']
    if desc is not None and len(desc):
        h.append(f"<h3>Registro de descartes ({len(desc)})</h3>")
        h.append(_tabla(desc.groupby('motivo').agg(n=('x', 'count')).reset_index()))
    pinos = tab[tab['especie'] == 'Pino'].sort_values('h_m', ascending=False)
    if len(pinos):
        h.append(f"<h3>Pinos (ápice ≥ {p['h_pino_m']:g} m)</h3>")
        h.append(_tabla(pinos, ['id', 'x', 'y', 'h_m', 'area_m2', 'diam_eq_m', 'vol_m3'], {'x': '{:.2f}', 'y': '{:.2f}'}, max_filas=40))

    # ---------------------------------------------------------- campo
    if campo:
        c = campo['cmp']
        h.append(f"<h2 class='salto'>6. Validación con el inventario de campo de {campo.get('anio', '')}</h2>")
        h.append("<div class='kpis'>" + _kpi(c['n_campo'], "sabinas vivas ≥ 1,3 m en campo") + _kpi(c['n_lidar'], "individuos LiDAR") +
                 _kpi(c['n_emparejados'], f"emparejados (≤ {c['radio_m']:g} m)") + _kpi(c['n_tapados'], "tapados bajo otra copa") +
                 _kpi(c['n_sin_copa'], "sin copa ≥ 1,3 m") + _kpi(f"{c['desplaza_x']:+.1f}/{c['desplaza_y']:+.1f} m", "desplazamiento campo→LiDAR (E/N)") + "</div>")
        h.append("<div class='dos'>" + _img(fg.mapa_campo(A, modelos, roi, campo['df']), "Fichas de campo (desplazadas) sobre las copas LiDAR.") +
                 _img(fg.barras_comparacion_campo(c), "Comparación de recuentos.") + "</div>")
        h.append(f"<p>El LiDAR cuenta desde arriba, así que los juveniles y plántulas que crecen <b>bajo copas mayores</b> no se ven: "
                 f"{c['n_tapados']} de las {c['n_campo']} fichas de campo caen dentro de una copa ya contada y se cuentan como un solo individuo "
                 f"(criterio pedido: lo tapado cuenta como uno). {c['n_copas_sin_ficha']} copas LiDAR no tienen ninguna ficha de {campo.get('anio', '')} "
                 f"(reclutas posteriores o individuos mayores no fichados). El inventario de campo es anterior a este vuelo ({campo.get('anio', '')}), "
                 f"por lo que no se espera una coincidencia exacta.</p>")
        if campo.get('ficha'):
            f_ = campo['ficha']
            h.append(f"<p>Inventario completo en la parcela: {f_['n_fichas']} fichas, {f_['n_muertos']} muertas, {f_['n_vivos']} vivas "
                     f"({f_['n_vivos_hmin']} ≥ 1,3 m y {f_['n_menores']} menores).</p>")

    # ---------------------------------------------------------- capas y volumen
    h.append("<h2 class='salto'>7. Capas de altura y volúmenes</h2>")
    gl = A['cap']['global_']
    h.append(f"<p>Cada copa se descompone en capas de {p['ancho_capa_m']:g} m. El volumen de cada polígono es Σ(h·A_celda) sobre el CHM; "
             f"el volumen por estrato es el espesor ocupado dentro de cada capa.</p>")
    h.append(_img(fg.volumen_por_capa(gl, p['ancho_capa_m']), "Superficie y volumen por capa de altura."))
    h.append(_tabla(gl, fmt={'h_min_m': '{:.2f}', 'h_max_m': '{:.2f}', 'area_clase_m2': '{:.1f}', 'vol_estrato_m3': '{:.1f}', 'vol_clase_m3': '{:.1f}'}))
    if 'vol_voxel_m3' in tab:
        h.append(f"<p>Volumen ocupado por vóxeles (0,25 m): {tab['vol_voxel_m3'].sum():.0f} m³ frente a "
                 f"{r['volumen_total_m3']:.0f} m³ de envolvente CHM.</p>")

    # ---------------------------------------------------------- infraestructura
    h.append("<h2 class='salto'>8. Infraestructura: antena, vallado y sensores</h2>")
    if te is not None and len(te):
        h.append(_img(fg.plano_estructuras(d['xyz'], ep['hag'], d['suelo'], estr[et], radio=18), "Plano de planta de los elementos detectados alrededor de la antena."))
        h.append(_img(fg.detalle_antena(d['xyz'], ep['hag'], d['suelo'], estr[et]), "Detalle de la antena y de su recinto: planta ampliada y cortes laterales (E–O y S–N)."))
        h.append(_tabla(te, ['id', 'tipo', 'x', 'y', 'h_m', 'longitud_m', 'confianza', 'excluir', 'notas'],
                        {'x': '{:.2f}', 'y': '{:.2f}', 'h_m': '{:.2f}', 'longitud_m': '{:.1f}'}))
        if ant is not None:
            h.append(_img(fg.perfil_vertical(d['xyz'], ep['hag'], d['suelo'], (ant['x'] - 15, ant['y']), (ant['x'] + 15, ant['y']), 4.0, estr[et]),
                          "Perfil vertical de la nube de puntos a través de la antena (banda de 4 m)."))
        h.append("<div class='aviso'>Detección heurística. Los elementos de confianza «baja» no se excluyen del recuento salvo que se marque «excluir». "
                 "Si no hay vallado en la nube, no se inventa: la tabla solo recoge lo que la nube respalda.</div>")
    else:
        h.append("<p>No se detectaron estructuras.</p>")

    # ---------------------------------------------------------- multitemporal
    if comp is not None:
        c = comp['resumen']
        h.append(f"<h2 class='salto'>9. Cambios {comp['a']} → {comp['b']}: muertos, crecidos y nuevos</h2><div class='kpis'>" +
                 _kpi(c['n_persisten'], "persisten") + _kpi(c['n_crecieron'], "crecieron (ΔH ≥ 0,1 m)") + _kpi(c['n_estables'], "estables") +
                 _kpi(c['n_perdieron'], "perdieron altura") + _kpi(c['n_mortalidad'], "mortalidad probable") +
                 _kpi(c['n_sin_corr'], "sin correspondencia (revisar)") + _kpi(c['n_nuevos'], "nuevos (reclutamiento)") +
                 _kpi(f"{c['dh_medio']:.2f} m", "ΔH medio de persistentes") + _kpi(f"{c['dvol_total']:+.0f} m³", "Δ volumen") + "</div>")
        sp_rows = []
        for sp in ("Sabina", "Pino"):
            s = sp.lower()
            sp_rows.append({'Especie': sp, 'Inicio': c.get(f'{s}_a', 0), 'Persisten': c.get(f'{s}_persisten', 0), 'Mortalidad probable': c.get(f'{s}_mortalidad', 0),
                            'Sin corresp.': c.get(f'{s}_sin_corr', 0), 'Nuevos': c.get(f'{s}_nuevos', 0),
                            'ΔH medio (m)': f"{c.get(f'{s}_dh', float('nan')):.2f}"})
        h.append("<h3>Por especie</h3>" + _tabla(pd.DataFrame(sp_rows)))
        h.append("<div class='dos'>" + _img(fg.mapa_estados(comp, m, res, roi, arb[comp['b']]['chm_eff'], "Estado de individuos"), "Estado de cada individuo entre épocas.") +
                 _img(fg.mapa_raster(comp['dh'], m, res, roi, "ΔH del CHM", etiqueta="m", divergente=True), "Diferencia de altura (positivo = crecimiento).") + "</div>")
        h.append("<div class='dos'>" + _img(fg.dispersion_h(comp), "Altura en A frente a B.") + _img(fg.demografia_especies(comp), "Demografía por especie.") + "</div>")
        h.append(_tabla(comp['tabla_dh'], fmt={'area_m2': '{:.1f}'}))
        pers = comp['df_a'][comp['df_a']['estado'] == 'persiste'].sort_values('dh', ascending=False)
        if len(pers):
            h.append("<h3>Mayores crecimientos</h3>" + _tabla(pers, ['id_a', 'id_b', 'especie_a', 'x', 'y', 'h_a', 'h_b', 'dh'],
                                                               {'x': '{:.2f}', 'y': '{:.2f}', 'h_a': '{:.2f}', 'h_b': '{:.2f}', 'dh': '{:+.2f}'}, max_filas=10))
        mu = comp['df_a'][comp['df_a']['estado'] == 'mortalidad probable']
        if len(mu):
            h.append("<h3>Mortalidad probable</h3>" + _tabla(mu, ['id_a', 'especie_a', 'x', 'y', 'h_a', 'area_a'], {'x': '{:.2f}', 'y': '{:.2f}', 'h_a': '{:.2f}', 'area_a': '{:.1f}'}, max_filas=60))
        nu = comp['df_b']
        if len(nu):
            h.append("<h3>Nuevos individuos</h3>" + _tabla(nu, ['id_b', 'especie_b', 'x', 'y', 'h_b', 'area_b'], {'x': '{:.2f}', 'y': '{:.2f}', 'h_b': '{:.2f}', 'area_b': '{:.1f}'}, max_filas=60))
        h.append(f"<div class='aviso'>Emparejamiento: ápices a ≤ {c['radio_m']:g} m (protocolo ICTS) o copas solapadas. «Mortalidad probable» y «sin correspondencia» "
                 "requieren verificación en campo u ortofoto: copas solapadas o desajustes entre vuelos pueden producir falsos positivos.</div>")

    # ---------------------------------------------------------- metodología
    h.append("<h2 class='salto'>10. Parámetros</h2>")
    h.append(_tabla(pd.DataFrame(sorted(p.items()), columns=['parámetro', 'valor'])))
    h.append("<h2>11. Limitaciones</h2><ul>"
             "<li>Copas muy solapadas o plántulas bajo copa no pueden separarse con un CHM: el recuento visible es una cota inferior de los individuos reales.</li>"
             "<li>El volumen CHM es una envolvente, no biomasa (el protocolo ICTS usa alpha-shapes 3D).</li>"
             "<li>La separación pino/sabina es solo por altura; copas de sabina &gt; 5 m se contarán como pino.</li>"
             "<li>La detección de estructuras es asistida; requiere revisión.</li>"
             "<li>La verificación debe hacerse con inventario de campo.</li></ul>")
    h.append("<h2>12. Referencias</h2><ul><li>Zhang et al. (2016). Remote Sensing 8(6):501 (CSF).</li>"
             "<li>Zhang et al. (2003). IEEE TGRS 41:872–882 (filtro morfológico progresivo).</li>"
             "<li>Khosravipour et al. (2014). Photogramm. Eng. Remote Sens. 80(9):863–872 (pit-free CHM).</li>"
             "<li>Popescu &amp; Wynne (2004). Photogramm. Eng. Remote Sens. 70:589–604.</li>"
             "<li>Roussel et al. (2020). lidR. Remote Sens. Environ. 251:112061.</li>"
             "<li>Vincent &amp; Soille (1991). IEEE TPAMI 13:583–598.</li>"
             "<li>Kuhn (1955). Naval Research Logistics Quarterly 2:83–97.</li></ul>")
    h.append("</div></body></html>")
    return "".join(h)