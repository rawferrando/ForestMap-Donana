"""
ForestMap Doñana V3 — LiDAR para el sabinar: modelos digitales, capas de altura de copa, individuos,
volúmenes, antena/vallado/sensores y cambios entre épocas.   Ejecutar:  streamlit run app.py
"""
import os
import tempfile

import numpy as np
import pandas as pd
import streamlit as st

import copas as cp
import exportar as ex
import figuras as fg
import graficos as gr
import informe as inf
import pipeline as pl
import procesado as pr
import roi as rm
import estructuras as es
import campo as cm

st.set_page_config(page_title="ForestMap Doñana V3", page_icon="🌲", layout="wide")
S = st.session_state

PASOS = ["1 · Configuración", "2 · Carga y parcela", "3 · Modelos digitales", "4 · Antena y estructuras",
         "5 · Árboles y capas", "6 · Cambios entre épocas", "7 · Exportar e informe"]
for k, v in dict(epocas=["2024"], epsg=25829, datos={}, roi=None, modelos=None, estr={}, tab_est={}, arb={}, comp=None,
                 sintetico=False, paquete=None, html=None, autor="", titulo="Sabinar de Doñana", campo_inv=None, campo=None, campo_anio=2011).items():
    S.setdefault(k, v)


def aviso_falta(txt):
    st.info(txt)
    st.stop()


def figura_png(b, **kw):
    if b:
        st.image(b, **kw)


with st.sidebar:
    st.title("🌲 ForestMap Doñana V3")
    paso = st.radio("Flujo de trabajo", PASOS, key="paso")
    st.divider()
    st.caption("Estado")
    st.write(f"{'✅' if S.datos else '⬜'} Nubes: {', '.join(S.datos) or '—'}")
    st.write(f"{'✅' if S.roi else '⬜'} Parcela: {S.roi.descripcion() if S.roi else '—'}")
    st.write(f"{'✅' if S.modelos else '⬜'} Modelos digitales")
    st.write(f"{'✅' if S.estr else '⬜'} Estructuras: {', '.join(S.estr) or '—'}")
    st.write(f"{'✅' if S.arb else '⬜'} Árboles: {', '.join(S.arb) or '—'}")
    if S.sintetico:
        st.warning("Datos SINTÉTICOS de demostración")

# ===================================================================== 1
if paso == PASOS[0]:
    st.header("1 · Configuración")
    txt = st.text_input("Épocas (etiquetas separadas por comas, de antigua a reciente)", ", ".join(S.epocas))
    ep = [e.strip() for e in txt.split(",") if e.strip()]
    if ep and ep != S.epocas:
        S.epocas = ep
    S.epsg = st.number_input("EPSG del sistema de coordenadas", 1000, 99999, int(S.epsg), help="Doñana: 25829 (ETRS89 / UTM 29N)")
    S.titulo = st.text_input("Título del informe", S.titulo)
    S.autor = st.text_input("Autor/a", S.autor)
    st.success(f"Épocas: {', '.join(S.epocas)}. Con dos o más épocas se activa el análisis de cambios (paso 6).")

# ===================================================================== 2
elif paso == PASOS[1]:
    st.header("2 · Carga de nubes de puntos y parcela")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("Nubes LAS/LAZ")
        st.caption("Las etiquetas de época se cambian en el paso 1 (p. ej. «2022, 2026»). Archivos de más de 2 GB: use la ruta en disco.")
        subidos = {}
        for e in S.epocas:
            f_ = st.file_uploader(f"Nube de la época {e}", type=["las", "laz"], key=f"up_{e}")
            ruta_ = st.text_input(f"…o ruta del archivo en este ordenador ({e})", key=f"ruta_{e}",
                                  placeholder="/home/usuario/datos/ojillo_22_w.las",
                                  help="Más rápido para archivos grandes: no hace falta subirlos por el navegador.")
            subidos[e] = f_ if f_ is not None else (os.path.expanduser(ruta_.strip()) if ruta_.strip() and os.path.isfile(os.path.expanduser(ruta_.strip())) else None)
            if ruta_.strip() and subidos[e] is None:
                st.error(f"No existe el archivo: {ruta_}")
        dens = st.slider("Densidad máxima (puntos/m²)", 50, 5000, 1000, 50, help="Submuestrea para acelerar; no afecta a la altura.")
        rig = st.select_slider("Rigidez del CSF", [1, 2, 3], 3)
        modo_suelo = st.radio("Suelo", ["Clasificar (CSF / morfológico)", "Usar la clase 2 del archivo (suelo ya clasificado)"], horizontal=True,
                              help="Si el LAS ya trae el suelo clasificado (clase 2) puede aprovecharse.")
        if st.button("Cargar y clasificar suelo", type="primary", disabled=not all(subidos.values())):
            S.datos, S.modelos, S.estr, S.tab_est, S.arb, S.comp, S.sintetico, S.campo = {}, None, {}, {}, {}, None, False, None
            if S.get('roi') is not None and 'sintética' in S.roi.nombre:
                S.roi = None
            for e, f in subidos.items():
                with st.spinner(f"Leyendo {e}…"):
                    xyz, cl = pr.leer_las(f, dens, con_clase=True)
                    if modo_suelo.startswith("Usar") and (cl == 2).any():
                        S.datos[e] = dict(xyz=xyz, suelo=(cl == 2))
                    else:
                        S.datos[e] = pl.preparar_epoca(xyz, rig)
            st.success("Nubes cargadas.")
        st.subheader("Inventario de campo (opcional)")
        st.caption("Excel con IDENT, LAT, LON, Altura y Cambio… (p. ej. datos_ojillo_11.xlsx). Se usa para validar el recuento.")
        fi = st.file_uploader("Inventario de campo (.xlsx)", type=["xlsx"], key="up_campo")
        S.campo_anio = st.number_input("Año del inventario", 1990, 2100, int(S.campo_anio))
        if fi is not None and st.button("Cargar inventario"):
            try:
                S.campo_inv = cm.leer_inventario(fi)
                S.campo = None
                st.success(f"{len(S.campo_inv)} fichas leídas ({int(S.campo_inv.vivo.sum())} vivas).")
            except Exception as ex_:
                st.error(f"No se pudo leer el inventario: {ex_}")
    with c2:
        st.subheader("Datos de demostración")
        st.caption("Escena sintética (no son datos reales) para probar el flujo completo.")
        if st.button("Cargar escena sintética"):
            import sintetico as sy
            ep2 = S.epocas if len(S.epocas) >= 2 else ["2024", "2026"]
            S.epocas = ep2
            esc = sy.escenario(epocas=tuple(ep2))
            S.datos = {e: dict(xyz=esc['xyz'][e], suelo=pr.clasificar_suelo_morfologico(esc['xyz'][e])) for e in ep2}
            S.modelos, S.estr, S.tab_est, S.arb, S.comp, S.sintetico = None, {}, {}, {}, None, True
            ext = rm.extension([d['xyz'] for d in S.datos.values()])
            cx, cy = (ext[0] + ext[1]) / 2, (ext[2] + ext[3]) / 2
            L, th = 99.4 / 2, np.radians(14)
            R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            esq = np.array([[-L, -L], [L, -L], [L, L], [-L, L], [-L, -L]]) @ R.T + [cx, cy]
            S.roi = rm.Roi((esq[:, 0].min(), esq[:, 0].max(), esq[:, 1].min(), esq[:, 1].max()), anillos=[esq], nombre="Parcela (sintética)")
            st.rerun()
    if S.datos:
        st.divider()
        st.subheader("Parcela de estudio (ROI)")
        st.caption("Todo lo que queda fuera de la parcela se descarta en mapas, estadísticas y exportaciones.")
        ext = rm.extension([d['xyz'] for d in S.datos.values()])
        modo = st.radio("Modo", ["Polígono (.gpkg / .geojson / .zip)", "Parcela cuadrada", "Coordenadas", "Sin borde", "Toda la nube"], horizontal=True)
        roi_nueva = None
        if modo.startswith("Polígono"):
            f = st.file_uploader("Archivo de la parcela", type=["gpkg", "geojson", "zip"], key="poly")
            if f is not None:
                try:
                    pol = rm.leer_poligono(f, f.name, int(S.epsg))
                    roi_nueva = rm.construir_roi(ext, "poligono", poligono=pol)
                except Exception as ex_:
                    st.error(f"No se pudo leer el polígono: {ex_}")
        elif modo == "Parcela cuadrada":
            ha = st.number_input("Hectáreas", 0.1, 50.0, 1.0, 0.1)
            dx = st.number_input("Desplazamiento X (m)", -500.0, 500.0, 0.0)
            dy = st.number_input("Desplazamiento Y (m)", -500.0, 500.0, 0.0)
            roi_nueva = rm.construir_roi(ext, "parcela", hectareas=ha, dx=dx, dy=dy)
        elif modo == "Coordenadas":
            cc = st.columns(4)
            co = [cc[i].number_input(n, value=float(v)) for i, (n, v) in enumerate(zip(["Xmin", "Xmax", "Ymin", "Ymax"], ext))]
            roi_nueva = rm.construir_roi(ext, "coords", coords=co)
        elif modo == "Sin borde":
            roi_nueva = rm.construir_roi(ext, "borde", borde=st.number_input("Borde (m)", 0.0, 100.0, 10.0))
        else:
            roi_nueva = rm.construir_roi(ext, "nube")
        if roi_nueva is not None and st.button("Aplicar parcela", type="primary"):
            S.roi = roi_nueva
            S.modelos, S.estr, S.tab_est, S.arb, S.comp = None, {}, {}, {}, None
            st.rerun()
        if S.roi:
            st.success(S.roi.descripcion())
        filas = []
        for e, d in S.datos.items():
            dentro = S.roi.puntos(d['xyz'][:, 0], d['xyz'][:, 1]) if S.roi else np.ones(len(d['xyz']), bool)
            filas.append(dict(época=e, puntos=len(d['xyz']), en_parcela=int(dentro.sum()), suelo_pct=round(100 * d['suelo'].mean(), 1)))
        st.dataframe(pd.DataFrame(filas), hide_index=True)

# ===================================================================== 3
elif paso == PASOS[2]:
    st.header("3 · Modelos digitales (DEM · DSM · CHM)")
    if not S.datos or S.roi is None:
        aviso_falta("Cargue las nubes y defina la parcela en el paso 2.")
    preset = st.radio("Resolución", ["Rápida (DEM 2 m · CHM 1 m)", "Estándar (DEM 1 m · CHM 0,5 m)", "Fina (DEM 1 m · CHM 0,25 m)"], index=1, horizontal=True)
    rd, rc = {0: (2.0, 1.0), 1: (1.0, 0.5), 2: (1.0, 0.25)}[["R", "E", "F"].index(preset[0])]
    with st.expander("Parámetros avanzados"):
        a, b, c, d_ = st.columns(4)
        eq = a.number_input("Equidistancia curvas (m)", 0.1, 5.0, 1.0)
        pit = b.number_input("Relleno de huecos del CHM (m)", 0.0, 1.0, 0.025, 0.005, format="%.3f")
        f_min = c.number_input("Filtro mínimo DEM (px)", 1, 15, 3)
        f_med = d_.number_input("Suavizado DEM (px)", 1, 41, 15)
    if st.button("Generar modelos", type="primary"):
        with st.spinner("Generando DEM, DSM y CHM…"):
            S.modelos = pl.generar_modelos(S.datos, rd, rc, eq, pit, f_min, f_med)
            S.estr, S.tab_est, S.arb, S.comp = {}, {}, {}, None
    mod = S.modelos
    if mod is None:
        st.stop()
    et = st.selectbox("Época", list(S.datos), index=len(S.datos) - 1)
    ep = mod['epocas'][et]
    t1, t2, t3, t4 = st.tabs(["DEM + sombreado", "DSM", "CHM", "Perfil vertical"])
    with t1:
        st.plotly_chart(gr.raster_plotly(ep['dem'], mod['malla_dem'], mod['res_dem'], S.roi, "DEM", "Earth", "m s.n.m."), use_container_width=True)
        figura_png(fg.mapa_dem_hillshade(ep['dem'], ep['hs'], mod['malla_dem'], mod['res_dem'], S.roi, f"DEM + sombreado — {et}"))
    with t2:
        st.plotly_chart(gr.raster_plotly(ep['dsm'], mod['malla_chm'], mod['res_chm'], S.roi, "DSM", "Cividis", "m"), use_container_width=True)
    with t3:
        st.plotly_chart(gr.raster_plotly(ep['chm'], mod['malla_chm'], mod['res_chm'], S.roi, "CHM", "Viridis", "m"), use_container_width=True)
    with t4:
        b = S.roi.bounds
        c1, c2, c3, c4, c5 = st.columns(5)
        x0 = c1.number_input("X inicio", value=float(b[0])); y0 = c2.number_input("Y inicio", value=float((b[2] + b[3]) / 2))
        x1 = c3.number_input("X fin", value=float(b[1])); y1 = c4.number_input("Y fin", value=float((b[2] + b[3]) / 2))
        w = c5.number_input("Ancho banda (m)", 0.5, 20.0, 3.0)
        d = S.datos[et]
        figura_png(fg.perfil_vertical(d['xyz'], ep['hag'], d['suelo'], (x0, y0), (x1, y1), w, S.estr.get(et)))

# ===================================================================== 4
elif paso == PASOS[3]:
    st.header("4 · Antena, vallado y sensores")
    if S.modelos is None:
        aviso_falta("Genere los modelos en el paso 3.")
    with st.expander("Parámetros de detección"):
        P = es.ParamsEstructuras()
        c = st.columns(3)
        P.h_semilla_m = c[0].number_input("Altura mínima del mástil (m)", 1.0, 30.0, P.h_semilla_m)
        P.sobresale_min_m = c[1].number_input("Debe sobresalir de su entorno (m)", 0.5, 10.0, P.sobresale_min_m)
        P.radio_busqueda_m = c[2].number_input("Radio de búsqueda alrededor (m)", 5.0, 80.0, P.radio_busqueda_m)
        P.largo_vallado_min_m = c[0].number_input("Longitud mínima del vallado (m)", 1.0, 50.0, P.largo_vallado_min_m)
        P.grosor_vegetacion_m = c[1].number_input("Grosor máximo estructura (m)", 0.3, 3.0, P.grosor_vegetacion_m)
        P.tam_max_instrumento_m = c[2].number_input("Tamaño máx. instrumento (m)", 0.5, 10.0, P.tam_max_instrumento_m)
    if st.button("Detectar estructuras", type="primary"):
        S.estr, S.tab_est, S.arb, S.comp = {}, {}, {}, None
        for e in S.datos:
            with st.spinner(f"Estructuras {e}…"):
                S.estr[e] = pl.detectar_estructuras_epoca(S.modelos, S.datos, e, P)
                S.tab_est[e] = S.estr[e].tabla.copy()
    if not S.estr:
        st.stop()
    et = st.selectbox("Época", list(S.estr), index=len(S.estr) - 1)
    est = S.estr[et]
    if est.vacio():
        st.warning("No se ha detectado ninguna estructura. Pruebe a bajar la altura mínima del mástil.")
        st.stop()
    st.caption("Marque **excluir** en lo que sea infraestructura (no se contará como árbol). Desmárquelo si es un falso positivo.")
    ed = st.data_editor(S.tab_est[et], hide_index=True, disabled=[c for c in S.tab_est[et].columns if c != 'excluir'],
                        key=f"ed_{et}", use_container_width=True)
    if not ed.equals(S.tab_est[et]):
        S.tab_est[et] = ed
        S.arb, S.comp = {}, None
    ep = S.modelos['epocas'][et]
    figura_png(fg.plano_estructuras(S.datos[et]['xyz'], ep['hag'], S.datos[et]['suelo'], est))
    for log in est.log:
        st.caption(log)

# ===================================================================== 5
elif paso == PASOS[4]:
    st.header("5 · Árboles, capas de altura y volúmenes")
    if S.modelos is None or S.roi is None:
        aviso_falta("Genere los modelos digitales (paso 3).")
    P = cp.ParamsArboles()
    PRE = {"Conservadora (menos copas)": (0.3, 0.15, 0.3), "Equilibrada": (0.2, 0.05, 0.3),
           "Máxima separación (más copas)": (0.15, 0.03, 0.3), "Censo máximo (≈ 700 en el Ojillo, riesgo de partir copas)": (0.10, 0.015, 0.5)}
    pre = st.radio("Sensibilidad", list(PRE), index=1, horizontal=True)
    P.sigma_m, P.prominencia_m, P.area_min_m2 = PRE[pre]
    st.caption("Por debajo de 1,3 m todo se trata como suelo (matojos). Más de 5 m = pino. Lo tapado bajo una copa cuenta como uno.")
    with st.expander("Parámetros avanzados"):
        a, b, c = st.columns(3)
        P.metodo = a.selectbox("Método de ápices", ["hmax", "lmf"], format_func=lambda m: "H-máximos (recomendado)" if m == "hmax" else "Ventana variable (lidR)")
        P.sigma_m = b.number_input("Suavizado (m)", 0.0, 2.0, P.sigma_m, 0.05)
        P.prominencia_m = c.number_input("Prominencia (m)", 0.02, 1.0, P.prominencia_m, 0.01)
        P.hmin_m = a.number_input("Altura mínima individuo / suelo (m)", 0.2, 5.0, P.hmin_m, 0.1)
        P.hcorte_m = P.hmin_m
        P.h_pino_m = b.number_input("Altura de pino (m)", 3.0, 30.0, P.h_pino_m, 0.5)
        P.fusion_pinos = c.checkbox("Un pino = un individuo", P.fusion_pinos)
        P.rel_min_copa = a.number_input("Límite relativo de copa", 0.0, 0.8, P.rel_min_copa, 0.05)
        P.ancho_capa_m = b.number_input("Grosor de capa (m)", 0.25, 2.0, P.ancho_capa_m, 0.25)
        P.umbral_alto_m = c.number_input("Umbral de auditoría 'alto' (m)", 3.0, 30.0, P.umbral_alto_m)
        P.area_min_m2 = a.number_input("Área mínima de copa (m²)", 0.05, 5.0, float(P.area_min_m2))
        voxel = b.checkbox("Calcular volumen ocupado por vóxeles", True)
    if st.button("Detectar árboles y calcular capas", type="primary"):
        S.arb, S.comp = {}, None
        for e in S.datos:
            with st.spinner(f"Árboles {e}…"):
                S.arb[e] = pl.analizar_arboles(S.modelos, S.datos, e, P, S.roi, S.estr.get(e), S.tab_est.get(e), voxel)
        S.campo = None
    if not S.arb:
        st.stop()
    et = st.selectbox("Época a visualizar", list(S.arb), index=len(S.arb) - 1)
    A = S.arb[et]; r = A['rodal']; p = A['params']
    npino = int((A['tabla']['especie'] == 'Pino').sum())
    k = st.columns(7)
    k[0].metric("Individuos", r['n_arboles']); k[1].metric("Sabinas", r['n_arboles'] - npino); k[2].metric("Pinos", npino)
    k[3].metric("Densidad (ind/ha)", f"{r['densidad_ha']:.0f}"); k[4].metric("H media (m)", f"{r['h_media']:.2f}")
    k[5].metric("H máx (m)", f"{r['h_max']:.1f}"); k[6].metric("Volumen (m³)", f"{r['volumen_total_m3']:.0f}")
    c1, c2 = st.columns(2)
    mc = c1.checkbox("Contornos de copa", True); ma = c2.checkbox("Ápices", True)
    st.plotly_chart(gr.capas_plotly(A, S.modelos, S.roi, S.estr.get(et), mostrar_copas=mc, mostrar_apices=ma), use_container_width=True)
    t1, t2, t3, t4, t5, t6 = st.tabs(["Individuos", "Auditoría de altos", "Capas y volúmenes", "Distribuciones", "Sensibilidad", "Campo"])
    with t1:
        st.dataframe(A['tabla'], hide_index=True, use_container_width=True)
    with t2:
        altos = A['tabla'][A['tabla']['alto']].sort_values('h_m', ascending=False)
        st.success(f"{len(altos)} individuos ≥ {p['umbral_alto_m']:g} m conservados (sin límite superior de altura).")
        st.dataframe(altos, hide_index=True)
        if A['n_forzados']:
            st.info(f"{A['n_forzados']} copas añadidas por la regla de completitud (columna 'forzado'): revíselas.")
        st.subheader("Descartes registrados")
        st.dataframe(A['descartes'], hide_index=True)
    with t3:
        st.plotly_chart(gr.volumen_capas(A['cap']['global_']), use_container_width=True)
        st.dataframe(A['cap']['global_'], hide_index=True)
        st.caption("Volumen por árbol y capa (CSV completo en la exportación):")
        st.dataframe(A['vol_capas_arbol'], hide_index=True)
    with t4:
        st.plotly_chart(gr.histograma_alturas(A['tabla'], p['umbral_alto_m']), use_container_width=True)
        st.plotly_chart(gr.dispersion(A['tabla']), use_container_width=True)
        st.dataframe(r['clases_altura'], hide_index=True)
    with t5:
        st.caption("Número de copas según la prominencia: una curva plana indica un recuento estable.")
        if st.button("Calcular curva de sensibilidad"):
            chm = np.nan_to_num(S.modelos['epocas'][et]['chm'])
            roi_m = S.roi.mascara(S.modelos['malla_chm'], S.modelos['res_chm'], chm.shape)
            xs, ns = [], []
            for pv in [0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50]:
                q = cp.ParamsArboles(**{**p, 'prominencia_m': pv})
                rr = cp.detectar_copas(chm, S.modelos['malla_chm'], S.modelos['res_chm'], q, excluir=A['excl'], roi=roi_m)
                xs.append(pv); ns.append(len(rr['tabla']))
            import plotly.graph_objects as go
            f = go.Figure(go.Scatter(x=xs, y=ns, mode="lines+markers"))
            f.update_layout(height=320, template="plotly_white", xaxis_title="Prominencia (m)", yaxis_title="N.º de copas")
            st.plotly_chart(f, use_container_width=True)

    with t6:
        if S.campo_inv is None:
            st.info("Cargue el inventario de campo en el paso 2 para validar el recuento.")
        else:
            inv = S.campo_inv
            if st.button("Comparar con el inventario de campo"):
                vivos = inv[inv.vivo & (inv.altura_m >= p['hmin_m'])]
                dx, dy, _n = cm.estimar_desplazamiento(A['tabla'], vivos)
                cmp_, cdf = cm.comparar_con_campo(A['tabla'], A['labels'], S.modelos['malla_chm'], S.modelos['res_chm'], inv, S.roi, (dx, dy), p['hmin_m'])
                ficha = cm.resumen_campo(inv.assign(x=inv.x + dx, y=inv.y + dy), S.roi, p['hmin_m'])
                S.campo = dict(cmp=cmp_, df=cdf, anio=int(S.campo_anio), desplaza=(dx, dy), ficha=ficha, et=et)
            if S.campo:
                c = S.campo['cmp']
                k = st.columns(5)
                k[0].metric("Fichas vivas ≥ 1,3 m", c['n_campo']); k[1].metric("Individuos LiDAR", c['n_lidar'])
                k[2].metric("Emparejados", c['n_emparejados']); k[3].metric("Tapados bajo copa", c['n_tapados'])
                k[4].metric("Desplazamiento E/N (m)", f"{c['desplaza_x']:+.1f} / {c['desplaza_y']:+.1f}")
                figura_png(fg.mapa_campo(A, S.modelos, S.roi, S.campo['df']))
                st.dataframe(S.campo['df'], hide_index=True)

# ===================================================================== 6
elif paso == PASOS[5]:
    st.header("6 · Cambios entre épocas")
    if len(S.arb) < 2:
        aviso_falta("Necesita al menos dos épocas con árboles detectados (paso 5).")
    ks = list(S.arb)
    c1, c2 = st.columns(2)
    a = c1.selectbox("Época inicial", ks, index=0)
    b = c2.selectbox("Época final", ks, index=len(ks) - 1)
    if a == b:
        aviso_falta("Elija dos épocas distintas.")
    radio = st.number_input("Radio de emparejamiento de ápices (m)", 0.3, 5.0, 1.2, 0.1, help="Protocolo ICTS: 1,2 m")
    if st.button("Comparar", type="primary"):
        S.comp = pl.comparar_epocas(S.modelos, S.arb, a, b, S.roi, radio)
    comp = S.comp
    if comp is None:
        st.stop()
    r = comp['resumen']
    k = st.columns(7)
    k[0].metric("Persisten", r['n_persisten']); k[1].metric("Crecieron", r['n_crecieron']); k[2].metric("Mortalidad probable", r['n_mortalidad'])
    k[3].metric("A revisar", r['n_sin_corr']); k[4].metric("Nuevos", r['n_nuevos'])
    k[5].metric("ΔH medio (m)", f"{r['dh_medio']:.2f}"); k[6].metric("Δ volumen (m³)", f"{r['dvol_total']:+.0f}")
    m, res = S.modelos['malla_chm'], S.modelos['res_chm']
    st.plotly_chart(gr.raster_plotly(comp['dh'], m, res, S.roi, f"ΔH del CHM {comp['a']} → {comp['b']}", unidad="m", divergente=True), use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        figura_png(fg.mapa_estados(comp, m, res, S.roi, S.arb[comp['b']]['chm_eff'], "Estado de individuos"))
    with c2:
        figura_png(fg.dispersion_h(comp)); figura_png(fg.demografia_especies(comp))
    st.dataframe(comp['tabla_dh'], hide_index=True)
    st.dataframe(comp['df_a'], hide_index=True)
    st.warning("«Mortalidad probable» y «sin correspondencia» deben verificarse (ortofoto o campo): el solape de copas y los desajustes entre vuelos generan falsos positivos.")

# ===================================================================== 7
else:
    st.header("7 · Exportar e informe")
    if not S.arb:
        aviso_falta("Detecte árboles primero (paso 5).")
    fmt = st.radio("Formato vectorial", ["geojson", "gpkg", "shp"], horizontal=True,
                   help="GeoJSON no requiere librerías extra. GPKG/SHP requieren geopandas.")
    if st.button("Generar informe y paquete QGIS", type="primary"):
        with st.spinner("Generando figuras, informe y archivos…"):
            meta = dict(titulo=S.titulo, autor=S.autor, crs=f"EPSG:{int(S.epsg)}")
            S.html = inf.generar_informe(S.modelos, S.datos, S.roi, S.estr, S.arb, S.comp, S.tab_est, meta, S.sintetico, campo=S.campo)
            with tempfile.TemporaryDirectory() as td:
                try:
                    S.creados = ex.exportar_todo(td, S.modelos, S.roi, S.arb, S.estr, S.tab_est, S.comp, int(S.epsg), fmt, S.html, campo=S.campo)
                    S.paquete = ex.comprimir(td)
                except Exception as e_:
                    S.paquete = None
                    st.error(f"Error al exportar ({fmt}): {e_}")
    if S.html:
        st.download_button("⬇ Informe (HTML autocontenido; imprimir → PDF)", S.html.encode("utf-8"), "Informe_ForestMap.html", "text/html")
    if S.paquete:
        st.download_button("⬇ Paquete completo (ZIP: RASTER, VECTORIALES, TABLAS, INFORME, QGIS)", S.paquete, "ForestMap_resultados.zip", "application/zip")
        st.caption("Archivos incluidos:")
        st.code("\n".join(S.get('creados', [])))
        st.info("En QGIS: cargue RASTER/CHM_*.tif (o .asc), pulse «Estilo → Cargar estilo» con QGIS/CHM_*.qml y añada VECTORIALES/Copas_*.")
