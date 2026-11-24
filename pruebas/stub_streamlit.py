"""Streamlit y Plotly simulados: ejecutan app.py de principio a fin sin interfaz (prueba de humo)."""
import sys, types, os, runpy, traceback
from unittest.mock import MagicMock
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import matplotlib; matplotlib.use("Agg")


class Parar(Exception):
    pass


class SS(dict):
    __getattr__ = lambda s, k: s[k] if k in s else (_ for _ in ()).throw(AttributeError(k))
    __setattr__ = dict.__setitem__


class ST:
    def __init__(self):
        self.session_state = SS(); self.paso = None; self.sidebar = self; self.llamadas = []
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __getattr__(self, n):                       # display genérico: title, write, caption, plotly_chart…
        return lambda *a, **k: None
    def set_page_config(self, **k): pass
    def radio(self, label, opts, index=0, **k):
        return self.paso if k.get("key") == "paso" else opts[index]
    def select_slider(self, l, opts, value=None, **k): return value if value is not None else opts[-1]
    def selectbox(self, l, opts, index=0, **k): return opts[index]
    def text_input(self, l, value="", **k): return value
    def checkbox(self, l, value=False, **k): return value
    def slider(self, l, a, b, v=None, *r, **k): return v if v is not None else a
    def number_input(self, l, a=None, b=None, v=None, *r, **k): return k.get("value", v if v is not None else a)
    def button(self, l, **k):
        ok = not k.get("disabled", False); self.llamadas.append((l, ok)); return ok
    def download_button(self, *a, **k): self.llamadas.append(("download", True)); return False
    def file_uploader(self, *a, **k):
        if k.get("key") == "up_campo" and os.path.exists(os.environ.get("CAMPO_XLSX", "")):
            return os.environ["CAMPO_XLSX"]
        return None
    def columns(self, spec, **k): return [self] * (spec if isinstance(spec, int) else len(spec))
    def tabs(self, names): return [self] * len(names)
    def expander(self, *a, **k): return self
    def spinner(self, *a, **k): return self
    def data_editor(self, df, **k): return df
    def stop(self): raise Parar()
    def rerun(self): pass
    def image(self, b, **k): assert isinstance(b, (bytes, bytearray)) and len(b) > 1000, "imagen vacía"


st = ST()
sys.modules["streamlit"] = st
for m in ("plotly", "plotly.graph_objects"):
    sys.modules[m] = MagicMock()
sys.modules["plotly"].graph_objects = sys.modules["plotly.graph_objects"]

import time, ast
app = os.path.join(os.path.dirname(__file__), "..", "app.py")


def leer_pasos(ruta):
    """Lee la lista PASOS de app.py sin ejecutarlo."""
    with open(ruta, encoding="utf-8") as f:
        arbol = ast.parse(f.read(), filename=ruta)
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "PASOS" for t in nodo.targets):
            return ast.literal_eval(nodo.value)
    sys.exit(f"ERROR: no se encuentra la asignación PASOS en {ruta}")


os.environ.setdefault("FORESTMAP_DEMO", "1")
PASOS = leer_pasos(app)
fallos = 0
for i, p in enumerate(PASOS):
    st.paso = p; t = time.time()
    try:
        runpy.run_path(app, run_name="__main__")
        print(f"OK   {p}  ({time.time()-t:.1f}s)")
    except Parar:
        if i >= 2:
            fallos += 1; print(f"FALLA {p}  (st.stop() inesperado: el paso no se ha recorrido)")
        else:
            print(f"STOP {p}  (st.stop — esperado en los pasos 1 y 2)")
    except Exception:
        fallos += 1; print(f"FALLA {p}"); traceback.print_exc()
S = st.session_state
print("épocas:", list(S.get("arb", {})), "| comp:", S.get("comp") is not None, "| zip KB:", len(S.get("paquete") or b"") // 1024)
if not S.get("arb"):
    fallos += 1; print("FALLA: no hay épocas con árboles (S['arb'] vacío o ausente)")
if S.get("comp") is None:
    fallos += 1; print("FALLA: no hay comparación entre épocas (S['comp'] es None)")
if not S.get("paquete"):
    fallos += 1; print("FALLA: no se ha generado el paquete de exportación (S['paquete'] vacío)")
if not fallos:
    print("TODO OK")
sys.exit(1 if fallos else 0)
