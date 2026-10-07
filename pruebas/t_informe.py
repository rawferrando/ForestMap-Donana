import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.dirname(__file__))
import fixture, informe
F = fixture.construir()
s = informe.generar_informe(F['mod'], F['datos'], F['roi'], F['estr'], F['arb'], F['comp'], sintetico=True)
open(sys.argv[1], "w", encoding="utf-8").write(s); print(len(s)//1024, "KB")
