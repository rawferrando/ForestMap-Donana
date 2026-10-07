import sys, os, json, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.dirname(__file__))
import fixture, informe, exportar
F = fixture.construir()
dest = sys.argv[1]; shutil.rmtree(dest, ignore_errors=True)
h = informe.generar_informe(F['mod'], F['datos'], F['roi'], F['estr'], F['arb'], F['comp'], sintetico=True)
c = exportar.exportar_todo(dest, F['mod'], F['roi'], F['arb'], F['estr'], comp=F['comp'], html=h)
print(len(c), "archivos"); [print(" ", x) for x in c[:40]]
g = json.load(open(f"{dest}/VECTORIALES/Copas_2026.geojson"))
print(len(g['features']), g['crs'])
print(len(exportar.comprimir(dest))//1024, "KB zip")
