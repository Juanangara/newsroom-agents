# newsroom-agents

Equipo de agentes autónomos que produce un reporte de noticias **con afirmaciones verificadas y fuentes citadas**.
Corre completo en la GPU T4 gratuita de Google Colab, sin claves de API.

```
tema ─► Investigador ─► Resumidor ─► Verificador ─► Editor ─► reporte.md
        (RSS + filtro)   (JSON)      (fact-check)    (citas [n])
```

| Agente | Función | Contrato |
|---|---|---|
| **Investigador** | Busca en Google News (RSS), deduplica, filtra por relevancia y descarga el texto completo | lista de ids (JSON) |
| **Resumidor** | Resume y extrae hechos verificables de cada artículo | `{summary, key_facts[]}` |
| **Verificador** | Contrasta cada hecho con el texto fuente; **descarta** los no respaldados | lista de booleanos |
| **Editor** | Redacta el reporte usando solo hechos verificados, con citas `[n]` | markdown |

## Por qué es más que un demo
- **Verificación explícita:** el reporte solo contiene hechos que el Verificador encontró en la fuente.
- **Métrica de grounding:** `report.grounding_rate` = hechos respaldados / hechos propuestos.
- **Traza completa** de cada paso (`report.trace`).
- **LLM intercambiable:** Qwen2.5 local en 4-bit (Colab) o Claude por API.
- **Robusto:** JSON con reintentos y autocorrección; un artículo que falla no tumba el pipeline.
- **Testeable sin GPU:** `FakeLLM` para pruebas deterministas.

## Uso en Colab
Abre `notebooks/colab.ipynb`, activa GPU T4 y ejecuta todo. O manualmente:

```python
!pip -q install -e '.[local]'
from newsroom import HFLocalLLM, run_newsroom
report = run_newsroom("inteligencia artificial en Colombia", HFLocalLLM())
print(report.grounding_rate); print(report.markdown)
```

## Desarrollo
```bash
pip install -e '.[dev]' && pytest
```

## Próximos pasos
- Benchmark de grounding: Qwen 3B vs 7B vs Claude.
- Ejecución en paralelo de Resumidor/Verificador.
- Envío del reporte por correo o Telegram de forma programada.
