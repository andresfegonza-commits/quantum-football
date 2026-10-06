# Football Quant Intelligence

MVP de inteligencia futbolística cuantitativa para investigar oportunidades en
hándicap asiático con datos históricos, cuotas pre-partido, modelos
probabilísticos, backtesting, motor de riesgo y una capa interpretativa.

El objetivo del proyecto es validar si existe ventaja estadística medible. No
garantiza rentabilidad ni debe usarse como sustituto de gestión de riesgo,
criterio profesional o cumplimiento regulatorio local.

## Alcance Inicial

- Mercado principal: hándicap asiático.
- Enfoque: híbrido, empezando por pre-partido y preparado para live.
- Datos: fuentes gratuitas o de bajo costo, con contratos para migrar a APIs
  premium.
- Competiciones iniciales: ligas top con mayor liquidez y mejor cobertura.

## Módulos

- `asian_handicap`: liquidación exacta de líneas enteras, medias y cuartos.
- `data_sources`: fuentes recomendadas y proveedor CSV normalizado.
- `dataset`: modelo histórico validado para partidos, resultados y cuotas.
- `features`: variables pre-partido y prioridades live.
- `models`: baseline probabilístico basado en Elo, forma, contexto y mercado.
- `edge`: cálculo de probabilidad implícita y valor esperado.
- `risk`: Kelly fraccional, límites de exposición y criterios de rechazo.
- `backtesting`: simulación histórica sin usar información posterior al evento.
- `ai_layer`: explicación textual de señales cuantitativas y riesgo.
- `scanner`: ranking diario de oportunidades auditables.

## Uso Rápido

```bash
python -m football_quant.cli scan-sample
python -m football_quant.cli backtest-sample
```

Para desarrollo local:

```bash
python -m pip install -e ".[dev]"
pytest
```

## Métricas De Validación

- ROI y yield por mercado, liga y rango de cuota.
- Closing line value.
- Drawdown máximo.
- Hit rate con liquidación real de hándicap asiático.
- Estabilidad fuera de muestra.
- Concentración de ganancias por liga, temporada o pocos partidos.

## Próximos Datos Recomendados

1. `football-data.co.uk` para histórico gratuito de resultados y cuotas.
2. Datasets curados de Kaggle para prototipado de variables.
3. The Odds API para escaneo diario de cuotas.
4. API-Football para eventos live cuando el pre-partido esté validado.
