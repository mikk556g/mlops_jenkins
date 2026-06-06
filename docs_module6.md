# Module 6: Monitoring – Dokumentation

---

## D6.1 – Carbon footprint ved modeltræning

Modeltræningen er instrumenteret med **CarbonTracker**, som tracker energiforbrug og CO₂-udledning per epoch direkte via GPU-strømmåling. CarbonTracker er initialiseret i `train.py` og logger til `carbontracker_logs/`, som efter endt træning automatisk uploades som MLflow artifact under `carbontracker/`.

Træningen blev udført på en **NVIDIA RTX 4000 Ada Generation** GPU med en lokal carbon-intensitet på **143.30 gCO₂eq/kWh** (Aalborg, North Denmark, DK, fetched live under træning). Der anvendes en PUE-koefficient (Power Usage Effectiveness) på **1.58**, som afspejler overhead fra køling og infrastruktur i serverrummet.

Måleresultater for træningskørslen med 2 epochs (early stopping):

| Metric                    | Epoch 1       | Epoch 2       |
|---------------------------|---------------|---------------|
| Varighed                  | 0:00:14.64    | 0:00:15.19    |
| Gennemsnitlig GPU-effekt  | 57.39 W       | 47.85 W       |

**Samlet faktisk forbrug (2 epochs, ~30 sekunder):**

- Energi: **0.000688 kWh**
- CO₂-ækvivalent: **0.0986 g CO₂eq**
- Svarer til: **0.000923 km kørt i bil**

Predicted consumption (CarbonTracker estimat inden træning) var 0.000738 kWh og 0.1057 g CO₂eq, hvilket er tæt på det faktiske forbrug og viser at estimaterne er pålidelige.

Disse resultater er logget til MLflow under run `evaluation in FP32 and FP16` og kan ses under Artifacts → carbontracker/\*\_carbontracker\_output.log.

---

## D6.2 – Estimeret carbon footprint for et års drift

**Antaget use-case:** Emotion classifieren er deployeret som en del af et HR-analyseværktøj, der analyserer videooptagelser af jobsamtaler i realtid. Systemet kører i en virksomhed med 8 timers arbejdsdag, 5 dage om ugen.

**Antagelser:**

| Parameter                            | Værdi                          |
|--------------------------------------|--------------------------------|
| Inferencer per time (arbejdstid)     | 500 billeder/time              |
| Arbejdstimer per dag                 | 8 timer                        |
| Arbejdsdage per år                   | 260 dage (5 × 52 uger)         |
| Inferencer per år                    | 500 × 8 × 260 = **1.040.000** |
| Inferens-tid per billede             | ~50 ms (p50 fra Grafana)       |
| GPU-effekt under inferens (estimat)  | ~30 W (lavere end træning)     |
| PUE-koefficient                      | 1.58                           |
| Carbon-intensitet (Aalborg, DK)      | 143.30 gCO₂eq/kWh             |

**Beregning – inferens:**

Energi per inferens (inkl. PUE):
```
E_inf = 30 W × 0.050 s / 3600 × 1.58 = 6.58 × 10⁻⁷ kWh
```

Samlet årlig energi fra inferens:
```
E_år = 1.040.000 × 6.58 × 10⁻⁷ = 0.684 kWh/år
```

CO₂-ækvivalent fra inferens per år:
```
CO₂_inf = 0.684 kWh × 143.30 gCO₂eq/kWh ≈ 98 g CO₂eq/år
```

**Beregning – retraining:**

Antaget månedlig retraining (12 gange/år). En fuld 4-epoch træning estimeres til ca. det dobbelte af 2-epoch kørslen:
```
CO₂_træning = 2 × 0.0986 g × 12 = ≈ 2.4 g CO₂eq/år
```

**Samlet estimat:**

| Kilde          | CO₂eq/år       |
|----------------|----------------|
| Inferens       | ~98 g          |
| Retraining     | ~2 g           |
| **Total**      | **~100 g CO₂eq/år** |

Til sammenligning svarer 100 g CO₂eq til ca. **0.9 km kørt i bil** per år, hvilket er et ekstremt lavt aftryk. Det skyldes primært den korte inferenstid (~50 ms) og den relativt lave carbon-intensitet i det nordjyske elnet.

Det bør dog bemærkes, at dette estimat udelukkende dækker GPU-forbrug til selve modellen. I et produktionsmiljø vil der også være overhead fra netværk, storage, load balancing og andre services, der vil øge det reelle fodaftryk.

---

## D6.3 – Driftdetektion i pipeline

### Tilgang

Driftdetektionen er implementeret i `drift_detection.py` og anvender biblioteket **TorchDrift** med en **Kernel Maximum Mean Discrepancy (MMD)**-detektor. MMD-metoden måler, om to fordelinger (kalibreringsdata og testdata) stammer fra den samme underliggende distribution ved at sammenligne kernelbaserede statistikker i et feature-rum.

**Feature-ekstraktion:** I stedet for at køre drift-test direkte på råpixels bruges en **ResNet50-backbone** (uden klassifikationshoved) til at ekstrahere 2048-dimensionale feature-vektorer per billede. Dette giver en mere semantisk meningsfuld repræsentation, der er robust over for uvæsentlige variationer som billedstørrelse.

**Kalibrering:** Detektoren fittes på 200 træningsbilleder med normal transform (resize + centercrop + normalisering), som udgør referencen for "normal" datadistribution.

**Simuleret drift:** Da datasættet ikke naturligt præsenterer en driftende stream, simuleres inputdrift ved at anvende en aggressiv **Gaussian blur** (kernel=23, sigma=5–10) på testbillederne. Dette efterligner fx et forringelsesscenarie i kamerakvalitet, som ville forekomme naturligt i et deployeret system.

**Beslutningsregel:** Drift flagges, hvis den beregnede p-vædi er under tærsklen **p < 0.05**.

### Resultater

Kørslen er logget til MLflow under run `drift_detection` og viser følgende:

| Datasæt         | MMD Score | P-værdi | Drift detekteret |
|-----------------|-----------|---------|------------------|
| Normal testdata | 0.0251    | 0.483   | **Nej**          |
| Driftet testdata (Gaussian blur) | 0.184 | ~0.000 | **Ja** |

Detektoren performer præcis som forventet: normale billeder giver en høj p-værdi (0.483 >> 0.05), mens de kunstigt blurrede billeder giver en p-værdi på ~0, langt under tærsklen. MMD-scoren er 7× højere for de driftede data (0.184 vs. 0.025), hvilket afspejler den store distributionsforskel.

### Reaktion på drift i produktionen

Hvis drift detekteres i et live system, ville den anbefalede reaktion følge en eskaleret procedure:

1. **Alerting:** Automatisk notifikation via Prometheus alertmanager (kan konfigureres som en Grafana alert-regel baseret på f.eks. løbende MMD-score).
2. **Undersøgelse:** Manuel gennemgang af flaggede samples for at identificere årsagen (kamerafejl, sæsonvariationer, demografisk skift i brugerbase).
3. **Retraining:** Hvis driften skyldes reel konceptdrift (ny datadistribution), genindsamles og annoteres repræsentative samples, og en ny modelversion trænes og deployes via CI/CD-pipelinen.
4. **Rollback:** Hvis en nylig modelopdatering har forårsaget driften, kan den forrige stabile version hentes fra MLflow Model Registry og sættes i Staging.

---

## D6.4 – Monitorering af ML/AI-pipeline

Monitorering af den deployerede model er implementeret via en **FastAPI**-applikation instrumenteret med **Prometheus** og visualiseret i **Grafana**. Stacken er defineret i `fastapi-prometheus-grafana/docker-compose.yaml` og kører med tre containere: FastAPI-appen (port 8000), Prometheus (port 9090) og Grafana (port 3000).

FastAPI-appen eksponerer custom Prometheus-metrics via `prometheus_fastapi_instrumentator`, herunder:
- `inference_requests_total` – antal inferensforespørgsler
- `inference_latency_seconds` – latens-histogram

Grafana-dashboardet (**FastAPI Dashboard**) viser over en observationsperiode fra 2026-06-05 12:50 til 2026-06-06 00:50 følgende nøglemetrics:

| Metric                              | Observation                               |
|-------------------------------------|-------------------------------------------|
| Total requests/min (GET /metrics)   | Mean: 11.7 req/min, Last: 3.33 req/min   |
| 2xx success rate                    | Mean: 0.195/min, Max: 0.200/min – ingen fejl |
| 4xx fejlrate                        | Mean: 0, Max: 0 – ingen client errors    |
| Average response time (/)           | 3.88 ms                                  |
| Average response time (/metrics)    | 7.39 ms (last: 8.33 ms)                  |
| Request duration p50 (/metrics)     | Mean: 50.1 ms, Max: 62.5 ms             |
| Request duration p90 (/metrics)     | Mean: 90.9 ms, Max: 300 ms              |
| Requests under 100 ms               | 100% af forespørgsler                    |
| Memory usage                        | Stabil ~44.3 MB                          |
| CPU usage                           | ~0.72% (lav og stabil)                  |

Dashboardet bekræfter, at systemet kører stabilt uden fejl, med lav og forudsigelig latens. Alle requests besvares under 100 ms (p50: ~50 ms), og der observeres ingen 4xx eller 5xx fejl i observationsperioden. Hukommelses- og CPU-forbruget er minimalt og stabilt, hvilket indikerer ingen memory leaks eller ressourcepres.

I en udvidet produktionsopsætning ville man yderligere tilføje modelspecifikke metrics som klassifikationskonfidenser per klasse, inputbilledernes pixelstatistikker, og løbende MMD-score fra drift-detektoren – alt loggable som custom Prometheus gauges direkte fra inferens-endpointet.
