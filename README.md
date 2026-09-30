# Predictor

Notebook Deepnote care propune până la 7 numere (1–20) pentru următoarea extragere orară (07:00–23:00).

## Utilizare în Deepnote

1. Importă `Predictor.ipynb` și urcă `history.csv` în același director.
2. În celula **1 · Configurare**:
   - `NUM_PROPUNERI`: câte numere vrei (1–7);
   - `NUMERE_NOI`: extragerile noi, de forma `"2026-04-29 23:00": 5`.
3. **Run all** (~10 secunde).
4. După fiecare extragere, rulează ultima celulă (**10 · Introdu ultimul număr extras**). În output apare o căsuță: scrii numărul (1–20) și apeși Enter. Numărul se salvează în istoric și primești imediat propunerile pentru extragerea următoare.

## Ce face

- Curăță istoricul: duplicate, ordine, sloturi lipsă. Adăugarea e idempotentă: rularea repetată nu dublează rânduri.
- Rulează teste de aleatorism: uniformitate, serial, oră din zi, zi a săptămânii.
- Șapte modele (uniform, frecvență globală și recentă, Markov-1/2, oră din zi, întârziere), evaluate walk-forward și combinate într-un ansamblu. Ponderile sunt calibrate pe date pe care ansamblul nu le testează.
- Afișează propunerile împreună cu rata reală de nimerire din backtest și cu șansa pură (K × 5%).
- Salvează propunerile în `predictions_log.csv` și le compară cu rezultatele reale pe măsură ce le adaugi.

> Pe istoricul actual (30.497 extrageri), niciun model nu bate semnificativ șansa pură: top-7 nimerește ~34,7%, față de 35% la întâmplare.

## Cercetare: se poate crește acuratețea?

Scripturile din `cercetare/` (se rulează din rădăcina repo-ului, cu `scikit-learn` și `lightgbm` instalate) au căutat sistematic un avantaj:

| Script | Ce testează | Rezultat |
|---|---|---|
| `explorare1.py` | dependență la distanța 1–60, diferențe, serii par/impar și mic/mare, derivă pe an, lună și zi, distribuția întârzierilor | nicio abatere (cel mai mic p corectat = 1,0) |
| `explorare2.py` | regresie logistică și LightGBM cu 264 de trăsături | nu bat uniformul; LightGBM se oprește după o iterație |
| `explorare3.py` | 30 de strategii „calde/reci/întârziate”, alese pe trecut și verificate pe viitor | toate între 34–36%; corelație trecut ↔ viitor ≈ 0 |
| `explorare4.py` | 17 teste Monte Carlo (1.000 simulări fiecare): structura zilnică, perechi în aceeași zi, periodicitate (FFT), legătura cu data/ora, poker, colecționarul de cupoane, triplete, compresibilitate | nicio abatere (cel mai mic p = 0,056; corectat = 0,95) |
| `explorare5.py` | rețea neuronală (MLP), vecini apropiați (kNN), potrivirea celei mai lungi secvențe repetate | niciuna nu bate uniformul pe test (top-7: 34,3% / 35,4% / 35,6%) |
| `explorare6.py` | 226.700 reguli aritmetice („anteriorul +1”, „acum 10 ture +1”, a·x+c, ±x[t-a] ± x[t-b] + c, aceeași oră de acum d zile) | cea mai bună pe trecut: 5,57% (explicabil prin noroc în 96% din cazuri); cele mai bune 100 în viitor: 5,02% |

Concluzie: generatorul se comportă aleator, iar acuratețea top-7 rămâne ~35%. Secțiunea **9 · Laborator** din notebook repetă verificarea pe măsură ce crește istoricul.
