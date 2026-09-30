# Predictor

Notebook Deepnote care propune până la 7 numere (1–20) pentru următoarea extragere orară (07:00–23:00).

## Utilizare în Deepnote

1. Importă `Predictor.ipynb` și urcă `history.csv` în același director.
2. În celula **1 · Configurare**:
   - `NUM_PROPUNERI`: câte numere vrei (1–7);
   - `NUMERE_NOI`: extragerile noi, de forma `"2026-04-29 23:00": 5`.
3. **Run all** (~10 secunde).
4. După fiecare extragere, completează în ultima celulă (**9 · Introdu ultimul număr extras**) `NUMAR_EXTRAS` și rulează doar celula aceea. Numărul se salvează în istoric și primești imediat propunerile pentru extragerea următoare.

## Ce face

- Curăță istoricul: duplicate, ordine, sloturi lipsă. Adăugarea e idempotentă: rularea repetată nu dublează rânduri.
- Rulează teste de aleatorism: uniformitate, serial, oră din zi, zi a săptămânii.
- Șapte modele (uniform, frecvență globală și recentă, Markov-1/2, oră din zi, întârziere), evaluate walk-forward și combinate într-un ansamblu. Ponderile sunt calibrate pe date pe care ansamblul nu le testează.
- Afișează propunerile împreună cu rata reală de nimerire din backtest și cu șansa pură (K × 5%).
- Salvează propunerile în `predictions_log.csv` și le compară cu rezultatele reale pe măsură ce le adaugi.

> Pe istoricul actual (30.497 extrageri), niciun model nu bate semnificativ șansa pură: top-7 nimerește ~34,7%, față de 35% la întâmplare.
