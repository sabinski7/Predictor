# Predictor

Notebook Deepnote care propune până la 9 numere (1–20) pentru următoarea extragere orară (07:00–23:00).

## Utilizare în Deepnote

1. Importă `Predictor.ipynb` și urcă `history.csv` în același director.
2. În celula **1 · Configurare**:
   - `NUM_PROPUNERI`: câte numere vrei (1–9; implicit 9);
   - `NUMERE_NOI`: extragerile noi, de forma `"2026-04-29 23:00": 5`.
3. **Run all** (~10 secunde).
4. După fiecare extragere, rulează ultima celulă (**10 · Introdu ultimul număr extras**). În output apare o căsuță: scrii numărul (1–20) și apeși Enter. Numărul se salvează în istoric și primești imediat propunerile pentru extragerea următoare. Tot acolo apare **alerta de serie**: la câte ratări la rând ești și cât de rară e seria (doar informativ: seria nu schimbă șansa următoarei ture).

## Ce face

- Curăță istoricul: duplicate, ordine, sloturi lipsă. Adăugarea e idempotentă: rularea repetată nu dublează rânduri.
- Rulează teste de aleatorism: uniformitate, serial, oră din zi, zi a săptămânii.
- Șapte modele (uniform, frecvență globală și recentă, Markov-1/2, oră din zi, întârziere), evaluate walk-forward și combinate într-un ansamblu. Ponderile sunt calibrate pe date pe care ansamblul nu le testează.
- Afișează propunerile împreună cu rata reală de nimerire din backtest și cu șansa pură (K × 5%).
- Secțiunea **9 · Câte ture sunt între nimeriri?** arată distribuția pauzelor dintre nimeriri (backtest vs. teorie), șansa de nimerire după k ratări la rând și pauzele tale reale din jurnal.
- Secțiunea **9b** urmărește tiparul „multe ratări → o nimerire → multe ratări” pe tot istoricul și în jurnalul tău.
- Secțiunea **9c · Detector de schimbare** verifică ultimele 500 și 1.700 de extrageri (uniformitate, repetări, diferențe, oră, modele vs. șansă) și te anunță dacă jocul începe să se comporte altfel.
- Ponderile ansamblului pun accent pe perioada recentă (`TIMP_INJUMATATIRE`, implicit 1.000 de extrageri), ca să se adapteze repede dacă apare un tipar nou.
- Secțiunea **9d · Simulator de sisteme** testează ipotetic orice sistem de pariere (când pariezi: mereu / după ratări / după nimeriri / la anumite ore; cum pariezi: fix / martingale / invers / procent / Kelly) pe istoricul real și pe serii aleatoare, cu bancă, obiectiv și limită de pierdere. Parametrul `sansa_ipotetica` arată cum ar merge un sistem dacă ar exista un avantaj.
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
| `explorare7.py` | „numerele rămase în urmă recuperează?” (ultimele 100, ultimele 500, tot istoricul) | nu: cele mai reci 7 ies 35,0–35,5% în următoarele 100 de extrageri; diferențele în bucăți cresc, doar procentele se egalează |
| `explorare8_pauze_consecutive.py` | după o pauză lungă urmează tot o pauză lungă? (ansamblul vs. 120 de serii aleatoare trecute prin același mecanism) | legătură slabă negativă (r ≈ −0,02; p 0,02–0,10) care nu apare pe seturi fixe sau pe serii aleatoare; prea mică pentru a fi folosită, urmărită în secțiunea 9b |
| `explorare9_reselectie.py` | re-alegerea celei mai bune dintre 48 de strategii după fiecare extragere (pe baza ultimelor 17–2.000 ture) | 44,8–45,4% cu 9 propuneri, față de 45% șansa pură; niciun câștig |
| `explorare10_combinatii.py` | 26 de modele (inclusiv variante „reci” și ferestre), toate submulțimile celor 6 modele din notebook × 4 metode de combinare, toate perechile; 2.316 strategii evaluate după profit cu cotele reale | nicio strategie nu e pe profit în test; cea mai bună pe trecut pierde 6–8% în test |
| `explorare11_detector_retro.py` | detectorul de schimbare (9c) rulat retroactiv săptămânal (248 verificări) și zilnic (1.732), comparat cu istorii simulate | zilnic: 22 🟡, 0 🚨 (1,3% din zile, față de 3,4% pe serii aleatoare); după cele două perioade în care ansamblul „bătea șansa” (mai 2024, apr. 2026) rata a revenit imediat la ~45% |
| `explorare12_reguli_propuneri.py` | 1.700 de reguli între propuneri și numărul ieșit („= primul propus − 1”, „= p3 + p7 − 3”, suma, min, max…), găsite în jurnal și verificate pe 31.340 de ture | cea mai bună din jurnal (9,6%) dă 5,1% pe istoric; nicio regulă nu bate 5% |
| `explorare13_valuri.py` | strategia „în valuri”: la fiecare tură se alege, dintre 1.700 de reguli, cea mai bună pe ultimele 5–3.000 de ture | 4,7–5,3% față de 5%; vârful de la fereastra de 3 zile (5,28%) nu rezistă (4,84% pe a doua jumătate a istoricului) |
| `explorare14_urmatoarea_regula.py` | (A) regula care, istoric, a urmat după regula cea mai bună recentă; (B) regula cea mai slabă recent | A: 4,6–5,3%, B: 4,7–5,2%, față de 5%; nicio variantă nu bate șansa constant |
| `explorare15_doar_istoric.py` | Markov de ordin 1–6 (stil PPM), modele cu regimuri ascunse (HMM cu 2–4 stări), 340 de combinații oră × lag, baterie tip NIST (entropie aproximativă, triplete, sume cumulative, cele mai lungi pauze) | top-9 între 44,5% și 45,5% pe ultimele 10.000 de ture, față de 45%; niciun test de aleatorism nu pică după corecția pentru teste multiple |
| `explorare16_retea_secventiala.py` | rețea neuronală secvențială (GRU): ultimele 68 de numere + ora, cu oprire timpurie | log-loss mai slab decât uniform, top-9 44,3% față de 45% |
| `explorare17_top_reguli_propuneri.py` | 10.040 de reguli de tipul „p1 + p8 + 9” (pi + c, 2·pi, pi ± pj, trei propuneri, suma/min/max), căutate pe jumătate din istoric și verificate pe cealaltă jumătate și pe jurnal | top 10 la căutare: 5,57% → 5,07% la verificare; cea mai bună pe tot istoricul (5,48%) e exact cât dă hazardul când numerele sunt amestecate (5,48%) |
| `explorare18_pauze_scurte.py` | top 10 reguli din propuneri (57.360: combinații ± de 1–4 propuneri, ±2·pi, + constantă) cu cea mai scurtă pauză maximă între nimeriri în jurnal, plus ce prezic pentru tura următoare | descrie jurnalul de până acum; pe numere amestecate cea mai bună regulă are tot ~27 de ture pauză maximă |
| `explorare19_dezvoltare_reguli.py` | 8 variante ale strategiei cu top 10 reguli (top 5/20, numere cu 2+ voturi, după nimeriri, „fierbinți”, „restante”, cele care au nimerit recent), pe jurnal și pe 31.544 de ture din istoric | eficiență 0,98–1,01 pe istoric (1,00 = hazard); cea mai bună pe jurnal (H, 1,095) e cea mai slabă pe istoric (0,981) |

Concluzie: generatorul se comportă aleator, iar acuratețea rămâne la nivelul șansei pure (top-7 ~35%, top-9 ~45%).

## Automatizare: GitHub Actions + Telegram

Workflow-ul `.github/workflows/predictor.yml` rulează la fiecare 10 minute, pentru că rezultatele apar pe site la ore variabile (de obicei la :15–:20, uneori abia la ora următoare):

1. citește Numerone din arhiva Win for Life Classico (`automat/sursa.py`, pagina https://www.winforlife.it/archivio-estrazioni-classico, ultimele ~30 de concursuri, ora Italiei); dacă apar deodată mai multe, de exemplu 12:00 întârziată împreună cu 13:00, le adaugă pe toate, în ordine;
2. le adaugă în `history.csv`, fără duplicate;
3. rulează `Predictor.ipynb`, care salvează propunerile în `predictions_log.csv`;
4. îți trimite pe Telegram propunerile pentru extragerea următoare și rezultatul ultimei extrageri;
   apoi un al treilea mesaj (`automat/pauze.py`): top 10 reguli din propuneri cu cele mai scurte pauze între nimeriri în jurnal, recalculat după fiecare tură, cu numărul prezis de fiecare regulă, ce reguli din top-ul anterior au nimerit și ce reguli au intrat sau au ieșit din top;
   top-ul fiecărei ture se salvează și în `top_reguli.csv` (tura, locul, regula, numărul prezis, pauza maximă, pauza curentă, nimeriri, câte ture avea jurnalul, numărul ieșit, nimerit da/nu), inclusiv pentru turele din trecut, de la a 50-a tură din jurnal;
5. salvează `history.csv` și `predictions_log.csv` în repo.

Dacă nu a apărut o extragere nouă, nu face nimic și nu trimite niciun mesaj.

### Configurare Telegram (o singură dată, ~5 minute)

1. În Telegram, deschide **@BotFather**, trimite `/newbot` și urmează pașii. La final primești un **token**, de forma `123456789:AA...`.
2. Deschide botul nou creat și apasă **Start**. Fără pasul ăsta, botul nu îți poate scrie.
3. Deschide **@userinfobot** și apasă **Start**. Îți răspunde cu **Id**-ul tău, un număr.
4. Pe GitHub, în repo: **Settings → Secrets and variables → Actions → New repository secret**, adaugă:
   - `TELEGRAM_TOKEN` = tokenul de la pasul 1
   - `TELEGRAM_CHAT_ID` = Id-ul de la pasul 3
5. Test: **Actions → Predictor automat → Run workflow**, bifează „Trimite propunerile acum” și apasă **Run workflow**. În ~1 minut primești mesajul.

### Introducere manuală, de pe telefon

Dacă sursa nu merge, poți introduce numărul din aplicația GitHub sau din browser: **Actions → Predictor automat → Run workflow**, scrii numărul în câmpul „Numărul extras” și pornești rularea.

### Observații

- GitHub poate întârzia rulările programate cu câteva minute, uneori mai mult în orele aglomerate.
- Repo-ul e public: istoricul și jurnalul se văd public. Tokenul Telegram rămâne secret.
- `history.csv` și `predictions_log.csv` din repo devin sursa principală. Pentru Deepnote, descarcă-le de aici.

## Vânătorul de reguli (automat/vanator.py)

Rulează zilnic, după extragerea de la 23:00, și caută reguli care ar prezice Numerone mai bine decât șansa. O regulă e acceptată doar dacă trece trei etape, fiecare pe date nevăzute de etapa anterioară:

1. **Căutare**: ~14.000 de reguli pe istoric, fără ultimele 10.000 de ture. Familii de reguli: numărul de acum L ture × a + c, combinații de două ture anterioare, „la ora H joacă N”, aceeași oră de acum D zile, locul în frecvență sau întârziere, seturi de 9 numere calde, reci sau întârziate. Se rețin cele mai bune 50.
2. **Validare**: acele 50, pe ultimele 10.000 de ture. Trec doar cele cu p < 0,0002 (corectat pentru 50 de încercări).
3. **Probă**: regula e înghețată și urmărită pe extrageri reale noi. E confirmată la p < 0,001, după cel puțin 100 de ture în care se aplică. E respinsă dacă nu e peste șansă sau dacă trec ~200 de zile fără confirmare.

Pe Telegram primești: „🔬 candidat în probă” (neconfirmat), „🚨 regulă confirmată” sau „❌ respins”, plus un raport scurt duminica. Regulile confirmate apar și în mesajul de la fiecare extragere, cu numărul pe care îl prezic. Starea e salvată în `automat/vanator_stare.json`, raportul în `automat/vanator_raport.md`.

Verificare: pe istoricul real nu trece nicio regulă. Pe istorice cu defecte plantate („la ora 15 iese 7 de două ori mai des”, „numărul anterior se repetă în 7% din ture”) defectele sunt găsite din prima zi și confirmate după ~110–120 de zile, iar candidații falși sunt respinși în probă.
