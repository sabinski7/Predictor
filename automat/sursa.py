"""De unde vin rezultatele extragerilor: arhiva Win for Life Classico de pe winforlife.it.

extrage_rezultate() întoarce o listă de perechi (slot, Numerone), de exemplu
[("2026-10-08 14:00", 15), ("2026-10-08 15:00", 8)].
Orele sunt cele de pe site, adică ora Italiei, la fel ca în history.csv.
Pagina arată ultimele ~30 de concursuri; cele care există deja în history.csv sunt ignorate.
"""
import re
import urllib.request

URL = "https://www.winforlife.it/archivio-estrazioni-classico"
LUNI = {"gennaio": 1, "febbraio": 2, "marzo": 3, "aprile": 4, "maggio": 5, "giugno": 6,
        "luglio": 7, "agosto": 8, "settembre": 9, "ottobre": 10, "novembre": 11, "dicembre": 12}

ROW = re.compile(r'<tr class="wfl-extraction-archive__details__table__row[^"]*">(.*?)</tr>', re.S)
DATA = re.compile(r"__body__label.*?del\s*<strong>\s*(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})\s*</strong>", re.S)
ORA = re.compile(r"__body__hour.*?<strong>\s*(\d{1,2}):(\d{2})\s*</strong>", re.S)
NUMERONE = re.compile(r'--numerone"[^>]*>\s*<span[^>]*>\s*(\d{1,2})\s*</span>', re.S)
COMB = re.compile(r'__body__combination">(.*?)</td>', re.S)
CONC = re.compile(r"N\D{0,3}(\d+)</strong>")
NUMS = re.compile(r'combination__text">\s*(\d{1,2})\s*<')


def parse(html):
    out = []
    for row in ROW.findall(html):
        d, o, n = DATA.search(row), ORA.search(row), NUMERONE.search(row)
        if not (d and o and n):
            continue
        luna = LUNI.get(d.group(2).lower())
        if not luna:
            continue
        slot = f"{int(d.group(3)):04d}-{luna:02d}-{int(d.group(1)):02d} {int(o.group(1)):02d}:{o.group(2)}"
        out.append((slot, int(n.group(1))))
        comb = COMB.search(row)
        nums = [int(v) for v in NUMS.findall(comb.group(1))] if comb else []
        conc = CONC.search(row)
        if len(nums) == 10:
            COMBINATII[slot] = (conc.group(1) if conc else "", nums, int(n.group(1)))
    return sorted(set(out))


COMBINATII = {}      # slot -> (nr. concurs, cele 10 numere, Numerone), completat de parse()


def save_combinations(path):
    """Adaugă în CSV combinațiile complete noi (slot, concurs, n1..n10, numerone)."""
    import csv, os
    known = set()
    if os.path.exists(path):
        with open(path) as f:
            known = {r[0] for r in csv.reader(f)}
    new = [(s, *COMBINATII[s]) for s in sorted(COMBINATII) if s not in known]
    if not new:
        return 0
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["slot", "concurs"] + [f"n{i}" for i in range(1, 11)] + ["numerone"])
        for s, conc, nums, nm in new:
            w.writerow([s, conc] + nums + [nm])
    return len(new)


def extrage_rezultate():
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0 (predictor personal; o cerere la 10 minute)"})
    with urllib.request.urlopen(req, timeout=30) as r:
        html = r.read().decode("utf-8", errors="replace")
    rez = parse(html)
    if not rez:
        raise RuntimeError("Nu am găsit niciun rezultat pe pagină: probabil s-a schimbat structura site-ului.")
    return rez


if __name__ == "__main__":
    for slot, n in extrage_rezultate():
        print(slot, n)
