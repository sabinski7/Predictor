"""Actualizare automată a predictorului.

1. Citește extragerile noi din sursă (automat/sursa.py) sau primește un număr cu --numar.
2. Le scrie în history.csv (fără duplicate).
3. Rulează Predictor.ipynb, care salvează propunerile pentru extragerea următoare în predictions_log.csv.
4. Trimite pe Telegram propunerile și rezultatul ultimei extrageri.

Rulare:
    python automat/actualizeaza.py               # citește sursa
    python automat/actualizeaza.py --numar 7     # adaugă 7 pentru următoarea extragere
    python automat/actualizeaza.py --test        # trimite propunerile actuale, fără extragere nouă
"""
import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import nbformat
import pandas as pd
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent
HISTORY = ROOT / "history.csv"
LOG = ROOT / "predictions_log.csv"
NOTEBOOK = ROOT / "Predictor.ipynb"
PRIMA_ORA, ULTIMA_ORA = 7, 23
K = 20

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sursa import extrage_rezultate  # noqa: E402


def load_history():
    df = pd.read_csv(HISTORY, dtype={"Date": str, "Time": str})
    df["dt"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%Y-%m-%d %H:%M")
    return df


def next_slot(dt):
    n = dt + pd.Timedelta(hours=1)
    if n.hour > ULTIMA_ORA:
        n = n.normalize() + pd.Timedelta(days=1, hours=PRIMA_ORA)
    elif n.hour < PRIMA_ORA:
        n = n.normalize() + pd.Timedelta(hours=PRIMA_ORA)
    return n


def add_draws(draws):
    """Adaugă extragerile care lipsesc. Întoarce lista celor adăugate."""
    df = load_history()
    existing = dict(zip(df["dt"], df["Drawn Number"]))
    added = []
    for slot, num in sorted(draws, key=lambda d: pd.Timestamp(d[0])):
        dt, num = pd.Timestamp(slot), int(num)
        if not 1 <= num <= K or not (PRIMA_ORA <= dt.hour <= ULTIMA_ORA and dt.minute == 0):
            print(f"⚠️  ignor {slot} → {num} (număr sau oră invalidă)")
            continue
        if dt in existing:
            if existing[dt] != num:
                print(f"⚠️  {slot} are deja {existing[dt]} în istoric (sursa spune {num}); nu suprascriu")
            continue
        existing[dt] = num
        added.append((dt, num))
    if added:
        new = pd.DataFrame({"Date": [d.strftime("%Y-%m-%d") for d, _ in added],
                            "Time": [d.strftime("%H:%M") for d, _ in added],
                            "Drawn Number": [n for _, n in added]})
        out = pd.concat([df[["Date", "Time", "Drawn Number"]], new], ignore_index=True)
        out["dt"] = pd.to_datetime(out["Date"] + " " + out["Time"])
        out.sort_values("dt").drop(columns="dt").to_csv(HISTORY, index=False)
        for d, n in added:
            print(f"✅ adăugat {d:%Y-%m-%d %H:%M} → {n}")
    return added


def run_notebook():
    nb = nbformat.read(NOTEBOOK, as_version=4)
    NotebookClient(nb, timeout=900, kernel_name="python3",
                   resources={"metadata": {"path": str(ROOT)}}).execute()
    print("📓 notebook rulat")


def build_message(added=()):
    df = load_history()
    real = dict(zip(df["dt"].dt.strftime("%Y-%m-%d %H:%M"), df["Drawn Number"]))
    log = pd.read_csv(LOG, dtype=str)
    nxt = f"{next_slot(df['dt'].iloc[-1]):%Y-%m-%d %H:%M}"
    row = log[log["slot"] == nxt]
    added = sorted(added)
    lines = []
    if len(row):
        props = row["propuneri"].iloc[0].split()
        lines.append(f"🎯 Propuneri pentru {pd.Timestamp(nxt):%d.%m %H:%M}")
        lines.append("   " + " · ".join(props))
    else:
        lines.append(f"⚠️ Nu găsesc propuneri pentru {nxt} în jurnal.")

    if not added:
        added = [(df["dt"].iloc[-1], int(df["Drawn Number"].iloc[-1]))]
    lines.append("")
    for dt, num in added[-5:]:
        prev = log[log["slot"] == f"{dt:%Y-%m-%d %H:%M}"]
        verdict = ""
        if len(prev):
            verdict = "  ✅ NIMERIT" if num in map(int, prev["propuneri"].iloc[0].split()) else "  ❌ ratat"
        lines.append(f"Extragerea {dt:%d.%m %H:%M} → {num}{verdict}")
    if len(added) > 5:
        lines.append(f"(+ încă {len(added) - 5} extrageri mai vechi adăugate)")

    log["rez"] = log["slot"].map(real)
    done = log.dropna(subset=["rez"]).sort_values("slot")
    if len(done):
        hits = [int(r) in map(int, p.split()) for r, p in zip(done["rez"], done["propuneri"])]
        streak = 0
        for h in reversed(hits):
            if h:
                break
            streak += 1
        k = int(done["k"].iloc[-1])
        q = 1 - k / K
        level = "🟢" if q ** streak > 0.10 else ("🟡" if q ** streak > 0.02 else "🔴")
        if streak:
            lines.append(f"Serie: {streak} {'tură ratată' if streak == 1 else 'ture ratate'} la rând {level}")
        exp = sum(int(v) / K for v in done["k"])
        lines.append(f"Jurnal: {sum(hits)}/{len(hits)} nimerite ({100 * sum(hits) / len(hits):.1f}%), "
                     f"șansa pură ≈ {100 * exp / len(hits):.0f}%")
    return "\n".join(lines)


def send_telegram(text):
    token = (os.environ.get("TELEGRAM_TOKEN") or "").strip()
    chat_raw = (os.environ.get("TELEGRAM_CHAT_ID") or "").strip()
    chat = "".join(re.findall(r"-?\d+", chat_raw)[:1])          # păstrează doar numărul (ex. „Id: 123” → 123)
    print("\n" + text + "\n")
    if not token or not chat:
        gh_note("warning", "TELEGRAM_TOKEN sau TELEGRAM_CHAT_ID lipsesc din secretele repo-ului: mesajul nu a fost trimis.")
        return
    data = urllib.parse.urlencode({"chat_id": chat, "text": text}).encode()
    try:
        with urllib.request.urlopen(f"https://api.telegram.org/bot{token}/sendMessage", data, timeout=30) as r:
            print("📨 trimis pe Telegram")
            gh_note("notice", "Mesaj trimis pe Telegram.")
    except urllib.error.HTTPError as e:
        try:
            desc = json.loads(e.read().decode()).get("description", "")
        except Exception:
            desc = ""
        hint = {
            401: "tokenul e greșit: copiază-l din nou de la @BotFather.",
            404: "tokenul e greșit sau incomplet: copiază-l din nou de la @BotFather.",
            403: "botul nu are voie să-ți scrie: deschide botul tău în Telegram și apasă Start.",
            400: "chat id-ul nu e bun: verifică numărul de la @userinfobot și apasă Start la botul tău.",
        }.get(e.code, "")
        raise RuntimeError(f"Telegram a refuzat mesajul (HTTP {e.code}: {desc}). {hint} "
                           f"[chat id folosit: {chat[:3]}…{chat[-2:]}, {len(chat)} cifre] "
                           + telegram_diagnostic(token, chat)) from None


def telegram_diagnostic(token, chat):
    """Ce știe botul: numele lui și de la cine a primit mesaje recent (fără să afișeze Id-uri complete)."""
    def call(method):
        with urllib.request.urlopen(f"https://api.telegram.org/bot{token}/{method}", timeout=30) as r:
            return json.loads(r.read().decode())["result"]
    try:
        me = call("getMe")
        upd = call("getUpdates")
    except Exception as e:
        return f"(diagnostic indisponibil: {e})"
    senders = {str(u[k]["chat"]["id"]) for u in upd for k in ("message", "my_chat_member") if k in u}
    bot = f"@{me.get('username')}"
    if not senders:
        return (f"Diagnostic: tokenul aparține botului {bot}, care NU a primit niciun mesaj recent. "
                f"Deschide {bot} în Telegram, apasă Start și trimite-i orice mesaj, apoi rulează din nou.")
    if chat in senders:
        return f"Diagnostic: {bot} a primit mesaje de la acest Id; încearcă din nou peste un minut."
    masked = ", ".join(f"{c[:3]}…{c[-2:]} ({len(c)} cifre)" for c in sorted(senders))
    return (f"Diagnostic: {bot} a primit mesaje de la alt Id: {masked}. "
            f"Verifică secretul TELEGRAM_CHAT_ID (să fie Id-ul contului cu care scrii botului).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--numar", type=int, help="numărul extras la următoarea extragere (1–20)")
    ap.add_argument("--test", action="store_true", help="rulează și trimite mesajul chiar fără extragere nouă")
    args = ap.parse_args()

    if args.numar is not None:
        slot = next_slot(load_history()["dt"].iloc[-1])
        draws = [(f"{slot:%Y-%m-%d %H:%M}", args.numar)]
    else:
        try:
            draws = extrage_rezultate()
        except NotImplementedError as e:
            print(f"ℹ️  {e}")
            draws = []
        except OSError as e:          # site indisponibil / rețea: încercăm din nou la următoarea rulare
            print(f"⚠️  Nu am putut citi site-ul ({e}); încerc din nou la următoarea verificare.")
            draws = []

    gh_note("notice", f"Sursa: {len(draws)} rezultate citite" + (f", ultimul {max(draws)[0]} → {max(draws)[1]}" if draws else ""))
    # Turele noi se procesează pe rând, în ordine cronologică: pentru fiecare se adaugă rezultatul,
    # se rulează notebook-ul (propunerile pentru tura următoare) și se trimit mesajele.
    # Așa, dacă 12:00 și 13:00 apar deodată pe site, vin întâi 12:00 + propunerile pentru 13:00,
    # apoi 13:00 (verificat față de acele propuneri) + propunerile pentru 14:00.
    any_added = False
    for slot, num in sorted(draws, key=lambda d: pd.Timestamp(d[0])):
        added = add_draws([(slot, num)])
        if not added:
            continue
        any_added = True
        run_notebook()
        send_both(added)
    if not any_added:
        if not args.test:
            print("Nicio extragere nouă: nu rulez nimic.")
            return
        run_notebook()
        send_both([])


def send_both(added):
    send_telegram(build_message(added))
    nums = numbers_only()
    if nums:
        send_telegram(nums)


SEPARATOR = "\t·\t"   # tab + punct vizibil: în Excel numerele ajung în celule separate, punctele în celulele dintre ele


def numbers_only():
    """Al doilea mesaj: doar propunerile pentru extragerea următoare, ușor de copiat în Excel."""
    df = load_history()
    nxt = f"{next_slot(df['dt'].iloc[-1]):%Y-%m-%d %H:%M}"
    log = pd.read_csv(LOG, dtype=str)
    row = log[log["slot"] == nxt]
    return SEPARATOR.join(row["propuneri"].iloc[0].split()) if len(row) else ""


def gh_note(level, msg):
    """Pe GitHub Actions, mesajul apare ca adnotare pe rulare (vizibilă și fără jurnalul complet)."""
    if os.environ.get("GITHUB_ACTIONS"):
        print(f"::{level}::" + msg.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A"))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        gh_note("error", f"{type(e).__name__}: {e}\n" + "".join(traceback.format_exc().splitlines(True)[-12:]))
        raise
