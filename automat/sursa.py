"""De unde vin rezultatele extragerilor.

extrage_rezultate() întoarce o listă de perechi (slot, număr), de exemplu
[("2026-10-08 14:00", 7), ("2026-10-08 15:00", 12)].
Slotul e ora extragerii, în formatul "YYYY-MM-DD HH:MM", ora României.
Pot fi și extrageri mai vechi: cele care există deja în history.csv sunt ignorate.
"""


def extrage_rezultate():
    raise NotImplementedError("Sursa rezultatelor nu e configurată încă (automat/sursa.py).")
