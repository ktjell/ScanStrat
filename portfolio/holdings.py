"""portfolio/holdings.py

Dine nuværende beholdninger.
Opdater listen naar du koeber/saelger.

Yahoo Finance ticker-format:
  - Kobenhavn:  NOVO-B.CO, ROCK-B.CO
  - Amsterdam:  AVTX.AS
  - XETRA:      WEBN.DE, QDVE.DE, VWCE.DE, EUDF.DE
  - USA:        SOUN
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Holding:
    ticker: str  # Yahoo Finance ticker
    name: str  # Visningsnavn
    shares: float = 0.0  # Antal aktier (valgfri — bruges til DKK-vaerdi)
    avg_cost: float = 0.0  # Gns. købskurs i lokal valuta (valgfri)


HOLDINGS: list[Holding] = [
    Holding("AVTX.AS", "Avantis (AMS)"),
    Holding("WEBN.DE", "WisdomTree Europe Smallcap Div (XETRA)"),
    Holding("QDVE.DE", "iShares S&P 500 IT Sector (XETRA)"),
    Holding("SOUN", "SoundHound AI (NASDAQ)"),
    Holding("VWCE.DE", "Vanguard FTSE All-World (XETRA)"),
    Holding("EUDF.DE", "iShares MSCI Europe Dividend (XETRA)"),
    Holding("NOVO-B.CO", "Novo Nordisk B (CPH)"),
    Holding("ROCK-B.CO", "Rockwool B (CPH)"),
]
