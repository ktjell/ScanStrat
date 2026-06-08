from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ExchangeRate:
    """Commission rate for a single exchange."""

    pct: float
    """Commission as a fraction of trade value, e.g. 0.001 = 0.10%."""

    min_usd: float
    """Minimum commission per trade, expressed in USD."""


@dataclass
class CommissionSchedule:
    """Per-exchange commission schedule.

    Usage
    -----
    schedule = CommissionSchedule.saxo_classic()
    effective_pct = schedule.effective_pct("AAPL", position_value_usd=5000)
    """

    rates: dict[str, ExchangeRate] = field(default_factory=dict)
    default: ExchangeRate = field(default_factory=lambda: ExchangeRate(0.001, 4.0))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def effective_pct(self, ticker: str, position_value_usd: float) -> float:
        """Return the effective commission as a fraction of *position_value_usd*.

        The minimum commission floor is applied when the percentage amount
        would fall below it.

        Parameters
        ----------
        ticker:
            Ticker symbol, e.g. ``"AAPL"``, ``"NOVO-B.CO"``, ``"SAP.DE"``.
        position_value_usd:
            Value of the position in USD at trade time.
        """
        if position_value_usd <= 0:
            return self.default.pct

        rate = self._rate_for(ticker)
        dollar_commission = rate.pct * position_value_usd
        actual = max(dollar_commission, rate.min_usd)
        return actual / position_value_usd

    def _rate_for(self, ticker: str) -> ExchangeRate:
        exchange = _detect_exchange(ticker)
        return self.rates.get(exchange, self.default)

    # ------------------------------------------------------------------
    # Saxo Bank presets
    # Rates as published at home.saxo/rates-and-conditions/stocks/commissions
    # Min amounts converted to USD at approximate mid-2025 rates.
    # ------------------------------------------------------------------

    @classmethod
    def saxo_classic(cls) -> CommissionSchedule:
        """Saxo Bank Classic account commission schedule."""
        return cls(
            rates=_SAXO_CLASSIC_RATES,
            default=ExchangeRate(0.0008, 1.0),  # US fallback
        )

    @classmethod
    def saxo_platinum(cls) -> CommissionSchedule:
        """Saxo Bank Platinum account commission schedule."""
        return cls(
            rates=_SAXO_PLATINUM_RATES,
            default=ExchangeRate(0.0005, 1.0),
        )

    @classmethod
    def saxo_vip(cls) -> CommissionSchedule:
        """Saxo Bank VIP account commission schedule."""
        return cls(
            rates=_SAXO_VIP_RATES,
            default=ExchangeRate(0.0003, 1.0),
        )

    @classmethod
    def zero(cls) -> CommissionSchedule:
        """No commission (for baseline comparisons)."""
        return cls(default=ExchangeRate(0.0, 0.0))


# ---------------------------------------------------------------------------
# Exchange detection
# ---------------------------------------------------------------------------

_SUFFIX_TO_EXCHANGE: dict[str, str] = {
    # Nordic
    "CO": "DK",  # Nasdaq Copenhagen
    "ST": "SE",  # Nasdaq Stockholm
    "HE": "FI",  # Nasdaq Helsinki
    "OL": "NO",  # Oslo Børs
    # UK
    "L": "UK",  # London Stock Exchange
    # Continental Europe
    "DE": "DE",  # XETRA
    "PA": "FR",  # Euronext Paris
    "AS": "NL",  # Euronext Amsterdam
    "BR": "BE",  # Euronext Brussels
    "MI": "IT",  # Borsa Italiana
    "MC": "ES",  # Bolsa de Madrid
    "SW": "CH",  # SIX Swiss Exchange
    # Asia-Pacific
    "T": "JP",  # Tokyo Stock Exchange
    "HK": "HK",  # Hong Kong Stock Exchange
    "AX": "AU",  # Australian Securities Exchange
    # Canada
    "TO": "CA",
    "V": "CA",
}


def _detect_exchange(ticker: str) -> str:
    """Return an exchange code from the ticker suffix.

    US tickers have no suffix → returns ``"US"``.
    """
    if "-" in ticker:
        # e.g. BRK-B → US (yfinance normalisation of BRK.B)
        # but NOVO-B.CO → strip the suffix first
        parts = ticker.rsplit(".", 1)
        if len(parts) == 2:
            suffix = parts[1].upper()
            return _SUFFIX_TO_EXCHANGE.get(suffix, "OTHER")
        return "US"

    if "." in ticker:
        suffix = ticker.rsplit(".", 1)[1].upper()
        return _SUFFIX_TO_EXCHANGE.get(suffix, "OTHER")

    return "US"


# ---------------------------------------------------------------------------
# Saxo rate tables — source: home.saxo/da-dk/rates-and-conditions/stocks/commissions
# Min amounts converted to USD at approximate mid-2026 rates:
#   DKK 10  ≈ $1.45   (DKK/USD ≈ 0.145)
#   SEK 10  ≈ $0.95   (SEK/USD ≈ 0.095)
#   NOK 10  ≈ $0.95   (NOK/USD ≈ 0.095)
#   EUR 2   ≈ $2.20   (EUR/USD ≈ 1.10)
#   EUR 3   ≈ $3.30
#   EUR 5   ≈ $5.50
#   GBP 3   ≈ $3.80   (GBP/USD ≈ 1.27)
#   CHF 3   ≈ $3.40   (CHF/USD ≈ 1.13)
#   AUD 3   ≈ $2.00   (AUD/USD ≈ 0.66)
#   HKD 15  ≈ $1.92   (HKD/USD ≈ 0.128)
#   CAD 5   ≈ $3.70   (CAD/USD ≈ 0.73)
#   JPY 800 ≈ $5.50   (JPY/USD ≈ 0.0069)
# ---------------------------------------------------------------------------

_SAXO_CLASSIC_RATES: dict[str, ExchangeRate] = {
    # 0.08% — same for US, Nordic, most European and Asia-Pacific
    "US": ExchangeRate(0.0008, 1.00),  # min 1 USD
    "DK": ExchangeRate(0.0008, 1.45),  # min DKK 10
    "SE": ExchangeRate(0.0008, 0.95),  # min SEK 10
    "NO": ExchangeRate(0.0008, 0.95),  # min NOK 10
    "FI": ExchangeRate(0.0008, 3.30),  # min EUR 3
    "UK": ExchangeRate(0.0008, 3.80),  # min GBP 3
    "DE": ExchangeRate(0.0008, 3.30),  # min EUR 3
    "FR": ExchangeRate(0.0008, 2.20),  # min EUR 2
    "NL": ExchangeRate(0.0008, 2.20),  # min EUR 2
    "BE": ExchangeRate(0.0008, 2.20),  # min EUR 2
    "IT": ExchangeRate(0.0008, 3.30),  # min EUR 3
    "ES": ExchangeRate(0.0012, 5.50),  # 0.12%, min EUR 5
    "CH": ExchangeRate(0.0008, 3.40),  # min CHF 3
    "HK": ExchangeRate(0.0008, 1.92),  # min HKD 15
    "AU": ExchangeRate(0.0008, 2.00),  # min AUD 3
    "CA": ExchangeRate(0.0008, 3.70),  # min CAD 5
    "JP": ExchangeRate(0.0008, 5.50),  # min JPY 800
}

_SAXO_PLATINUM_RATES: dict[str, ExchangeRate] = {
    "US": ExchangeRate(0.0005, 1.00),
    "DK": ExchangeRate(0.0005, 1.45),
    "SE": ExchangeRate(0.0005, 0.95),
    "NO": ExchangeRate(0.0005, 0.95),
    "FI": ExchangeRate(0.0005, 3.30),
    "UK": ExchangeRate(0.0005, 3.80),
    "DE": ExchangeRate(0.0005, 3.30),
    "FR": ExchangeRate(0.0005, 2.20),
    "NL": ExchangeRate(0.0005, 2.20),
    "BE": ExchangeRate(0.0005, 2.20),
    "IT": ExchangeRate(0.0005, 3.30),
    "ES": ExchangeRate(0.0008, 5.50),  # 0.08%, min EUR 5
    "CH": ExchangeRate(0.0005, 3.40),
    "HK": ExchangeRate(0.0005, 1.92),
    "AU": ExchangeRate(0.0005, 2.00),
    "CA": ExchangeRate(0.0005, 3.70),
    "JP": ExchangeRate(0.0005, 5.50),
}

_SAXO_VIP_RATES: dict[str, ExchangeRate] = {
    "US": ExchangeRate(0.0003, 1.00),
    "DK": ExchangeRate(0.0003, 1.45),
    "SE": ExchangeRate(0.0003, 0.95),
    "NO": ExchangeRate(0.0003, 0.95),
    "FI": ExchangeRate(0.0003, 3.30),
    "UK": ExchangeRate(0.0003, 3.80),
    "DE": ExchangeRate(0.0003, 3.30),
    "FR": ExchangeRate(0.0003, 2.20),
    "NL": ExchangeRate(0.0003, 2.20),
    "BE": ExchangeRate(0.0003, 2.20),
    "IT": ExchangeRate(0.0003, 3.30),
    "ES": ExchangeRate(0.0005, 5.50),  # 0.05%, min EUR 5
    "CH": ExchangeRate(0.0003, 3.40),
    "HK": ExchangeRate(0.0003, 1.92),
    "AU": ExchangeRate(0.0003, 2.00),
    "CA": ExchangeRate(0.0003, 3.70),
    "JP": ExchangeRate(0.0003, 5.50),
}
