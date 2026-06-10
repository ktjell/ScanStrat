"""Tests for backtest.commission — exchange detection and commission calculation."""

from __future__ import annotations

import pytest

from backtest.commission import CommissionSchedule, _detect_exchange


# ---------------------------------------------------------------------------
# Exchange detection
# ---------------------------------------------------------------------------


class TestDetectExchange:
    def test_us_no_suffix(self):
        assert _detect_exchange("AAPL") == "US"

    def test_us_hyphen_no_dot(self):
        assert _detect_exchange("BRK-B") == "US"

    def test_dk(self):
        assert _detect_exchange("NOVO-B.CO") == "DK"

    def test_dk_simple(self):
        assert _detect_exchange("CARL-B.CO") == "DK"

    def test_se(self):
        assert _detect_exchange("VOLV-B.ST") == "SE"

    def test_no(self):
        assert _detect_exchange("EQNR.OL") == "NO"

    def test_de(self):
        assert _detect_exchange("SAP.DE") == "DE"

    def test_uk(self):
        assert _detect_exchange("SHEL.L") == "UK"

    def test_fr(self):
        assert _detect_exchange("MC.PA") == "FR"

    def test_nl(self):
        assert _detect_exchange("ASML.AS") == "NL"

    def test_ch(self):
        assert _detect_exchange("NESN.SW") == "CH"

    def test_jp(self):
        assert _detect_exchange("7203.T") == "JP"

    def test_hk(self):
        assert _detect_exchange("0700.HK") == "HK"

    def test_au(self):
        assert _detect_exchange("BHP.AX") == "AU"

    def test_ca(self):
        assert _detect_exchange("SHOP.TO") == "CA"

    def test_unknown_suffix(self):
        assert _detect_exchange("XYZ.ZZ") == "OTHER"


# ---------------------------------------------------------------------------
# CommissionSchedule.effective_pct
# ---------------------------------------------------------------------------


class TestEffectivePct:
    def test_percentage_applies_when_above_minimum(self):
        """Large position: percentage > minimum, so percentage applies."""
        schedule = CommissionSchedule.saxo_classic()
        # US: 0.08%, min $1. Position $10 000 → 0.0008 * 10000 = $8 >> $1
        result = schedule.effective_pct("AAPL", 10_000.0)
        assert result == pytest.approx(0.0008, rel=1e-6)

    def test_minimum_applies_when_position_is_small(self):
        """Small US position: minimum $1 kicks in."""
        schedule = CommissionSchedule.saxo_classic()
        # Position $500 → 0.0008 * 500 = $0.40 < $1 minimum
        result = schedule.effective_pct("AAPL", 500.0)
        assert result == pytest.approx(1.0 / 500.0, rel=1e-6)

    def test_dk_ticker_uses_dk_rate(self):
        """Danish ticker uses DK rate (min DKK 10 ≈ $1.45)."""
        schedule = CommissionSchedule.saxo_classic()
        # Position $2000 → 0.0008 * 2000 = $1.60 > $1.45 → percentage applies
        result = schedule.effective_pct("NOVO-B.CO", 2_000.0)
        assert result == pytest.approx(0.0008, rel=1e-6)

    def test_dk_ticker_minimum_kicks_in(self):
        """Small DK position: minimum $1.45 applies."""
        schedule = CommissionSchedule.saxo_classic()
        # Position $1000 → 0.0008 * 1000 = $0.80 < $1.45
        result = schedule.effective_pct("NOVO-B.CO", 1_000.0)
        assert result == pytest.approx(1.45 / 1_000.0, rel=1e-6)

    def test_zero_schedule_returns_zero(self):
        result = CommissionSchedule.zero().effective_pct("AAPL", 5_000.0)
        assert result == 0.0

    def test_zero_position_returns_default_pct(self):
        """Guard against division by zero."""
        schedule = CommissionSchedule.saxo_classic()
        result = schedule.effective_pct("AAPL", 0.0)
        assert result == schedule.default.pct


# ---------------------------------------------------------------------------
# Factory methods
# ---------------------------------------------------------------------------


class TestFactories:
    def test_saxo_classic_creates_instance(self):
        s = CommissionSchedule.saxo_classic()
        assert isinstance(s, CommissionSchedule)
        assert s.rates["US"].pct == pytest.approx(0.0008)

    def test_saxo_platinum_creates_instance(self):
        s = CommissionSchedule.saxo_platinum()
        assert isinstance(s, CommissionSchedule)
        assert s.rates["US"].pct == pytest.approx(0.0005)

    def test_saxo_vip_creates_instance(self):
        s = CommissionSchedule.saxo_vip()
        assert isinstance(s, CommissionSchedule)
        assert s.rates["US"].pct == pytest.approx(0.0003)

    def test_zero_creates_instance(self):
        s = CommissionSchedule.zero()
        assert s.default.pct == 0.0
        assert s.default.min_usd == 0.0

    def test_classic_cheaper_than_blank_for_large_positions(self):
        """Sanity: Classic < Blank impossible; Classic < Platinum for US."""
        classic = CommissionSchedule.saxo_classic().effective_pct("AAPL", 10_000.0)
        platinum = CommissionSchedule.saxo_platinum().effective_pct("AAPL", 10_000.0)
        vip = CommissionSchedule.saxo_vip().effective_pct("AAPL", 10_000.0)
        assert vip < platinum < classic
