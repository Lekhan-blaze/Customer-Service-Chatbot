"""Tests for support_bot.clock — FakeClock, business hours, holidays, scheduling."""

import datetime as dt
from zoneinfo import ZoneInfo

from support_bot.clock import (
    FakeClock,
    SystemClock,
    add_business_time,
    is_business_hours,
    next_working_day,
)
from support_bot.config import load_defaults

IST = ZoneInfo("Asia/Kolkata")


def _cfg(**overrides):
    cfg = load_defaults()
    cfg.update(overrides)
    return cfg


# ── SystemClock ─────────────────────────────────────────────────────────

class TestSystemClock:
    def test_now_returns_aware_datetime(self):
        clock = SystemClock()
        now = clock.now()
        assert now.tzinfo is not None

    def test_today_returns_date(self):
        clock = SystemClock()
        assert isinstance(clock.today(), dt.date)


# ── FakeClock ───────────────────────────────────────────────────────────

class TestFakeClock:
    def test_frozen_time(self, fake_clock):
        t1 = fake_clock.now()
        t2 = fake_clock.now()
        assert t1 == t2

    def test_set(self, fake_clock):
        new_time = dt.datetime(2025, 6, 15, 14, 0, tzinfo=IST)
        fake_clock.set(new_time)
        assert fake_clock.now() == new_time

    def test_advance(self, fake_clock):
        t0 = fake_clock.now()
        fake_clock.advance(hours=2)
        assert fake_clock.now() == t0 + dt.timedelta(hours=2)

    def test_today(self, fake_clock):
        assert fake_clock.today(IST) == dt.date(2025, 1, 6)


# ── Business hours ──────────────────────────────────────────────────────

class TestBusinessHours:
    def test_weekday_during_hours(self):
        # Mon 10:00 IST
        clock = FakeClock(dt.datetime(2025, 1, 6, 10, 0, tzinfo=IST))
        assert is_business_hours(clock, _cfg()) is True

    def test_weekday_before_hours(self):
        clock = FakeClock(dt.datetime(2025, 1, 6, 8, 0, tzinfo=IST))
        assert is_business_hours(clock, _cfg()) is False

    def test_weekday_after_hours(self):
        clock = FakeClock(dt.datetime(2025, 1, 6, 19, 0, tzinfo=IST))
        assert is_business_hours(clock, _cfg()) is False

    def test_weekend(self):
        # Sat 2025-01-11
        clock = FakeClock(dt.datetime(2025, 1, 11, 12, 0, tzinfo=IST))
        assert is_business_hours(clock, _cfg()) is False

    def test_holiday(self):
        # Mon 2025-01-06 is a holiday in this config
        clock = FakeClock(dt.datetime(2025, 1, 6, 10, 0, tzinfo=IST))
        cfg = _cfg(holidays=["2025-01-06"])
        assert is_business_hours(clock, cfg) is False


# ── next_working_day ────────────────────────────────────────────────────

class TestNextWorkingDay:
    def test_from_weekday(self):
        # Mon → Tue
        clock = FakeClock(dt.datetime(2025, 1, 6, 10, 0, tzinfo=IST))
        assert next_working_day(clock, _cfg()) == dt.date(2025, 1, 7)

    def test_from_friday(self):
        # Fri → Mon
        clock = FakeClock(dt.datetime(2025, 1, 10, 10, 0, tzinfo=IST))
        assert next_working_day(clock, _cfg()) == dt.date(2025, 1, 13)

    def test_skips_holiday(self):
        # Mon, with Tue as holiday → Wed
        clock = FakeClock(dt.datetime(2025, 1, 6, 10, 0, tzinfo=IST))
        cfg = _cfg(holidays=["2025-01-07"])
        assert next_working_day(clock, cfg) == dt.date(2025, 1, 8)


# ── add_business_time ───────────────────────────────────────────────────

class TestAddBusinessTime:
    def test_within_same_day(self):
        start = dt.datetime(2025, 1, 6, 10, 0, tzinfo=IST)  # Mon 10:00
        result = add_business_time(start, 60, _cfg())
        assert result == dt.datetime(2025, 1, 6, 11, 0, tzinfo=IST)

    def test_spans_overnight(self):
        start = dt.datetime(2025, 1, 6, 17, 0, tzinfo=IST)  # Mon 17:00, 1h left
        # 120 biz minutes = 60 today + 60 tomorrow
        result = add_business_time(start, 120, _cfg())
        assert result == dt.datetime(2025, 1, 7, 10, 0, tzinfo=IST)

    def test_spans_weekend(self):
        start = dt.datetime(2025, 1, 10, 17, 0, tzinfo=IST)  # Fri 17:00, 1h left
        result = add_business_time(start, 120, _cfg())
        # 60 Fri + 60 Mon morning → Mon 10:00
        assert result == dt.datetime(2025, 1, 13, 10, 0, tzinfo=IST)

    def test_starts_before_business_hours(self):
        start = dt.datetime(2025, 1, 6, 7, 0, tzinfo=IST)  # Mon 07:00
        result = add_business_time(start, 30, _cfg())
        assert result == dt.datetime(2025, 1, 6, 9, 30, tzinfo=IST)
