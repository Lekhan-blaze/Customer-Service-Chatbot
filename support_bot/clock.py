"""Clock abstraction — SystemClock for production, FakeClock for tests.

Business-hours helpers respect weekends and configured holidays.
"""

from __future__ import annotations

import abc
import datetime as _dt
from zoneinfo import ZoneInfo


class Clock(abc.ABC):
    """Interface — every module that needs the current time receives a Clock."""

    @abc.abstractmethod
    def now(self, tz: ZoneInfo | None = None) -> _dt.datetime:
        """Return the current datetime, optionally in *tz*."""

    @abc.abstractmethod
    def today(self, tz: ZoneInfo | None = None) -> _dt.date:
        """Return the current date, optionally in *tz*."""


class SystemClock(Clock):
    """Real wall-clock time."""

    def now(self, tz: ZoneInfo | None = None) -> _dt.datetime:
        return _dt.datetime.now(tz or ZoneInfo("UTC"))

    def today(self, tz: ZoneInfo | None = None) -> _dt.date:
        return self.now(tz).date()


class FakeClock(Clock):
    """Deterministic clock for tests — set or advance time explicitly."""

    def __init__(self, frozen: _dt.datetime | None = None):
        self._now = frozen or _dt.datetime(
            2025, 1, 6, 10, 0, 0, tzinfo=ZoneInfo("Asia/Kolkata")
        )

    def now(self, tz: ZoneInfo | None = None) -> _dt.datetime:
        if tz and self._now.tzinfo != tz:
            return self._now.astimezone(tz)
        return self._now

    def today(self, tz: ZoneInfo | None = None) -> _dt.date:
        return self.now(tz).date()

    def set(self, dt: _dt.datetime) -> None:
        self._now = dt

    def advance(self, **kwargs) -> _dt.datetime:
        """Advance time by a timedelta (accepts timedelta keyword args)."""
        self._now += _dt.timedelta(**kwargs)
        return self._now


# ── Business-hours helpers ──────────────────────────────────────────────


_DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _parse_time(s: str) -> _dt.time:
    return _dt.datetime.strptime(s, "%H:%M").time()


def is_business_hours(
    clock: Clock,
    cfg: dict,
) -> bool:
    """True if the clock's current time falls within configured business hours."""
    tz = ZoneInfo(cfg.get("company_timezone", "Asia/Kolkata"))
    now = clock.now(tz)
    bh = cfg.get("business_hours", {})
    day_name = _DAY_NAMES[now.weekday()]
    if day_name not in bh.get("days", ["Mon", "Tue", "Wed", "Thu", "Fri"]):
        return False
    holidays = [str(h) for h in cfg.get("holidays", [])]
    if now.date().isoformat() in holidays:
        return False
    start = _parse_time(bh.get("start", "09:00"))
    end = _parse_time(bh.get("end", "18:00"))
    return start <= now.time() < end


def _is_working_day(d: _dt.date, cfg: dict) -> bool:
    """True if *d* is a configured working day (not weekend, not holiday)."""
    bh = cfg.get("business_hours", {})
    allowed = bh.get("days", ["Mon", "Tue", "Wed", "Thu", "Fri"])
    day_name = _DAY_NAMES[d.weekday()]
    if day_name not in allowed:
        return False
    holidays = [str(h) for h in cfg.get("holidays", [])]
    return d.isoformat() not in holidays


def next_working_day(
    clock: Clock,
    cfg: dict,
) -> _dt.date:
    """Return the next working day after *clock.today()*."""
    tz = ZoneInfo(cfg.get("company_timezone", "Asia/Kolkata"))
    d = clock.today(tz) + _dt.timedelta(days=1)
    while not _is_working_day(d, cfg):
        d += _dt.timedelta(days=1)
    return d


def add_business_time(
    start: _dt.datetime,
    minutes: int,
    cfg: dict,
) -> _dt.datetime:
    """Add *minutes* of business time to *start*, skipping non-working periods.

    Returns a timezone-aware datetime in the company timezone.
    """
    tz = ZoneInfo(cfg.get("company_timezone", "Asia/Kolkata"))
    bh = cfg.get("business_hours", {})
    bh_start = _parse_time(bh.get("start", "09:00"))
    bh_end = _parse_time(bh.get("end", "18:00"))

    current = start.astimezone(tz)
    remaining = minutes

    # If before business hours on a working day, snap to start
    if _is_working_day(current.date(), cfg) and current.time() < bh_start:
        current = current.replace(
            hour=bh_start.hour, minute=bh_start.minute, second=0, microsecond=0
        )
    # If outside working day/hours, jump to next working day start
    if not _is_working_day(current.date(), cfg) or current.time() >= bh_end:
        d = current.date()
        if current.time() >= bh_end or not _is_working_day(d, cfg):
            d += _dt.timedelta(days=1)
        while not _is_working_day(d, cfg):
            d += _dt.timedelta(days=1)
        current = _dt.datetime.combine(d, bh_start, tzinfo=tz)

    while remaining > 0:
        end_of_day = current.replace(
            hour=bh_end.hour, minute=bh_end.minute, second=0, microsecond=0
        )
        available = int((end_of_day - current).total_seconds() // 60)
        if remaining <= available:
            current += _dt.timedelta(minutes=remaining)
            remaining = 0
        else:
            remaining -= available
            # Jump to next working day start
            d = current.date() + _dt.timedelta(days=1)
            while not _is_working_day(d, cfg):
                d += _dt.timedelta(days=1)
            current = _dt.datetime.combine(d, bh_start, tzinfo=tz)

    return current


def is_maintenance_window(
    clock: Clock,
    cfg: dict,
) -> bool:
    """True if the clock's current time falls within configured maintenance window."""
    tz = ZoneInfo(cfg.get("company_timezone", "Asia/Kolkata"))
    now = clock.now(tz)
    mw = cfg.get("maintenance_window", {})
    start = _parse_time(mw.get("start", "02:00"))
    end = _parse_time(mw.get("end", "04:00"))
    if start <= end:
        return start <= now.time() < end
    return now.time() >= start or now.time() < end
