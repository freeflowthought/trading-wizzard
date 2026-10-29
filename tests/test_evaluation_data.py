import sys
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from twiz.ingest import evaluation_data as data


def provider(monkeypatch, days, on_return=None):
    rows = [(datetime.fromisoformat(day), {
        'Open': 100, 'High': 102, 'Low': 99, 'Close': 101, 'Volume': 1000,
    }) for day in days]

    def history(**kwargs):
        if on_return:
            on_return()
        return SimpleNamespace(iterrows=lambda: iter(rows))

    monkeypatch.setitem(sys.modules, 'yfinance', SimpleNamespace(
        Ticker=lambda ticker: SimpleNamespace(history=history)))


@pytest.mark.parametrize('now,expected', [
    ('2026-10-28T19:00:00+00:00', ['2026-10-27']),
    ('2026-10-28T20:10:00+00:00', ['2026-10-27']),
    ('2026-10-28T20:16:00+00:00', ['2026-10-27', '2026-10-28']),
])
def test_finite_intraday_bars_do_not_become_completed_research_data(monkeypatch, now, expected):
    provider(monkeypatch, ['2026-10-27', '2026-10-28', '2026-10-29'])
    result = data.fetch_research_asset('QQQ', '2026-10-27', '2026-10-29',
                                       now=datetime.fromisoformat(now))
    assert [row['date'] for row in result['rows']] == expected
    assert result['completed_session_cutoff'] == expected[-1]
    assert all(row['session_complete'] is True for row in result['rows'])


def test_holiday_and_weekend_rows_are_not_research_sessions(monkeypatch):
    provider(monkeypatch, ['2026-06-18', '2026-06-19', '2026-06-20', '2026-06-22'])
    result = data.fetch_research_asset('QQQ', '2026-06-18', '2026-06-22',
                                       now=datetime(2026, 6, 22, 21, tzinfo=timezone.utc))
    assert [row['date'] for row in result['rows']] == ['2026-06-18', '2026-06-22']


def test_naive_clock_rejected_before_provider_access(monkeypatch):
    monkeypatch.setitem(sys.modules, 'yfinance', None)
    with pytest.raises(ValueError, match='timezone'):
        data.fetch_research_asset('QQQ', '2026-10-27', '2026-10-28',
                                  now=datetime(2026, 10, 28))


def test_price_receipt_is_after_http_return_not_request_start(monkeypatch):
    state = {'returned': False}
    before = datetime(2026, 10, 28, 19, tzinfo=timezone.utc)
    after = datetime(2026, 10, 28, 20, 16, tzinfo=timezone.utc)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return after if state['returned'] else before

    monkeypatch.setattr(data, 'datetime', Clock)
    provider(monkeypatch, ['2026-10-28'], lambda: state.update(returned=True))
    result = data.fetch_research_asset('QQQ', '2026-10-28', '2026-10-28')
    assert result['request_started_at'] == before.isoformat()
    assert result['fetched_at'] == result['rows'][0]['fetched_at'] == after.isoformat()
    assert result['completed_session_cutoff'] == '2026-10-28'


def test_macro_receipt_is_after_response_and_still_marks_revised_history(monkeypatch, tmp_path):
    state = {'returned': False}
    before = datetime(2026, 10, 28, 19, tzinfo=timezone.utc)
    after = datetime(2026, 10, 28, 19, 1, tzinfo=timezone.utc)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return after if state['returned'] else before

    def get(url):
        state['returned'] = True
        series = url.split('id=')[1].split('&')[0]
        return f'observation_date,{series}\n2026-10-27,4.0\n'.encode()

    monkeypatch.setattr(data, 'datetime', Clock)
    monkeypatch.setattr(data, '_get', get)
    rows = data.fetch_research_macro(tmp_path/'macro', '2026-10-27', '2026-10-27')
    assert rows and all(row['fetched_at'] == after.isoformat() for row in rows)
    assert rows[0]['request_started_at'] == before.isoformat()
    assert all(row['pit_status'] == 'revised_history_not_historical_vintage' for row in rows)
