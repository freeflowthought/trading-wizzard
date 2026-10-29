import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import unquote, urlparse

import pytest

from twiz.ingest import swing_data as data

NOW = datetime(2026, 10, 28, 13, tzinfo=timezone.utc)


def epoch(at):
    return int(datetime.fromisoformat(at).timestamp())


def chart(ticker='QQQ', days=None, events=None):
    days = days or ['2026-10-26T13:30:00+00:00', '2026-10-27T13:30:00+00:00']
    quotes = {'open': [100] * len(days), 'high': [102] * len(days),
              'low': [99] * len(days), 'close': [101] * len(days), 'volume': [0] * len(days)}
    return {'chart': {'error': None, 'result': [{
        'meta': {'symbol': ticker, 'instrumentType': 'INDEX' if ticker.startswith('^') else 'ETF',
                 'exchangeTimezoneName': 'America/New_York', 'currency': 'USD'},
        'timestamp': [epoch(day) for day in days],
        'indicators': {'quote': [quotes], 'adjclose': [{'adjclose': [90] * len(days)}]},
        'events': events or {},
    }]}}


def action(at, **kwargs):
    timestamp = epoch(at)
    return {str(timestamp): {'date': timestamp, **kwargs}}


def inputs(tmp_path, candidates=None):
    market, annotation = tmp_path / 'old-market.json', tmp_path / 'annotation.json'
    market.write_text(json.dumps({'QQQ': [{'date': '2016-01-04', 'close': 999}]}))
    annotation.write_text(json.dumps({
        'candidates': candidates or [], 'outcome_join_performed': False}))
    return market, annotation


def collect(tmp_path, fetcher=None, **kwargs):
    market, annotation = inputs(tmp_path)

    def default_fetch(url, timeout):
        ticker = unquote(urlparse(url).path.rsplit('/', 1)[-1])
        return json.dumps(chart(ticker)).encode()

    return data.collect_swing_data(market, annotation, tmp_path / 'runs', now=NOW,
                                   fetcher=fetcher or default_fetch, **kwargs)


def test_exchange_local_dates_and_no_double_price_adjustment():
    payload = chart(days=['2026-10-27T00:30:00+00:00'], events={
        'splits': action('2026-10-27T00:30:00+00:00', numerator=4, denominator=1, splitRatio='4:1'),
        'dividends': action('2026-10-27T00:30:00+00:00', amount=0.25),
    })
    result = data.parse_chart(payload, 'QQQ', '2016-01-01', '2026-10-27')
    assert result['rows'][0]['date'] == '2026-10-26'  # not UTC's October27
    assert result['rows'][0]['close'] == 101
    assert result['rows'][0]['provider_adjusted_close'] == 90
    assert result['rows'][0]['price_basis'] == data.PRICE_BASIS
    split, dividend = sorted(result['actions'], key=lambda row: row['type'], reverse=True)
    assert split['ratio'] == 4 and split['price_adjustment_applied'] is False
    assert split['effective_date'] == dividend['effective_date'] == '2026-10-26'
    assert dividend['amount'] == 0.25 and dividend['known_at'] is None
    assert dividend['payment_date'] is dividend['declaration_date'] is None
    assert dividend['currency'] is None  # listing USD does not invent action currency


def test_only_completed_us_sessions_and_finite_ohlcv_are_exported():
    payload = chart(days=['2026-06-18T13:30:00+00:00', '2026-06-19T13:30:00+00:00',
                          '2026-06-20T13:30:00+00:00', '2026-06-22T13:30:00+00:00',
                          '2026-06-23T13:30:00+00:00'])
    payload['chart']['result'][0]['indicators']['quote'][0]['close'][0] = float('nan')
    result = data.parse_chart(payload, 'QQQ', '2016-01-01', '2026-06-22')
    assert [row['date'] for row in result['rows']] == ['2026-06-22']
    assert result['rejected_bars'] == {
        'invalid_ohlcv': 1, 'outside_completed_session_window': 3}


@pytest.mark.parametrize('now,cutoff', [
    ('2026-10-28T19:00:00+00:00', '2026-10-27'),
    ('2026-10-28T20:10:00+00:00', '2026-10-27'),
    ('2026-10-28T20:16:00+00:00', '2026-10-28'),
])
def test_collector_session_cutoff_is_aware_and_intraday_safe(tmp_path, now, cutoff):
    market, annotation = inputs(tmp_path)
    receipt = data.collect_swing_data(market, annotation, tmp_path / 'runs',
                                     now=datetime.fromisoformat(now), max_requests=0)
    manifest = json.loads((Path(receipt['path']) / 'manifest.json').read_text())
    assert manifest['completed_session_cutoff'] == cutoff


def test_raw_and_normalized_bars_actions_are_one_response_and_old_prices_not_mixed(tmp_path):
    receipt = collect(tmp_path)
    run = Path(receipt['path'])
    manifest = json.loads((run / 'manifest.json').read_text())
    market = json.loads((run / 'market.json').read_text())
    assert len(market) == 4 and len(market['QQQ']) == 2
    assert all(row['close'] == 101 for row in market['QQQ'])
    assert all(row['date'] != '2016-01-04' for row in market['QQQ'])
    assert market['QQQ'][0]['raw_body_file'] == 'raw/QQQ-0.body'
    assert market['QQQ'][0]['raw_json_pointer'] == '/chart/result/0/indicators/quote/0'
    assert market['QQQ'][0]['raw_array_index'] == 0
    assert market['QQQ'][0]['raw_timestamp_pointer'] == '/chart/result/0/timestamp/0'
    assert manifest['outcome_join_performed'] is False
    assert hashlib.sha256((run / 'manifest.json').read_bytes()).hexdigest() == receipt[
        'manifest_sha256']
    for relative, digest in manifest['files'].items():
        assert hashlib.sha256((run / relative).read_bytes()).hexdigest() == digest
    reference = json.loads((run / 'inputs/previous-market-reference.json').read_text())
    old_digest = hashlib.sha256((tmp_path / 'old-market.json').read_bytes()).hexdigest()
    assert reference['sha256'] == old_digest


def test_literal_universe_does_not_guess_ambiguous_symbols_or_openai():
    def candidate(ticker, clause, context=''):
        return {'id': ticker + clause, 'ticker': ticker,
                'instrument_mapping_basis': 'literal_index_or_symbol',
                'evidence': {'exact_clause': clause}, 'same_post_context': context}

    result = data.build_swing_universe({'NVDA': [], 'CBRS': []}, {'candidates': [
        candidate('SOXL', 'SOXL might hold support'), candidate('SPCX', 'SPCX target'),
        candidate('CBRS', 'CBRS target'), candidate('OPEN', 'Open AI future IPO'),
        candidate('OPEN', 'OPEN这只房地产股票'), candidate('IREN', 'target30', 'Iren today26'),
        candidate('UNKNOWN', 'sector may rise'),
    ]})
    assert {row['ticker'] for row in result['assets']} == data.CORE | {
        'NVDA', 'SOXL', 'OPEN', 'IREN'}
    assert len(result['excluded_candidate_mappings']) == 4


def test_per_asset_failure_is_unknown_and_other_assets_survive(tmp_path):
    def fetch(url, timeout):
        ticker = unquote(urlparse(url).path.rsplit('/', 1)[-1])
        if ticker == 'QQQ':
            raise HTTPError(url, 429, 'rate limited', {}, io.BytesIO(b'{"error":"rate"}'))
        return json.dumps(chart(ticker)).encode()

    receipt = collect(tmp_path, fetcher=fetch)
    assert receipt['summary']['asset_statuses'] == {'unavailable': 1, 'downloaded': 3}
    index = json.loads((Path(receipt['path']) / 'index.json').read_text())
    qqq = next(row for row in index['assets'] if row['ticker'] == 'QQQ')
    assert qqq['action_status'] == 'unknown' and len(qqq['attempts']) == 2
    assert (Path(receipt['path']) / 'raw/QQQ-0.body').read_bytes() == b'{"error":"rate"}'


def test_missing_timezone_or_wrong_symbol_is_not_a_valid_chart():
    payload = chart()
    payload['chart']['result'][0]['meta']['exchangeTimezoneName'] = 'UTC'
    with pytest.raises(ValueError, match='timezone'):
        data.parse_chart(payload, 'QQQ', '2016-01-01', '2026-10-27')
    with pytest.raises(ValueError, match='identity'):
        data.parse_chart(chart('SPY'), 'QQQ', '2016-01-01', '2026-10-27')


@pytest.mark.parametrize('numerator,denominator', [(None, 1), (4, 0), (True, 1), (1, float('nan'))])
def test_unknown_actions_are_not_imputed_to_zero(numerator, denominator):
    result = data.parse_chart(chart(events={'splits': action(
        '2026-10-27T13:30:00+00:00', numerator=numerator, denominator=denominator)}),
        'QQQ', '2016-01-01', '2026-10-27')
    assert result['actions'] == [] and result['action_status'] == 'partial_invalid_events'


def test_mismatched_action_key_future_action_and_invalid_dividend_rejected():
    payload = chart(events={'dividends': {
        **action('2026-10-28T13:30:00+00:00', amount=1),
        **action('2026-10-27T13:30:00+00:00', amount=-1),
        '123': {'date': epoch('2026-10-26T13:30:00+00:00'), 'amount': 1},
    }})
    result = data.parse_chart(payload, 'QQQ', '2016-01-01', '2026-10-27')
    assert result['actions'] == []
    assert sum(result['rejected_actions'].values()) == 3


def test_no_events_and_index_not_applicable_differ_from_fetch_unknown():
    equity = data.parse_chart(chart(), 'QQQ', '2016-01-01', '2026-10-27')
    index = data.parse_chart(chart('^SOX'), '^SOX', '2016-01-01', '2026-10-27')
    assert equity['action_status'] == 'no_events_returned'
    assert index['action_status'] == 'not_applicable_index'
    assert index['rows'][0]['price_basis'] == 'Yahoo index level'


def test_request_budget_and_elapsed_timeout_are_bounded(tmp_path, monkeypatch):
    clock, timeouts = {'value': 100}, []
    monkeypatch.setattr(data.time, 'monotonic', lambda: clock['value'])

    def fetch(url, timeout):
        timeouts.append(timeout)
        clock['value'] += 6
        ticker = unquote(urlparse(url).path.rsplit('/', 1)[-1])
        return json.dumps(chart(ticker)).encode()

    receipt = collect(tmp_path, fetcher=fetch, total_budget_seconds=5, request_timeout=15)
    assert timeouts == [5] and receipt['summary']['requests'] == 1
    assert receipt['summary']['asset_statuses'] == {'downloaded': 1, 'unavailable': 3}


def test_unique_runs_never_overwrite_raw_history(tmp_path):
    first, second = collect(tmp_path), collect(tmp_path)
    assert first['path'] != second['path']
    assert (Path(first['path']) / 'manifest.json').is_file()


def test_naive_clock_rejected_before_creating_archive(tmp_path):
    market, annotation = inputs(tmp_path)
    with pytest.raises(ValueError, match='timezone'):
        data.collect_swing_data(market, annotation, tmp_path / 'runs', now=datetime(2026, 10, 28))
    assert not (tmp_path / 'runs').exists()


def test_text_split_ratio_must_agree_with_numeric_ratio():
    result = data.parse_chart(chart(events={'splits': action(
        '2026-10-27T13:30:00+00:00', numerator=1, denominator=5, splitRatio='5:1')}),
        'QQQ', '2016-01-01', '2026-10-27')
    assert result['actions'] == []
    assert result['rejected_actions'] == {'conflicting_split_ratio': 1}


def test_new_short_history_is_not_filled_back_to_2016():
    result = data.parse_chart(chart(), 'QQQ', '2016-01-01', '2026-10-27')
    assert len(result['rows']) == 2 and result['coverage_start'] == '2026-10-26'


def test_same_session_duplicate_bars_remain_unknown_instead_of_choosing_last():
    result = data.parse_chart(chart(days=['2026-10-27T13:30:00+00:00'] * 2),
                              'QQQ', '2016-01-01', '2026-10-27')
    assert result['rows'] == []
    assert result['rejected_bars'] == {'duplicate_session': 1}
