import hashlib
import json
from datetime import datetime, timezone
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse

import pytest

from twiz.ingest import historical_predictions as hp

NOW = datetime(2026, 10, 28, 12, tzinfo=timezone.utc)


def source(tmp_path, **changes):
    market = {
        'id': '10', 'question': 'Will there be no change in Fed interest rates after '
                                 'the June 2026 meeting?',
        'outcomes': '["No", "Yes"]', 'clobTokenIds': '["101", "102"]',
        'createdAt': '2026-06-01T00:00:00Z', 'startDate': '2026-06-01T03:00:00Z',
        'endDate': '2026-06-17T00:00:00Z', 'description': 'Current downloaded rules.',
        'outcomePrices': '["0", "1"]', 'liquidity': 0, 'spread': 1,
    }
    market.update(changes)
    event = {'id': '1', 'slug': 'fed-decision-in-june', 'title': 'Fed decision in June?',
             'createdAt': '2026-05-31T00:00:00Z', 'startDate': '2026-06-01T00:00:00Z',
             'endDate': '2026-06-17T00:00:00Z', 'markets': [market]}
    folder = tmp_path / 'sources'
    folder.mkdir(exist_ok=True)
    path = folder / 'fed-decision-in-june.json'
    path.write_text(json.dumps({'fetched_at': NOW.isoformat(),
                               'request_url': 'https://gamma-api.polymarket.com/events?'
                                              'slug=fed-decision-in-june',
                               'response': [event]}))
    return folder, path


def point(at='2026-06-02T00:00:00+00:00', price=0.4, resolution=10800):
    return {'timestamp': int(datetime.fromisoformat(at).timestamp()), 'price': price,
            'resolution_seconds': resolution}


def envelope(points, more=False, cursor=None):
    return json.dumps({'data': points, 'pagination': {
        'has_more': more, 'next_cursor': cursor, 'limit': 10000, 'offset': 0}}).encode()


def collect(tmp_path, folder=None, fetcher=None, **kwargs):
    if folder is None:
        folder, _ = source(tmp_path)
    return hp.collect_historical_predictions(
        folder, tmp_path / 'runs', now=NOW,
        fetcher=fetcher or (lambda url, timeout: envelope([point()])), **kwargs)


def index(receipt):
    return json.loads((hp.Path(receipt['path']) / 'index.json').read_text())


def history(receipt, outcome='Yes'):
    return json.loads((hp.Path(receipt['path']) / f'history/10-{outcome.lower()}.json').read_text())


def test_exact_source_binding_and_immutable_raw_capture(tmp_path):
    folder, original = source(tmp_path)
    calls = []
    raw = envelope([point()])

    def fetch(url, timeout):
        calls.append((parse_qs(urlparse(url).query), timeout))
        return raw

    receipt = collect(tmp_path, folder, fetch)
    assert calls[0][0]['token_id'] == ['101']  # label order is No, Yes in the source
    assert calls[1][0]['token_id'] == ['102']
    assert all(call[0]['interval'] == ['max'] for call in calls)
    run = hp.Path(receipt['path'])
    assert (run / 'sources' / original.name).read_bytes() == original.read_bytes()
    assert (run / 'raw/10-yes-00.body').read_bytes() == raw
    row = history(receipt)
    assert row['history'][0]['p'] == 0.4  # never source's current outcomePrices=1
    assert row['history'][0]['available_at'] == '2026-06-02T03:00:00+00:00'
    assert row['history'][0]['historical_publication_time_verified'] is False
    assert row['history'][0]['raw_references'] == [
        {'body_file': 'raw/10-yes-00.body', 'data_index': 0}]
    assert row['rules_version'] == 'current_download_not_verified_historical'
    assert 'liquidity' not in row and 'spread' not in row
    assert hp.verify_event_history_run(run)['valid'] is True


def test_second_run_does_not_overwrite_first_and_tamper_is_rejected(tmp_path):
    first, second = collect(tmp_path), collect(tmp_path)
    assert first['path'] != second['path']
    run = hp.Path(first['path'])
    (run / 'collector.py').write_text('tampered')
    assert hp.verify_event_history_run(run)['valid'] is False
    assert hp.verify_event_history_run(hp.Path(second['path']))['valid'] is True


def test_creation_start_and_conservative_ambiguous_end_gate(tmp_path):
    points = [point('2026-06-01T00:00:00+00:00'),
              point('2026-06-01T03:00:00+00:00'),
              point('2026-06-16T21:00:00+00:00'),
              point('2026-06-16T23:00:00+00:00'),
              point('2026-06-17T00:00:00+00:00'),
              point('2026-06-17T18:00:00+00:00', 1, 0)]
    receipt = collect(tmp_path, fetcher=lambda url, timeout: envelope(points))
    row = history(receipt)
    assert len(row['history']) == 2
    assert row['history'][-1]['available_at'] == row['source_end_date']
    assert row['rejected_points'] == {
        'before_creation_or_start': 1, 'after_cutoff_or_bucket_not_available': 2,
        'zero_resolution_terminal': 1}


def test_explicit_as_of_cuts_off_future_history_even_before_contract_end(tmp_path):
    folder, path = source(tmp_path)
    wrapper = json.loads(path.read_text())
    wrapper['fetched_at'] = '2026-06-05T00:00:00+00:00'
    path.write_text(json.dumps(wrapper))
    receipt = hp.collect_historical_predictions(
        folder, tmp_path / 'runs', now=datetime(2026, 6, 5, tzinfo=timezone.utc),
        fetcher=lambda url, timeout: envelope([
            point('2026-06-04T21:00:00+00:00'), point('2026-06-05T00:00:00+00:00')]))
    assert len(history(receipt)['history']) == 1


@pytest.mark.parametrize('invalid', [True, float('nan'), float('inf'), -0.1, 1.01, None])
def test_invalid_prices_cannot_create_probability_observations(tmp_path, invalid):
    receipt = collect(tmp_path, fetcher=lambda url, timeout: envelope([point(price=invalid)]))
    assert history(receipt)['status'] == 'unknown'
    assert history(receipt)['history'] == []


def test_conflicting_duplicate_timestamp_is_excluded_not_last_write_wins(tmp_path):
    receipt = collect(tmp_path, fetcher=lambda url, timeout: envelope([
        point(price=0.4), point(price=0.4), point(price=0.6)]))
    row = history(receipt)
    assert row['history'] == []
    assert row['rejected_points']['duplicate'] == 1
    assert row['rejected_points']['conflicting_timestamp'] == 1


def test_follow_cursor_and_use_actual_resolution_conservatively(tmp_path):
    calls = []

    def fetch(url, timeout):
        query = parse_qs(urlparse(url).query)
        calls.append(query)
        if 'cursor' not in query:
            return envelope([point(resolution=43200)], True, 'opaque-next')
        return envelope([point('2026-06-03T00:00:00+00:00', 0.5)])

    receipt = collect(tmp_path, fetcher=fetch)
    row = history(receipt)
    assert row['pagination_complete'] is True and len(row['history']) == 2
    assert row['history'][0]['available_at'] == '2026-06-02T12:00:00+00:00'
    assert calls[1]['cursor'] == ['opaque-next']


def test_budget_and_partial_pagination_are_explicit(tmp_path):
    receipt = collect(tmp_path, max_requests=1,
                      fetcher=lambda url, timeout: envelope([point()], True, 'next'))
    rows = index(receipt)['contracts']
    assert rows[0]['status'] == 'partial' and rows[0]['pagination_complete'] is False
    assert rows[1]['status'] == 'unknown' and rows[1]['request_count'] == 0
    assert receipt['summary']['requests'] == 1


def test_zero_request_budget_never_calls_provider_or_backfills_current_prices(tmp_path):
    def forbidden(*args):
        pytest.fail('Provider called with a zero request budget')

    receipt = collect(tmp_path, max_requests=0, fetcher=forbidden)
    assert receipt['summary']['tokens'] == 2
    assert receipt['summary']['tokens_with_history'] == receipt['summary']['requests'] == 0


def test_http_failure_preserves_body_and_does_not_abort_other_token(tmp_path):
    import io

    def fetch(url, timeout):
        if parse_qs(urlparse(url).query)['token_id'] == ['101']:
            raise HTTPError(url, 429, 'Rate limited', {}, io.BytesIO(b'{"error":"rate limit"}'))
        return envelope([point()])

    receipt = collect(tmp_path, fetcher=fetch)
    assert history(receipt, 'No')['status'] == 'unknown'
    assert history(receipt)['status'] == 'downloaded'
    body = (hp.Path(receipt['path']) / 'raw/10-no-00.body').read_bytes()
    assert body == b'{"error":"rate limit"}'


@pytest.mark.parametrize('change', ['url', 'slug', 'tokens', 'date', 'future_source'])
def test_unverified_identity_and_chronology_never_reach_provider(tmp_path, change):
    folder, path = source(tmp_path)
    wrapper = json.loads(path.read_text())
    if change == 'url':
        wrapper['request_url'] = wrapper['request_url'].replace(
            'gamma-api.polymarket.com', 'fake.test')
    elif change == 'slug':
        wrapper['response'][0]['slug'] = 'a-different-event'
    elif change == 'tokens':
        wrapper['response'][0]['markets'][0]['clobTokenIds'] = '["101", "101"]'
    elif change == 'date':
        wrapper['response'][0]['markets'][0]['createdAt'] = '2026-06-01T00:00:00'
    else:
        wrapper['fetched_at'] = '2027-01-01T00:00:00Z'
    path.write_text(json.dumps(wrapper))
    receipt = collect(tmp_path, folder, lambda *args: pytest.fail('Unverified source fetched'))
    assert receipt['summary']['requests'] == 0
    assert receipt['summary']['source_statuses'] == {'invalid_source': 1}


def test_all_brackets_and_both_tokens_preserved_no_event_sum_normalization(tmp_path):
    folder, path = source(tmp_path)
    wrapper = json.loads(path.read_text())
    second = dict(wrapper['response'][0]['markets'][0], id='11',
                  question='Will the Fed decrease interest rates by 25 bps after '
                           'the June 2026 meeting?', clobTokenIds='["103", "104"]')
    wrapper['response'][0]['markets'].append(second)
    path.write_text(json.dumps(wrapper))
    receipt = collect(tmp_path, folder, lambda url, timeout: envelope([point(price=0.6)]))
    assert receipt['summary']['markets'] == 2 and receipt['summary']['tokens'] == 4
    diagnostic = index(receipt)['event_sum_diagnostics'][0]
    assert diagnostic['yes_sum_min'] == diagnostic['yes_sum_max'] == 1.2
    assert diagnostic['yes_sum_outside_0_95_to_1_05'] == 1
    assert history(receipt)['history'][0]['p'] == 0.6


def test_empty_discovery_response_is_an_audited_gap(tmp_path):
    folder, path = source(tmp_path)
    wrapper = json.loads(path.read_text())
    wrapper['response'] = []
    path.write_text(json.dumps(wrapper))
    receipt = collect(tmp_path, folder)
    assert receipt['summary']['source_statuses'] == {'empty_response': 1}
    assert receipt['summary']['requests'] == 0
    manifest = hp.Path(receipt['path']) / 'manifest.json'
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == receipt['manifest_sha256']


def test_naive_analysis_clock_and_unsafe_density_rejected_before_archive(tmp_path):
    folder, _ = source(tmp_path)
    with pytest.raises(ValueError, match='timezone'):
        hp.collect_historical_predictions(folder, tmp_path / 'runs', now=datetime(2026, 6, 1))
    with pytest.raises(ValueError, match='density'):
        collect(tmp_path, folder, bucket_seconds=60)
    assert not (tmp_path / 'runs').exists()


def test_elapsed_budget_bounds_following_requests_and_request_timeout(tmp_path, monkeypatch):
    clock, timeouts = {'value': 100.0}, []
    monkeypatch.setattr(hp.time, 'monotonic', lambda: clock['value'])

    def fetch(url, timeout):
        timeouts.append(timeout)
        clock['value'] += 6
        return envelope([point()])

    receipt = collect(tmp_path, fetcher=fetch, total_budget_seconds=5, request_timeout=10)
    assert timeouts == [5]
    assert receipt['summary']['requests'] == 1
    assert history(receipt)['status'] == 'unknown'


def test_repeated_cursor_does_not_loop_or_imply_complete_history(tmp_path):
    receipt = collect(tmp_path, fetcher=lambda url, timeout: envelope([point()], True, 'same'))
    assert receipt['summary']['requests'] == 4  # two requests per outcome, then repeated cursor
    assert history(receipt)['status'] == 'partial'
    assert 'repeated' in history(receipt)['error']


def test_external_receipt_rejects_rewritten_manifest_even_with_recomputed_local_hash(tmp_path):
    receipt = collect(tmp_path)
    run = hp.Path(receipt['path'])
    manifest = json.loads((run / 'manifest.json').read_text())
    manifest['summary']['observations'] = 123456
    body = json.dumps(manifest).encode()
    (run / 'manifest.json').write_bytes(body)
    (run / 'manifest.sha256').write_text(hashlib.sha256(body).hexdigest())
    assert hp.verify_event_history_run(run, receipt['manifest_sha256'])['valid'] is False


def test_duplicate_token_bindings_fail_closed_and_are_counted(tmp_path):
    folder, path = source(tmp_path)
    wrapper = json.loads(path.read_text())
    duplicate = dict(wrapper['response'][0]['markets'][0], id='11')
    wrapper['response'][0]['markets'].append(duplicate)
    path.write_text(json.dumps(wrapper))
    receipt = collect(tmp_path, folder)
    assert receipt['summary']['duplicate_bindings_excluded'] == 4
    assert receipt['summary']['requests'] == 0


def test_malformed_provider_envelope_is_unknown_and_raw_response_is_still_archived(tmp_path):
    receipt = collect(tmp_path, fetcher=lambda url, timeout: b'{"outcomePrices":["1","0"]}')
    assert history(receipt)['status'] == 'unknown'
    assert history(receipt)['history'] == []
    assert (hp.Path(receipt['path']) / 'raw/10-yes-00.body').exists()


def test_terminal_point_caps_later_positive_resolution_rows_without_inferring_outcome(tmp_path):
    points = [point('2026-06-09T21:00:00+00:00', 0.6),
              point('2026-06-10T00:00:00+00:00', 0.7, 0),
              point('2026-06-11T00:00:00+00:00', 1)]
    receipt = collect(tmp_path, fetcher=lambda url, timeout: envelope(points))
    row = history(receipt)
    assert len(row['history']) == 1
    assert row['history'][0]['p'] == 0.6
    assert row['terminal_cutoff'] == '2026-06-10T00:00:00+00:00'
    assert row['rejected_points']['after_terminal_or_bucket_not_available'] == 1


@pytest.mark.parametrize('key', ['closedTime', 'umaEndDate', 'resolvedAt'])
def test_earlier_source_closure_is_stricter_than_declared_end_date(tmp_path, key):
    folder, _ = source(tmp_path, **{key: '2026-06-10T00:00:00+00:00'})
    receipt = collect(tmp_path, folder, fetcher=lambda url, timeout: envelope([
        point('2026-06-09T21:00:00+00:00'), point('2026-06-11T00:00:00+00:00', 1)]))
    row = history(receipt)
    assert row['source_end_date'] == '2026-06-17T00:00:00+00:00'
    assert row['cutoff'] == '2026-06-10T00:00:00+00:00'
    assert len(row['history']) == 1
