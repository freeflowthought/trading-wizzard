"""Near-term national joint-control relevance is separate from an equity direction hypothesis."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from twiz.ingest import prediction_discovery as discovery
from twiz.ingest.prediction_markets import fetch_prediction_markets
from twiz.provenance import begin_analysis_snapshot, finalize_analysis_snapshot, verify_snapshot
from twiz.scenarios import build_scenario_context
from twiz.storage import ProjectPaths, read_json, write_json

NOW = datetime(2026, 10, 28, 12, tzinfo=timezone.utc)
DAY = '2026-10-28'
TITLE = 'Balance of Power: 2026 Midterms'
CONFIG = {'main_case_threshold': .65, 'discovery': {'enabled': True, 'active_pages': 1,
          'search_queries': [], 'max_events': 6}}


def event(identity, title, question, probability=.8, days=5, **overrides):
    payload = {'id': str(identity), 'question': question, 'active': True, 'closed': False,
               'archived': False, 'outcomes': ['Yes', 'No'],
               'outcomePrices': [probability, 1-probability],
               'updatedAt': (NOW-timedelta(minutes=5)).isoformat(),
               'endDate': (NOW+timedelta(days=days)).isoformat(),
               'description': 'Exact chamber-control settlement conditions.',
               'liquidityNum': 20000, 'spread': .02, **overrides}
    return {'id': str(identity), 'title': title, 'slug': 'event-'+str(identity),
            'markets': [payload]}


def collect(monkeypatch, groups, config=None):
    calls = []

    def endpoint(path, params, timeout):
        calls.append((path, params, timeout))
        return groups

    monkeypatch.setattr(discovery, '_fetch_json', endpoint)
    audit = {}
    rows, warnings = fetch_prediction_markets(config or CONFIG, DAY, live=True, now=NOW,
                                              discovery_audit=audit)
    return rows, audit, calls, warnings


@pytest.mark.parametrize('senate,house', [('D', 'D'), ('R', 'R'), ('D', 'R'), ('R', 'D')])
@pytest.mark.parametrize('days,expected', [(14, 3), (14.0001, 4), (0, 4), (-1, 4)])
def test_joint_priority_is_party_neutral_with_fourteen_day_bound(senate, house, days, expected):
    question = f'2026 Balance of Power: {senate} Senate, {house} House'
    base = discovery.classify_market(question, TITLE)
    rank = discovery.screening_priority(base, question, TITLE, NOW+timedelta(days=days), NOW)
    assert rank['base_importance_rank'] == 4
    assert rank['effective_importance_rank'] == rank['importance_rank'] == expected
    assert rank['election_scope'] == 'us_congress_joint_control'
    assert base['driver'] == 'observe_only'
    assert rank['urgency_window_days'] == 14


@pytest.mark.parametrize('question,title', [
    ('Will the Democratic Party control the state House and state Senate?', 'US 2026 Midterms'),
    ('Will Republicans win the NC-03 House seat?', 'Congress House and Senate race'),
    ('Will Democrats control the House and Senate?', 'Michigan State Legislature'),
    ('Will Democrats win a Senate seat?', 'US midterm Senate election'),
    ('Will the US House and Senate hold a Republican hearing?', 'US Congress'),
    ('Will Republicans control the US House or Senate?', 'US Congress'),
    ('Will Republicans control the House and Senate?', 'Philippines Congress'),
    ('Will Democrats control the House?', 'US House and Senate control 2026'),
    ('Will Republicans control the Texas House and Senate?', 'US 2026 Midterms'),
])
def test_local_or_single_chamber_races_do_not_receive_joint_control_boost(question, title):
    base = discovery.classify_market(question, title)
    rank = discovery.screening_priority(base, question, title, NOW+timedelta(days=5), NOW)
    assert rank['effective_importance_rank'] == 4 and rank['urgency_applied'] is False
    assert rank['election_scope'] is None


def test_joint_control_replaces_fed_background_without_displacing_core_fed(monkeypatch):
    groups = [event(1, 'Fed decision', 'Will there be no change in Fed interest rates?'),
              event(2, 'Fed decision background',
                    'Will two members dissent the October Fed decision?'),
              event(3, 'US inflation', 'Will US annual inflation be 3.7%?'),
              event(4, 'US tariffs', 'Will the US lower tariffs?'),
              event(5, 'US-Iran conflict', 'Will the US attack Iran?'),
              event(6, 'WTI oil', 'Will WTI crude oil reach $100?'),
              event(7, TITLE, '2026 Balance of Power: D Senate, D House', .655)]
    rows, audit, calls, _ = collect(monkeypatch, groups)
    assert len(rows) == 6 and rows[0]['id'] == '1'
    assert '7' in [row['id'] for row in rows] and '2' not in [row['id'] for row in rows]
    assert len(calls) == 1  # No extra search or network budget.
    joint = next(row for row in rows if row['id'] == '7')
    assert joint['driver'] == 'observe_only' and joint['importance_rank'] == 3
    assert discovery.discovery_contract(joint)['importance_rank'] == 3
    reasons = {row['market_id']: row['final_disposition'] for row in audit['candidates']}
    assert reasons['7'] == 'selected' and reasons['2'] == 'max_events'


def test_audit_distinguishes_nonrepresentative_channel_quota_and_quality(monkeypatch):
    group = event(1, 'Fed decision', 'Will the Fed cut interest rates?', .2)
    group['markets'] += event(2, 'Fed decision', 'Will the Fed hike interest rates?', .8)['markets']
    groups = [group, event(3, 'Another Fed event', 'Will Fed interest rates remain unchanged?'),
              event(4, 'WTI oil', 'Will WTI oil reach $100?'),
              event(5, TITLE, '2026 Balance of Power: D Senate, D House', liquidityNum=10)]
    config = deepcopy(CONFIG)
    config['discovery']['max_events'] = 1
    _, audit, _, _ = collect(monkeypatch, groups, config)
    reasons = {row['market_id']: row['final_disposition'] for row in audit['candidates']}
    assert reasons == {'1': 'not_representative', '2': 'selected', '3': 'channel_limit',
                       '4': 'max_events', '5': 'ineligible'}
    assert all(row['final_reason'] for row in audit['candidates'])


def test_saved_priority_is_rederived_and_cached_score_changes_are_rejected(monkeypatch):
    rows, _, _, _ = collect(monkeypatch, [event(
        7, TITLE, '2026 Balance of Power: D Senate, D House', .655)])
    row = rows[0]
    row['discovery']['days_to_end'] = 365  # Cached convenience value is not authority.
    assert discovery.discovery_contract(row)['importance_rank'] == 3
    row['importance_rank'] = 0
    assert discovery.discovery_contract(row) is None
    row = deepcopy(rows[0])
    row['importance_rank'] = 3
    row['discovery']['urgency_applied'] = False
    assert discovery.discovery_contract(row) is None


def test_joint_probability_changes_do_not_become_equity_direction_or_sizing(monkeypatch):
    policies = []
    for probability in (.99, .01):
        rows, _, _, _ = collect(monkeypatch, [
            event(1, 'Fed decision', 'Will there be no change in Fed interest rates?', .9),
            event(7, TITLE, '2026 Balance of Power: D Senate, D House', probability),
        ])
        scenario = build_scenario_context(DAY, [], rows, {}, now=NOW, contracts=[])
        policies.append(scenario['policy'])
        election = next(row for row in scenario['events'] if row['id'] == '7')
        assert election['driver'] == 'observe_only'
    assert policies[0] == policies[1]


def test_legacy_saved_rows_keep_their_original_unversioned_base_priority(monkeypatch):
    rows, _, _, _ = collect(monkeypatch, [event(
        7, TITLE, '2026 Balance of Power: D Senate, D House', .655)])
    row = rows[0]
    base = discovery.classify_market(row['question'], TITLE)
    keys = discovery.screening_priority(base, row['question'], TITLE, row['end_date'], NOW)
    for key in keys:
        row.pop(key, None)
        row['discovery'].pop(key, None)
    row.update(base)
    row['discovery'].update(importance_rank=4, selection_reason=base['selection_reason'])
    assert discovery.discovery_contract(row)['importance_rank'] == 4


def test_candidate_audit_is_frozen_with_report_inputs(tmp_path):
    paths = ProjectPaths(tmp_path)
    paths.ensure()
    audit_path = paths.processed/f'prediction-discovery-{DAY}.json'
    write_json(audit_path, {'candidates': [
        {'market_id': '562828', 'final_disposition': 'selected'}
    ]})
    draft = begin_analysis_snapshot(paths, DAY, source_root=tmp_path)
    write_json(audit_path, {'candidates': []})
    finalize_analysis_snapshot(draft, {})
    captured = Path(draft['input_root'])/f'data/processed/prediction-discovery-{DAY}.json'
    assert read_json(captured)['candidates'][0]['market_id'] == '562828'
    assert verify_snapshot(Path(draft['path']))['valid']
