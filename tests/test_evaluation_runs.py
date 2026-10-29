"""Runner tests use a tiny copied engine; no real model tuning or market data is evaluated."""
import json
import shutil
from pathlib import Path
from unittest.mock import Mock

import pytest

from twiz import cli
from twiz import evaluation_runs as runs

ENGINE = '''from pathlib import Path
from . import sessions
class EvaluationV2Protocol:
    @classmethod
    def from_config(cls, config):
        value = cls()
        value.config = config
        return value

def select_pretest_models_v2(market, macro=None, protocol=None):
    assert list(market) == ['pretest_only']
    assert macro == ['pretest_macro']
    return {'pretest_keys': list(market), 'helper': sessions.VALUE, 'protocol': protocol.config}

def run_walk_forward_evaluation_v2(market, macro=None, protocol=None, pretest_selection=None):
    run = Path(__file__).resolve().parents[3]
    assert (run/'pretest-selection-seal.json').exists()
    assert pretest_selection['pretest_keys'] == ['pretest_only']
    assert market == {'full_only': [99]} and macro == ['full_macro']
    return {'status': 'target_not_met', 'model_version': 'fixture', 'helper': sessions.VALUE,
            'goal_completion_proven': False, 'selection': pretest_selection,
            'primary': {'primary_horizon': {'accuracy': .5}},
            'individual': {'primary_horizon': {'accuracy': .4}}}
'''


@pytest.fixture
def fixture(tmp_path):
    source, dataset, root = tmp_path/'source', tmp_path/'datasets', tmp_path/'experiments'
    package = source/'src/twiz'
    package.mkdir(parents=True)
    (package/'__init__.py').write_text('')
    (package/'sessions.py').write_text('VALUE = 7\n')
    (package/'evaluation_v2.py').write_text(ENGINE)
    shutil.copyfile(runs.__file__, package/'evaluation_runs.py')
    dataset.mkdir()
    for name, data in [('pretest-market.json', {'pretest_only': [1]}),
                       ('pretest-macro.json', ['pretest_macro']),
                       ('market.json', {'full_only': [99]}), ('macro.json', ['full_macro'])]:
        (dataset/name).write_text(json.dumps(data))
    protocol = tmp_path/'protocol.yml'
    protocol.write_text('protocol_version: test-fixture\n')
    return source, dataset, root, protocol


def execute(fixture, **kwargs):
    source, dataset, root, protocol = fixture
    return runs.run_evaluation_experiment(
        protocol, dataset, experiment_root=root, source_root=source, **kwargs
    )


def test_direction_runner_cannot_silently_substitute_daily_labels_for_swing_protocol(fixture):
    _, _, root, protocol = fixture
    protocol.write_text('protocol_version: author-swing-v1\n')
    with pytest.raises(ValueError, match="cannot run a swing protocol"):
        execute(fixture)
    assert not root.exists()


def test_two_phase_run_uses_captured_helpers_and_sealed_pretest_selection(fixture, monkeypatch):
    source, _, root, _ = fixture
    original = runs._launch_phase

    def mutate_live_after_capture(run, phase, selection_sha256=None):
        if phase == 'selection':
            (source/'src/twiz/sessions.py').write_text('VALUE = 999\n')
        else:
            assert (run/'pretest-selection-seal.json').is_file()
        original(run, phase, selection_sha256)

    monkeypatch.setattr(runs, '_launch_phase', mutate_live_after_capture)
    receipt = execute(fixture)
    run = Path(receipt['path'])
    assert runs.verify_evaluation_experiment(run, receipt['manifest_sha256'])['valid']
    result = json.loads((run/'result.json').read_text())
    assert result['helper'] == result['selection']['helper'] == 7
    assert result['selection']['pretest_keys'] == ['pretest_only']
    assert (run/'frozen/src/twiz/sessions.py').read_text() == 'VALUE = 7\n'
    assert json.loads((run/'summary.json').read_text())['goal_completion_proven'] is False
    assert json.loads((root.parent/'latest-experiment-v2.json').read_text()) == receipt
    inputs = json.loads((run/'input-manifest.json').read_text())
    assert all(name in inputs['runtime'] for name in ('python', 'numpy', 'PyYAML'))
    assert inputs['runtime']['numpy']['sha256']
    assert inputs['protocol_preregistration']['status'] == 'not_provided_custom_protocol'
    assert 'captured_engine=' in (run/'selection.log').read_text()
    assert not list((run/'frozen/src').rglob('*.pyc'))


def test_same_inputs_get_new_archives_without_replacing_v1_or_prior_results(fixture):
    _, _, root, _ = fixture
    v1 = root/'v1-existing/result.json'
    v1.parent.mkdir(parents=True)
    v1.write_text('original-v1')
    old_pointer = root.parent/'latest-experiment.json'
    old_pointer.write_text('original-v1-pointer')
    first, second = execute(fixture), execute(fixture)
    assert first['run_id'] != second['run_id']
    assert runs.verify_evaluation_experiment(Path(first['path']), first['manifest_sha256'])['valid']
    assert v1.read_text() == 'original-v1' and old_pointer.read_text() == 'original-v1-pointer'


def test_completed_archive_rejects_helper_selection_data_and_manifest_tampering(fixture):
    receipt = execute(fixture)
    run = Path(receipt['path'])
    for name in ['frozen/src/twiz/sessions.py', 'pretest-selection.json',
                 'pretest-selection-seal.json', 'frozen/datasets/market.json', 'summary.json',
                 'manifest.json']:
        path = run/name
        original = path.read_bytes()
        path.write_bytes(original+b' ')
        assert not runs.verify_evaluation_experiment(run, receipt['manifest_sha256'])['valid'], name
        path.write_bytes(original)
    assert runs.verify_evaluation_experiment(run)['valid']


def test_started_archive_is_not_proof_of_activity_and_audit_never_restarts(fixture, monkeypatch):
    _, _, root, _ = fixture
    monkeypatch.setattr(runs, '_launch_phase', Mock(side_effect=KeyboardInterrupt))
    with pytest.raises(KeyboardInterrupt):
        execute(fixture)
    archives = list(root.iterdir())
    assert len(archives) == 1 and (archives[0]/'started.json').exists()
    audit = runs.verify_evaluation_experiment(archives[0])
    assert audit['status'] == 'incomplete' and not audit['valid']
    assert 'not proof' in audit['errors'][0]
    assert list(root.iterdir()) == archives
    assert not (root.parent/'latest-experiment-v2.json').exists()


def test_changed_selection_is_rejected_before_phase_two(fixture, monkeypatch):
    original = runs._launch_phase

    def corrupt_before_phase_two(run, phase, selection_sha256=None):
        if phase == 'evaluation':
            (run/'pretest-selection.json').write_text('{}')
        original(run, phase, selection_sha256)

    monkeypatch.setattr(runs, '_launch_phase', corrupt_before_phase_two)
    with pytest.raises(ValueError, match='selection changed'):
        execute(fixture)
    run = next(fixture[2].iterdir())
    assert not (run/'result.json').exists()
    assert runs.verify_evaluation_experiment(run)['status'] == 'integrity_failed'


def test_cli_evaluate_and_audit_evaluation_flags(monkeypatch):
    evaluate = Mock(return_value=Path('/tmp/v2-run'))
    audit = Mock(return_value=0)
    monkeypatch.setattr(cli, 'cmd_evaluate', evaluate)
    monkeypatch.setattr(cli, 'cmd_audit_evaluation', audit)
    assert cli._argparse_main(['evaluate', '--protocol', '/tmp/p.yml',
                              '--dataset-dir', '/tmp/data']) == 0
    evaluate.assert_called_once_with(Path('/tmp/p.yml'), Path('/tmp/data'))
    assert cli._argparse_main(['audit-evaluation', '--run', '/tmp/v2-run', '--hash', 'abc']) == 0
    audit.assert_called_once_with(Path('/tmp/v2-run'), 'abc')
    if cli.typer is not None:
        from typer.testing import CliRunner
        result = CliRunner().invoke(cli.app, ['evaluate', '--protocol', '/tmp/p.yml',
                                             '--dataset-dir', '/tmp/data'])
        assert result.exit_code == 0, result.output
        result = CliRunner().invoke(cli.app, ['audit-evaluation', '--run', '/tmp/v2-run'])
        assert result.exit_code == 0, result.output


def test_rewriting_selection_and_its_seal_cannot_override_controller_receipt(fixture, monkeypatch):
    original = runs._launch_phase

    def corrupt_both(run, phase, selection_sha256=None):
        if phase == 'evaluation':
            selection = run/'pretest-selection.json'
            selection.write_text('{}')
            seal_path = run/'pretest-selection-seal.json'
            seal = json.loads(seal_path.read_text())
            seal['selection_sha256'] = runs._hash(selection.read_bytes())
            seal_path.write_text(json.dumps(seal))
        original(run, phase, selection_sha256)

    monkeypatch.setattr(runs, '_launch_phase', corrupt_both)
    with pytest.raises(ValueError, match='selection changed'):
        execute(fixture)
    assert not (next(fixture[2].iterdir())/'result.json').exists()


def test_worker_failure_preserves_log_and_does_not_publish_latest_success(fixture):
    source, _, root, _ = fixture
    engine = source/'src/twiz/evaluation_v2.py'
    engine.write_text(ENGINE.replace("assert list(market) == ['pretest_only']",
                                     "raise ValueError('intentional fixture failure')"))
    with pytest.raises(RuntimeError, match='preserved archive and log'):
        execute(fixture)
    run = next(root.iterdir())
    assert (run/'failed-selection.json').exists()
    assert 'intentional fixture failure' in (run/'selection.log').read_text()
    assert runs.verify_evaluation_experiment(run)['status'] == 'failed'
    assert not (root.parent/'latest-experiment-v2.json').exists()


def test_modified_upstream_pretest_dataset_seal_is_rejected_before_selection(fixture):
    _, dataset, root, _ = fixture
    (dataset/'pretest-seal.json').write_text(json.dumps({'files': {
        'pretest-market.json': {'sha256': 'incorrect-prior-digest'}
    }}))
    with pytest.raises(ValueError, match='upstream seal'):
        execute(fixture)
    run = next(root.iterdir())
    assert not (run/'pretest-selection.json').exists()
    assert runs.verify_evaluation_experiment(run)['status'] == 'integrity_failed'


def test_explicit_preregistration_receipt_is_captured_and_audited(fixture):
    source, _, _, protocol = fixture
    receipt_path = source/'registered.json'
    receipt_path.write_text(json.dumps({'sha256': runs._hash(protocol.read_bytes()),
                                       'frozen_at': '2026-10-28T10:00:00+00:00'}))
    receipt = execute(fixture, preregistration_receipt=receipt_path)
    run = Path(receipt['path'])
    inputs = json.loads((run/'input-manifest.json').read_text())
    proof = inputs['protocol_preregistration']
    assert proof['status'] == 'matched'
    assert proof['protocol_sha256'] == runs._hash(protocol.read_bytes())
    assert (run/'protocol-preregistration.json').read_bytes() == receipt_path.read_bytes()
    assert any(item['role'] == 'protocol_preregistration' for item in inputs['files'])
    (run/'protocol-preregistration.json').write_text('{}')
    assert not runs.verify_evaluation_experiment(run, receipt['manifest_sha256'])['valid']


def test_default_protocol_requires_matching_freeze_receipt_before_execution(fixture, monkeypatch):
    source, dataset, root, original_protocol = fixture
    protocol = source/'config/evaluation-v2.yml'
    protocol.parent.mkdir()
    protocol.write_bytes(original_protocol.read_bytes())
    monkeypatch.setattr(runs, '__file__', str(source/'src/twiz/evaluation_runs.py'))
    default_fixture = source, dataset, root, protocol
    with pytest.raises(ValueError, match='preregistration receipt is required'):
        execute(default_fixture)
    receipt_path = source/'data/processed/evaluation/protocol-v2-freeze.json'
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_text(json.dumps({'sha256': 'changed-config-digest'}))
    with pytest.raises(ValueError, match='differs from the preregistration'):
        execute(default_fixture)
    assert not root.exists()
    receipt_path.write_text(json.dumps({'sha256': runs._hash(protocol.read_bytes())}))
    result = execute(default_fixture)
    assert runs.verify_evaluation_experiment(Path(result['path']))['valid']
    archived = json.loads((Path(result['path'])/'input-manifest.json').read_text())
    assert archived['protocol_preregistration']['status'] == 'matched'
