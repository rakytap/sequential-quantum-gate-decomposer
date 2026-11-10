import copy
import importlib.util
from pathlib import Path

import pytest

path=Path(__file__).resolve().parents[2]/'examples/decomposition/symbolic_objective_migration.py'
spec=importlib.util.spec_from_file_location('objective_migration',path)
migration=importlib.util.module_from_spec(spec)
spec.loader.exec_module(migration)


def test_migration_requires_exact_coverage_and_preserves_historical_sources():
    legacy=dict(objective=['native_two_qubit_operations','original_parameter_sensitive_score','single_qubit_operations'])
    report=dict(comparison_contract=legacy,rows=[dict(phase_exact=dict(operations=[dict(parameters=['pi/2'])]))],
                synthesis_runs=[dict(comparison_contract=copy.deepcopy(legacy),source_sha256={'actual':'old'})])
    audit=[migration.audit_parameter('pi/2')]
    result=migration.migrate(copy.deepcopy(report),audit)
    assert result['objective_migration']['unchanged_retained_scores']
    assert result['synthesis_runs']==report['synthesis_runs']
    with pytest.raises(ValueError):migration.migrate(copy.deepcopy(report),[])
    with pytest.raises(ValueError):migration.migrate(copy.deepcopy(report),[dict(parameter='pi/2',unchanged=False)])


def test_audit_rejects_the_historical_negative_cycle_parameter():
    assert migration.audit_parameter('1/log(2049/2048)')['unchanged'] is False
