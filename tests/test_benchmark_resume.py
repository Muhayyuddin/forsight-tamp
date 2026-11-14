import json

import pytest

from generate_results import _sha256_file, _validate_resume_checkpoint


def _checkpoint(tmp_path):
    trace_path = tmp_path / "traces" / "Method" / "Scenario" / "trial_000.json"
    trace_path.parent.mkdir(parents=True)
    trace_path.write_text(
        json.dumps({
            "method": "Method",
            "scenario": "Scenario",
            "trial_index": 0,
            "trial_seed": 41,
            "trace": {},
        }),
        encoding="utf-8",
    )
    raw = {
        "Method": {
            "Scenario": [{
                "trial_index": 0,
                "trial_seed": 41,
                "llm_api_requests": 3,
                "trace_file": str(trace_path.relative_to(tmp_path)),
                "trace_sha256": _sha256_file(trace_path),
            }]
        }
    }
    return raw, trace_path


def test_resume_validation_binds_trials_to_seed_and_trace(tmp_path):
    raw, trace_path = _checkpoint(tmp_path)
    _validate_resume_checkpoint(
        raw,
        out_dir=tmp_path,
        methods=["Method"],
        scenarios=["Scenario"],
        trials_per_cell=10,
        max_llm_requests=10,
        base_seed=41,
    )

    trace_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="trace hash mismatch"):
        _validate_resume_checkpoint(
            raw,
            out_dir=tmp_path,
            methods=["Method"],
            scenarios=["Scenario"],
            trials_per_cell=10,
            max_llm_requests=10,
            base_seed=41,
        )


def test_resume_validation_rejects_mutated_foresight_artifacts(tmp_path):
    method = "ForeSight"
    scenario = "A1_easy_few"
    trace_path = (
        tmp_path / "traces" / method / scenario / "trial_000.json"
    )
    trace_path.parent.mkdir(parents=True)
    trace_path.write_text(json.dumps({
        "method": method,
        "scenario": scenario,
        "trial_index": 0,
        "trial_seed": 41,
        "trace": {},
    }), encoding="utf-8")
    artifact_directory = (
        tmp_path / "artifacts" / method / scenario / "trial_000"
    )
    artifact_directory.mkdir(parents=True)
    artifact_path = artifact_directory / "executed_joint_space_paths.json"
    artifact_path.write_text("{}", encoding="utf-8")
    raw = {method: {scenario: [{
        "trial_index": 0,
        "trial_seed": 41,
        "llm_api_requests": 3,
        "trace_file": str(trace_path.relative_to(tmp_path)),
        "trace_sha256": _sha256_file(trace_path),
        "artifact_directory": str(artifact_directory.relative_to(tmp_path)),
        "artifact_files_sha256": {
            artifact_path.name: _sha256_file(artifact_path),
        },
    }]}}

    _validate_resume_checkpoint(
        raw,
        out_dir=tmp_path,
        methods=[method],
        scenarios=[scenario],
        trials_per_cell=10,
        max_llm_requests=10,
        base_seed=41,
    )

    artifact_path.write_text('{"tampered": true}', encoding="utf-8")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        _validate_resume_checkpoint(
            raw,
            out_dir=tmp_path,
            methods=[method],
            scenarios=[scenario],
            trials_per_cell=10,
            max_llm_requests=10,
            base_seed=41,
        )
