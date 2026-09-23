from __future__ import annotations

import copy
import hashlib
import itertools
import json
import string
from pathlib import Path
from typing import Any

import orjson
import yaml

from agentic_rl_forge.experiments.models import (
    ExperimentArtifactPlan,
    ExperimentMatrixSpec,
    ExperimentMetricGate,
    ExperimentPlan,
    ExperimentScalar,
    ExperimentStagePlan,
    ExperimentTrialPlan,
)
from agentic_rl_forge.pipelines import SearchR1CollectionConfig


class ExperimentPlanError(ValueError):
    pass


def load_experiment_matrix_spec(path: Path) -> ExperimentMatrixSpec:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ExperimentPlanError("experiment matrix is not valid readable YAML") from error
    if not isinstance(payload, dict):
        raise ExperimentPlanError("experiment matrix must contain a YAML object")
    try:
        return ExperimentMatrixSpec.model_validate(payload)
    except ValueError as error:
        raise ExperimentPlanError("experiment matrix contract is invalid") from error


def build_experiment_plan(path: Path) -> ExperimentPlan:
    spec_path = path.resolve(strict=True)
    source_bytes = spec_path.read_bytes()
    spec = load_experiment_matrix_spec(spec_path)
    base_path = Path(spec.base_config)
    if not base_path.is_absolute():
        base_path = spec_path.parent / base_path
    base_path = base_path.resolve(strict=True)
    try:
        base_bytes = base_path.read_bytes()
        base_payload = yaml.safe_load(base_bytes)
    except (OSError, yaml.YAMLError) as error:
        raise ExperimentPlanError("experiment base config is not valid readable YAML") from error
    if not isinstance(base_payload, dict) or any(not isinstance(key, str) for key in base_payload):
        raise ExperimentPlanError("experiment base config must contain a string-keyed YAML object")
    _canonical_json(base_payload, "experiment base config")

    axis_names = tuple(sorted(spec.axes))
    parameter_sets = []
    for combination in itertools.product(*(spec.axes[name] for name in axis_names)):
        parameters: dict[str, ExperimentScalar] = {
            **spec.fixed_parameters,
            **dict(zip(axis_names, combination, strict=True)),
        }
        sorted_parameters = dict(sorted(parameters.items()))
        if any(_selector_matches(sorted_parameters, selector) for selector in spec.exclude):
            continue
        parameter_sets.append(sorted_parameters)
    if not parameter_sets:
        raise ExperimentPlanError("experiment matrix exclusions removed every trial")
    if len(parameter_sets) > spec.max_trials:
        raise ExperimentPlanError("experiment matrix exceeds its maximum trial count")

    try:
        trials = tuple(
            sorted(
                (_build_trial(spec, base_payload, parameters) for parameters in parameter_sets),
                key=lambda trial: trial.trial_id,
            )
        )
    except ExperimentPlanError:
        raise
    except ValueError as error:
        raise ExperimentPlanError("expanded experiment trial is invalid") from error
    source_digest = hashlib.sha256(source_bytes).hexdigest()
    base_digest = hashlib.sha256(base_bytes).hexdigest()
    plan_id = ExperimentPlan.expected_plan_id(
        name=spec.name,
        source_spec_sha256=source_digest,
        base_config_sha256=base_digest,
        trials=trials,
    )
    try:
        return ExperimentPlan(
            plan_id=plan_id,
            name=spec.name,
            source_spec_sha256=source_digest,
            base_config_sha256=base_digest,
            trials=trials,
            trial_count=len(trials),
        )
    except ValueError as error:
        raise ExperimentPlanError("expanded experiment plan is invalid") from error


def _build_trial(
    spec: ExperimentMatrixSpec,
    base_payload: dict[str, Any],
    parameters: dict[str, ExperimentScalar],
) -> ExperimentTrialPlan:
    resolved = copy.deepcopy(base_payload)
    for config_path, parameter_name in sorted(spec.config_bindings.items()):
        _set_existing_config_value(resolved, config_path, parameters[parameter_name])
    try:
        resolved_config = SearchR1CollectionConfig.model_validate(resolved).model_dump(
            mode="json",
            exclude_none=True,
        )
    except ValueError as error:
        raise ExperimentPlanError("experiment resolved collection config is invalid") from error
    config_bytes = _canonical_json(resolved_config, "experiment resolved config")
    config_digest = hashlib.sha256(config_bytes).hexdigest()
    trial_id = ExperimentTrialPlan.expected_trial_id(
        experiment_name=spec.name,
        parameters=parameters,
        config_digest=config_digest,
    )
    render_values = {
        **{name: _scalar_text(value) for name, value in parameters.items()},
        "trial_id": trial_id,
        "config_path": "{config_path}",
    }
    stages = []
    for template in spec.stages:
        inputs = tuple(
            ExperimentArtifactPlan(
                kind=item.kind,
                path=_render(item.path, render_values),
                required=item.required,
            )
            for item in template.inputs
        )
        outputs = tuple(
            ExperimentArtifactPlan(
                kind=item.kind,
                path=_render(item.path, render_values),
                required=item.required,
            )
            for item in template.outputs
        )
        gates = tuple(
            ExperimentMetricGate(
                name=gate.name,
                artifact_path=_render(gate.artifact_path, render_values),
                metric=gate.metric,
                statistic=gate.statistic,
                minimum=gate.minimum,
                maximum=gate.maximum,
            )
            for gate in template.gates
        )
        stages.append(
            ExperimentStagePlan(
                name=template.name,
                command=tuple(_render(argument, render_values) for argument in template.command),
                depends_on=template.depends_on,
                inputs=inputs,
                outputs=outputs,
                gates=gates,
            )
        )
    return ExperimentTrialPlan(
        trial_id=trial_id,
        parameters=parameters,
        resolved_config=resolved_config,
        config_digest=config_digest,
        stages=tuple(stages),
    )


def _set_existing_config_value(
    payload: dict[str, Any],
    dotted_path: str,
    value: ExperimentScalar,
) -> None:
    parts = dotted_path.split(".")
    current = payload
    for part in parts[:-1]:
        nested = current.get(part)
        if not isinstance(nested, dict) or any(not isinstance(key, str) for key in nested):
            raise ExperimentPlanError(
                f"experiment config binding {dotted_path!r} does not resolve to an object"
            )
        current = nested
    final = parts[-1]
    if final not in current:
        raise ExperimentPlanError(
            f"experiment config binding {dotted_path!r} does not exist in the base config"
        )
    current[final] = value


def _render(template: str, values: dict[str, str]) -> str:
    formatter = string.Formatter()
    for _, field_name, format_spec, conversion in formatter.parse(template):
        if field_name is None:
            continue
        if field_name not in values:
            raise ExperimentPlanError(
                f"experiment template references unknown value {field_name!r}"
            )
        if format_spec or conversion:
            raise ExperimentPlanError("experiment templates do not support format modifiers")
    try:
        return template.format_map(values)
    except (KeyError, ValueError) as error:
        raise ExperimentPlanError("experiment template is invalid") from error


def _selector_matches(
    parameters: dict[str, ExperimentScalar],
    selector: dict[str, ExperimentScalar],
) -> bool:
    return all(
        _scalar_identity(parameters[name]) == _scalar_identity(value)
        for name, value in selector.items()
    )


def _scalar_identity(value: ExperimentScalar) -> str:
    return json.dumps(
        {"type": type(value).__name__, "value": value},
        sort_keys=True,
        separators=(",", ":"),
    )


def _scalar_text(value: ExperimentScalar) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _canonical_json(payload: Any, label: str) -> bytes:
    try:
        return orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)
    except (TypeError, ValueError) as error:
        raise ExperimentPlanError(f"{label} contains non-JSON values") from error
