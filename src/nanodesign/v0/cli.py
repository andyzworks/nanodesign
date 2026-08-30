"""Command-line entry points for validating NanoDesign v0 infrastructure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nanodesign.v0.config import (
    load_config,
    validate_resolved_assets,
    validate_v0_config,
)
from nanodesign.v0.data.adapters import ADAPTERS
from nanodesign.v0.data.inventory import (
    audit_rna_complex_inventory,
    load_rna_complex_inventory,
)
from nanodesign.v0.data.manifest import audit_manifest, load_manifest
from nanodesign.v0.evaluation import PROTOCOLS
from nanodesign.v0.model import NanoDesignTiny, NanoDesignTinyConfig
from nanodesign.v0.spec import get_v0_spec


def _model_from_config(path: str | Path) -> NanoDesignTiny:
    config = load_config(path)
    validate_v0_config(config)
    model_values = {key: config["model"][key] for key in NanoDesignTinyConfig.__dataclass_fields__}
    model = NanoDesignTiny(NanoDesignTinyConfig.from_mapping(model_values))
    model.validate_parameter_budget()
    return model


def command_spec(_: argparse.Namespace) -> int:
    payload = get_v0_spec().to_dict()
    payload["data_adapters"] = {
        source.value: {
            "task": adapter.task.name.lower(),
            "purpose": adapter.purpose.value,
            "status": adapter.implementation_status,
        }
        for source, adapter in ADAPTERS.items()
    }
    payload["evaluation"] = {
        task.name.lower(): [metric.name for metric in protocol.metrics]
        for task, protocol in PROTOCOLS.items()
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def command_validate_config(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    config = load_config(config_path)
    report = validate_v0_config(config)
    payload: dict[str, object] = {
        "valid": report.valid,
        "ready": report.ready,
        "blockers": report.blockers,
    }
    if not args.allow_tbd:
        report.require_ready()
        payload["rna_inventory"] = validate_resolved_assets(
            config, config_directory=config_path.parent
        )
    print(json.dumps(payload, indent=2))
    return 0


def command_validate_manifest(args: argparse.Namespace) -> int:
    result = audit_manifest(load_manifest(args.manifest))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def command_validate_rna_inventory(args: argparse.Namespace) -> int:
    result = audit_rna_complex_inventory(load_rna_complex_inventory(args.inventory))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def command_model_summary(args: argparse.Namespace) -> int:
    model = _model_from_config(args.config)
    print(
        json.dumps(
            {
                "architecture": "RosettaCommons Foundry rfd3na.model.RFD3.RFD3",
                "parameter_count": model.parameter_count,
                "within_v0_budget": True,
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    spec = subparsers.add_parser("spec", help="print the immutable v0 contract")
    spec.set_defaults(function=command_spec)
    validate_config = subparsers.add_parser(
        "validate-config", help="validate exact decisions and list unresolved blockers"
    )
    validate_config.add_argument("--config", default="configs/v0.yaml")
    validate_config.add_argument("--allow-tbd", action="store_true")
    validate_config.set_defaults(function=command_validate_config)
    validate_manifest = subparsers.add_parser(
        "validate-manifest", help="audit source and split leakage constraints"
    )
    validate_manifest.add_argument("--manifest", required=True)
    validate_manifest.set_defaults(function=command_validate_manifest)
    validate_inventory = subparsers.add_parser(
        "validate-rna-inventory",
        help="audit usable RNA-target complex counts before freezing the data pool",
    )
    validate_inventory.add_argument("--inventory", required=True)
    validate_inventory.set_defaults(function=command_validate_rna_inventory)
    model_summary = subparsers.add_parser("model-summary")
    model_summary.add_argument("--config", default="configs/v0.yaml")
    model_summary.set_defaults(function=command_model_summary)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(args.function(args))


if __name__ == "__main__":
    main()
