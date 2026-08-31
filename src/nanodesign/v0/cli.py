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
from nanodesign.v0.evaluators import evaluate_protein_binder
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


def _binder_generation_context(
    args: argparse.Namespace,
) -> tuple[list[str], str, dict[str, object] | None]:
    if args.generation_metadata is None:
        if not args.target_chains or not args.binder_chain:
            raise ValueError(
                "target-chains and binder-chain are required without generation metadata"
            )
        return list(args.target_chains), args.binder_chain, None
    metadata_path = Path(args.generation_metadata).resolve()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise TypeError("generation metadata must be a JSON object")
    tasks = metadata.get("tasks", metadata.get("generation"))
    if not isinstance(tasks, dict) or not isinstance(tasks.get("protein_binder"), dict):
        raise TypeError("generation metadata lacks tasks.protein_binder")
    binder = tasks["protein_binder"]
    generated_path = Path(binder.get("structure_path", "")).resolve()
    if generated_path != Path(args.generated_complex).resolve():
        raise ValueError("generated complex does not match generation metadata")
    target_chains = binder.get("target_chains")
    binder_chain = binder.get("binder_chain")
    if (
        not isinstance(target_chains, list)
        or not target_chains
        or not all(isinstance(chain, str) and chain for chain in target_chains)
        or not isinstance(binder_chain, str)
        or not binder_chain
        or binder_chain in target_chains
    ):
        raise ValueError("generation metadata has invalid protein binder chain roles")
    if args.target_chains is not None and target_chains != list(args.target_chains):
        raise ValueError("target chains do not match generation metadata")
    if args.binder_chain is not None and binder_chain != args.binder_chain:
        raise ValueError("binder chain does not match generation metadata")
    required = (
        "checkpoint",
        "samples_seen",
        "optimizer_step",
        "manifest_sha256",
        "config_sha256",
    )
    missing = [key for key in required if key not in metadata]
    missing.extend(
        f"tasks.protein_binder.{key}" for key in ("sample_id", "seed") if key not in binder
    )
    if missing:
        raise ValueError(f"generation metadata lacks provenance fields: {missing}")
    provenance = {
        "metadata_path": str(metadata_path),
        "sample_id": binder["sample_id"],
        "seed": int(binder["seed"]),
        **{key: metadata[key] for key in required},
    }
    return target_chains, binder_chain, provenance


def command_evaluate_protein_binder(args: argparse.Namespace) -> int:
    """Run the frozen binder evaluator and persist its per-design result."""

    target_chains, binder_chain, provenance = _binder_generation_context(args)
    result = evaluate_protein_binder(
        args.generated_complex,
        target_chains=target_chains,
        binder_chain=binder_chain,
        output_dir=args.output_dir,
        colabfold_executable=args.colabfold_executable,
        rosetta_executable=args.rosetta_executable,
        pyrosetta_python=args.pyrosetta_python,
        pyrosetta_analyzer_script=args.pyrosetta_analyzer_script,
    )
    payload = {
        "generated_complex": str(Path(args.generated_complex).resolve()),
        "target_chains": target_chains,
        "binder_chain": binder_chain,
        "metrics": result.metrics,
        "passed": result.passed,
        "in_silico_success_rate": float(result.passed),
    }
    if provenance is not None:
        payload["generation_provenance"] = provenance
    result_json = Path(args.result_json)
    result_json.parent.mkdir(parents=True, exist_ok=True)
    result_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
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
    evaluate_binder = subparsers.add_parser(
        "evaluate-protein-binder",
        help="run the frozen ColabFold/Rosetta binder protocol for one generated complex",
    )
    evaluate_binder.add_argument("--generated-complex", required=True)
    evaluate_binder.add_argument("--target-chains", nargs="+")
    evaluate_binder.add_argument("--binder-chain")
    evaluate_binder.add_argument("--output-dir", required=True)
    evaluate_binder.add_argument("--result-json", required=True)
    evaluate_binder.add_argument("--colabfold-executable", default="colabfold_batch")
    evaluate_binder.add_argument(
        "--rosetta-executable", default="InterfaceAnalyzer.linuxgccrelease"
    )
    evaluate_binder.add_argument("--pyrosetta-python")
    evaluate_binder.add_argument("--pyrosetta-analyzer-script")
    evaluate_binder.add_argument(
        "--generation-metadata",
        help="optional milestone metadata.json used to validate and preserve provenance",
    )
    evaluate_binder.set_defaults(function=command_evaluate_protein_binder)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(args.function(args))


if __name__ == "__main__":
    main()
