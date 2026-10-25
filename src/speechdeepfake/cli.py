"""Command-line interface: ``speechdeepfake <command> [options]``.

Run ``speechdeepfake --help`` or ``speechdeepfake <command> --help`` for details.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from speechdeepfake import __version__, artifacts
from speechdeepfake.config import MODEL_NAMES, ConfigError, load_config
from speechdeepfake.utils.logging import get_logger, setup_logging

log = get_logger("cli")
DEFAULT_CONFIG = "configs/default.yaml"
MODEL_CHOICES = [*MODEL_NAMES, "all"]


def _cfg(args):
    return load_config(args.config)


def _start(cfg, name: str) -> None:
    path = setup_logging(artifacts.logs_dir(cfg), name)
    cfg.save_snapshot(cfg.results_dir / "config_snapshots" / f"{name}.yaml")
    log.info("run=%s type=%s config=%s log=%s", cfg.run_name, cfg.run_type, cfg.config_path, path)


# ----------------------------------------------------------------- commands


def cmd_inspect(args) -> int:
    from speechdeepfake.utils.reproducibility import environment_info

    setup_logging()
    info = environment_info()
    print(json.dumps(info, indent=2))
    if not info.get("cuda_available"):
        print("\nNote: CUDA is not available to torch. Training and wav2vec extraction will run on CPU. "
              "If this machine has an NVIDIA GPU, install a CUDA build of torch (see README).")
    if args.config:
        from speechdeepfake.pipeline import dataset_status

        cfg = _cfg(args)
        print(f"\nData directory: {cfg.data_dir}")
        print(json.dumps(dataset_status(cfg), indent=2))
    return 0


def cmd_check_data(args) -> int:
    from speechdeepfake.pipeline import dataset_status, required_corpora

    setup_logging()
    cfg = _cfg(args)
    status = dataset_status(cfg)
    print(json.dumps(status, indent=2))
    missing = [c for c in required_corpora(cfg) if not (cfg.manifests_dir / f"{c}.csv").exists()]
    if missing:
        print(f"\nManifests missing for: {missing}. See docs/dataset_setup.md, then run "
              f"`speechdeepfake prepare-data --config {args.config}`.")
        return 1
    return 0


def cmd_download(args) -> int:
    from speechdeepfake.data import loaders

    setup_logging()
    cfg = _cfg(args)
    if args.dataset == "cdadd":
        out = cfg.data_path(cfg.data.cdadd_root)
        print(f"Downloading CD-ADD (~3.9 GB) to {out}")
        loaders.download_cdadd(out)
    else:
        target = cfg.data_path(cfg.data.asvspoof_root)
        out = target.parent  # LA.zip extracts to <parent>/LA
        print(f"Downloading ASVspoof 2019 LA.zip (~7.1 GB) to {out}; it extracts to {out / 'LA'}")
        la = loaders.download_asvspoof2019_la(out, keep_zip=args.keep_zip, connections=args.connections)
        if la.resolve() != target.resolve():
            print(f"Note: extracted to {la}; set data.asvspoof_root accordingly.")
    print("Done. Next: speechdeepfake prepare-data --config", args.config)
    return 0


def cmd_download_subset(args) -> int:
    from speechdeepfake.data import subsets

    setup_logging()
    cfg = _cfg(args)
    if args.dataset in ("asvspoof2019_la", "all"):
        out = cfg.data_path(cfg.data.asvspoof_root).parent
        la = subsets.fetch_asvspoof2019_la_subset(
            out,
            n_bonafide={"train": args.bonafide, "dev": args.bonafide, "eval": args.bonafide},
            n_per_attack={"train": args.per_attack, "dev": args.per_attack, "eval": args.per_attack_eval},
            seed=cfg.seed, connections=args.connections)
        print(f"ASVspoof 2019 LA subset written to {la} (see SUBSET.json)")
    if args.dataset in ("cdadd", "all"):
        out = subsets.fetch_cdadd_subset(cfg.data_path(cfg.data.cdadd_root), args.cdadd_row_groups, cfg.seed)
        print(f"CD-ADD subset written to {out} (see SUBSET.json)")
    print("Next: speechdeepfake prepare-data --config", args.config)
    return 0


def cmd_prepare(args) -> int:
    from speechdeepfake.pipeline import prepare_data

    cfg = _cfg(args)
    _start(cfg, "prepare-data")
    res = prepare_data(cfg, args.corpus)
    print(json.dumps(res, indent=2))
    return 0 if not all(v.startswith("NOT AVAILABLE") for v in res.values()) else 1


def cmd_validate(args) -> int:
    from speechdeepfake.data.validation import summary_markdown
    from speechdeepfake.pipeline import validate_data

    cfg = _cfg(args)
    _start(cfg, "validate-data")
    summaries = validate_data(cfg, workers=args.workers, compute_hash=not args.no_hash, force=args.force)
    if not summaries:
        print("No manifests to validate. Run `speechdeepfake prepare-data` first (see docs/dataset_setup.md).")
        return 1
    failed = False
    for corpus, s in summaries.items():
        print(f"\n## {corpus}\n{summary_markdown(s)}")
        for e in s["integrity"]["errors"]:
            print("ERROR:", e)
            failed = True
        for w in s["integrity"]["warnings"]:
            print("WARNING:", w)
    return 1 if failed else 0


def cmd_smoke(args) -> int:
    from speechdeepfake.pipeline import smoke_test

    base = _cfg(args)
    out = Path(args.out) if args.out else Path(base.project_root) / "results" / "smoke_test"
    setup_logging(out / "logs", "smoke-test")
    checks = smoke_test(base, out, wav2vec_model=args.wav2vec_model)
    print("\nSMOKE TEST PASSED (synthetic fixture - software check only, not research results)")
    print(json.dumps(checks, indent=2))
    return 0


def cmd_extract(args) -> int:
    from speechdeepfake.pipeline import expand_models, extract_features

    cfg = _cfg(args)
    _start(cfg, "extract-features")
    extract_features(cfg, expand_models(args.model))
    return 0


def cmd_train(args) -> int:
    from speechdeepfake.pipeline import expand_models, train

    cfg = _cfg(args)
    _start(cfg, f"train-{args.model}")
    train(cfg, expand_models(args.model), resume=args.resume)
    return 0


def cmd_evaluate(args) -> int:
    from speechdeepfake.pipeline import evaluate, expand_models

    cfg = _cfg(args)
    model = getattr(args, "model", "all")
    _start(cfg, f"evaluate-{model}")
    table = evaluate(cfg, expand_models(model))
    cols = ["model", "dataset", "n", "eer", "eer_ci_low", "eer_ci_high", "roc_auc", "balanced_accuracy", "f1", "threshold"]
    print(table[cols].to_string(index=False))
    if cfg.run_type != "full":
        print(f"\n[{cfg.run_type.upper()}] These are not final research results.")
    return 0


def cmd_plots(args) -> int:
    from speechdeepfake.evaluation.figures import generate_figures

    cfg = _cfg(args)
    _start(cfg, "plots")
    made, skipped = generate_figures(cfg)
    print(f"{len(made)} figure(s) written to {artifacts.figures_dir(cfg)}")
    for k, v in skipped.items():
        print(f"  skipped {k}: {v}")
    return 0


def cmd_report(args) -> int:
    from speechdeepfake.evaluation.report import write_report

    cfg = _cfg(args)
    _start(cfg, "report")
    if not args.no_plots:
        from speechdeepfake.evaluation.figures import generate_figures

        generate_figures(cfg)
    path = write_report(cfg)
    print(f"Report written to {path}")
    return 0


def cmd_run_all(args) -> int:
    from speechdeepfake.pipeline import run_all

    cfg = _cfg(args)
    _start(cfg, "run-all")
    print(f"Report written to {run_all(cfg, resume=args.resume)}")
    return 0


# ------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="speechdeepfake", description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="Typical order: check-data -> prepare-data -> validate-data -> train -> evaluate-all -> report")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True, metavar="<command>")

    def add(name, func, help_text, config=True):
        sp = sub.add_parser(name, help=help_text, description=help_text)
        if config:
            sp.add_argument("--config", default=DEFAULT_CONFIG, help=f"YAML config (default: {DEFAULT_CONFIG})")
        sp.set_defaults(func=func)
        return sp

    sp = add("inspect", cmd_inspect, "Show Python, package versions, CPU/GPU and (with --config) dataset availability.", config=False)
    sp.add_argument("--config", default=None, help="optionally also report dataset availability for this config")
    add("check-data", cmd_check_data, "Report which datasets and manifests are present (no audio is read).")
    sp = add("download", cmd_download, "Explicitly download a dataset (ASVspoof 2019 LA ~7.1 GB, CD-ADD ~3.9 GB).")
    sp.add_argument("--dataset", required=True, choices=["asvspoof2019_la", "cdadd"])
    sp.add_argument("--keep-zip", action="store_true", help="keep LA.zip after extraction")
    sp.add_argument("--connections", type=int, default=12,
                    help="parallel HTTP connections for LA.zip (DataShare throttles each connection)")
    sp = add("download-subset", cmd_download_subset,
             "Fetch a small stratified subset (~100-300 MB) via HTTP range requests instead of the full datasets.")
    sp.add_argument("--dataset", choices=["asvspoof2019_la", "cdadd", "all"], default="all")
    sp.add_argument("--bonafide", type=int, default=300, help="ASVspoof bona fide utterances per split")
    sp.add_argument("--per-attack", type=int, default=50, help="ASVspoof utterances per attack in train/dev")
    sp.add_argument("--per-attack-eval", type=int, default=25, help="ASVspoof utterances per attack in eval")
    sp.add_argument("--cdadd-row-groups", type=int, default=1, help="100-row groups per CD-ADD shard (8 shards)")
    sp.add_argument("--connections", type=int, default=12, help="parallel HTTP connections")
    sp = add("prepare-data", cmd_prepare, "Parse dataset protocols/packages into normalised manifests (exports CD-ADD audio).")
    sp.add_argument("--corpus", nargs="*", help="limit to these corpora (asvspoof2019_la, cdadd, custom)")
    sp = add("validate-data", cmd_validate, "Decode every file, check labels/duplicates/leakage, write validated manifests.")
    sp.add_argument("--workers", type=int, default=8, help="parallel file readers")
    sp.add_argument("--no-hash", action="store_true", help="skip SHA-1 hashing (faster; disables duplicate-audio check)")
    sp.add_argument("--force", action="store_true", help="re-validate even if the manifest is unchanged")
    sp = add("smoke-test", cmd_smoke, "Run all three pipelines end to end on a synthetic fixture (software check only).")
    sp.add_argument("--out", default=None, help="output directory (default: results/smoke_test)")
    sp.add_argument("--wav2vec-model", default=None, help="override wav2vec.model_name, e.g. 'random-tiny' for offline use")
    sp = add("extract-features", cmd_extract, "Compute and cache MFCC and/or wav2vec 2.0 features for all partitions.")
    sp.add_argument("--model", choices=MODEL_CHOICES, default="all")
    sp = add("train", cmd_train, "Train one model or all three on the configured train/dev partitions.")
    sp.add_argument("--model", choices=MODEL_CHOICES, required=True)
    sp.add_argument("--resume", action="store_true", help="resume CNN training from last.pt")
    sp = add("evaluate", cmd_evaluate, "Evaluate one trained model on the configured test sets.")
    sp.add_argument("--model", choices=MODEL_CHOICES, required=True)
    add("evaluate-all", cmd_evaluate, "Evaluate all three trained models on the same test records and build comparison tables.")
    add("plots", cmd_plots, "Generate all figures supported by the saved outputs.")
    sp = add("report", cmd_report, "Generate figures and the Markdown report from saved results.")
    sp.add_argument("--no-plots", action="store_true", help="only rewrite the Markdown report")
    sp = add("run-all", cmd_run_all, "Train, evaluate, plot and report for all three models.")
    sp.add_argument("--resume", action="store_true", help="resume CNN training from last.pt")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except (ConfigError, FileNotFoundError, RuntimeError, ValueError) as e:
        if os.environ.get("SPEECHDEEPFAKE_DEBUG"):
            raise
        log.error("%s", e)
        print(f"\nERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
