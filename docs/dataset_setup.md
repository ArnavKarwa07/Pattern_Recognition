# Dataset setup

Nothing in this project downloads data implicitly — not the tests, not the
smoke test, not training. Datasets are fetched only by the explicit commands
below. All paths are relative to `paths.data_dir` (default `data/`).

## Storage

| Item | Approx. size | Location (default) |
|---|---|---|
| ASVspoof 2019 `LA.zip` (deleted after extraction unless `--keep-zip`) | 7.1 GB | `data/raw/ASVspoof2019/LA.zip` |
| ASVspoof 2019 LA extracted | ~7.6 GB | `data/raw/ASVspoof2019/LA/` |
| CD-ADD parquet shards | 3.9 GB | `data/raw/cdadd/data/` |
| CD-ADD exported FLAC files | ~3.9 GB | `data/raw/cdadd/audio/` |
| MFCC feature cache (all partitions) | < 0.1 GB | `data/processed/mfcc_stats/` |
| wav2vec 2.0 embedding cache (768 float32 per recording) | ~0.5 GB | `data/processed/wav2vec2_meanpool/` |
| `facebook/wav2vec2-base` checkpoint | 0.38 GB | Hugging Face cache (`~/.cache/huggingface`) |

Plan on **~25 GB free** for everything (less once `LA.zip` and the CD-ADD
parquet shards are deleted after preparation). After `prepare-data` has
exported CD-ADD you may delete `data/raw/cdadd/data/*.parquet` manually to
reclaim 3.9 GB, but keep it if you might need to rebuild the manifest.

> **OneDrive / cloud-synced folders.** This repository lives under a OneDrive
> folder on the development machine. Syncing 20+ GB of audio and caches is
> slow and wastes cloud quota. Put the data elsewhere, e.g.
>
> ```powershell
> $env:SPEECHDEEPFAKE_DATA_DIR = "D:\speechdeepfake-data"   # PowerShell, current session
> ```
> ```bash
> export SPEECHDEEPFAKE_DATA_DIR=/d/speechdeepfake-data     # bash
> ```
>
> or set `paths.data_dir` to an absolute path in the config. Manifests store
> audio paths relative to the data directory, so the folder can be moved.

## Small subsets instead (demos, pilots)

```bash
speechdeepfake download-subset --dataset all --config configs/demo.yaml \
    --bonafide 300 --per-attack 50 --per-attack-eval 25 --cdadd-row-groups 1
```

- ASVspoof: reads the remote `LA.zip` with HTTP range requests (central
  directory ≈ 12 MB, the three protocol files, then only the chosen FLAC
  members, each CRC-checked). Per split it selects `--bonafide` genuine
  utterances and `--per-attack` (eval: `--per-attack-eval`) utterances of every
  attack, with the config seed. The filtered official protocol lines are written
  to `ASVspoof2019_LA_cm_protocols/`; the full originals go to `.../full/`.
- CD-ADD: reads `--cdadd-row-groups` random 100-row groups from each of the 8
  shards (both LibriTTS and TED, all generators) into
  `data/raw/cdadd/data/test-subset-00000.parquet`.
- `SUBSET.json` in each dataset folder records the seed and the exact selection.
  Use `configs/demo.yaml` (run type `pilot`) so that every figure and the
  report are labelled as subset results.

DataShare throttles each connection to roughly 25 KB/s (observed Oct 2026),
which is why both the subset fetcher and `download --dataset asvspoof2019_la`
use parallel connections (`--connections`, default 12).

## A. ASVspoof 2019 Logical Access

- Source: Edinburgh DataShare, <https://datashare.ed.ac.uk/handle/10283/3336>
  (doi:10.7488/ds/2555). Licence: Open Data Commons Attribution (ODC-By).
  Challenge page: <https://www.asvspoof.org/index2019.html>.
- The CM protocol files *with keys* for train, dev **and eval** are part of the
  database (`LA/ASVspoof2019_LA_cm_protocols/`). If your copy has an eval
  protocol without `bonafide`/`spoof` keys, the eval split is skipped with a
  note — labels are never invented.

Download and extract (≈7.1 GB download):

```bash
speechdeepfake download --dataset asvspoof2019_la --config configs/default.yaml
```

Manual alternative: download `LA.zip` from the DataShare page and extract it
so that `data/raw/ASVspoof2019/LA/ASVspoof2019_LA_cm_protocols/` exists.

```powershell
# PowerShell manual alternative
New-Item -ItemType Directory -Force data\raw\ASVspoof2019
Invoke-WebRequest https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip -OutFile data\raw\ASVspoof2019\LA.zip
Expand-Archive data\raw\ASVspoof2019\LA.zip -DestinationPath data\raw\ASVspoof2019
```

The parser identifies protocol fields by their content (utterance id
`LA_[TDE]_…`, speaker `LA_####`, attack `A##` or `-`, key `bonafide|spoof`)
rather than assuming column positions, and rejects inconsistent lines (e.g. a
spoof without an attack id).

## B. CD-ADD (external test set)

- Paper: Li et al., EMNLP 2024, <https://aclanthology.org/2024.emnlp-main.286/>;
  original repository <https://github.com/leolya/CD-ADD>.
- Package used: <https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/CD-ADD>
  (CC BY 4.0). Verified schema (Hub API, Oct 2026): a single `test` split in 8
  parquet shards with columns `path`, `audio` (16 kHz mono FLAC bytes),
  `label` (`ClassLabel[bonafide, spoof]`) and `notes` (JSON with
  `utterance_id`, `source` ∈ {libritts, ted}, `system` ∈ {real, openvoice,
  valle, whisperSpeech, yourTTS, seamless}, `speaker_id`, `transcript`), plus
  `data/labels.parquet` (utterance_id → label).
- It contains LibriTTS **test-clean** and TED-sourced utterances only, and no
  train/dev split, so it is used purely as an external test set.

```bash
speechdeepfake download --dataset cdadd --config configs/default.yaml   # ~3.9 GB
```

`prepare-data` then streams the shards (never loading all audio into RAM),
writes each clip to `data/raw/cdadd/audio/<path>.flac`, checks that
`label` agrees with `notes.system` (`real` ⇔ bona fide) and with
`labels.parquet`, and records `source_domain`, `generator_id`, `speaker_id`
and `group_id` (the source-utterance directory shared by a genuine clip and
its spoofs, used for leakage checks and cluster bootstrap CIs).

## C. Your own data (custom CSV)

Set `data.custom_manifest: path/to/manifest.csv`. Required columns:
`audio_path`, `label`, `split`; optional: `record_id`, `corpus`,
`utterance_id`, `speaker_id`, `source_domain`, `generator_id`, `group_id`.
Relative audio paths are resolved against the CSV's folder. Accepted labels:
`bonafide/bona-fide/genuine/real/human/0` and
`spoof/fake/synthetic/generated/deepfake/ai-generated/tts/1`
(case-insensitive); anything else is rejected.

## D. Prepare and validate

```bash
speechdeepfake check-data    --config configs/default.yaml   # what is present
speechdeepfake prepare-data  --config configs/default.yaml   # manifests
speechdeepfake validate-data --config configs/default.yaml   # decode + checks
```

Validation decodes every file and writes
`data/manifests/validated/<corpus>.csv` (status, duration, sample rate, peak,
silence flag, SHA-1), `<corpus>_summary.json` (class counts, duration
statistics, metadata coverage, failures, integrity findings) and
`<corpus>_failures.csv`. Training refuses to start on an unvalidated or stale
manifest (`data.require_validation: true`). Statuses: `ok`, `missing`,
`corrupt`, `empty`, `non_finite`, `too_short`; silent clips are flagged but
kept.

The integrity checks report duplicate record ids, conflicting labels, and —
per corpus — utterance ids, source groups, audio paths and byte-identical
files (SHA-1) shared across splits (errors), and speakers shared across
splits (warning). When an experiment is assembled, the same checks run across
all its partitions (train, dev and every test set, across corpora).
