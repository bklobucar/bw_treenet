# BWTreeNet: historical urban canopy mapping

Maps urban tree canopy change from the 1960s to the present in Malmö, Gothenburg and Stockholm. It runs deep learning segmentation (an adapted BWTreeNet, Gui et al. 2025) on archival panchromatic aerial imagery. Everything is in EPSG:3006 (SWEREF99 TM) at 0.5 m.

- Formas-funded to the end of 2027.
- NAISS allocation `naiss2026-4-1108`, ends 2027-07-01.
- Repo: github.com/bklobucar/bw_treenet. Upstream: GrayVictoria/BWTreeNet, cloned at `~/BWTreeNet`.

## Where things run

| Machine | Role |
|---|---|
| This PC (EndeavourOS, RTX 4070 Ti Super) | Local repo at `~/bw_treenet`; edit code here |
| Arrhenius HPC | Data, inference and analysis. Project dir: `/nobackup/proj/disk/naiss2026-4-1108/personal/bklobucar/bw_treenet` |

### Reaching the cluster

- **Do not run commands on the cluster via `ssh arr` until NAISS has authorised AI-agent access for `naiss2026-4-1108`.** Until then, write the cluster commands as code blocks for Blaž to run himself. This overrides the points below.
- Use `ssh arr '<cmd>'`. The connection is shared (ControlMaster), and Blaž logs in himself each day with a password plus a code.
- If `ssh arr` asks for a password, stop and ask him to log in. Never try to authenticate.
- Move code with git: commit and push here, then `ssh arr 'cd <project dir> && git pull'`. Don't scp scripts.

### Cluster architecture (important)

- **The login node is x86 (`el9_epyc9005`). The conda envs are GH200 (ARM) only**, so they don't work on the login node.
  - On the login node, use only plain shell, git, `squeue`, `sbatch`, `cat`, `ls`, `sed` and `python3` (system Python without GDAL).
  - Anything with GDAL, PyTorch or the envs must run on a GPU node.
- Batch jobs:
  - Header: `#SBATCH -A naiss2026-4-1108-gpu`, `-p gpu`, `--gpus 1`.
  - Logs go to `results/logs/%x_%j.out`.
  - Env setup inside a job:
    ```
    module load GPU/Miniforge/26.3.2-2-eb > /dev/null 2>&1
    eval "$(conda shell.bash hook)"
    conda activate gis        # GDAL CLI + Python bindings, no PyTorch
    # or: source ~/bw_setup.sh  (bwtreenet env: PyTorch)
    ```
- `interactive` sessions are for Blaž to run by hand. Use `sbatch` jobs instead; a short `srun` with the same account flags is fine for quick checks.
- **Lantmäteriet/Geotorget downloads** (`36_fetch_build_era.py`) need his Geotorget login. He runs these himself; never ask for or store the password.

## Data conventions

- Cities: `malmo`, `gtb` = Gothenburg (~11°E), `sth` = Stockholm (~17°E). The gtb and sth folders were swapped once, so check coordinates if in doubt.
- Imagery: `data/raw/<city>/<era>_OF_gray_<suffix>/<sheet>_<year>.tif` (single band, 0.5 m).
- Inference outputs: `results/inference_v19/<city>_<era>/<sheet>_<year>_canopy.tif` and `_prob.tif`.
  - **The canopy maps carry `NoData Value=0`.** Always warp them with `srcNodata="None"`, or every non-canopy pixel is dropped. This caused a bug in `38`.
  - `prob = -1` where production masks: outside the 2023 tätort, and water. Measure only where `prob >= 0`.
- Eras:
  - `1960`, `1970` (film); **`1990r05`** (1990s film at 0.5 m; use this one); `1990` (1 m, superseded, under-calls by about 4 pp).
  - `2005` (Gothenburg uses the 2008 flight); `2010`; `2020`.
  - Test eras: `2005cmp` (Gothenburg 2005 leaf-off), `2005deg` (degraded ~2005), `pan1999` and `pan2000` (late film; Gothenburg's is 2001).
- RGB to gray uses a plain band mean, not BT.601. This was fixed in `19`.
- Production model: v19.

## Key scripts

| Script | Purpose |
|---|---|
| `19_production_inference.py --city --era` | inference |
| `20_aggregate_canopy.py` | city totals |
| `32_leafon_scan.py`, `36_fetch_build_era.py` | find and build post-2000 eras from the Lantmäteriet STAC |
| `38_eval_1990_labels.py` | 1990s label clips (in training, so not independent) |
| `39_common_footprint.py --extra ... --own-footprint ...` | series on a fixed footprint |
| `40_film_digital_windows.py` | film → digital step by woodedness, plus image strips |

## Current state (Oct 2026)

Notes are in `notes/paper2_notes_2026-10.md`.

- **The film → digital step (1990s/2001 → ~2005) in Gothenburg and Stockholm is a source effect, not real gain.** It is concentrated in closed stands: film-era predictions under-call dense stands.
- Open question: is it model failure on film or leaf-off film flights? To be decided from the image strips (`results/film_windows_*.png`).
- Planned next:
  - stratified point-based accuracy assessment per era
  - a ranked labelling queue for a research assistant
  - DeSO 2025 aggregation (`41`, canopy % over measured area plus coverage %)

## Rules

- `/nobackup` has **no backup**. Never delete or overwrite data or results on the cluster without asking.
- Don't run heavy work on the login node.
- Write terminal steps for Blaž as copy-pasteable code blocks.
- Verify claims against command output before stating them.
