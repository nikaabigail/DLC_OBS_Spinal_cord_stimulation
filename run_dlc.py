from pathlib import Path
import sys
import traceback

import numpy as np
import pandas as pd
import yaml
import deeplabcut
import re

CONFIG_PATH = Path(r"C:\dlc\project\r_tm_side-og-2024-10-25\config.yaml")
PROJECT_PATH = CONFIG_PATH.parent
LABELED_DATA_DIR = PROJECT_PATH / "labeled-data"

# Текущий рабочий shuffle. После create_training_dataset может измениться.
SHUFFLE = 5

# Этот scorer нужен для metrics. При смене best snapshot можно обновить вручную.
SCORER = "DLC_Resnet50_r_tm_sideOct25shuffle5_snapshot_best-380"

# Здесь оставь только те видео, которые реально есть у тебя на диске
# и которые хочешь прогонять в analyze / labeled / metrics.
VIDEOS = [
    Path(r"C:\dlc\videos\1_MER2-230-168U3C(FDE22070174)_20240604_152156.avi"),
    #Path(r"C:\dlc\videos\4_MER2-230-168U3C(FDE22070174)_20240604_164749.avi"),
    #Path(r"C:\dlc\videos\5_MER2-230-168U3C(FDE22070174)_20240604_165132.avi"),
    #Path(r"C:\dlc\videos\6_MER2-230-168U3C(FDE22070174)_20240604_170308.avi"),
    #Path(r"C:\dlc\videos\7_MER2-230-168U3C(FDE22070174)_20240604_171951.avi"),
    #Path(r"C:\dlc\videos\8_MER2-230-168U3C(FDE22070174)_20240604_172323.avi"),
    #Path(r"C:\dlc\videos\10_MER2-230-168U3C(FDE22070174)_20240604_174954.avi"),
    #Path(r"C:\dlc\videos\13_MER2-230-168U3C(FDE22070174)_20240604_181719.avi"),
    #Path(r"C:\dlc\videos\14_MER2-230-168U3C(FDE22070174)_20240604_181947.avi"),
    #Path(r"C:\dlc\videos\16_MER2-230-168U3C(FDE22070174)_20240604_192145.avi"),
    #Path(r"C:\dlc\videos\17_MER2-230-168U3C(FDE22070174)_20240604_192344.avi"),
]


# -----------------------------
# Helpers
# -----------------------------

def get_train_dir() -> Path:
    return (
        PROJECT_PATH
        / "dlc-models-pytorch"
        / "iteration-0"
        / f"r_tm_sideOct25-trainset95shuffle{SHUFFLE}"
        / "train"
    )


def get_pytorch_config_path() -> Path:
    return get_train_dir() / "pytorch_config.yaml"


def get_snapshot_epochs() -> list[int]:
    train_dir = get_train_dir()
    epochs = []

    for p in train_dir.glob("snapshot-*.pt"):
        m = re.fullmatch(r"snapshot-(\d+)\.pt", p.name)
        if m:
            epochs.append(int(m.group(1)))

    return sorted(epochs)

def filter_mild() -> None:
    existing_videos = existing_video_strings()

    print("=== Applying mild median filter ===")
    deeplabcut.filterpredictions(
        str(CONFIG_PATH),
        existing_videos,
        shuffle=SHUFFLE,
        filtertype="median",
        windowlength=3,
        save_as_csv=True,
    )
    print("Mild filtering finished.")

def despike_mild(jump_threshold: float = 120.0) -> None:
    existing_videos = [v for v in VIDEOS if v.exists()]
    if not existing_videos:
        raise FileNotFoundError("No videos found for despike.")

    print("=== Applying mild despike ===")
    print(f"Using scorer: {SCORER}")
    print(f"jump_threshold={jump_threshold}")

    for video in existing_videos:
        h5_path = video.with_name(video.stem + SCORER + ".h5")
        if not h5_path.exists():
            print(f"[SKIP] H5 not found: {h5_path}")
            continue

        print(f"\nProcessing: {h5_path}")
        df = pd.read_hdf(h5_path)

        if not isinstance(df.columns, pd.MultiIndex):
            raise ValueError("Unexpected H5 format: columns are not MultiIndex.")

        scorers = list(dict.fromkeys(df.columns.get_level_values(0)))
        scorer = SCORER if SCORER in scorers else scorers[0]

        bodyparts = sorted(set(df.columns.get_level_values(1)))
        out = df.copy()

        print("\nPer-bodypart despike summary:")
        total_spikes = 0

        for bp in bodyparts:
            x_col = (scorer, bp, "x")
            y_col = (scorer, bp, "y")
            l_col = (scorer, bp, "likelihood")

            x = out[x_col].copy()
            y = out[y_col].copy()
            l = out[l_col].copy().astype(np.float32, copy=False)

            dx = x.diff()
            dy = y.diff()
            jump = np.sqrt(dx**2 + dy**2)

            spikes = jump > jump_threshold
            spike_count = int(spikes.sum())
            total_spikes += spike_count

            x.loc[spikes] = np.nan
            y.loc[spikes] = np.nan
            l.loc[spikes] = np.float32(0.0)

            out[x_col] = x
            out[y_col] = y
            out[l_col] = l

            pct = spike_count / len(x) * 100.0
            print(f"{bp:12s} | spikes: {spike_count:5d} | pct: {pct:6.2f}%")

        print(f"\nOverall spikes removed: {total_spikes}")

        out_h5 = video.with_name(video.stem + SCORER + "_despike.h5")
        out_csv = video.with_name(video.stem + SCORER + "_despike.csv")

        out.to_hdf(out_h5, key="df_with_missing", mode="w")
        out.to_csv(out_csv)

        print(f"Saved: {out_h5}")
        print(f"Saved: {out_csv}")

    print("Mild despike finished.")

def get_best_snapshot_epoch() -> int | None:
    train_dir = get_train_dir()
    best_epochs = []

    for p in train_dir.glob("snapshot-best-*.pt"):
        m = re.fullmatch(r"snapshot-best-(\d+)\.pt", p.name)
        if m:
            best_epochs.append(int(m.group(1)))

    if not best_epochs:
        return None
    return max(best_epochs)

def finetune_from_300() -> None:
    from deeplabcut.pose_estimation_pytorch.apis import training as dlc_pt_training

    train_dir = get_train_dir()
    snapshot_path = train_dir / "snapshot-330.pt"

    if not snapshot_path.exists():
        raise FileNotFoundError(f"Snapshot not found: {snapshot_path}")

    print("=== Fine-tune from snapshot-330.pt ===")
    print(f"Snapshot path: {snapshot_path}")

    dlc_pt_training.train_network(
        str(CONFIG_PATH),
        shuffle=SHUFFLE,
        snapshot_path=str(snapshot_path),
    )

    print("Fine-tune finished.")

def training_status() -> None:
    train_dir = get_train_dir()
    cfg_path = get_pytorch_config_path()

    print("=== Training status ===")
    print(f"Train dir: {train_dir}")
    print(f"Exists:    {train_dir.exists()}")

    if not cfg_path.exists():
        print(f"pytorch_config.yaml not found: {cfg_path}")
        return

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    epochs_target = cfg.get("train_settings", {}).get("epochs", None)
    save_every = cfg.get("runner", {}).get("snapshots", {}).get("save_epochs", None)

    snapshot_epochs = get_snapshot_epochs()
    best_epoch = get_best_snapshot_epoch()

    print(f"Target epochs: {epochs_target}")
    print(f"Snapshot every: {save_every}")
    print(f"Regular snapshots: {snapshot_epochs if snapshot_epochs else 'none'}")
    print(f"Best snapshot epoch: {best_epoch}")

    if snapshot_epochs:
        latest = max(snapshot_epochs)
        print(f"Latest saved epoch: {latest}")
        if epochs_target is not None:
            remaining = max(0, epochs_target - latest)
            print(f"Remaining epochs to target: {remaining}")
    else:
        print("No regular snapshots found. Training will start from scratch.")

def set_save_every_10() -> None:
    cfg_path = get_pytorch_config_path()
    if not cfg_path.exists():
        raise FileNotFoundError(f"pytorch_config.yaml not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg.setdefault("runner", {})
    cfg["runner"].setdefault("snapshots", {})
    cfg["runner"]["snapshots"]["save_epochs"] = 10

    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    print("=== Updated snapshot saving ===")
    print("save_epochs = 10")

def resume_training() -> None:
    from deeplabcut.pose_estimation_pytorch.apis import training as dlc_pt_training

    train_dir = get_train_dir()
    cfg_path = get_pytorch_config_path()

    if not cfg_path.exists():
        raise FileNotFoundError(f"pytorch_config.yaml not found: {cfg_path}")

    snapshot_epochs = get_snapshot_epochs()
    if not snapshot_epochs:
        raise FileNotFoundError("No regular snapshots found to resume from.")

    latest = max(snapshot_epochs)
    snapshot_path = train_dir / f"snapshot-{latest}.pt"

    print("=== True resume training (PyTorch) ===")
    print(f"Train dir:     {train_dir}")
    print(f"Resume epoch:  {latest}")
    print(f"Snapshot path: {snapshot_path}")

    if not snapshot_path.exists():
        raise FileNotFoundError(f"Snapshot not found: {snapshot_path}")

    dlc_pt_training.train_network(
        str(CONFIG_PATH),
        shuffle=SHUFFLE,
        snapshot_path=str(snapshot_path),
    )

    print("Resume training finished.")

def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(cfg: dict) -> None:
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def existing_video_strings() -> list[str]:
    existing = [str(v) for v in VIDEOS if v.exists()]
    if not existing:
        raise FileNotFoundError("No videos found in VIDEOS.")
    return existing


def existing_video_paths() -> list[Path]:
    existing = [v for v in VIDEOS if v.exists()]
    if not existing:
        raise FileNotFoundError("No videos found in VIDEOS.")
    return existing


def find_collected_data_pair(folder: Path) -> tuple[Path, Path] | None:
    csv_candidates = sorted(folder.glob("CollectedData_*.csv"))
    for csv_path in csv_candidates:
        suffix = csv_path.stem.removeprefix("CollectedData_")
        h5_path = folder / f"CollectedData_{suffix}.h5"
        if h5_path.exists():
            return csv_path, h5_path
    return None


def discover_labeled_video_sets() -> dict:
    """
    Собирает video_sets по всем папкам в labeled-data.
    Имя папки трактуется как имя видео без расширения.
    Путь к видео формируется как C:/dlc/videos/<folder>.avi
    Crop ставим единый, как в текущем проекте.
    """
    if not LABELED_DATA_DIR.exists():
        raise FileNotFoundError(f"labeled-data not found: {LABELED_DATA_DIR}")

    video_sets = {}

    for folder in sorted(LABELED_DATA_DIR.iterdir()):
        if not folder.is_dir():
            continue

        # Берем только реально размеченные папки
        pair = find_collected_data_pair(folder)
        if pair is None:
            continue

        video_name = folder.name + ".avi"
        video_path = f"C:/dlc/videos/{video_name}"

        video_sets[video_path] = {
            "crop": "0, 1920, 0, 220"
        }

    return video_sets


def sync_video_sets_from_labeled_data() -> None:
    """
    Переписывает config.yaml -> video_sets по всем папкам из labeled-data.
    """
    cfg = load_config()
    old_count = len(cfg.get("video_sets", {}))
    new_video_sets = discover_labeled_video_sets()
    new_count = len(new_video_sets)

    cfg["video_sets"] = new_video_sets
    save_config(cfg)

    print("=== Synced video_sets from labeled-data ===")
    print(f"Old video_sets count: {old_count}")
    print(f"New video_sets count: {new_count}")
    for vp in new_video_sets:
        print(f"  - {vp}")


def summarize_labeled_data() -> None:
    """
    Показывает, сколько реально размеченных кадров есть в labeled-data.
    """
    if not LABELED_DATA_DIR.exists():
        raise FileNotFoundError(f"labeled-data not found: {LABELED_DATA_DIR}")

    total_frames = 0
    total_folders = 0

    print("=== Labeled-data summary ===")
    for folder in sorted(LABELED_DATA_DIR.iterdir()):
        if not folder.is_dir():
            continue

        pair = find_collected_data_pair(folder)
        if pair is None:
            continue

        try:
            csv_path, _ = pair
            df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
            n = len(df)
            total_frames += n
            total_folders += 1
            print(f"{folder.name}: {n}")
        except Exception as e:
            print(f"{folder.name}: ERROR -> {e}")

    print(f"\nTOTAL folders: {total_folders}")
    print(f"TOTAL labeled frames: {total_frames}")


def _get_series(df: pd.DataFrame, scorer: str, bodypart: str, coord: str) -> pd.Series:
    series_or_df = df.xs((scorer, bodypart, coord), axis=1)
    if isinstance(series_or_df, pd.Series):
        return series_or_df
    return series_or_df.iloc[:, 0]


# -----------------------------
# Main actions
# -----------------------------
def check_paths() -> None:
    print("=== Checking paths ===")
    print(f"Config exists:       {CONFIG_PATH.exists()} -> {CONFIG_PATH}")
    print(f"Project path exists: {PROJECT_PATH.exists()} -> {PROJECT_PATH}")
    print(f"Labeled-data exists: {LABELED_DATA_DIR.exists()} -> {LABELED_DATA_DIR}")
    print(f"Shuffle:             {SHUFFLE}")
    print(f"Scorer:              {SCORER}")

    cfg = load_config()
    print(f"video_sets in config: {len(cfg.get('video_sets', {}))}")

    for video in VIDEOS:
        print(f"Video exists: {video.exists()} -> {video}")


def create_dataset() -> None:
    print("=== Syncing config from labeled-data ===")
    sync_video_sets_from_labeled_data()

    print("\n=== Creating training dataset (PyTorch) ===")
    deeplabcut.create_training_dataset(
        str(CONFIG_PATH),
        net_type="resnet_50",
        augmenter_type="imgaug",
        engine=deeplabcut.Engine.PYTORCH,
    )
    print("Training dataset created.")


def train() -> None:
    print("=== Training network (PyTorch) ===")
    deeplabcut.train_network(
        str(CONFIG_PATH),
        shuffle=SHUFFLE,
        engine=deeplabcut.Engine.PYTORCH,
    )
    print("Training finished.")


def evaluate() -> None:
    print("=== Evaluating network (PyTorch) ===")
    deeplabcut.evaluate_network(
        str(CONFIG_PATH),
        shuffle=SHUFFLE,
        engine=deeplabcut.Engine.PYTORCH,
    )
    print("Evaluation finished.")


def analyze() -> None:
    existing_videos = existing_video_strings()

    print("=== Analyzing videos (PyTorch) ===")
    deeplabcut.analyze_videos(
        str(CONFIG_PATH),
        existing_videos,
        shuffle=SHUFFLE,
        engine=deeplabcut.Engine.PYTORCH,
        save_as_csv=True,
    )
    print("Video analysis finished.")


def labeled_video() -> None:
    existing_videos = existing_video_strings()

    print("=== Creating labeled videos (PyTorch) ===")
    deeplabcut.create_labeled_video(
        str(CONFIG_PATH),
        existing_videos,
        shuffle=SHUFFLE,
        filtered=True,
        draw_skeleton=False,
        engine=deeplabcut.Engine.PYTORCH,
    )
    print("Labeled video creation finished.")

def interpolate_mild(pcutoff: float = 0.6, max_gap: int = 7) -> None:
    existing_videos = existing_video_paths()
    if not existing_videos:
        raise FileNotFoundError("No videos found for interpolation.")

    print("=== Applying mild interpolation ===")
    print(f"Using scorer: {SCORER}")
    print(f"pcutoff={pcutoff}, max_gap={max_gap}")

    for video in existing_videos:
        h5_path = video.with_name(video.stem + SCORER + ".h5")
        if not h5_path.exists():
            print(f"[SKIP] H5 not found: {h5_path}")
            continue

        print(f"\nProcessing: {h5_path}")
        df = pd.read_hdf(h5_path)

        if not isinstance(df.columns, pd.MultiIndex):
            raise ValueError("Unexpected H5 format: columns are not MultiIndex.")

        scorers = list(dict.fromkeys(df.columns.get_level_values(0)))
        scorer = SCORER if SCORER in scorers else scorers[0]

        bodyparts = sorted(set(df.columns.get_level_values(1)))

        out = df.copy()

        total_bad_frames = 0
        total_filled_frames = 0

        print("\nPer-bodypart interpolation summary:")

        for bp in bodyparts:
            x_col = (scorer, bp, "x")
            y_col = (scorer, bp, "y")
            l_col = (scorer, bp, "likelihood")

            x = out[x_col].copy()
            y = out[y_col].copy()
            l = out[l_col].copy()

            # чтобы не было warning по dtype
            l = l.astype(np.float32, copy=False)

            bad = l < pcutoff
            bad_count = int(bad.sum())

            x[bad] = np.nan
            y[bad] = np.nan

            x_interp = x.interpolate(method="linear", limit=max_gap, limit_direction="both")
            y_interp = y.interpolate(method="linear", limit=max_gap, limit_direction="both")

            # кадры, которые были плохими, но после интерполяции получили координаты
            filled = bad & x_interp.notna() & y_interp.notna()
            filled_count = int(filled.sum())

            x = x_interp
            y = y_interp

            # поднимаем confidence у реально интерполированных кадров
            l.loc[filled] = np.float32(pcutoff + 0.1)

            out[x_col] = x
            out[y_col] = y
            out[l_col] = l

            total_bad_frames += bad_count
            total_filled_frames += filled_count

            filled_pct = (filled_count / bad_count * 100.0) if bad_count > 0 else 0.0

            print(
                f"{bp:12s} | bad<{pcutoff:.2f}: {bad_count:5d} | "
                f"filled: {filled_count:5d} | filled_pct: {filled_pct:6.2f}%"
            )

        total_filled_pct = (
            total_filled_frames / total_bad_frames * 100.0 if total_bad_frames > 0 else 0.0
        )

        print("\nOverall interpolation summary:")
        print(f"  Total bad frames:    {total_bad_frames}")
        print(f"  Total filled frames: {total_filled_frames}")
        print(f"  Filled percent:      {total_filled_pct:.2f}%")

        out_h5 = video.with_name(video.stem + SCORER + "_interp.h5")
        out_csv = video.with_name(video.stem + SCORER + "_interp.csv")

        out.to_hdf(out_h5, key="df_with_missing", mode="w")
        out.to_csv(out_csv)

        print(f"Saved: {out_h5}")
        print(f"Saved: {out_csv}")

    print("Mild interpolation finished.")

def inference_metrics() -> None:
    existing_videos = existing_video_paths()

    print("=== Inference metrics ===")
    print(f"Using scorer: {SCORER}")

    for video in existing_videos:
        h5_path = video.with_name(video.stem + SCORER + ".h5")

        if not h5_path.exists():
            print(f"\n[SKIP] H5 not found for {video.name}")
            print(f"Expected: {h5_path}")
            continue

        print("\n" + "=" * 100)
        print(f"VIDEO: {video.name}")
        print(f"H5:    {h5_path}")

        df = pd.read_hdf(h5_path)

        if not isinstance(df.columns, pd.MultiIndex):
            raise ValueError("Unexpected H5 format: columns are not MultiIndex.")

        scorers = list(dict.fromkeys(df.columns.get_level_values(0)))
        if SCORER not in scorers:
            print(f"[WARN] Expected scorer '{SCORER}' not found.")
            print(f"Available scorers: {scorers}")
            scorer = scorers[0]
            print(f"Using scorer from file: {scorer}")
        else:
            scorer = SCORER

        bodyparts = sorted(set(df.columns.get_level_values(1)))
        summary_rows = []

        for bp in bodyparts:
            x = _get_series(df, scorer, bp, "x")
            y = _get_series(df, scorer, bp, "y")
            l = _get_series(df, scorer, bp, "likelihood")

            dx = x.diff()
            dy = y.diff()
            jump = np.sqrt(dx**2 + dy**2)

            # Можно считать прыжки на всех кадрах, чтобы видеть полный шум
            mean_jump = float(jump.mean())
            p95_jump = float(jump.quantile(0.95))
            max_jump = float(jump.max())

            row = {
                "bodypart": bp,
                "mean_lik": float(l.mean()),
                "median_lik": float(l.median()),
                "pct_below_0.6": float((l < 0.6).mean() * 100),
                "pct_below_0.9": float((l < 0.9).mean() * 100),
                "mean_jump": mean_jump,
                "p95_jump": p95_jump,
                "max_jump": max_jump,
            }
            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows).sort_values(
            by=["mean_lik", "pct_below_0.6", "p95_jump"],
            ascending=[False, True, True],
        )

        print("\nPer-bodypart summary:")
        for _, row in summary_df.iterrows():
            print(
                f"{row['bodypart']:12s} | "
                f"mean_lik={row['mean_lik']:.3f} | "
                f"median_lik={row['median_lik']:.3f} | "
                f"<0.6={row['pct_below_0.6']:6.2f}% | "
                f"<0.9={row['pct_below_0.9']:6.2f}% | "
                f"mean_jump={row['mean_jump']:7.2f} | "
                f"p95_jump={row['p95_jump']:7.2f} | "
                f"max_jump={row['max_jump']:7.2f}"
            )

        all_lik = []
        for bp in bodyparts:
            all_lik.append(_get_series(df, scorer, bp, "likelihood"))
        all_lik = pd.concat(all_lik, axis=1)

        all_lik_values = all_lik.to_numpy(dtype=float).ravel()
        all_lik_values = all_lik_values[~np.isnan(all_lik_values)]

        mean_lik_all = float(np.mean(all_lik_values))
        median_lik_all = float(np.median(all_lik_values))
        pct_below_06_all = float(np.mean(all_lik_values < 0.6) * 100)
        pct_below_09_all = float(np.mean(all_lik_values < 0.9) * 100)

        worst_bp_lowconf = summary_df.sort_values("pct_below_0.6", ascending=False).iloc[0]
        worst_bp_jump = summary_df.sort_values("p95_jump", ascending=False).iloc[0]

        print("\nOverall inference summary:")
        print(f"  Mean likelihood:      {mean_lik_all:.3f}")
        print(f"  Median likelihood:    {median_lik_all:.3f}")
        print(f"  All points < 0.6:     {pct_below_06_all:.2f}%")
        print(f"  All points < 0.9:     {pct_below_09_all:.2f}%")
        print(
            f"  Worst by low conf:    {worst_bp_lowconf['bodypart']} "
            f"({worst_bp_lowconf['pct_below_0.6']:.2f}% below 0.6)"
        )
        print(
            f"  Worst by jumps:       {worst_bp_jump['bodypart']} "
            f"(p95 jump {worst_bp_jump['p95_jump']:.2f})"
        )

        print("\nInterpretation:")
        print("  - mean/median likelihood ближе к 1.0 — лучше")
        print("  - высокий % < 0.6 означает нестабильную точку")
        print("  - большие jumps часто указывают на скачки/срывы трекинга")
        print("  - это не mAP/mAR/RMSE по GT, а proxy-метрики качества инференса")


def main() -> None:
    if len(sys.argv) < 2:
        print(
            "Usage:\n"
            "  python C:\\dlc\\run_dlc.py check\n"
            "  python C:\\dlc\\run_dlc.py summary\n"
            "  python C:\\dlc\\run_dlc.py sync\n"
            "  python C:\\dlc\\run_dlc.py dataset\n"
            "  python C:\\dlc\\run_dlc.py train\n"
            "  python C:\\dlc\\run_dlc.py evaluate\n"
            "  python C:\\dlc\\run_dlc.py analyze\n"
            "  python C:\\dlc\\run_dlc.py labeled\n"
            "  python C:\\dlc\\run_dlc.py metrics\n"
            "  python C:\\dlc\\run_dlc.py status\n"
            "  python C:\\dlc\\run_dlc.py set_save10\n"
            "  python C:\\dlc\\run_dlc.py resume\n"
            "  python C:\\dlc\\run_dlc.py interp_mild\n"
            "  python C:\\dlc\\run_dlc.py filter_mild\n"
            "  python C:\\dlc\\run_dlc.py finetune300\n"
            "  python C:\\dlc\\run_dlc.py despike_mild\n"
        )
        sys.exit(1)

    command = sys.argv[1].lower()

    try:
        if command == "check":
            check_paths()
        elif command == "summary":
            summarize_labeled_data()
        elif command == "sync":
            sync_video_sets_from_labeled_data()
        elif command == "dataset":
            create_dataset()
        elif command == "train":
            train()
        elif command == "evaluate":
            evaluate()
        elif command == "analyze":
            analyze()
        elif command == "labeled":
            labeled_video()
        elif command == "metrics":
            inference_metrics()
        elif command == "status":
            training_status()
        elif command == "set_save10":
            set_save_every_10()
        elif command == "resume":
            resume_training()
        elif command == "finetune300":
            finetune_from_300()
        elif command == "filter_mild":
            filter_mild()
        elif command == "interp_mild":
            interpolate_mild()
        elif command == "despike_mild":
            despike_mild()
        else:
            raise ValueError(f"Unknown command: {command}")
    except Exception:
        print("\n=== ERROR ===")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
