from pathlib import Path
import csv

PROJECT_PATH = Path(r"C:\dlc\project\r_tm_side-og-2024-10-25")
MODELS_DIR = PROJECT_PATH / "dlc-models-pytorch" / "iteration-0"


def count_csv_rows(csv_path: Path) -> int:
    if not csv_path.exists():
        return 0
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return 0
    # минус заголовок, если он есть
    return max(0, len(rows) - 1)


def main():
    print("=" * 90)
    print("DLC SHUFFLE CHECK")
    print("=" * 90)
    print(f"Models dir: {MODELS_DIR}")
    print()

    if not MODELS_DIR.exists():
        print("ERROR: models directory does not exist.")
        return

    shuffle_dirs = sorted([p for p in MODELS_DIR.iterdir() if p.is_dir()])

    if not shuffle_dirs:
        print("No shuffle directories found.")
        return

    for shuffle_dir in shuffle_dirs:
        train_dir = shuffle_dir / "train"
        test_dir = shuffle_dir / "test"

        pytorch_cfg = train_dir / "pytorch_config.yaml"
        pose_cfg = train_dir / "pose_cfg.yaml"
        learning_stats = train_dir / "learning_stats.csv"

        snapshots = sorted(train_dir.glob("snapshot*.pt"))
        best_snapshots = [p for p in snapshots if "best" in p.name]

        print("-" * 90)
        print(f"SHUFFLE DIR: {shuffle_dir.name}")
        print(f"Path: {shuffle_dir}")
        print(f"Train dir exists: {train_dir.exists()}")
        print(f"Test dir exists:  {test_dir.exists()}")
        print(f"pytorch_config.yaml: {pytorch_cfg.exists()}")
        print(f"pose_cfg.yaml:       {pose_cfg.exists()}")
        print(f"learning_stats.csv:  {learning_stats.exists()}")

        if learning_stats.exists():
            try:
                nrows = count_csv_rows(learning_stats)
                print(f"learning_stats rows: {nrows}")
            except Exception as e:
                print(f"learning_stats rows: ERROR -> {e}")

        print(f"Total snapshot*.pt:  {len(snapshots)}")
        if snapshots:
            for s in snapshots[:10]:
                print(f"  - {s.name}")
            if len(snapshots) > 10:
                print(f"  ... and {len(snapshots) - 10} more")

        if best_snapshots:
            print("Best snapshots:")
            for s in best_snapshots:
                print(f"  * {s.name}")

    print("-" * 90)
    print("TIP:")
    print("Ищи тот shuffle, где есть train/pytorch_config.yaml и актуальные snapshot*.pt.")
    print("Именно его потом надо указывать в train/analyze/labeled, если используешь shuffle вручную.")


if __name__ == "__main__":
    main()