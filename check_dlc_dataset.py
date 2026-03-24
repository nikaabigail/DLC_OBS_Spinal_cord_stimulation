from pathlib import Path
import pandas as pd
import yaml

PROJECT_PATH = Path(r"C:\dlc\project\r_tm_side-og-2024-10-25")
CONFIG_PATH = PROJECT_PATH / "config.yaml"
LABELED_DATA_DIR = PROJECT_PATH / "labeled-data"


def read_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def count_labeled_frames_in_csv(csv_path: Path) -> int:
    """
    Для DLC CSV обычно multi-header (3 строки заголовка).
    Индексом является путь к изображению.
    """
    df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
    return len(df)


def find_collected_data_pair(folder: Path, scorer: str | None) -> tuple[Path, Path] | None:
    if scorer:
        preferred_csv = folder / f"CollectedData_{scorer}.csv"
        preferred_h5 = folder / f"CollectedData_{scorer}.h5"
        if preferred_csv.exists() and preferred_h5.exists():
            return preferred_csv, preferred_h5

    for csv_path in sorted(folder.glob("CollectedData_*.csv")):
        suffix = csv_path.stem.removeprefix("CollectedData_")
        h5_path = folder / f"CollectedData_{suffix}.h5"
        if h5_path.exists():
            return csv_path, h5_path
    return None


def main():
    print("=" * 80)
    print("DLC DATASET CHECK")
    print("=" * 80)
    print(f"Project path: {PROJECT_PATH}")
    print(f"Config path:  {CONFIG_PATH}")
    print(f"Labeled dir:  {LABELED_DATA_DIR}")
    print()

    if not CONFIG_PATH.exists():
        print("ERROR: config.yaml not found")
        return

    if not LABELED_DATA_DIR.exists():
        print("ERROR: labeled-data directory not found")
        return

    cfg = read_config()

    scorer = cfg.get("scorer", None)
    video_sets = cfg.get("video_sets", {})

    print(f"Scorer in config: {scorer}")
    print(f"Videos in config.yaml: {len(video_sets)}")
    for video_path in video_sets:
        print(f"  - {video_path}")
    print()

    total_frames = 0
    folders_with_csv = 0
    folders_without_csv = 0

    print("=" * 80)
    print("LABELED-DATA FOLDERS")
    print("=" * 80)

    for folder in sorted(LABELED_DATA_DIR.iterdir()):
        if not folder.is_dir():
            continue

        pair = find_collected_data_pair(folder, scorer)

        png_count = len(list(folder.glob("*.png")))

        if pair is not None:
            try:
                csv_path, h5_path = pair
                n_frames = count_labeled_frames_in_csv(csv_path)
                total_frames += n_frames
                folders_with_csv += 1
                print(
                    f"[OK] {folder.name}\n"
                    f"     labeled frames in CSV: {n_frames}\n"
                    f"     png files:             {png_count}\n"
                    f"     csv:                   {csv_path.name}\n"
                    f"     h5 exists:             {h5_path.exists()}\n"
                )
            except Exception as e:
                print(
                    f"[ERROR] {folder.name}\n"
                    f"        could not read {csv_path.name}: {e}\n"
                )
        else:
            folders_without_csv += 1
            expected = f"CollectedData_{scorer}.csv/.h5" if scorer else "CollectedData_<scorer>.csv/.h5"
            print(
                f"[NO CSV] {folder.name}\n"
                f"         png files: {png_count}\n"
                f"         expected pair: {expected}\n"
            )

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Folders with CollectedData CSV: {folders_with_csv}")
    print(f"Folders without CSV:           {folders_without_csv}")
    print(f"TOTAL labeled frames in CSVs:  {total_frames}")
    print()

    expected_train = int(total_frames * 0.95)
    expected_test = total_frames - expected_train
    print("If TrainingFraction = 0.95, expected split would be approximately:")
    print(f"  train: {expected_train}")
    print(f"  test:  {expected_test}")
    print()

    print("IMPORTANT:")
    print("- Если TOTAL близок к 119, значит dataset собрался примерно весь.")
    print("- Если TOTAL сильно больше 119, значит часть разметки не вошла в обучение.")
    print("- Если папок много, но CSV мало, значит часть папок фактически не размечена для DLC.")


if __name__ == "__main__":
    main()
