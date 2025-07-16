from pathlib import Path

def sorted_files(folder: Path):
    return sorted(
        folder.glob('*.jpg'),
        key=lambda f: int(f.name.split('_')[0])
    )