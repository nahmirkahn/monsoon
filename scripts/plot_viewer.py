#!/usr/bin/env python3
import argparse
from pathlib import Path

from PIL import Image


def list_plots(plots_dir: Path) -> list[Path]:
    if not plots_dir.exists():
        return []
    return sorted(p for p in plots_dir.glob('*.png'))


def show_plots(plots: list[Path]) -> None:
    # Lazy import to avoid dependency errors if PIL not installed
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print('matplotlib is required to show plots:', e)
        return

    for p in plots:
        try:
            img = Image.open(p)
            plt.figure(figsize=(7, 5))
            plt.imshow(img)
            plt.axis('off')
            plt.title(p.name)
            plt.show()
        except Exception as e:
            print(f'Failed to display {p.name}: {e}')


def main() -> None:
    parser = argparse.ArgumentParser(description='View saved plots from final_solution artifacts directory')
    parser.add_argument('--root', default='/home/miso/Documents/WINDOWS/monsoon/final_solution', help='final_solution root (absolute path)')
    parser.add_argument('--list', action='store_true', help='Only list plot files')
    parser.add_argument('--show', action='store_true', help='Display plots using matplotlib')
    args = parser.parse_args()

    root = Path(args.root)
    plots_dir = root / 'artifacts' / 'plots'
    print('Plots directory:', plots_dir)

    plots = list_plots(plots_dir)
    if not plots:
        print('No .png plots found.')
        return

    print('Found plots:')
    for p in plots:
        try:
            size_mb = p.stat().st_size / (1024 * 1024)
        except Exception:
            size_mb = 0.0
        print(f'- {p.name}\t{size_mb:.2f} MB')

    if args.show:
        show_plots(plots)


if __name__ == '__main__':
    main()


