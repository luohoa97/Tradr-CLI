"""HMR dev runner — watches for .py changes and auto-restarts the trading CLI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from watchfiles import watch


def main() -> None:
    project_root = Path(__file__).parent.resolve()
    target_dir = project_root / "trading_cli"

    print(f"🔄 Watching {target_dir} for changes (Ctrl+C to stop)\n")

    for changes in watch(target_dir, watch_filter=None):
        for change_type, path in changes:
            if not path.endswith((".py", ".pyc")):
                continue
            action = "Added" if change_type.name == "added" else \
                     "Modified" if change_type.name == "modified" else "Deleted"
            rel = Path(path).relative_to(project_root)
            print(f"\n📝 {action}: {rel}")
            print("⟳  Restarting...\n")
            break  # restart on first matching change
        subprocess.run([sys.executable, "-m", "trading_cli"])


if __name__ == "__main__":
    main()
