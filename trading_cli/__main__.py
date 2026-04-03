"""Entry point — run with `trading-cli` or `uv run trading-cli`."""

import logging
import signal
import sys
from pathlib import Path


def main() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(
                Path("~/.config/trading-cli/app.log").expanduser(),
                mode="a",
                encoding="utf-8",
            )
        ],
    )
    from trading_cli.app import TradingApp

    app = TradingApp()

    # Handle SIGINT (Ctrl+C) for clean shutdown
    def handle_sigint(signum, frame):
        logging.getLogger(__name__).info("Received SIGINT, shutting down...")
        app.exit()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigint)

    try:
        app.run()
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("KeyboardInterrupt, exiting...")
        sys.exit(0)
    finally:
        # Ensure clean shutdown
        logging.getLogger(__name__).info("Trading CLI shutdown complete")
        sys.exit(0)


if __name__ == "__main__":
    main()
