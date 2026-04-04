"""Entry point — run with `trading-cli` or `uv run trading-cli`."""

import os
import sys

# CRITICAL: Lower file descriptor limit EARLY to avoid subprocess fds_to_keep error
# Must be set BEFORE importing transformers or any library that uses subprocess
try:
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    # Lower to 1024 to avoid fds_to_keep errors while still allowing normal operation
    target_limit = 1024
    if soft > target_limit:
        new_soft = min(target_limit, hard)
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
        print(f"Adjusted FD limit: {soft} -> {new_soft}", file=sys.stderr)
except Exception as e:
    print(f"Could not adjust FD limit: {e}", file=sys.stderr)

# CRITICAL: Disable all parallelism before importing transformers
# These MUST be set before any transformers/tokenizers import
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['TQDM_DISABLE'] = '1'

import logging
import signal
import threading
import time
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

    # Track if we've already started shutdown
    _shutdown_started = False
    _shutdown_lock = threading.Lock()

    def force_kill():
        """Force kill after timeout."""
        time.sleep(3)
        print("\n⚠️  Force-killing process (shutdown timeout exceeded)", file=sys.stderr)
        os._exit(1)  # Force kill, bypassing all handlers

    def handle_sigint(signum, frame):
        """Handle SIGINT (Ctrl+C) with force-kill fallback."""
        nonlocal _shutdown_started

        with _shutdown_lock:
            if _shutdown_started:
                # Already shutting down, skip force kill
                print("\n⚠️  Already shutting down, waiting...", file=sys.stderr)
                return

            _shutdown_started = True
            logger = logging.getLogger(__name__)
            logger.info("Received SIGINT (Ctrl+C), initiating shutdown...")
            print("\n🛑 Shutting down... (press Ctrl+C again to force-kill)", file=sys.stderr)

            # Start force-kill timer
            killer_thread = threading.Thread(target=force_kill, daemon=True)
            killer_thread.start()

            # Try clean shutdown
            try:
                app.exit()
            except Exception as e:
                logger.error(f"Error during exit: {e}")
            finally:
                # Give it a moment then exit
                time.sleep(0.5)
                sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigint)

    try:
        app.run()
    except KeyboardInterrupt:
        # This handles the case where Textual catches it first
        logging.getLogger(__name__).info("KeyboardInterrupt caught at top level, exiting...")
        sys.exit(0)
    finally:
        # Ensure clean shutdown
        logging.getLogger(__name__).info("Trading CLI shutdown complete")
        sys.exit(0)


if __name__ == "__main__":
    main()
