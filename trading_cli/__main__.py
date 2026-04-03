"""Entry point — run with `trading-cli` or `uv run trading-cli`."""
 
import logging
import sys
 
 
def main() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(
                __import__("pathlib").Path("~/.config/trading-cli/app.log").expanduser(),
                mode="a",
                encoding="utf-8",
            )
        ],
    )
    from trading_cli.app import TradingApp
    app = TradingApp()
    app.run()
 
 
if __name__ == "__main__":
    main()
