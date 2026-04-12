from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Live and in-play MLB pitcher strikeout pipeline scaffold.")
    parser.add_argument("--date", required=True)
    parser.add_argument("--data-dir", default="data_rebuild")
    parser.add_argument("--system-path", default="data_rebuild/pitcher_k_system.joblib")
    parser.add_argument("--out-dir", default="outputs/live_board")
    args = parser.parse_args()

    raise SystemExit(
        "Live in-play pipeline scaffold created. Next step is to implement live state building, live quote joins, and conditional PMF pricing."
    )


if __name__ == "__main__":
    main()
