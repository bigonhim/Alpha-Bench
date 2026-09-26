"""Command-line entry point: serve the app, warm caches, build data."""

from __future__ import annotations

import argparse
import logging
import socket
import sys
import threading
import time
import webbrowser


def _free_port(host: str, port: int) -> int:
    for p in range(port, port + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, p))
                return p
            except OSError:
                continue
    return port


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    port = _free_port(args.host, args.port)
    url = f"http://{args.host}:{port}"
    print(f"\n  Alpha Foundry is starting at {url}\n")
    if args.open:
        threading.Thread(target=lambda: (time.sleep(2.5), webbrowser.open(url)), daemon=True).start()
    uvicorn.run("alphafoundry.api.app:app", host=args.host, port=port, log_level="warning", workers=1)


def cmd_warmup(args: argparse.Namespace) -> None:
    t0 = time.time()
    from .workspace import get_workspace

    ws = get_workspace()
    for e in ("rank(-ts_delta(close, 5))", "group_rank(ts_rank(ts_mean(returns, 20), 60), sector)",
              "ts_corr(rank(close), rank(volume), 10)", "ts_decay_linear(zscore(returns), 5)",
              "trade_when(volume > adv20, rank(-returns), -1)", "ts_regression(returns, close, 20, rettype=2)",
              "ts_median(returns, 10) + ts_std_dev(returns, 10) + ts_skewness(returns, 20)",
              "group_neutralize(quantile(ts_arg_max(close, 10)), bucket(rank(cap), range=\"0.1,1,0.1\"))",
              "hump(ts_backfill(ts_av_diff(close, 5), 20), hump=0.01) + kth_element(close, 5, k=2)",
              "days_from_last_change(close) + last_diff_value(close, 5) + ts_product(1 + returns, 5)",
              "ts_decay_exp_window(close, 10, factor=0.7) + ts_decay_linear(returns, 5, dense=true)"
              " + ts_target_tvr_decay(rank(returns), target_tvr=0.2) + ts_moment(returns, 10, k=3)",
              "group_median(returns, sector) + group_vector_neut(returns, close, sector)"
              " + group_backfill(returns, sector, 5) + regression_neut(returns, close) + ts_backfill(returns, 20, k=2)"):
        ws.simulate(e, None, extras=True)
    print(f"warm-up complete in {time.time() - t0:.1f}s (numba kernels compiled and cached)")


def cmd_build_data(args: argparse.Namespace) -> None:
    from .data.build import build_real

    def prog(phase, done, total, msg):
        print(f"\r[{phase}] {done}/{total} {msg[:40]:40s}", end="", flush=True)

    info = build_real(prog, max_tickers=args.max_tickers)
    print("\n", info)


def cmd_demo(args: argparse.Namespace) -> None:
    from . import config
    from .data.demo import build_demo

    build_demo(config.DEMO_PANELS_DIR, n_stocks=args.stocks, n_days=args.days)
    print("demo dataset written to", config.DEMO_PANELS_DIR)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(prog="alphafoundry", description="Offline WorldQuant BRAIN alpha studio")
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("serve", help="run the web app")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--open", action="store_true", help="open the browser")
    s.set_defaults(fn=cmd_serve)
    w = sub.add_parser("warmup", help="compile numba kernels")
    w.set_defaults(fn=cmd_warmup)
    b = sub.add_parser("build-data", help="download free data and build the real panel")
    b.add_argument("--max-tickers", type=int, default=None)
    b.set_defaults(fn=cmd_build_data)
    d = sub.add_parser("demo", help="rebuild the synthetic demo dataset")
    d.add_argument("--stocks", type=int, default=300)
    d.add_argument("--days", type=int, default=2000)
    d.set_defaults(fn=cmd_demo)
    args = ap.parse_args(argv)
    if not getattr(args, "fn", None):
        args = ap.parse_args(["serve", "--open"])
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
