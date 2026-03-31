#!/usr/bin/env python3
"""
Bulk-download 10 months of observational data (Feb–Nov 2025) from a remote
SSH host via rsync.

Data types
----------
  joyce_nc   – JOYCE parsivel NetCDF
               /data/obs/site/jue/parsivel/l1/{y}/{m}/{d}/sups_joy_dm00_l1_any_v00_{ymd}.nc
  tower_csv  – Tower parsivel CSV
               /data/obs/site/jue-tower1/parsivel_452070/l1/{y}/{m}/{d}/{ymd}_parsivel_tower.csv
  bonn_log   – BONN parsivel log
               /data/obs/site/jue/parsivel_304640/l1/{y}/{m}/{d}/parsivel_jue_{ymd}.log
  cloudnet   – Cloudnet categorize NetCDF
               /data/obs/site/jue/cloudnet/clu_processing/categorize/{y}/{ymd}_juelich_categorize.nc
  joyrad35   – Tower MIRA-35 radar (all *tower.znc files in daily directory)
               /data/obs/site/jue/joyrad35/{y}/{m}/{d}/*tower.znc

Usage
-----
    python3 download_data.py --host USER@HOST [options]

    # Download everything (Feb–Nov 2025):
    python3 download_data.py --host myuser@juelich.example.com

    # Only Joyce NC + CloudNet, dry-run first:
    python3 download_data.py --host myuser@juelich.example.com \\
        --types joyce_nc,cloudnet --dry-run

    # Different period:
    python3 download_data.py --host myuser@juelich.example.com \\
        --start 202503 --end 202508

Options
-------
    --host        SSH user@host or alias from ~/.ssh/config  [REQUIRED]
    --local-root  Local base directory  (default: ./data)
    --start       Start month  YYYYMM   (default: 202502)
    --end         End month    YYYYMM   (default: 202511)
    --types       Comma-separated list  (default: all five types)
    --dry-run     Pass --dry-run to rsync – nothing is written
    --ssh-port    SSH port if not 22
    -v/--verbose  Print per-file progress lines
    -j/--jobs     Parallel rsync processes per day  (default: 4)

Local directory layout mirrors the remote tree under --local-root:
    <local-root>/
      parsivel/joyce_nc/{year}/{month}/{day}/sups_joy_dm00_l1_any_v00_{ymd}.nc
      parsivel/tower_csv/{year}/{month}/{day}/{ymd}_parsivel_tower.csv
      parsivel/bonn_log/{year}/{month}/{day}/parsivel_jue_{ymd}.log
      cloudnet/{year}/{ymd}_juelich_categorize.nc
      joyrad35/{year}/{month}/{day}/*tower.znc
"""

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

# ── defaults ──────────────────────────────────────────────────────────────────

DEFAULT_LOCAL_ROOT = Path(__file__).parent / "data"
DEFAULT_START      = "202502"   # Feb 2025
DEFAULT_END        = "202511"   # Nov 2025
ALL_TYPES          = ["joyce_nc", "tower_csv", "bonn_log", "cloudnet", "joyrad35"]


# ── remote path builders ──────────────────────────────────────────────────────

def remote_joyce_nc(d: date) -> str:
    ymd = f"{d.year:04d}{d.month:02d}{d.day:02d}"
    return (
        f"/data/obs/site/jue/parsivel/l1"
        f"/{d.year:04d}/{d.month:02d}/{d.day:02d}"
        f"/sups_joy_dm00_l1_any_v00_{ymd}.nc"
    )

def remote_tower_csv(d: date) -> str:
    ymd = f"{d.year:04d}{d.month:02d}{d.day:02d}"
    return (
        f"/data/obs/site/jue-tower1/parsivel_452070/l1"
        f"/{d.year:04d}/{d.month:02d}/{d.day:02d}"
        f"/{ymd}_parsivel_tower.csv"
    )

def remote_bonn_log(d: date) -> str:
    ymd = f"{d.year:04d}{d.month:02d}{d.day:02d}"
    return (
        f"/data/obs/site/jue/parsivel_304640/l1"
        f"/{d.year:04d}/{d.month:02d}/{d.day:02d}"
        f"/parsivel_jue_{ymd}.log"
    )

def remote_cloudnet(d: date) -> str:
    ymd = f"{d.year:04d}{d.month:02d}{d.day:02d}"
    return (
        f"/data/obs/site/jue/cloudnet/clu_processing/categorize"
        f"/{d.year:04d}"
        f"/{ymd}_juelich_categorize.nc"
    )

def remote_joyrad35_dir(d: date) -> str:
    """Remote *directory* that holds all *tower.znc files for the day."""
    return (
        f"/data/obs/site/jue/joyrad35"
        f"/{d.year:04d}/{d.month:02d}/{d.day:02d}"
    )


# ── local mirror paths ────────────────────────────────────────────────────────

def local_file(dtype: str, d: date, root: Path) -> Path:
    """Return the local destination file (or directory for joyrad35)."""
    ymd = f"{d.year:04d}{d.month:02d}{d.day:02d}"
    y   = f"{d.year:04d}"
    m   = f"{d.month:02d}"
    dd  = f"{d.day:02d}"

    if dtype == "joyce_nc":
        return root / "parsivel" / "joyce_nc" / y / m / dd / f"sups_joy_dm00_l1_any_v00_{ymd}.nc"
    if dtype == "tower_csv":
        return root / "parsivel" / "tower_csv" / y / m / dd / f"{ymd}_parsivel_tower.csv"
    if dtype == "bonn_log":
        return root / "parsivel" / "bonn_log"  / y / m / dd / f"parsivel_jue_{ymd}.log"
    if dtype == "cloudnet":
        return root / "cloudnet" / y / f"{ymd}_juelich_categorize.nc"
    if dtype == "joyrad35":
        return root / "joyrad35" / y / m / dd   # directory
    raise ValueError(f"Unknown dtype: {dtype!r}")


# ── rsync wrappers ────────────────────────────────────────────────────────────

def _base_cmd(host: str, ssh_port: Optional[int], verbose: bool, dry_run: bool) -> list:
    cmd = ["rsync", "-az"]
    if ssh_port and ssh_port != 22:
        cmd += ["-e", f"ssh -p {ssh_port}"]
    if verbose:
        cmd.append("--progress")
    if dry_run:
        cmd.append("--dry-run")
    return cmd


def rsync_single(host: str, remote: str, local: Path,
                 ssh_port: Optional[int], dry_run: bool, verbose: bool) -> bool:
    """Download one file.  Returns True on success."""
    local.parent.mkdir(parents=True, exist_ok=True)
    cmd = _base_cmd(host, ssh_port, verbose, dry_run)
    cmd += [f"{host}:{remote}", str(local)]
    r = subprocess.run(cmd, capture_output=not verbose)
    return r.returncode == 0


def rsync_dir_pattern(host: str, remote_dir: str, local_dir: Path,
                      pattern: str,
                      ssh_port: Optional[int], dry_run: bool, verbose: bool) -> bool:
    """Download all files matching `pattern` from `remote_dir/` into `local_dir/`."""
    local_dir.mkdir(parents=True, exist_ok=True)
    cmd = _base_cmd(host, ssh_port, verbose, dry_run)
    # include matching files, exclude everything else
    cmd += ["--include", pattern, "--exclude", "*"]
    cmd += [f"{host}:{remote_dir}/", str(local_dir) + "/"]
    r = subprocess.run(cmd, capture_output=not verbose)
    return r.returncode == 0


# ── date helpers ──────────────────────────────────────────────────────────────

def parse_month(yyyymm: str) -> date:
    return date(int(yyyymm[:4]), int(yyyymm[4:6]), 1)


def last_day_of_month(d: date) -> date:
    if d.month == 12:
        return date(d.year + 1, 1, 1) - timedelta(days=1)
    return date(d.year, d.month + 1, 1) - timedelta(days=1)


def iter_dates(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


# ── download task (one dtype × one day) ──────────────────────────────────────

def download_task(dtype: str, d: date, host: str, root: Path,
                  ssh_port: Optional[int], dry_run: bool, verbose: bool) -> tuple:
    """Returns (dtype, d, status) where status is 'ok' | 'skip' | 'fail'."""
    local = local_file(dtype, d, root)
    ymd   = f"{d.year:04d}{d.month:02d}{d.day:02d}"

    if dtype == "joyrad35":
        ok = rsync_dir_pattern(
            host, remote_joyrad35_dir(d), local,
            "*tower.znc", ssh_port, dry_run, verbose,
        )
        return (dtype, d, "ok" if ok else "fail")

    # single-file types: skip if already present and non-empty
    if not dry_run and local.exists() and local.stat().st_size > 0:
        return (dtype, d, "skip")

    remote_fn = {
        "joyce_nc":  remote_joyce_nc,
        "tower_csv": remote_tower_csv,
        "bonn_log":  remote_bonn_log,
        "cloudnet":  remote_cloudnet,
    }[dtype]

    ok = rsync_single(host, remote_fn(d), local, ssh_port, dry_run, verbose)
    return (dtype, d, "ok" if ok else "fail")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Bulk-download 10-month observational dataset via SSH/rsync.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--host",       required=True,
                    help="SSH alias or user@host (must match ~/.ssh/config or known_hosts)")
    ap.add_argument("--local-root", default=str(DEFAULT_LOCAL_ROOT),
                    help=f"Local base directory  (default: {DEFAULT_LOCAL_ROOT})")
    ap.add_argument("--start",      default=DEFAULT_START,
                    help="Start month YYYYMM  (default: %(default)s)")
    ap.add_argument("--end",        default=DEFAULT_END,
                    help="End   month YYYYMM  (default: %(default)s)")
    ap.add_argument("--types",      default=",".join(ALL_TYPES),
                    help="Comma-separated data types  (default: all)")
    ap.add_argument("--ssh-port",   type=int, default=None,
                    help="SSH port if not 22")
    ap.add_argument("--dry-run",    action="store_true",
                    help="Pass --dry-run to rsync; no files are written")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="Print per-file rsync progress")
    ap.add_argument("-j", "--jobs", type=int, default=4,
                    help="Parallel rsync workers  (default: 4)")
    args = ap.parse_args()

    start = parse_month(args.start)
    end   = last_day_of_month(parse_month(args.end))
    types = [t.strip() for t in args.types.split(",")]
    root  = Path(args.local_root)

    unknown = [t for t in types if t not in ALL_TYPES]
    if unknown:
        sys.exit(f"ERROR: unknown type(s): {unknown}  (allowed: {ALL_TYPES})")

    print("=" * 62)
    print("  DATA DOWNLOAD  –  rsync over SSH")
    print("=" * 62)
    print(f"  Host       : {args.host}")
    print(f"  Local root : {root}")
    print(f"  Period     : {start}  →  {end}")
    print(f"  Types      : {', '.join(types)}")
    print(f"  Workers    : {args.jobs}")
    if args.dry_run:
        print("  ** DRY RUN – rsync --dry-run, nothing written **")
    print("=" * 62)
    print()

    # Build the full task list
    all_days  = list(iter_dates(start, end))
    n_days    = len(all_days)
    tasks     = [(dtype, d) for d in all_days for dtype in types]
    totals    = {t: {"ok": 0, "fail": 0, "skip": 0} for t in types}
    n_done    = 0

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(
                download_task,
                dtype, d, args.host, root, args.ssh_port, args.dry_run, args.verbose,
            ): (dtype, d)
            for dtype, d in tasks
        }

        for fut in as_completed(futures):
            dtype, d, status = fut.result()
            totals[dtype][status] += 1
            n_done += 1
            ymd = f"{d.year:04d}{d.month:02d}{d.day:02d}"

            # always print failures; print ok/skip only when verbose
            if status == "fail":
                print(f"  FAIL  {dtype:<12}  {ymd}", flush=True)
            elif args.verbose:
                print(f"  {status.upper():<4}  {dtype:<12}  {ymd}", flush=True)

            # simple inline progress (overwritten each line)
            pct = 100 * n_done / len(tasks)
            print(f"\r  Progress: {n_done}/{len(tasks)}  ({pct:.0f}%)   ", end="", flush=True)

    print()  # newline after progress line
    print(f"\n{'='*62}")
    print(f"  SUMMARY  ({n_days} days,  {start} – {end})")
    print(f"{'='*62}")
    print(f"  {'Type':<14}  {'OK':>6}  {'Skip':>6}  {'Fail':>6}")
    print(f"  {'-'*42}")
    for dtype in types:
        st = totals[dtype]
        print(f"  {dtype:<14}  {st['ok']:>6}  {st['skip']:>6}  {st['fail']:>6}")
    print(f"{'='*62}")
    print(f"\n  Files saved under: {root}")


if __name__ == "__main__":
    main()
