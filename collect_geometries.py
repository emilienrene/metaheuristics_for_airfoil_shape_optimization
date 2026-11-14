#!/usr/bin/env python3
"""Collect best airfoil coordinates by seed and case (Python 3.9+, no dependencies).

Run from the clone containing studies/ and optimization_<index>/:
    python collect_geometries.py
    python collect_geometries.py --repo /path/to/clone --dry-run
    python collect_geometries.py --seeds 103409 293794 --case constrained

Outputs default to <repo>/geometries/<case>/<method>_<optimizer>/seed_<seed>/.
Existing identical files
are accepted; different files require --overwrite. All inputs are checked before
any copies are made. Missing runs/geometries are reported and skipped.
best_airfoil.dat must contain a path, not coordinates.
"""
import argparse
import csv
from pathlib import Path, PurePosixPath
import re
import shutil
import sys


def portable(value):
    return str(value).strip().strip('\"\'').replace('\\', '/')


def resolve_pointer(value, repo, run, recorded_run):
    """Prefer this clone; retain an existing absolute path as a final fallback."""
    value = portable(value)
    path = Path(value)
    parts = PurePosixPath(value).parts
    candidates = []
    # Absolute paths recorded before moving/cloning the repo can be rebased
    # using the optimization directory or the original repo root from opt_dir.
    for i, part in enumerate(parts):
        if re.fullmatch(r'optimization_\d+', part):
            candidates.append(repo.joinpath(*parts[i:]))
            break
    old_run = portable(recorded_run).rstrip('/')
    old_repo = old_run.rsplit('/', 1)[0] if '/' in old_run else ''
    if old_repo and value.startswith(old_repo + '/'):
        candidates.append(repo / value[len(old_repo) + 1:])
    if not path.is_absolute() and not re.match(r'^[A-Za-z]:/', value):
        candidates.extend((run / path, repo / path))
    candidates.append(path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise ValueError(f'Cannot resolve geometry path {value!r}; tried: ' +
                     ', '.join(str(p) for p in candidates))


def build_plan(args):
    repo = args.repo.resolve()
    studies = repo / 'studies'
    if not studies.is_dir():
        raise ValueError(f'Studies directory does not exist: {studies}')
    output = args.output_root.resolve() if args.output_root else repo
    requested = set(args.seeds or [])
    found = set()
    plan = []
    destinations = set()
    groups = 0
    missing = 0
    for study in sorted(studies.glob(args.study_glob)):
        match = re.search(r'_seed_(\d+)$', study.name)
        if not study.is_dir() or not match:
            continue
        seed = match.group(1)
        if requested and seed not in requested:
            continue
        cases = ('constrained', 'unconstrained') if args.case == 'auto' else (args.case,)
        for case in cases:
            index = study / case / 'study_index.csv'
            if not index.is_file():
                continue
            found.add(seed)
            with index.open(newline='', encoding='utf-8-sig') as handle:
                reader = csv.DictReader(handle)
                required = {'run_index', 'method', 'optimizer', 'opt_dir'}
                if not required.issubset(reader.fieldnames or []):
                    raise ValueError(f'{index}: missing columns {sorted(required - set(reader.fieldnames or []))}')
                rows = [row for row in reader if any(row.values())]
            if len(rows) != args.expected_count:
                print(f'WARNING: {index}: expected {args.expected_count} rows, found {len(rows)}; collecting available rows', file=sys.stderr)
            for row_number, row in enumerate(rows, start=2):
                try:
                    run_id = (row['run_index'] or '').strip()
                    if not run_id:
                        recorded = portable(row['opt_dir'] or '').rstrip('/')
                        match = re.search(r'(?:^|/)optimization_(\d+)$', recorded)
                        if match:
                            run_id = match.group(1)
                            print(f'WARNING: {index}: CSV row {row_number}: '
                                  f'empty run_index; using {run_id} from opt_dir', file=sys.stderr)
                    if not run_id.isdigit():
                        raise ValueError(
                            f'{index}: CSV row {row_number}, '
                            f"{row['method']}__{row['optimizer']}: "
                            f'invalid or missing run_index {run_id!r}; '
                            f"opt_dir={row['opt_dir']!r}, status={row.get('status', '')!r}. "
                            'No identified optimization run.')
                    # A populated run_index is authoritative, including numbering gaps.
                    run = repo / f'optimization_{run_id}'
                    pointer = run / 'best_airfoil.dat'
                    if not pointer.is_file():
                        raise ValueError(f'Missing best-airfoil pointer: {pointer}')
                    lines = [line.strip() for line in pointer.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
                    if len(lines) != 1:
                        raise ValueError(f'{pointer}: expected one nonempty line containing a geometry path')
                    source = resolve_pointer(lines[0], repo, run, row['opt_dir'])
                except (OSError, ValueError) as exc:
                    missing += 1
                    print(f"MISSING: seed={seed}, case={case}, "
                          f"method={row['method']}__{row['optimizer']}, "
                          f'CSV row={row_number}: {exc}', file=sys.stderr)
                    continue
                method, optimizer = row['method'].strip(), row['optimizer'].strip()
                if not all(re.fullmatch(r'[A-Za-z0-9_-]+', label) for label in (method, optimizer)):
                    raise ValueError(f'{index}: invalid method or optimizer name')
                target = output / 'geometries' / case / f'{method}_{optimizer}' / f'seed_{seed}' / f'{method}__{optimizer}_{case}_seed_{seed}.dat'
                if target in destinations:
                    raise ValueError(f'Duplicate output name: {target}')
                destinations.add(target)
                if target.exists():
                    if not target.is_file():
                        raise ValueError(f'Output is not a file: {target}')
                    if target.read_bytes() != source.read_bytes() and not args.overwrite:
                        raise ValueError(f'Output differs (use --overwrite to replace): {target}')
                plan.append((source, target))
            groups += 1
    if requested - found:
        raise ValueError(f'No matching case/index for requested seeds: {", ".join(sorted(requested - found))}')
    if not groups:
        raise ValueError(f'No matching study indexes found under {studies}')
    return plan, groups, missing


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--repo', type=Path, default=Path.cwd(), help='Directory containing studies/ and optimization_*/ (default: current directory)')
    parser.add_argument('--output-root', type=Path, help='Parent of geometries/ (default: repo)')
    parser.add_argument('--study-glob', default='dae11_seed_*', help='Study folder pattern (default: dae11_seed_*)')
    parser.add_argument('--seeds', nargs='+', help='Only collect these seed identifiers (default: all matching studies)')
    parser.add_argument('--case', choices=('auto', 'constrained', 'unconstrained'), default='auto', help='auto discovers either or both cases')
    parser.add_argument('--expected-count', type=int, default=25, help='Expected row count per seed/case; warns on mismatch (default: 25)')
    parser.add_argument('--overwrite', action='store_true', help='Replace existing files with different contents')
    parser.add_argument('--dry-run', action='store_true', help='Validate and print the copy plan without writing files')
    args = parser.parse_args()
    try:
        plan, groups, missing = build_plan(args)
        for source, target in plan:
            if args.dry_run:
                print(f'{source} -> {target}')
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                if source != target.resolve():
                    shutil.copy2(source, target)
        verb = 'Validated' if args.dry_run else 'Collected'
        print(f'{verb} {len(plan)} geometries across {groups} seed/case groups; skipped {missing} missing entries.')
    except (OSError, ValueError, csv.Error) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())

