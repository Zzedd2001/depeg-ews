#!/usr/bin/env python3
"""Assemble the public release: a clean code tree for GitHub and a data archive for Zenodo.

  release/depeg-ews-v<version>/             code, configs, SQL templates, tests, figures, READMEs, LICENSE, CITATION.cff
  release/depeg-ews-data-v<version>.zip     the data files that reproduce the paper without re-fetching (data/README.md)
  release/MANIFEST.tsv, release/SHA256SUMS   every packaged file with its size and SHA-256; checksums of the two outputs

Left out: API caches (data/raw), archives of earlier versions, the rebuildable modeling tables (*.pkl.gz),
superseded files, scratch folders, and local working notes such as README.zh.md (the release is in English). Before writing anything the script scans every file it would package for
an RPC URL that carries a key (alchemy.com/v2/<key>, infura.io/v3/<key>, ...) and stops if it finds one; the
placeholder `<your key>` in the docs does not match. Absolute paths of the machine that produced a report
(for example /Users/<name>/.../depeg-ews/) are cut to repository-relative paths in the data .md and .json files.
A rebuild overwrites the earlier output in place and deletes nothing; files left from an earlier build are listed.

Usage:
  python scripts/make_release.py --list          # what would go where, with sizes; writes nothing
  python scripts/make_release.py                 # code tree and data archive
  python scripts/make_release.py --code-only
"""
import argparse
import fnmatch
import gzip
import hashlib
import re
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERSION = '1.1'

CODE = ['README.md', 'LICENSE', 'CITATION.cff', 'requirements.txt', 'requirements-lock.txt', '.gitignore',
        'config/*.json', 'scripts/*.py', 'sql/*.sql.tmpl', 'tests/*.py', 'tests/fixtures/**/*',
        'figures/*.pdf', 'figures/*.png', 'data/README.md']
DATA = ['data/README.md', 'data/prices_llama_hourly.csv.gz', 'data/token_meta_llama.csv', 'data/rates.csv.gz',
        'data/dex_prices_hourly.csv', 'data/morpho/*', 'data/lending/*', 'data/graph/*', 'data/graph_hourly/*',
        'data/labels/*', 'data/labels_robust/*', 'data/model/*', 'data/model_robust/*']
EXCLUDE = ['*/__pycache__/*', '*.pyc', '*.pkl.gz', '*.DS_Store', 'data/model/results_v0.5.*',
           'data/model/contagion_power_first_run.json']

SECRET = re.compile(rb'(alchemy\.com/v2/|infura\.io/v3/|quiknode\.pro/|ankr\.com/rpc/[a-z_]+/|'
                    rb'blastapi\.io/|chainstack\.com/|getblock\.io/)[A-Za-z0-9_-]{8,}')
LOCAL_PATH = re.compile(r'(?:/sessions|/Users|/home|/mnt|/tmp)/[^\s`\'"()\[\]]*?depeg-ews/')
TEXT_SUFFIXES = {'.py', '.md', '.json', '.csv', '.txt', '.tmpl', '.sql', '.cff', '.yml', '.yaml', ''}


def collect(root, patterns):
    """Files under root matching the patterns (glob, relative), minus EXCLUDE; sorted, unique."""
    out = set()
    for pat in patterns:
        for p in root.glob(pat):
            if p.is_file():
                rel = p.relative_to(root).as_posix()
                if not any(fnmatch.fnmatch(rel, x) or fnmatch.fnmatch('/' + rel, x) for x in EXCLUDE):
                    out.add(rel)
    return sorted(out)


def find_secret(path, gz_scan_mb):
    """The first RPC-with-key match in a file (text read whole, .gz read up to gz_scan_mb MB), or None."""
    if path.suffix == '.gz':
        with gzip.open(path, 'rb') as f:
            data = f.read(int(gz_scan_mb * 1e6))
    elif path.suffix in TEXT_SUFFIXES or path.name.startswith('.'):
        data = path.read_bytes()
    else:
        return None
    m = SECRET.search(data)
    return m.group(1).decode() if m else None


def scrub(rel, raw, root=None):
    """Cut machine-specific absolute paths in data reports (.md, .json) to repository-relative ones: any path through
    a folder named depeg-ews, and the absolute path of the repository being released, whatever its name."""
    if rel.startswith('data/') and rel.endswith(('.md', '.json')):
        text = raw.decode('utf-8')
        new = LOCAL_PATH.sub('', text)
        if root is not None:
            for prefix in sorted({Path(root).as_posix(), Path(root).resolve().as_posix()}, key=len, reverse=True):
                new = new.replace(prefix.rstrip('/') + '/', '')
        if new != text:
            return new.encode('utf-8'), True
    return raw, False


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def human(n):
    for unit in ('B', 'KB', 'MB', 'GB'):
        if n < 1024 or unit == 'GB':
            return f'{n:.0f} {unit}' if unit == 'B' else f'{n:.1f} {unit}'
        n /= 1024


def build(args):
    root = Path(args.root)
    out = Path(args.out)
    code = collect(root, CODE)
    data = [] if args.code_only else collect(root, DATA)
    missing = [p for p in ('README.md', 'LICENSE', 'CITATION.cff', 'data/README.md') if p not in code]
    if missing:
        sys.exit(f'missing release files: {", ".join(missing)}')
    size = {rel: (root / rel).stat().st_size for rel in set(code) | set(data)}
    print(f'code tree: {len(code)} files, {human(sum(size[r] for r in code))}')
    if data:
        print(f'data archive: {len(data)} files, {human(sum(size[r] for r in data))} before compression')
    if args.list:
        for kind, files in (('code', code), ('data', data)):
            for rel in files:
                print(f'{kind}\t{human(size[rel]):>10}\t{rel}')
        return {'code': code, 'data': data}

    hits = [(rel, h) for rel in sorted(set(code) | set(data)) if (h := find_secret(root / rel, args.gz_scan_mb))]
    if hits:
        for rel, host in hits:
            print(f'  {rel}: an RPC URL with a key ({host}...)')
        sys.exit('stopped: remove the key from these files before releasing (nothing was written)')

    tree = out / f'depeg-ews-v{args.version}'
    # files are overwritten in place (nothing is deleted); files left over from an earlier build are reported
    stale = sorted(p.relative_to(tree).as_posix() for p in tree.rglob('*') if p.is_file()) if tree.exists() else []
    stale = [rel for rel in stale if rel not in set(code)]
    manifest, scrubbed = [], []
    for rel in code:
        dst = tree / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / rel, dst)
        manifest.append(('code', rel, dst.stat().st_size, sha256(dst)))
    written = [tree]
    if data:
        zpath = out / f'depeg-ews-data-v{args.version}.zip'
        zpath.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zpath, 'w') as z:
            for rel in data:
                raw_path = root / rel
                if rel.endswith(('.md', '.json')):
                    raw, changed = scrub(rel, raw_path.read_bytes(), root)
                    if changed:
                        scrubbed.append(rel)
                    z.writestr(zipfile.ZipInfo.from_file(raw_path, rel), raw, compress_type=zipfile.ZIP_DEFLATED)
                    digest, n = hashlib.sha256(raw).hexdigest(), len(raw)
                else:
                    kind = zipfile.ZIP_STORED if rel.endswith('.gz') else zipfile.ZIP_DEFLATED
                    z.write(raw_path, rel, compress_type=kind)
                    digest, n = sha256(raw_path), size[rel]
                manifest.append(('data', rel, n, digest))
        written.append(zpath)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'MANIFEST.tsv').write_text('part\tpath\tbytes\tsha256\n' + ''.join(
        f'{k}\t{r}\t{n}\t{h}\n' for k, r, n, h in manifest))
    sums = [f'{sha256(p)}  {p.name}' for p in written if p.is_file()]
    (out / 'SHA256SUMS').write_text('\n'.join(sums) + ('\n' if sums else ''))
    if scrubbed:
        print(f'local paths cut to relative ones in {len(scrubbed)} file(s): {", ".join(scrubbed)}')
    if stale:
        print(f'not part of this release, delete by hand from {tree}: {", ".join(stale)}')
    print(f'wrote {tree}' + (f' and {written[-1]} ({human(written[-1].stat().st_size)})' if data else ''))
    return {'code': code, 'data': data, 'scrubbed': scrubbed, 'stale': stale}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=str(ROOT))
    ap.add_argument('--out', default=str(ROOT / 'release'))
    ap.add_argument('--version', default=VERSION)
    ap.add_argument('--code-only', action='store_true')
    ap.add_argument('--list', action='store_true', help='print the plan and write nothing')
    ap.add_argument('--gz-scan-mb', type=float, default=8.0,
                    help='how much of each .gz file the key scan reads (decompressed MB)')
    return build(ap.parse_args(argv))


if __name__ == '__main__':
    main()
