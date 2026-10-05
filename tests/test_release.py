"""Offline tests for make_release.py on a small fake repository.

python tests/test_release.py
"""
import contextlib
import gzip
import io
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
import make_release as mr  # noqa: E402


def fake_repo(d, key_in=None):
    files = {
        'README.md': 'export ETH_RPC_URL=https://eth-mainnet.g.alchemy.com/v2/<your key>\n',
        'README.zh.md': 'x\n', 'LICENSE': 'MIT\n', 'CITATION.cff': 'cff-version: 1.2.0\n', 'requirements.txt': 'pandas\n',
        '.gitignore': 'data/*\n', 'config/assets.json': '{}', 'scripts/a.py': 'print(1)\n', 'sql/02.sql.tmpl': 'SELECT 1\n',
        'tests/test_a.py': 'x = 1\n', 'tests/fixtures/f.csv': 'a\n1\n', 'data/README.md': '# data\n',
        'data/labels/source_check.md': 'Label source: `/sessions/rcw-1/mnt/depeg-ews/data/labels`\n',
        'data/labels/episodes.csv': 'symbol,start\nUSDe,2025-10-10\n',
        'data/model/results.json': '{"path": "/Users/someone/Downloads/depeg-ews/data/model"}',
        'data/model/results_v0.5.md': 'old\n', 'data/raw/llama/cache.json': '{}', 'data/archive_v07_morpho/x.md': 'old\n',
        'data/dex_prices_hourly_vwap_v1.csv': 'old\n', 'scripts/__pycache__/a.cpython-310.pyc': 'bin',
    }
    for rel, text in files.items():
        p = Path(d) / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    for rel in ('data/rates.csv.gz', 'data/model/dataset.pkl.gz'):
        with gzip.open(Path(d) / rel, 'wt') as f:
            f.write('symbol,rate\nwstETH,1.2\n')
    (Path(d) / 'figures').mkdir(exist_ok=True)
    (Path(d) / 'figures' / 'fig2.png').write_bytes(b'\x89PNG')
    if key_in:
        fake = 'https://eth-mainnet.g.alchemy.com/v2/' + 'AbCdEf' + '0123456789xyz'      # split so this file passes the scan
        (Path(d) / key_in).write_text(f'url = "{fake}"\n')


def run(d, *extra):
    with contextlib.redirect_stdout(io.StringIO()):
        return mr.main(['--root', d, '--out', str(Path(d) / 'release'), *extra])


def test_release_keeps_code_and_data_and_leaves_out_caches():
    with tempfile.TemporaryDirectory() as d:
        fake_repo(d)
        (Path(d) / 'data' / 'model' / 'report.md').write_text(f'Source: `{Path(d).resolve()}/data/labels`\n')
        res = run(d)
        tree = Path(d) / 'release' / f'depeg-ews-v{mr.VERSION}'
        assert (tree / 'scripts' / 'a.py').exists() and (tree / 'tests' / 'fixtures' / 'f.csv').exists()
        assert (tree / 'figures' / 'fig2.png').exists() and (tree / 'data' / 'README.md').exists()
        assert not (tree / 'scripts' / '__pycache__').exists()
        assert not (tree / 'README.zh.md').exists()                                 # local working notes stay out
        assert not (tree / 'data' / 'labels').exists()                              # data go to the archive only
        with zipfile.ZipFile(Path(d) / 'release' / f'depeg-ews-data-v{mr.VERSION}.zip') as z:
            names = set(z.namelist())
            assert {'data/README.md', 'data/rates.csv.gz', 'data/labels/episodes.csv', 'data/model/results.json'} <= names
            assert not any(n.endswith('.pkl.gz') or 'raw/' in n or 'archive_' in n or 'results_v0.5' in n
                           or 'vwap_v1' in n for n in names)
            assert z.read('data/labels/source_check.md').decode() == 'Label source: `data/labels`\n'
            assert z.read('data/model/results.json').decode() == '{"path": "data/model"}'
            assert z.read('data/model/report.md').decode() == 'Source: `data/labels`\n'   # any repo name
            assert z.getinfo('data/rates.csv.gz').compress_type == zipfile.ZIP_STORED
        assert sorted(res['scrubbed']) == ['data/labels/source_check.md', 'data/model/report.md',
                                           'data/model/results.json']
        sums = (Path(d) / 'release' / 'SHA256SUMS').read_text().split('\n')
        assert sums[0].endswith(f'depeg-ews-data-v{mr.VERSION}.zip')
        manifest = (Path(d) / 'release' / 'MANIFEST.tsv').read_text()
        assert 'code\tscripts/a.py' in manifest and 'data\tdata/rates.csv.gz' in manifest
        (Path(d) / 'scripts' / 'a.py').write_text('print(3)\n')          # a second build overwrites in place
        (tree / 'scripts' / 'old.py').write_text('x\n')
        res = run(d)
        assert (tree / 'scripts' / 'a.py').read_text() == 'print(3)\n' and res['stale'] == ['scripts/old.py']


def test_release_stops_on_an_rpc_key_and_writes_nothing():
    with tempfile.TemporaryDirectory() as d:
        fake_repo(d, key_in='scripts/b.py')
        try:
            run(d)
            raise AssertionError('a file with an RPC key was packaged')
        except SystemExit as e:
            assert 'stopped' in str(e)
        assert not (Path(d) / 'release').exists()
        fake_repo(d)                                    # the placeholder in README.md alone does not stop it
        (Path(d) / 'scripts' / 'b.py').write_text('print(2)\n')
        run(d, '--code-only')
        assert (Path(d) / 'release' / f'depeg-ews-v{mr.VERSION}' / 'scripts' / 'b.py').exists()


if __name__ == '__main__':
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            try:
                fn()
                print('ok  ', name)
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                fails += 1
                print('FAIL', name, repr(e))
    sys.exit(1 if fails else 0)
