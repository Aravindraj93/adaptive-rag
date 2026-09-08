"""Clean wheel installs across explicitly supplied local CPython executables."""
import argparse
import concurrent.futures
import hashlib
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--python', action='append', required=True)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--wheel', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    wheel = args.wheel.resolve()
    def check(executable):
        version = subprocess.check_output([executable,'-I','-c',
                     'import platform; print(platform.python_version())'], text=True).strip()
        directory = args.work.resolve()/version
        commands = [[executable,'-m','venv',str(directory)]]
        python = str(directory/'Scripts/python.exe')
        commands += [[python,'-m','pip','install','--no-index','--no-deps',str(wheel)],
                     [python,'-I','-m','unittest','discover','-s','tests','-q'],
                     [python,'-m','pip','check'],
                     [python,'-I','-c', 'import adaptive_rag; assert adaptive_rag.__version__ == "0.11.0"; print(adaptive_rag.__file__)']]
        commands += [[python,'-I',str(example)] for example in sorted(Path('examples').glob('*.py'))]
        logs = []
        for command in commands:
            result = subprocess.run(command, capture_output=True, text=True, timeout=240)
            logs.append(dict(command=command, exit_code=result.returncode,
                             stdout=result.stdout, stderr=result.stderr))
            if result.returncode:
                break
        passed = all(log['exit_code'] == 0 for log in logs)
        print(version, 'PASS' if passed else 'FAIL', flush=True)
        return dict(python=version, os='Windows', passed=passed, checks=logs)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(check, args.python))
    report = dict(wheel=wheel.name, wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
                  platforms_not_executed=['Linux','macOS'], results=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    if not all(r['passed'] for r in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
