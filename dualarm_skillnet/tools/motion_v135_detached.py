"""Launch a local supervisor outside the calling terminal/host Windows Job.

Only the supervisor breaks away. Its individual simulation workers must still
use the existing owned-process Job Objects for deadline and cleanup control.
"""
from __future__ import annotations

import argparse
import base64
import ctypes
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.overnight_batch import atomic_json, execution_lock, process_birth, sha256, utc_now

BASE = ROOT/'results/sim_v1_3/motion_v135'
FLAGS = 0x00000008 | 0x00000200 | 0x01000000  # DETACHED, NEW_GROUP, BREAKAWAY
REQUEST_KEYS = ('name', 'command', 'python_executable', 'target_sha256', 'helper_sha256',
                'helper', 'log', 'stderr_log', 'cwd', 'creationflags', 'via')


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _confined(value):
    path = Path(value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('后台启动路径必须位于本项目目录内')
    return path


def _inside_job():
    if os.name != 'nt':
        raise RuntimeError('此独立启动器仅支持Windows')
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.IsProcessInJob.argtypes = [wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)]
    answer = wintypes.BOOL()
    if not kernel.IsProcessInJob(kernel.GetCurrentProcess(), None, ctypes.byref(answer)):
        raise ctypes.WinError(ctypes.get_last_error())
    return bool(answer.value)


def _job_evidence():
    """Read only: distinguish a service Job from the launching tool's Job."""
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int,
        ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
    buffer = ctypes.create_string_buffer(65536)
    returned = wintypes.DWORD()
    ok = kernel.QueryInformationJobObject(None, 3, buffer, len(buffer), ctypes.byref(returned))
    if not ok:
        return {'query_ok': False, 'error': ctypes.get_last_error()}
    assigned, count = ctypes.cast(buffer, ctypes.POINTER(wintypes.DWORD))[0:2]
    offset = ctypes.sizeof(ctypes.c_void_p)
    pointers = ctypes.cast(ctypes.addressof(buffer)+offset, ctypes.POINTER(ctypes.c_size_t))
    answer = {'query_ok': True, 'assigned': assigned, 'count': count,
              'pids': [pointers[index] for index in range(count)]}
    class BASIC(ctypes.Structure):
        _fields_ = [('process_time', ctypes.c_int64), ('job_time', ctypes.c_int64),
            ('flags', wintypes.DWORD), ('minimum', ctypes.c_size_t), ('maximum', ctypes.c_size_t),
            ('active', wintypes.DWORD), ('affinity', ctypes.c_size_t),
            ('priority', wintypes.DWORD), ('scheduling', wintypes.DWORD)]
    class IO(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint64) for name in
                    ('read_ops', 'write_ops', 'other_ops', 'read_bytes', 'write_bytes', 'other_bytes')]
    class EXT(ctypes.Structure):
        _fields_ = [('basic', BASIC), ('io', IO), ('process_memory', ctypes.c_size_t),
                    ('job_memory', ctypes.c_size_t), ('peak_process', ctypes.c_size_t), ('peak_job', ctypes.c_size_t)]
    info = EXT()
    if not kernel.QueryInformationJobObject(None, 9, ctypes.byref(info), ctypes.sizeof(info), ctypes.byref(returned)):
        answer['limits_error'] = ctypes.get_last_error()
    else:
        answer.update(flags=info.basic.flags, flags_hex=hex(info.basic.flags),
                      kill_on_close=bool(info.basic.flags & 0x2000))
    return answer


def _confirm_detached(receipt, spec):
    """Check the calling terminal's current Job, not every outer desktop Job."""
    launcher_in_job = _inside_job()
    job = _job_evidence() if launcher_in_job else {'query_ok': True, 'pids': [], 'kill_on_close': False}
    if not job.get('query_ok') or 'kill_on_close' not in job:
        raise RuntimeError('无法核对启动者Job范围，不能确认独立后台')
    if job['kill_on_close'] and receipt['pid'] in job['pids']:
        raise RuntimeError('后台PID仍在启动者会关闭的Job中；保留现场，目标未执行')
    proof = {'pid': receipt['pid'], 'birth': receipt['birth'], 'at': utc_now(),
             'detached_from_launcher_job': True, 'launcher_pid': os.getpid(),
             'launcher_job_after_spawn': job, 'launcher_in_any_job': launcher_in_job,
             'scope': 'Independent from this calling tool/terminal Job; outer Windows desktop Job may remain'}
    atomic_json(Path(spec['receipt_path']).parent/'approval.json', proof)
    return proof


def _wait_approval(spec, receipt, timeout=10.):
    path = Path(spec['receipt_path']).parent/'approval.json'
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        if path.is_file():
            proof = _read(path)
            if (proof.get('pid') != receipt['pid'] or proof.get('birth') != receipt['birth']
                    or proof.get('detached_from_launcher_job') is not True):
                raise RuntimeError('后台独立性确认与本进程身份不一致')
            return proof
        time.sleep(.02)
    raise TimeoutError('父启动器未确认独立Job范围；目标未执行')


def _validate_receipt(receipt, spec):
    for key in ('name', 'command', 'python_executable', 'target_sha256', 'helper_sha256'):
        if receipt.get(key) != spec.get(key):
            raise ValueError('后台进程收据身份/版本mismatch：'+key)
    actual = process_birth(receipt.get('pid', 0))
    if actual is None or str(actual) != str(receipt.get('birth')):
        raise RuntimeError('后台进程identity已结束或PID重用；保留原尝试，使用新名称')
    return receipt


def _wait_receipt(process, path, spec, timeout):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        if path.is_file():
            return _validate_receipt(_read(path), spec)
        if process.poll() is not None:
            raise RuntimeError('后台监督进程在提交独立启动收据前退出；详见exit.json和stderr日志')
        time.sleep(.05)
    raise TimeoutError('10秒内未取得可验证后台收据；保留PID现场，禁止盲目重复启动')


class _ExternalProcess:
    """Track a WMI-created process by PID and creation time, without owning its parent service."""
    def __init__(self, pid):
        self.pid, self.birth = int(pid), process_birth(pid)

    def poll(self):
        current = process_birth(self.pid)
        return None if current is not None and (self.birth is None or current == self.birth) else 1


def _spawn_wmi(command, timeout):
    """Local WMI service creation avoids inheriting the caller's nested Jobs."""
    data = base64.b64encode(json.dumps({'command': subprocess.list2cmdline(command),
        'cwd': str(ROOT), 'flags': FLAGS}).encode('utf-8')).decode('ascii')
    source = "$ErrorActionPreference='Stop'\n"+(
        "$data=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('"+data+"'))|ConvertFrom-Json\n"
        "$startup=New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0;CreateFlags=[uint32]$data.flags}\n"
        "$answer=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=[string]$data.command;CurrentDirectory=[string]$data.cwd;ProcessStartupInformation=$startup}\n"
        "if($answer.ReturnValue -ne 0){throw ('Win32_Process.Create failed: '+$answer.ReturnValue)}\n"
        "$answer|Select-Object ReturnValue,ProcessId|ConvertTo-Json -Compress\n")
    encoded = base64.b64encode(source.encode('utf-16-le')).decode('ascii')
    result = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-EncodedCommand', encoded],
        stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding='utf-8',
        errors='replace',
        timeout=max(.1, min(8., timeout)), creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    if result.returncode:
        raise RuntimeError('本机WMI后台创建失败：'+result.stderr.strip())
    answer = json.loads(result.stdout.strip())
    if answer.get('ReturnValue') != 0 or not answer.get('ProcessId'):
        raise RuntimeError('本机WMI未返回有效后台PID')
    return _ExternalProcess(answer['ProcessId'])


def launch(name, log, command, *, timeout=10., via='direct'):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,99}', name):
        raise ValueError('name必须为单一安全标识')
    if not command or not 0 < timeout <= 10 or via not in ('direct', 'wmi'):
        raise ValueError('需要项目内Python脚本和不超过10秒的启动确认预算')
    target = _confined(command[0])
    if target.suffix != '.py' or not target.is_file():
        raise FileNotFoundError('后台目标必须是存在的项目Python脚本：'+str(target))
    log = _confined(log)
    stderr = log.with_name(log.stem+'.err'+log.suffix)
    helper = Path(__file__).resolve()
    request = {'name': name, 'command': [str(target), *map(str, command[1:])],
        'python_executable': sys.executable, 'target_sha256': sha256(target),
        'helper_sha256': sha256(helper), 'helper': str(helper), 'log': str(log),
        'stderr_log': str(stderr), 'cwd': str(ROOT), 'creationflags': FLAGS, 'via': via}
    folder = BASE/'launches'/name
    folder.mkdir(parents=True, exist_ok=True)
    with execution_lock(folder/'launcher.lock'):
        spec_path, receipt_path = folder/'spec.json', folder/'receipt.json'
        if spec_path.exists():
            old = _read(spec_path)
            if any(old.get(key) != request[key] for key in REQUEST_KEYS):
                raise ValueError('已登记启动的命令、日志或版本改变；保留现场并使用新name')
            if receipt_path.exists():
                receipt = _validate_receipt(_read(receipt_path), request)
                if receipt.get('detached_from_launcher_job') is not True:
                    raise RuntimeError('旧启动未完成Job独立性确认，保留现场并检查，不重复启动')
                return {**receipt, 'status': 'already_running', 'receipt': str(receipt_path)}
            raise RuntimeError('启动名称已有未确认/finished现场；保留并检查PID，禁止重置或重复启动')
        spec = {**request, 'schema_version': 1, 'created_at': utc_now(),
                'launcher_pid': os.getpid(), 'receipt_path': str(receipt_path),
                'launcher_job_before_spawn': _job_evidence() if _inside_job() else None}
        atomic_json(spec_path, spec)
        log.parent.mkdir(parents=True, exist_ok=True)
        try:
            launched_at = time.monotonic()
            worker_command = [sys.executable, '-B', '-X', 'utf8', '-u', str(helper), '--worker', str(spec_path)]
            if via == 'wmi':
                process = _spawn_wmi(worker_command, timeout)
            else:
                with log.open('ab') as out, stderr.open('ab') as err:
                    process = subprocess.Popen(worker_command, cwd=ROOT, stdin=subprocess.DEVNULL,
                        stdout=out, stderr=err, close_fds=True, creationflags=FLAGS)
            atomic_json(folder/'launch.json', {'status': 'launching', 'pid': process.pid,
                'birth': process_birth(process.pid), 'at': utc_now(), 'creationflags': FLAGS})
            remaining = timeout-(time.monotonic()-launched_at)
            if remaining <= 0:
                raise TimeoutError('后台创建已用完10秒启动确认预算，保留PID现场')
            receipt = _wait_receipt(process, receipt_path, spec, remaining)
            _validate_receipt(receipt, spec)
            proof = _confirm_detached(receipt, spec)
            deadline = launched_at+timeout
            while receipt.get('detached_from_launcher_job') is not True:
                if time.monotonic() >= deadline:
                    raise TimeoutError('子进程未确认独立Job握手，保留现场')
                time.sleep(.02)
                receipt = _validate_receipt(_read(receipt_path), spec)
            result = {**receipt, 'status': 'started', 'receipt': str(receipt_path)}
            atomic_json(folder/'launch.json', result)
            return result
        except BaseException as exc:
            atomic_json(folder/'launch_failure.json', {'at': utc_now(),
                'reason': f'{type(exc).__name__}: {exc}',
                'note': 'Original files and any child PID retained. Inspect before resuming; do not duplicate.'})
            raise


def worker(spec_path):
    spec_path = _confined(spec_path)
    if (not spec_path.is_relative_to(BASE/'launches') or spec_path.name != 'spec.json'):
        raise ValueError('后台worker spec必须在本轮launches独立目录')
    spec = _read(spec_path)
    began = time.monotonic()
    failure, code = None, 1
    argv_before, path_before = list(sys.argv), list(sys.path)
    streams, stdout_before, stderr_before = [], sys.stdout, sys.stderr
    try:
        # WMI-created processes have no inherited terminal/file descriptors.
        # Own these streams so every Python output persists without the caller.
        for key in ('log', 'stderr_log'):
            path = _confined(spec[key])
            path.parent.mkdir(parents=True, exist_ok=True)
            streams.append(path.open('a', encoding='utf-8', buffering=1))
        sys.stdout, sys.stderr = streams
        target = _confined(spec['command'][0])
        if (sha256(target) != spec['target_sha256']
                or sha256(Path(__file__)) != spec['helper_sha256']):
            raise ValueError('后台入口源文件在启动前改变，拒绝加载')
        inside = _inside_job()
        job = _job_evidence() if inside else None
        pid = os.getpid()
        identity = process_birth(pid)
        if identity is None:
            raise RuntimeError('无法取得后台进程启动身份')
        receipt = {key: spec[key] for key in REQUEST_KEYS}
        receipt.update(status='running', pid=pid, birth=identity, parent_pid=os.getppid(),
                       is_in_job=inside, in_any_job=inside, job=job, started_at=utc_now(),
                       detached_from_launcher_job=False, status_detail='awaiting_launcher_job_check',
                       scope='Detached from invoking tool Job; outer desktop Job membership is reported separately')
        atomic_json(spec_path.parent/'receipt.json', receipt)
        proof = _wait_approval(spec, receipt)
        receipt.update(detached_from_launcher_job=True, status_detail='target_starting',
                       launcher_detachment_proof=proof)
        atomic_json(spec_path.parent/'receipt.json', receipt)
        sys.argv = list(spec['command'])
        sys.path.insert(0, str(target.parent))
        # runpy keeps the independently verified PID. os.exec on Windows can
        # create a replacement PID and would invalidate the launch receipt.
        try:
            runpy.run_path(str(target), run_name='__main__')
            code = 0
        except SystemExit as exc:
            code = 0 if exc.code is None else int(exc.code) if isinstance(exc.code, int) else 1
            if exc.code is not None and not isinstance(exc.code, int):
                failure = str(exc.code)
    except BaseException as exc:
        failure = f'{type(exc).__name__}: {exc}'
        traceback.print_exc()
        code = 1
    finally:
        sys.argv, sys.path = argv_before, path_before
        atomic_json(spec_path.parent/'exit.json', {'at': utc_now(), 'pid': os.getpid(),
            'exit_code': code, 'reason': failure, 'wall_seconds': time.monotonic()-began})
        sys.stdout, sys.stderr = stdout_before, stderr_before
        for stream in streams:
            stream.close()
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description='脱离聊天终端Job的本地Windows后台启动器')
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--name')
    parser.add_argument('--log', type=Path)
    parser.add_argument('--via', choices=('direct', 'wmi'), default='direct')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.worker:
        return worker(args.worker)
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not args.name or not args.log or not command:
        parser.error('需要 --name NAME --log out/NAME.log -- SCRIPT.py ARGS')
    result = launch(args.name, args.log, command, via=args.via)
    # Full Job evidence is in receipt.json; keep the launch command readable.
    print(json.dumps({key: result.get(key) for key in ('name', 'status', 'pid', 'birth',
        'started_at', 'receipt', 'log', 'stderr_log', 'via', 'in_any_job',
        'detached_from_launcher_job')}, ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
