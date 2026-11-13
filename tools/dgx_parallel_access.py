"""Owner-issued ACLs for immutable sources and new delegated result folders."""
import argparse
import ctypes
import ctypes.util
import json
import os
from pathlib import Path
import pwd
import socket
import stat
import subprocess

PRIMARY = Path('/dgxa_home/se23uari167/third_eye_dgx_20261026')
HORIZON = Path('/dgxa_home/se23ucse205/third_eye_horizon_20261109')
ROOTS = {'positive': (1231, PRIMARY / 'runs/positive_regime_20261109'),
         'horizon': (1339, HORIZON)}
TEACHER = 1189


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


class ACL:
    def __init__(self):
        library = ctypes.util.find_library('acl')
        if not library:
            raise RuntimeError('POSIX ACL library unavailable; do not broaden public permissions')
        self.lib = ctypes.CDLL(library, use_errno=True)
        self.lib.acl_get_file.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self.lib.acl_get_file.restype = ctypes.c_void_p
        self.lib.acl_to_text.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ssize_t)]
        self.lib.acl_to_text.restype = ctypes.c_void_p
        self.lib.acl_from_text.argtypes = [ctypes.c_char_p]
        self.lib.acl_from_text.restype = ctypes.c_void_p
        self.lib.acl_calc_mask.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.lib.acl_valid.argtypes = [ctypes.c_void_p]
        self.lib.acl_set_file.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p]
        self.lib.acl_free.argtypes = [ctypes.c_void_p]

    def grant(self, path, uid, permission, default=False):
        path = Path(path)
        if path.is_symlink() or path.stat().st_uid != os.getuid():
            raise ValueError('ACL target must be directly owned and not a symlink')
        kind = 0x4000 if default else 0x8000
        current = self.lib.acl_get_file(os.fsencode(path), kind)
        if not current:
            raise OSError(ctypes.get_errno(), 'Cannot read ACL')
        text = self.lib.acl_to_text(current, None)
        if not text:
            self.lib.acl_free(current)
            raise OSError(ctypes.get_errno(), 'Cannot serialize ACL')
        previous = ctypes.string_at(text).decode()
        self.lib.acl_free(text)
        self.lib.acl_free(current)
        qualifiers = (f'user:{uid}:', f'user:{pwd.getpwuid(uid).pw_name}:')
        lines = [line for line in previous.splitlines() if not line.startswith(qualifiers)]
        if default and not lines:
            lines = ['user::rwx', 'group::r-x', 'other::r-x']
        lines.append(f'user:{uid}:{permission}')
        replacement = ctypes.c_void_p(self.lib.acl_from_text(('\n'.join(lines) + '\n').encode()))
        if not replacement.value:
            raise OSError(ctypes.get_errno(), 'Cannot construct scoped ACL')
        try:
            if (self.lib.acl_calc_mask(ctypes.byref(replacement)) or
                    self.lib.acl_valid(replacement) or
                    self.lib.acl_set_file(os.fsencode(path), kind, replacement)):
                raise OSError(ctypes.get_errno(), 'Cannot install scoped ACL: ' + str(path))
        finally:
            self.lib.acl_free(replacement)
        return {'path': str(path), 'uid': uid, 'permissions': permission,
                'default': default, 'previous_acl': previous}


def shared_parent(acl, path, owner, records):
    path.mkdir(parents=True, exist_ok=True)
    records.append(acl.grant(path, TEACHER, 'rwx'))
    records.append(acl.grant(path, owner, 'rwx', default=True))
    os.chmod(path, stat.S_IMODE(path.stat().st_mode) | stat.S_ISVTX)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('kind', choices=ROOTS)
    args = parser.parse_args()
    uid, root = ROOTS[args.kind]
    if (os.getuid() != uid or not os.environ.get('SLURM_JOB_ID')
            or 'login' in socket.gethostname().lower()):
        raise RuntimeError('Owning account and CPU Slurm allocation required')
    control = root / 'temporary_worker_20261112'
    control.mkdir(exist_ok=True)
    if (control / 'access_completed.json').exists():
        raise FileExistsError('Preserve a previous access receipt')
    acl, records = ACL(), []
    probe = control / 'acl_probe'
    probe.mkdir(exist_ok=True)
    (probe / 'owner_protected.txt').write_text('owner-only fixture\n')
    shared_parent(acl, probe, uid, records)
    if args.kind == 'horizon':
        records.append(acl.grant(Path.home(), TEACHER, '--x'))
        for directory, folders, files in os.walk(root, followlinks=False):
            folders[:] = [name for name in folders if not (Path(directory) / name).is_symlink()]
            parent = Path(directory)
            records.append(acl.grant(parent, TEACHER, 'r-x'))
            for name in files:
                path = parent / name
                if not path.is_symlink() and path.stat().st_uid == uid:
                    records.append(acl.grant(path, TEACHER, 'r--'))
        # Reinstate the dedicated fixture's write ACL after read-only traversal.
        shared_parent(acl, probe, uid, records)
        for model in ('qwen3-8b', 'llama3.1-8b'):
            shared_parent(acl, root / 'parents' / model, uid, records)
    else:
        shared_parent(acl, root / 'parents', uid, records)
    (control / 'status').mkdir(exist_ok=True)
    allocated = subprocess.run(['du', '-x', '-B1', '-s', str(Path.home())],
                               capture_output=True, text=True, check=True)
    write(control / 'access_completed.json', {'passed': True, 'job_id': os.environ['SLURM_JOB_ID'],
          'owner_uid': uid, 'delegate_uid': TEACHER, 'kind': args.kind,
          'whole_home_bytes': int(allocated.stdout.split()[0]), 'changes': records,
          'scientific_content_changed': False, 'faculty_owner_data_touched': False})
    print(json.dumps({'passed': True, 'kind': args.kind, 'acl_changes': len(records)}), flush=True)


if __name__ == '__main__':
    main()
