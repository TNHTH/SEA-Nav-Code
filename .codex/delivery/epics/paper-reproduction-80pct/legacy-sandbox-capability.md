# Legacy-converter sandbox capability probe

2026-09-07; controller, primary HEAD `c3db30d661f685c6c756b2957ba62a09d375849e`.
Read-only preparation for Task 7, **not converter implementation or acceptance**.

`/usr/bin/bwrap` reports `bubblewrap 0.6.1`; `/usr/bin/timeout` and
`/usr/bin/prlimit` exist. `/lib` and `/lib64` link to `usr/lib` and `usr/lib64`.
A trusted `/usr/bin/true` process completed with all namespaces unshared.
The more relevant fresh isolated dependency probe was:

```bash
timeout 15s bwrap --unshare-all --die-with-parent --new-session --clearenv \
  --ro-bind /usr /usr --symlink usr/lib /lib --symlink usr/lib64 /lib64 \
  --proc /proc --dev /dev --tmpfs /tmp --dir /run --dir /runtime \
  --ro-bind /home/twyc/Documents/Codex/2026-09-04/https-github-com-tnhth-sea-nav-2/work/sea-nav-cpu-venv /runtime/cpu-venv \
  --setenv PYTHONDONTWRITEBYTECODE 1 /runtime/cpu-venv/bin/python -I -B -c \
  'import inspect, os, socket, torch; print("torch", torch.__version__); print("weights_only", "weights_only" in inspect.signature(torch.load).parameters); print("network_interfaces", socket.if_nameindex()); print("home_visible", os.path.exists("/home")); print("env_keys", sorted(os.environ))'
```

Exit 0, stdout:

```text
torch 2.6.0+cpu
weights_only True
network_interfaces [(1, 'lo')]
home_visible False
env_keys ['LC_CTYPE', 'PWD', 'PYTHONDONTWRITEBYTECODE']
```

This verifies that this host currently permits a new namespace sandbox and
that the task CPU interpreter can import its real Torch dependency there.
Only trusted inline inspection ran: no checkpoint was loaded, no repository
or home directory was mounted, no output directory was made writable, and no
network request was made. It does not prove the future converter's input
hash pinning, descriptor/path defenses, resource limits, output restrictions,
safe output revalidation or handling of malicious pickle. Those remain Task 7
implementation/tests; unavailable isolation elsewhere must still fail closed.
It establishes no simulator, optimizer-resume or public redistribution rights.
The absolute CPU-venv path is this probe's evidence identity, not a portable
production default or a request to reproduce a different machine's layout.

## Later pinned-input and resource-limit capability probe

Primary HEAD `73972ad503fde0f95a323487bb03fd5272b18d2e`; Task 6 fix remains
the only source writer. `bwrap --help` actually lists `--ro-bind-fd`.
A trusted probe used an already-open no-follow regular-file descriptor for
`/usr/bin/true` (data only; no legacy checkpoint). Initial write-open checking
incorrectly required EROFS: actual EACCES arrived first, so that diagnostic
exited 1 after its matching-hash output, before reporting limits. The corrected
probe checks the mount's read-only flag independently and accepts the relevant
write-denial errors. It never opens with O_TRUNC or performs a write.

Exact corrected invocation from the task workspace (the parent of `work/`):

```bash
PYTHONDONTWRITEBYTECODE=1 ./work/sea-nav-cpu-venv/bin/python -I -B - <<'PY'
import hashlib, os, stat, subprocess
fd=os.open('/usr/bin/true',os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC)
try:
    assert stat.S_ISREG(os.fstat(fd).st_mode)
    h=hashlib.sha256()
    while True:
        chunk=os.read(fd,8192)
        if not chunk: break
        h.update(chunk)
    os.lseek(fd,0,os.SEEK_SET)
    code='''import errno, hashlib, os, resource, sys
with open('/input/pinned','rb') as f: digest=hashlib.sha256(f.read()).hexdigest()
print('pinned_hash_matches',digest==sys.argv[1])
assert digest==sys.argv[1]
readonly=bool(os.statvfs('/input/pinned').f_flag & os.ST_RDONLY)
print('mount_readonly',readonly)
assert readonly
try:
    output_fd=os.open('/input/pinned',os.O_WRONLY)
except OSError as exc:
    print('write_open_errno',exc.errno)
    assert exc.errno in (errno.EACCES,errno.EROFS,errno.EPERM)
else:
    os.close(output_fd)
    raise AssertionError('input writable')
for name,value in (('CPU',3),('AS',2147483648),('FSIZE',1048576),('NOFILE',32)):
    actual=resource.getrlimit(getattr(resource,'RLIMIT_'+name))
    print(name,actual)
    assert actual==(value,value)
print('home_visible',os.path.exists('/home'))
assert not os.path.exists('/home')
'''
    args=['/usr/bin/timeout','10s','/usr/bin/prlimit','--cpu=3:3','--as=2147483648:2147483648','--fsize=1048576:1048576','--nofile=32:32',
          '/usr/bin/bwrap','--unshare-all','--die-with-parent','--new-session','--clearenv','--ro-bind','/usr','/usr',
          '--symlink','usr/lib','/lib','--symlink','usr/lib64','/lib64','--proc','/proc','--dev','/dev','--tmpfs','/tmp',
          '--dir','/input','--ro-bind-fd',str(fd),'/input/pinned','/usr/bin/python3','-I','-B','-c',code,h.hexdigest()]
    result=subprocess.run(args,pass_fds=(fd,),text=True,capture_output=True)
    print('sandbox_probe_exit',result.returncode)
    print(result.stdout,end='')
    print(result.stderr,end='')
    assert result.returncode==0
finally:
    os.close(fd)
PY
```

Exit 0, output:

```text
sandbox_probe_exit 0
pinned_hash_matches True
mount_readonly True
write_open_errno 13
CPU (3, 3)
AS (2147483648, 2147483648)
FSIZE (1048576, 1048576)
NOFILE (32, 32)
home_visible False
```

This separately proves descriptor-based read-only mounting and that the trusted
child sees the requested hard/soft limits. It does not exercise those limits
under load, wall-time termination, descriptor/path races, malicious pickle,
the future converter's exact mount surface, or safe v2 output verification.
The child in this later probe is system Python doing hash/OS inspection, not
Torch: the earlier real Torch import had no such resource limits. These probe
limits therefore are **not** claimed as usable Torch-converter defaults.
