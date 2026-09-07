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
