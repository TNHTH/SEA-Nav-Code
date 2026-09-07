# Task 5 replay push performance harness

Date: 2026-09-07. This is a bounded CPU-only preparation note. It uses standalone synthetic tensors and does not import the simulator, `legged_gym`, the adapter trainer, or any production replay implementation. No current production defect is inferred by this probe.

## Practical conclusion

A scoped `TorchDispatchMode` is a suitable deterministic CPU regression harness for the replay push hot path. It can assert the operator-level contract directly, without wall-clock thresholds or monkeypatching `torch`:

- an indexed ring write accounts for only `active_rows * fixed_payload_width` source elements;
- known full-capacity materializers (`aten.clone`, `aten.roll`, and `aten.cat`) are rejected when their output spans the ring storage;
- `Tensor.item()` / implicit tensor-to-Python scalar conversion is detected as `aten._local_scalar_dense`;
- a `view` of the preallocated storage is not treated as a copy merely because its tensor metadata describes all storage elements.

The test should run the same active rows and payload against a small and a much larger capacity. It should assert stable indexed-write source volume, not elapsed time.

## Exact validation command

Run from the repository root with the required CPU environment. The following is the exact standalone program executed for this note:

```bash
PYTHONDONTWRITEBYTECODE=1 ../sea-nav-cpu-venv/bin/python - <<'PY'
import torch
from torch.utils._python_dispatch import TorchDispatchMode

NUM_ENVS, WIDTH, ACTIVE = 4, 5, 2


def tensors(value):
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (tuple, list)):
        result = []
        for child in value:
            result.extend(tensors(child))
        return result
    if isinstance(value, dict):
        result = []
        for child in value.values():
            result.extend(tensors(child))
        return result
    return []


class ReplayPushProbe(TorchDispatchMode):
    MATERIALIZERS = {
        "aten.clone.default",
        "aten.roll.default",
        "aten.cat.default",
    }

    def __init__(self, storage_numel):
        super().__init__()
        self.storage_numel = storage_numel
        self.full_capacity_materializations = []
        self.indexed_write_elements = 0
        self.scalar_extractions = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        name = str(func)
        if name == "aten._local_scalar_dense.default":
            self.scalar_extractions.append(name)
        if name in self.MATERIALIZERS:
            output_elements = max((x.numel() for x in tensors(result)), default=0)
            if output_elements >= self.storage_numel:
                self.full_capacity_materializations.append((name, output_elements))
        if name == "aten.index_copy_.default":
            self.indexed_write_elements += args[3].numel()
        elif name == "aten.index_put_.default":
            self.indexed_write_elements += args[2].numel()
        return result


def run(capacity, kind):
    storage = torch.zeros((NUM_ENVS, capacity, WIDTH))
    env_ids = torch.tensor([0, 2], dtype=torch.long)
    slot_ids = torch.tensor([1, 3], dtype=torch.long)
    payload = torch.arange(ACTIVE * WIDTH, dtype=torch.float32).view(ACTIVE, WIDTH)
    probe = ReplayPushProbe(storage.numel())
    with probe:
        if kind == "indexed":
            linear_ids = env_ids * capacity + slot_ids
            storage.view(-1, WIDTH).index_copy_(0, linear_ids, payload)
        elif kind == "view":
            storage.view(-1, WIDTH)
        elif kind == "clone":
            storage.clone()
        elif kind == "roll":
            torch.roll(storage, 1, 1)
        elif kind == "cat":
            torch.cat((storage[:, 1:], storage[:, :1]), 1)
        else:
            raise AssertionError(kind)
    if kind == "indexed":
        assert probe.indexed_write_elements == ACTIVE * WIDTH
        assert not probe.full_capacity_materializations
        assert not probe.scalar_extractions
        assert torch.equal(storage[env_ids, slot_ids], payload)
    elif kind == "view":
        assert not probe.full_capacity_materializations
    else:
        assert probe.full_capacity_materializations == [
            ("aten.%s.default" % kind, storage.numel())
        ]
    print(
        "cap=%d kind=%s indexed_elems=%d full=%s scalar=%d"
        % (
            capacity,
            kind,
            probe.indexed_write_elements,
            probe.full_capacity_materializations,
            len(probe.scalar_extractions),
        )
    )


for capacity in (8, 4096):
    for kind in ("indexed", "view", "clone", "roll", "cat"):
        run(capacity, kind)

x = torch.tensor([7], dtype=torch.long)
probe = ReplayPushProbe(x.numel())
with probe:
    x[0].item()
assert probe.scalar_extractions == ["aten._local_scalar_dense.default"]
print("item scalar=%d ops=%s" % (len(probe.scalar_extractions), probe.scalar_extractions))
print("PASS torch=%s" % torch.__version__)
PY
```

Exact output:

```text
cap=8 kind=indexed indexed_elems=10 full=[] scalar=0
cap=8 kind=view indexed_elems=0 full=[] scalar=0
cap=8 kind=clone indexed_elems=0 full=[('aten.clone.default', 160)] scalar=0
cap=8 kind=roll indexed_elems=0 full=[('aten.roll.default', 160)] scalar=0
cap=8 kind=cat indexed_elems=0 full=[('aten.cat.default', 160)] scalar=0
cap=4096 kind=indexed indexed_elems=10 full=[] scalar=0
cap=4096 kind=view indexed_elems=0 full=[] scalar=0
cap=4096 kind=clone indexed_elems=0 full=[('aten.clone.default', 81920)] scalar=0
cap=4096 kind=roll indexed_elems=0 full=[('aten.roll.default', 81920)] scalar=0
cap=4096 kind=cat indexed_elems=0 full=[('aten.cat.default', 81920)] scalar=0
item scalar=1 ops=['aten._local_scalar_dense.default']
PASS torch=2.6.0+cpu
```

The cases use two active rows and a five-float payload throughout. The indexed source volume therefore remains 10 elements when capacity grows from 8 to 4,096. By contrast, each known full-capacity materializer grows from `4 * 8 * 5 = 160` to `4 * 4096 * 5 = 81,920` output elements. The `view` also describes 160 or 81,920 elements, but it is a metadata/aliasing operator and correctly produces no full-copy flag.

## Recommended CPU test shape

Keep the probe local to the test and place it only around the replay `push` call. The production-facing assertion can be compact:

```python
@pytest.mark.parametrize("capacity", [8, 4096])
def test_push_work_scales_with_active_payload_not_capacity(capacity):
    ring = make_ring(num_envs=4, capacity=capacity, payload_width=5)
    env_ids = torch.tensor([0, 2])
    payload = torch.arange(10, dtype=torch.float32).view(2, 5)
    probe = ReplayPushProbe(ring.storage.numel())

    with probe:
        ring.push(env_ids=env_ids, payload=payload)

    assert probe.indexed_write_elements == payload.numel()
    assert probe.full_capacity_materializations == []
    assert probe.scalar_extractions == []
```

Use explicit overload names observed under the pinned Torch 2.6 environment. If Task 5's final hot-path contract also rejects `stack` or full-buffer `where`, add their observed ATen overloads to a separately named forbidden/materializing set and validate them with a synthetic positive-control case. Do not classify every capacity-sized tensor result as copied: operations such as `view`, `slice`, and other metadata aliases can legitimately expose the entire preallocated storage size without moving it.

## What this proves, and what it does not

This proves, for eager Torch 2.6 CPU execution of the covered path, which ATen operators were dispatched; that the indexed write's source payload has `active_rows * payload_width` elements; that the selected destination rows received the intended values; that the named full-capacity positive controls are caught; and that `_local_scalar_dense` extraction is caught. Running small and large capacities with identical active input makes accidental capacity-dependent materialization visible without a timing threshold.

This does **not** prove CUDA latency, kernel launch behavior, device memory bandwidth, asynchronous synchronization cost, or simulator/runtime performance. Tensor shape or `numel` metadata alone is not actual copy volume, especially for views and slices. The indexed source-element count is a semantic test oracle for the chosen indexed-write overload, not a hardware byte counter. The whitelist catches the operator families it names; it does not automatically detect future materializers, custom extensions, compiler fusion, non-Torch writes, or work hidden inside a runtime outside the scoped dispatch context. CUDA/runtime evidence remains a separate higher-rung requirement.

## Corrected capacity-relative recommendation

The original probe above remains valid narrow evidence for whole-storage `clone`/`roll`/`cat`, view handling, indexed writes, and scalar extraction. Its `output_elements >= storage.numel()` full-materialization threshold is **not sufficient** for the production test. A selected-row history copy such as `storage[env_ids].clone()` is `O(active_rows * capacity * width)`, yet each intermediate is smaller than the whole `num_envs * capacity * width` storage whenever `active_rows < num_envs`.

The corrected probe should therefore record output volume for explicitly materializing overloads without first comparing it to whole-storage size. For the same active rows, fixed payload, and bounded per-row metadata, require both:

1. observed materializing output volume does not grow when capacity grows; and
2. observed materializing output volume and indexed-write source volume stay within an explicit `active_rows * fixed_per_row_budget`.

Basic aliasing operations (`view`, `slice`, `select`) are deliberately absent from the materializer set. Advanced indexing is different: Torch 2.6 emits `aten.index.Tensor`, which creates an output. A fixed slot gather `storage[env_ids, slot_ids]` is acceptable bounded work, while `storage[env_ids]` copies complete selected histories and grows with capacity.

### Exact correction validation command

```bash
PYTHONDONTWRITEBYTECODE=1 ../sea-nav-cpu-venv/bin/python - <<'PY'
import torch
from torch.utils._python_dispatch import TorchDispatchMode

NUM_ENVS, ACTIVE, WIDTH, METADATA_PER_ROW = 4, 2, 5, 3
FIXED_BUDGET = ACTIVE * (WIDTH + METADATA_PER_ROW)


def tensors(value):
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (tuple, list)):
        result = []
        for child in value:
            result.extend(tensors(child))
        return result
    return []


class CapacityProbe(TorchDispatchMode):
    MATERIALIZING_OUTPUTS = {
        "aten.index.Tensor",          # advanced indexing copies; basic slices do not
        "aten.clone.default",
        "aten.roll.default",
        "aten.cat.default",
        "aten.stack.default",
        "aten.where.self",
    }

    def __init__(self):
        super().__init__()
        self.materializations = []
        self.indexed_write_elements = 0
        self.scalar_extractions = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        name = str(func)
        if name in self.MATERIALIZING_OUTPUTS:
            output_elements = sum(x.numel() for x in tensors(result))
            self.materializations.append((name, output_elements))
        if name == "aten.index_copy_.default":
            self.indexed_write_elements += args[3].numel()
        elif name == "aten.index_put_.default":
            self.indexed_write_elements += args[2].numel()
        if name == "aten._local_scalar_dense.default":
            self.scalar_extractions.append(name)
        return result

    @property
    def materialized_elements(self):
        return sum(elements for _, elements in self.materializations)


def run(capacity, kind):
    storage = torch.zeros((NUM_ENVS, capacity, WIDTH))
    env_ids = torch.tensor([0, 2], dtype=torch.long)
    slot_ids = torch.tensor([1, 3], dtype=torch.long)
    payload = torch.arange(ACTIVE * WIDTH, dtype=torch.float32).view(ACTIVE, WIDTH)
    mask = torch.zeros_like(storage, dtype=torch.bool)
    probe = CapacityProbe()
    with probe:
        if kind == "indexed_push":
            linear_ids = env_ids * capacity + slot_ids
            storage.view(-1, WIDTH).index_copy_(0, linear_ids, payload)
        elif kind == "slot_index":
            storage[env_ids, slot_ids]
        elif kind == "selected_clone":
            storage[env_ids].clone()
        elif kind == "slice_alias":
            storage[:ACTIVE]
        elif kind == "stack_history":
            torch.stack((storage[0], storage[2]), 0)
        elif kind == "full_where":
            torch.where(mask, storage, storage)
        else:
            raise AssertionError(kind)
    print(
        "cap=%d kind=%s write=%d materialized=%d ops=%s budget=%d"
        % (
            capacity,
            kind,
            probe.indexed_write_elements,
            probe.materialized_elements,
            probe.materializations,
            FIXED_BUDGET,
        )
    )
    return probe


by_kind = {}
for capacity in (8, 4096):
    for kind in (
        "indexed_push",
        "slot_index",
        "selected_clone",
        "slice_alias",
        "stack_history",
        "full_where",
    ):
        by_kind.setdefault(kind, []).append(run(capacity, kind))

for kind in ("indexed_push", "slot_index", "slice_alias"):
    small, large = by_kind[kind]
    assert small.materialized_elements == large.materialized_elements
    assert large.materialized_elements <= FIXED_BUDGET
for kind in ("selected_clone", "stack_history", "full_where"):
    small, large = by_kind[kind]
    assert large.materialized_elements > small.materialized_elements
    assert large.materialized_elements > FIXED_BUDGET
assert all(p.indexed_write_elements <= FIXED_BUDGET for p in by_kind["indexed_push"])
assert all(not p.scalar_extractions for probes in by_kind.values() for p in probes)
print("PASS corrected-capacity-probe torch=%s" % torch.__version__)
PY
```

Exact output:

```text
cap=8 kind=indexed_push write=10 materialized=0 ops=[] budget=16
cap=8 kind=slot_index write=0 materialized=10 ops=[('aten.index.Tensor', 10)] budget=16
cap=8 kind=selected_clone write=0 materialized=160 ops=[('aten.index.Tensor', 80), ('aten.clone.default', 80)] budget=16
cap=8 kind=slice_alias write=0 materialized=0 ops=[] budget=16
cap=8 kind=stack_history write=0 materialized=80 ops=[('aten.stack.default', 80)] budget=16
cap=8 kind=full_where write=0 materialized=160 ops=[('aten.where.self', 160)] budget=16
cap=4096 kind=indexed_push write=10 materialized=0 ops=[] budget=16
cap=4096 kind=slot_index write=0 materialized=10 ops=[('aten.index.Tensor', 10)] budget=16
cap=4096 kind=selected_clone write=0 materialized=81920 ops=[('aten.index.Tensor', 40960), ('aten.clone.default', 40960)] budget=16
cap=4096 kind=slice_alias write=0 materialized=0 ops=[] budget=16
cap=4096 kind=stack_history write=0 materialized=40960 ops=[('aten.stack.default', 40960)] budget=16
cap=4096 kind=full_where write=0 materialized=81920 ops=[('aten.where.self', 81920)] budget=16
PASS corrected-capacity-probe torch=2.6.0+cpu
```

The selected-row copy is the important correction: advanced indexing materialized 80 then cloning materialized another 80 elements at capacity 8; both grew to 40,960 at capacity 4,096. `stack` over the same two selected histories grew from 80 to 40,960, and full-buffer `where` grew from 160 to 81,920. In contrast, advanced indexing of two exact slots remained 10 elements. The basic two-row slice exposed capacity-dependent metadata but emitted only `aten.slice.Tensor`, so the corrected materializer accounting remained zero.

### Adoption guidance for the actual ring

Do not assert `indexed_write_elements == payload.numel()` once the real replay schema is present. A correct push may also perform bounded indexed writes for per-environment write positions, valid lengths, episode/step/task-generation IDs, collision-onset flags, and other fixed-count metadata. Instead calculate a documented per-row upper bound from the compact payload field widths plus the number of metadata scalars written per active row:

```python
fixed_write_budget = active_rows * (
    compact_payload_scalars_per_row + metadata_write_scalars_per_row
)
assert small_probe.indexed_write_elements <= fixed_write_budget
assert large_probe.indexed_write_elements <= fixed_write_budget
assert small_probe.indexed_write_elements == large_probe.indexed_write_elements
assert small_probe.materialized_elements <= fixed_materialization_budget
assert large_probe.materialized_elements <= fixed_materialization_budget
assert small_probe.materialized_elements == large_probe.materialized_elements
assert small_probe.scalar_extractions == large_probe.scalar_extractions == []
```

Derive both budgets from the implemented compact schema rather than copying the synthetic `WIDTH + 3` example. Inventory the actual Torch 2.6 overloads emitted by the final `push`; count bounded advanced-index reads and all indexed-write value sources. Keep named structural assertions for prohibited `stack`/`cat` and capacity-sized `where`, while the capacity comparison catches selected-row `index`/`clone` histories that a whole-storage threshold misses. This remains operator-semantic CPU evidence, not a measurement of physical bytes copied or CUDA latency.
