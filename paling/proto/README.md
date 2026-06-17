# Protobuf Schemas

Source of truth for Paling's event schemas. Generated language bindings are
**not** committed: `protoc` emits a `NO CHECKED-IN PROTOBUF GENCODE` marker, and
committed bindings drift from their source. Generate them locally instead.

## Schemas

| File | Package | Purpose |
|------|---------|---------|
| `banchan_event.proto` | `paling.events.v1` | Banchan lifecycle event envelope emitted by the orchestration daemon over Kafka. |

## Regenerating bindings

```bash
./scripts/gen_proto.sh
```

Output is written to `paling/proto/*_pb2.py` and ignored by Git. Re-run after
editing any `.proto`.
