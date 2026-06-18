package emit

import (
	"encoding/binary"
	"testing"

	"google.golang.org/protobuf/proto"

	observabilityv1 "paling-sidecar/gen/go/observability/v1"
	palingeventsv1 "paling-sidecar/gen/go/paling/events/v1"
)

// TestEncode_FirstMessageIndex pins the index-0 case: ServiceHealthHeartbeat is
// the first message in observability.proto, so its message-index is the single
// 0x00 optimization byte.
func TestEncode_FirstMessageIndex(t *testing.T) {
	hb := &observabilityv1.ServiceHealthHeartbeat{ServiceName: "paling"}
	frame, err := encode(7, hb)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	if frame[0] != 0x00 {
		t.Errorf("magic = 0x%02x, want 0x00", frame[0])
	}
	if id := binary.BigEndian.Uint32(frame[1:5]); id != 7 {
		t.Errorf("schema id = %d, want 7", id)
	}
	if frame[5] != 0x00 {
		t.Errorf("message-index = 0x%02x, want 0x00 (first message)", frame[5])
	}
	var got observabilityv1.ServiceHealthHeartbeat
	if err := proto.Unmarshal(frame[6:], &got); err != nil {
		t.Fatalf("payload round-trip: %v", err)
	}
	if got.GetServiceName() != "paling" {
		t.Errorf("round-trip mismatch: %q", got.GetServiceName())
	}
}

// TestEncode_SecondMessageIndex is the sharp edge: BanchanLifecycleEvent is the
// SECOND message in banchan_event.proto (index 1), so the message-index is NOT
// the 0x00 optimization -- it's a zig-zag varint count (1 -> 0x02) followed by
// the zig-zag varint index (1 -> 0x02). Hardcoding 0x00 here would silently
// break every consumer.
func TestEncode_SecondMessageIndex(t *testing.T) {
	ev := &palingeventsv1.BanchanLifecycleEvent{BentoId: "b1", BanchanName: "train"}
	frame, err := encode(9, ev)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	if id := binary.BigEndian.Uint32(frame[1:5]); id != 9 {
		t.Errorf("schema id = %d, want 9", id)
	}
	if frame[5] != 0x02 || frame[6] != 0x02 {
		t.Errorf("message-index = [0x%02x 0x%02x], want [0x02 0x02] (count=1, index=1, zig-zag)", frame[5], frame[6])
	}
	var got palingeventsv1.BanchanLifecycleEvent
	if err := proto.Unmarshal(frame[7:], &got); err != nil {
		t.Fatalf("payload round-trip: %v", err)
	}
	if got.GetBentoId() != "b1" || got.GetBanchanName() != "train" {
		t.Errorf("round-trip mismatch: %+v", &got)
	}
}
