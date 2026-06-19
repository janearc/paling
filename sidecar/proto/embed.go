// Package palingproto embeds the vendored contract source so the sidecar can
// register schemas with the Schema Registry at runtime. The .proto files are
// vendored copies: observability.v1 from kafka-svc, paling.events.v1 from
// paling/proto (both refreshed via `task sync-proto`).
package palingproto

import _ "embed"

// ObservabilitySchema is the PROTOBUF schema registered under the
// observability.v1.ServiceHealthHeartbeat subject (RecordNameStrategy).
//
//go:embed observability/v1/observability.proto
var ObservabilitySchema string

// BanchanSchema is the PROTOBUF schema registered under the
// paling.events.v1.BanchanLifecycleEvent subject (RecordNameStrategy).
//
//go:embed paling/events/v1/banchan_event.proto
var BanchanSchema string

// OrchestrationSchema is the PROTOBUF schema for the inbound
// paling.events.v1.OrchestrationCommand the sidecar consumes off Kafka and
// relays to the bare-metal daemon's control plane.
//
//go:embed paling/events/v1/orchestration_command.proto
var OrchestrationSchema string
