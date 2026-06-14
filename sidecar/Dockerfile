FROM golang:1.22 AS builder

WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -o sidecar main.go

FROM debian:bookworm-slim
WORKDIR /app
COPY --from=builder /app/sidecar /app/sidecar

EXPOSE 9090
CMD ["/app/sidecar"]
