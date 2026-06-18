FROM golang:1.25 AS builder

WORKDIR /app

# codegen toolchain (cached layer). buf is fetched as a prebuilt release binary:
# `go install`-ing buf compiles its entire dependency graph (a slow multi-minute
# step that looks like a hang from outside); the release binary is an instant
# download. protoc-gen-go stays a small go install. Bindings are generated at
# build from the vendored proto/, never committed.
RUN curl -fsSL "https://github.com/bufbuild/buf/releases/download/v1.71.0/buf-$(uname -s)-$(uname -m)" -o /usr/local/bin/buf \
 && chmod +x /usr/local/bin/buf \
 && go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
ENV PATH="/go/bin:${PATH}"

COPY go.mod go.sum ./
RUN go mod download

COPY . .

# generate bindings (gen/ is gitignored) before compiling the whole package
RUN buf generate
RUN CGO_ENABLED=0 GOOS=linux go build -o sidecar .

FROM debian:bookworm-slim
WORKDIR /app
COPY --from=builder /app/sidecar /app/sidecar

EXPOSE 9090
CMD ["/app/sidecar"]
