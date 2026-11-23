# syntax=docker/dockerfile:1

FROM rust:1-bookworm AS build
WORKDIR /src
COPY . .
# .cargo/config.toml sets target-cpu=native; an empty RUSTFLAGS overrides it so
# the image runs on any CPU of its architecture.
ENV RUSTFLAGS=""
RUN cargo build --release --locked -p cogzilla-server

FROM gcr.io/distroless/cc-debian12:nonroot
COPY --from=build /src/target/release/cogzilla-server /usr/local/bin/cogzilla-server
EXPOSE 8080
# The server defaults to 127.0.0.1, which is unreachable from outside the
# container, so the bind address is fixed here. Remap the port with `-p`.
ENTRYPOINT ["cogzilla-server", "--bind", "0.0.0.0:8080"]
