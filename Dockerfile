# syntax=docker/dockerfile:1

FROM rust:1-bookworm AS build
WORKDIR /src
# Install the tiny (~150 KB, libc-only) Vulkan loader so `docker run --gpus all`
# can auto-detect hardware GPUs while CPU-only hosts still fall back cleanly to CPU.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libvulkan1 \
    && rm -rf /var/lib/apt/lists/*
COPY . .
# .cargo/config.toml sets target-cpu=native; an empty RUSTFLAGS overrides it so
# the image runs on any CPU of its architecture.
ENV RUSTFLAGS=""
RUN cargo build --release --locked -p cogzilla-server \
    && mkdir -p /out-lib \
    && cp -a /usr/lib/*-linux-gnu/libvulkan.so.1* /out-lib/

FROM gcr.io/distroless/cc-debian12:nonroot
COPY --from=build /out-lib/ /usr/lib/
COPY --from=build /src/target/release/cogzilla-server /usr/local/bin/cogzilla-server
EXPOSE 8080
# The server defaults to 127.0.0.1, which is unreachable from outside the
# container, so the bind address is fixed here. Remap the port with `-p`.
ENTRYPOINT ["cogzilla-server", "--bind", "0.0.0.0:8080"]
