# ember chat on a Hugging Face Space (Docker SDK, free CPU tier).
# Build context: the folder tools/build_hf_space.sh assembles (src/ = an
# ember.cpp source snapshot, model/chat.gguf, www/, start.sh).
FROM ubuntu:24.04 AS build
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates libboost-regex-dev python3-dev \
    && rm -rf /var/lib/apt/lists/*
COPY src /src
# Two CPU builds: portable x86-64, and x86-64-v3 (AVX2/FMA), which halves
# prompt processing with identical greedy output (40/40 on the eval prompts).
# start.sh picks the v3 binary only if the CPU reports avx2 and fma.
RUN cmake -S /src -B /build -DCMAKE_BUILD_TYPE=Release -DHAS_CUDA=OFF \
    && cmake --build /build --target tinyllama_server -j2 \
    && cmake -S /src -B /build_v3 -DCMAKE_BUILD_TYPE=Release -DHAS_CUDA=OFF -DCMAKE_CXX_FLAGS="-march=x86-64-v3" \
    && cmake --build /build_v3 --target tinyllama_server -j2

FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 libboost-regex1.83.0 ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
# Spaces run the container as uid 1000.
RUN useradd -m -u 1000 user
WORKDIR /app
COPY --from=build /build/tinyllama_server /app/tinyllama_server
COPY --from=build /build_v3/tinyllama_server /app/tinyllama_server_v3
COPY model/chat.gguf /app/model.gguf
COPY www /app/www
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh && chown -R user:user /app
USER user
# Public mode: warnings-only logging (nothing a visitor types is logged),
# request caps, bounded queue (server.cpp). Two threads for two vCPUs.
ENV EMBER_PUBLIC=1 OMP_NUM_THREADS=2
EXPOSE 7860
HEALTHCHECK --interval=60s --timeout=5s CMD curl -fs http://127.0.0.1:7860/health || exit 1
CMD ["/app/start.sh"]
