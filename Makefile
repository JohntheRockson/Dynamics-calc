# Convenience targets for the Rust engine/server + web frontend.
# See README.md for the full architecture and dev-workflow writeup.

.PHONY: build run test engine-test server-test web-install web-build \
        dev-server dev-web fmt clippy parity-golden

# --- Production: one binary serving the API + built frontend on :8080 ---
build: web-build
	cargo build --release

run: web-build
	cargo run --release -p attitude-server

# --- Tests ---
test:
	cargo test --workspace

engine-test:
	cargo test -p attitude-engine

server-test:
	cargo test -p attitude-server

# --- Frontend ---
web-install:
	cd web && npm install

web-build: web-install
	cd web && npm run build

# --- Dev loop: run these two in separate terminals ---
# (Vite proxies /api to :8080; attitude-server ignores the missing web/dist.)
dev-server:
	cargo run -p attitude-server

dev-web: web-install
	cd web && npm run dev

# --- Lint / format ---
fmt:
	cargo fmt --all

clippy:
	cargo clippy --workspace --all-targets

# --- Regenerate the Python golden fixtures used by engine/tests/python_parity.rs ---
parity-golden:
	python3 -m pip install -e legacy-python[dev]
	python3 engine/tests/golden/generate_golden.py
