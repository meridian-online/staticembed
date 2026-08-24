# staticembed — build, package and test the DuckDB extension.
#
# `make check` is the whole gate, and it is what .github/workflows/ci.yml runs.

EXTENSION_NAME := staticembed
# The stable C_STRUCT ABI floor the packaged artifact declares, so one build
# loads on DuckDB 1.2 and later. Mirrors MIN_DUCKDB_VERSION in
# crates/staticembed-duckdb/src/lib.rs.
TARGET_DUCKDB_VERSION := v1.2.0
EXTENSION_VERSION := $(shell sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -1)

UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_S),Darwin)
	LIB_FILE := libstaticembed.dylib
	UNAME_PLATFORM := $(if $(filter arm64,$(UNAME_M)),osx_arm64,osx_amd64)
else
	LIB_FILE := libstaticembed.so
	UNAME_PLATFORM := $(if $(filter aarch64,$(UNAME_M)),linux_arm64,linux_amd64)
endif

DUCKDB ?= duckdb

# The platform string stamped into the metadata trailer must be the one the
# DuckDB that loads the artifact reports, or LOAD refuses it. Ask DuckDB rather
# than deriving it from uname, so the two cannot disagree; fall back to uname
# when the CLI is absent, and let a cross-compile override it outright.
CLI_PLATFORM := $(shell command -v $(DUCKDB) >/dev/null 2>&1 && $(DUCKDB) -init /dev/null -noheader -csv -c "SELECT platform FROM pragma_platform();" 2>/dev/null)
DUCKDB_PLATFORM ?= $(if $(CLI_PLATFORM),$(CLI_PLATFORM),$(UNAME_PLATFORM))

BUILD_DIR := build
EXTENSION := $(BUILD_DIR)/$(EXTENSION_NAME).duckdb_extension

.PHONY: all check build extension test test-sql no-network fmt clippy clean

all: extension

## Everything CI runs, in the order it runs it.
check: fmt clippy test extension no-network test-sql

build:
	cargo build --workspace --release

## The packaged artifact: the cdylib plus the DuckDB metadata trailer. Without
## the trailer DuckDB rejects the file rather than loading it, so this is the
## step that makes the build testable at all.
extension: build
	python3 scripts/append_extension_metadata.py \
		--library-file target/release/$(LIB_FILE) \
		--out-file $(EXTENSION) \
		--platform $(DUCKDB_PLATFORM) \
		--duckdb-version $(TARGET_DUCKDB_VERSION) \
		--extension-version $(EXTENSION_VERSION)

## Rust unit tests: the model, the cache and the embedding contract.
test:
	cargo test --workspace

## AC2's other half: nothing that can open a socket is compiled into the
## artifact. The self-test runs first, so a checker that has gone blind is
## caught before its clean report is believed.
no-network: extension
	python3 scripts/check_no_network_deps.py --self-test
	python3 scripts/check_no_network_deps.py --artifact $(EXTENSION)

## SQL tests: LOAD the packaged artifact into a real DuckDB and query it.
## Needs the duckdb CLI on PATH.
test-sql: extension
	python3 scripts/run_sql_tests.py --extension $(EXTENSION) --duckdb $(DUCKDB)

fmt:
	cargo fmt --all -- --check

clippy:
	cargo clippy --workspace --all-targets -- -D warnings

clean:
	cargo clean
	rm -rf $(BUILD_DIR)
