//! The `staticembed` DuckDB extension.
//!
//! The whole registered surface is four functions:
//!
//! | function | returns | why it exists |
//! |---|---|---|
//! | `embed(text VARCHAR)` | `FLOAT[]` | the product: one row in, one vector out |
//! | `staticembed_version()` | `VARCHAR` | which build, which model, which width |
//! | `staticembed_cache_stats()` | `STRUCT(hits, misses, encoded, entries, capacity)` | makes "did it re-embed?" answerable in SQL |
//! | `staticembed_cache_clear()` | `BIGINT` | vectors dropped; lets a session start from a known state |
//!
//! `embed` is a **scalar**, and that is the product argument rather than an
//! implementation detail: a scalar composes with `WHERE` and `LIMIT`, so a
//! filtered subset of a table can be embedded without materialising the rest,
//! and it behaves the same over a local Parquet file as over a remote table.
//!
//! There is deliberately no similarity or nearest-neighbour function here. The
//! measured position of this model is that a map built from its vectors keeps
//! the cluster structure and does not keep the neighbourhoods, so shipping a
//! "rows most like this one" surface would promise something the model does not
//! deliver. `README.md` says the same thing to the person reading the page.
//!
//! # NULL
//!
//! `embed(NULL)` is `NULL`. Text with no in-vocabulary tokens — `''`,
//! whitespace, a string of symbols outside the vocabulary — is a zero vector of
//! full width, because that is what the mean over zero tokens is. The two cases
//! are different on purpose: absence of a value is not the same as a value that
//! carries no signal. A caller that wants the pipeline's single behaviour for
//! both writes `embed(coalesce(t, ''))`.

use std::error::Error;
use std::ffi::CString;
use std::sync::Arc;

use duckdb::Result;
use duckdb::core::{DataChunkHandle, Inserter, LogicalTypeHandle, LogicalTypeId};
use duckdb::types::DuckString;
use duckdb::vscalar::{ScalarFunctionSignature, VScalar};
use duckdb::vtab::arrow::WritableVector;
use libduckdb_sys::duckdb_string_t;

/// The stable C API floor this artifact declares, so one build stays loadable
/// across DuckDB 1.2 and later.
const MIN_DUCKDB_VERSION: &str = "v1.2.0";

/// One input cell: SQL NULL, or the text of the row.
enum Cell<'a> {
    Null,
    Text(std::borrow::Cow<'a, str>),
}

/// Read column `col` of `input` as VARCHAR, distinguishing NULL from empty.
fn read_varchar_column(input: &mut DataChunkHandle, col: usize) -> Vec<Cell<'static>> {
    let rows = input.len();
    let vector = input.flat_vector(col);
    // SAFETY: `rows` is the row count DuckDB supplied for this invocation, and
    // the column is VARCHAR by the registered signature, so its physical
    // storage is `duckdb_string_t`.
    let values = unsafe { vector.as_slice_with_len::<duckdb_string_t>(rows) };
    let mut cells = Vec::with_capacity(rows);
    for (row, value) in values.iter().enumerate().take(rows) {
        if vector.row_is_null(row as u64) {
            cells.push(Cell::Null);
        } else {
            let mut owned = *value;
            let text = DuckString::new(&mut owned).as_str().into_owned();
            cells.push(Cell::Text(std::borrow::Cow::Owned(text)));
        }
    }
    cells
}

/// Write one `FLOAT[]` per row, with NULL rows left null.
fn write_float_lists(
    output: &mut dyn WritableVector,
    rows: &[Option<Arc<[f32]>>],
) -> Result<(), Box<dyn Error>> {
    let total: usize = rows.iter().flatten().map(|v| v.len()).sum();
    let mut list = output.list_vector();
    list.try_set_len(total)?;

    if total > 0 {
        let mut child = list.child(total);
        // SAFETY: `total` floats were just reserved in the child vector, and the
        // child is FLOAT by the registered return type.
        let slice = unsafe { child.as_mut_slice_with_len::<f32>(total) };
        let mut offset = 0usize;
        for vector in rows.iter().flatten() {
            slice[offset..offset + vector.len()].copy_from_slice(vector);
            offset += vector.len();
        }
    }

    let mut offset = 0usize;
    for (row, entry) in rows.iter().enumerate() {
        match entry {
            Some(vector) => {
                list.set_entry(row, offset, vector.len());
                offset += vector.len();
            }
            None => {
                // A null list entry still needs a well-formed (offset, length),
                // or a consumer that ignores validity reads a stale entry.
                list.set_entry(row, offset, 0);
                list.set_null(row);
            }
        }
    }
    Ok(())
}

/// `embed(text VARCHAR) → FLOAT[]`
///
/// The vector for one string, from the model bundled in this binary. No network
/// call, no API key, no configuration: the model is parsed out of the binary on
/// the first call and reused for the rest of the session.
struct Embed;

impl VScalar for Embed {
    type State = ();

    fn invoke(
        _state: &Self::State,
        input: &mut DataChunkHandle,
        output: &mut dyn WritableVector,
    ) -> Result<(), Box<dyn Error>> {
        let cells = read_varchar_column(input, 0);
        let mut vectors: Vec<Option<Arc<[f32]>>> = Vec::with_capacity(cells.len());
        for cell in &cells {
            match cell {
                Cell::Null => vectors.push(None),
                Cell::Text(text) => vectors.push(Some(staticembed_core::embed(text)?)),
            }
        }
        write_float_lists(output, &vectors)
    }

    fn signatures() -> Vec<ScalarFunctionSignature> {
        vec![ScalarFunctionSignature::exact(
            vec![LogicalTypeHandle::from(LogicalTypeId::Varchar)],
            LogicalTypeHandle::list(&LogicalTypeHandle::from(LogicalTypeId::Float)),
        )]
    }
}

/// `staticembed_version() → VARCHAR`
struct Version;

impl VScalar for Version {
    type State = ();

    fn invoke(
        _state: &Self::State,
        input: &mut DataChunkHandle,
        output: &mut dyn WritableVector,
    ) -> Result<(), Box<dyn Error>> {
        let text = CString::new(staticembed_core::describe())?;
        let vector = output.flat_vector();
        for row in 0..input.len().max(1) {
            vector.insert(row, text.clone());
        }
        Ok(())
    }

    fn signatures() -> Vec<ScalarFunctionSignature> {
        vec![ScalarFunctionSignature::exact(
            vec![],
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
        )]
    }
}

/// `staticembed_cache_stats() → STRUCT(hits, misses, encoded, entries, capacity)`
///
/// `encoded` is the number of times the encoder has actually run since the last
/// `staticembed_cache_clear()`. It is the observable behind "a repeated query
/// does not re-embed": run a query twice and `encoded` does not move.
struct CacheStats;

/// Field order of the STRUCT `staticembed_cache_stats()` returns. The order is
/// part of the signature, so it is written once and used for both the type and
/// the write.
const STATS_FIELDS: [&str; 5] = ["hits", "misses", "encoded", "entries", "capacity"];

impl VScalar for CacheStats {
    type State = ();

    fn invoke(
        _state: &Self::State,
        input: &mut DataChunkHandle,
        output: &mut dyn WritableVector,
    ) -> Result<(), Box<dyn Error>> {
        let stats = staticembed_core::stats();
        let values = [
            stats.hits,
            stats.misses,
            stats.encoded,
            stats.entries,
            stats.capacity,
        ];
        let rows = input.len().max(1);
        let parent = output.struct_vector();
        for (index, value) in values.iter().enumerate() {
            let mut child = parent.child(index, rows);
            // SAFETY: `rows` slots were reserved above and the child is BIGINT
            // by the registered return type.
            let slice = unsafe { child.as_mut_slice_with_len::<i64>(rows) };
            for slot in slice.iter_mut() {
                *slot = *value as i64;
            }
        }
        Ok(())
    }

    fn signatures() -> Vec<ScalarFunctionSignature> {
        let bigint = LogicalTypeHandle::from(LogicalTypeId::Bigint);
        let fields: Vec<(&str, LogicalTypeHandle)> = STATS_FIELDS
            .iter()
            .map(|name| (*name, LogicalTypeHandle::from(LogicalTypeId::Bigint)))
            .collect();
        drop(bigint);
        vec![ScalarFunctionSignature::exact(
            vec![],
            LogicalTypeHandle::struct_type(&fields),
        )]
    }

    /// Without this DuckDB folds a zero-argument scalar to a constant and the
    /// counters would be read once and reused for the rest of the session.
    fn volatile() -> bool {
        true
    }
}

/// `staticembed_cache_clear() → BIGINT` — vectors dropped.
struct CacheClear;

impl VScalar for CacheClear {
    type State = ();

    fn invoke(
        _state: &Self::State,
        input: &mut DataChunkHandle,
        output: &mut dyn WritableVector,
    ) -> Result<(), Box<dyn Error>> {
        let dropped = staticembed_core::clear_cache() as i64;
        let rows = input.len().max(1);
        let mut vector = output.flat_vector();
        // SAFETY: the output vector holds at least `rows` BIGINT slots by the
        // registered return type and DuckDB's chunk size.
        let slice = unsafe { vector.as_mut_slice_with_len::<i64>(rows) };
        for slot in slice.iter_mut() {
            *slot = dropped;
        }
        Ok(())
    }

    fn signatures() -> Vec<ScalarFunctionSignature> {
        vec![ScalarFunctionSignature::exact(
            vec![],
            LogicalTypeHandle::from(LogicalTypeId::Bigint),
        )]
    }

    fn volatile() -> bool {
        true
    }
}

/// Register the whole surface.
///
/// # Safety
/// The connection must be valid for the lifetime of the extension.
pub unsafe fn extension_entrypoint(con: duckdb::Connection) -> Result<(), Box<dyn Error>> {
    con.register_scalar_function::<Embed>("embed")?;
    con.register_scalar_function::<Version>("staticembed_version")?;
    con.register_scalar_function::<CacheStats>("staticembed_cache_stats")?;
    con.register_scalar_function::<CacheClear>("staticembed_cache_clear")?;
    Ok(())
}

/// The load path.
///
/// # Safety
/// Called by DuckDB at load with a live extension-info handle and access table.
pub unsafe fn init_extension(
    info: duckdb::ffi::duckdb_extension_info,
    access: *const duckdb::ffi::duckdb_extension_access,
) -> std::result::Result<bool, Box<dyn Error>> {
    let have_api_struct = duckdb::ffi::duckdb_rs_extension_api_init(info, access, MIN_DUCKDB_VERSION)?;
    if !have_api_struct {
        // The API version did not match; DuckDB already knows why.
        return Ok(false);
    }

    let get_database = (*access)
        .get_database
        .ok_or("get_database function pointer is null in duckdb_extension_access")?;
    let db_ptr = get_database(info);
    if db_ptr.is_null() {
        return Ok(false);
    }
    let db: duckdb::ffi::duckdb_database = *db_ptr;

    let connection = duckdb::Connection::open_from_raw(db.cast())?;
    extension_entrypoint(connection)?;
    Ok(true)
}

/// # Safety
/// The symbol DuckDB calls when loading the extension.
#[no_mangle]
pub unsafe extern "C" fn staticembed_init_c_api(
    info: duckdb::ffi::duckdb_extension_info,
    access: *const duckdb::ffi::duckdb_extension_access,
) -> bool {
    match init_extension(info, access) {
        Ok(loaded) => loaded,
        Err(error) => {
            if let Some(set_error) = (*access).set_error {
                match CString::new(error.to_string()) {
                    Ok(message) => set_error(info, message.as_ptr()),
                    Err(_) => set_error(
                        info,
                        c"staticembed failed to load, and the reason could not be converted to a C string"
                            .as_ptr(),
                    ),
                }
            }
            false
        }
    }
}
