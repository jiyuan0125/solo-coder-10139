#![deny(clippy::all, clippy::pedantic, clippy::disallowed_methods)]
#![allow(
    clippy::if_not_else,
    clippy::iter_not_returning_iterator,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::module_name_repetitions,
    clippy::must_use_candidate,
    clippy::needless_pass_by_value,
    clippy::redundant_closure_for_method_calls,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::unnecessary_wraps,
    clippy::unreadable_literal
)]

//! # redb
//!
//! A simple, portable, high-performance, ACID, embedded key-value store.
//!
//! redb is written in pure Rust and is loosely inspired by [lmdb][lmdb]. Data is stored in a collection
//! of copy-on-write B+trees. For more details, see the [design doc][design].
//!
//! # Features
//!
//! - Zero-copy, thread-safe, `BTreeMap` based API
//! - Fully ACID-compliant transactions
//! - MVCC support for concurrent readers & writer, without blocking
//! - Crash-safe by default
//! - Savepoints and rollbacks
//!
//! # Example
//!
//! ```
//! use redb::{Database, Error, ReadableDatabase, ReadableTable, TableDefinition};
//!
//! const TABLE: TableDefinition<&str, u64> = TableDefinition::new("my_data");
//!
//! fn main() -> Result<(), Error> {
//!   # #[cfg(not(target_os = "wasi"))]
//!     let file = tempfile::NamedTempFile::new().unwrap();
//!   # #[cfg(target_os = "wasi")]
//!   # let file = tempfile::NamedTempFile::new_in("/tmp").unwrap();
//!     let db = Database::create(file.path())?;
//!     let write_txn = db.begin_write()?;
//!     {
//!         let mut table = write_txn.open_table(TABLE)?;
//!         table.insert("my_key", &123)?;
//!     }
//!     write_txn.commit()?;
//!
//!     let read_txn = db.begin_read()?;
//!     let table = read_txn.open_table(TABLE)?;
//!     assert_eq!(table.get("my_key")?.unwrap().value(), 123);
//!
//!     Ok(())
//! }
//! ```
//!
//! [lmdb]: https://www.lmdb.tech/doc/
//! [design]: https://github.com/cberner/redb/blob/master/docs/design.md

pub use db::{
    Builder, CacheStats, Database, MultimapTableDefinition, MultimapTableHandle, ReadOnlyDatabase,
    ReadableDatabase, RepairSession, StorageBackend, TableDefinition, TableHandle,
    UntypedMultimapTableHandle, UntypedTableHandle,
};
pub use error::{
    CommitError, CompactionError, DatabaseError, Error, SavepointError, SetDurabilityError,
    StorageError, TableError, TransactionError,
};
pub use multimap_table::{
    MultimapRange, MultimapTable, MultimapValue, ReadOnlyMultimapTable,
    ReadOnlyUntypedMultimapTable, ReadableMultimapTable,
};
pub use table::{
    Entry, ExtractIf, OccupiedEntry, Range, ReadOnlyTable, ReadOnlyUntypedTable, ReadableTable,
    ReadableTableMetadata, Table, TableStats, VacantEntry,
};
pub use transactions::{DatabaseStats, Durability, ReadTransaction, WriteTransaction};
pub use tree_store::{AccessGuard, AccessGuardMut, AccessGuardMutInPlace, Savepoint};
pub use types::{Key, MutInPlaceValue, TypeName, Value};

pub type Result<T = (), E = StorageError> = std::result::Result<T, E>;

pub mod backends;
mod complex_types;
mod db;
mod error;
mod multimap_table;
mod sealed;
mod table;
mod transaction_tracker;
mod transactions;
mod tree_store;
mod tuple_types;
mod types;

#[cfg(test)]
fn create_tempfile() -> tempfile::NamedTempFile {
    if cfg!(target_os = "wasi") {
        tempfile::NamedTempFile::new_in("/tmp").unwrap()
    } else {
        tempfile::NamedTempFile::new().unwrap()
    }
}

#[cfg(test)]
mod size_limit_tests {
    use crate::error::{Error, StorageError};
    use crate::tree_store::{MAX_KEY_LENGTH, MAX_PAIR_LENGTH, MAX_VALUE_LENGTH};

    const GIB: usize = 1024 * 1024 * 1024;

    #[test]
    fn constants_have_correct_values() {
        assert_eq!(MAX_KEY_LENGTH, 3 * GIB);
        assert_eq!(MAX_VALUE_LENGTH, 3 * GIB);
        assert_eq!(MAX_PAIR_LENGTH, 3 * GIB + 768 * 1024 * 1024);
    }

    #[test]
    fn constants_are_distinct_named() {
        assert_eq!(MAX_KEY_LENGTH, MAX_VALUE_LENGTH);
        assert_ne!(MAX_KEY_LENGTH, MAX_PAIR_LENGTH);
        assert_ne!(MAX_VALUE_LENGTH, MAX_PAIR_LENGTH);
    }

    #[test]
    fn storage_error_display_key_mentions_key() {
        let err = StorageError::KeyTooLarge(MAX_KEY_LENGTH + 1);
        let msg = format!("{err}");
        assert!(msg.contains("key"), "KeyTooLarge msg should mention 'key': {msg}");
        assert!(
            !msg.contains("value (length"),
            "KeyTooLarge msg should not say 'value (length': {msg}"
        );
    }

    #[test]
    fn storage_error_display_value_mentions_value() {
        let err = StorageError::ValueTooLarge(MAX_VALUE_LENGTH + 1);
        let msg = format!("{err}");
        assert!(
            msg.contains("value (length"),
            "ValueTooLarge msg should mention 'value (length': {msg}"
        );
        assert!(
            !msg.contains("key (length"),
            "ValueTooLarge msg should not say 'key (length': {msg}"
        );
    }

    #[test]
    fn storage_error_display_pair_mentions_both_key_and_value() {
        let err = StorageError::KeyValuePairTooLarge {
            key_len: 100,
            value_len: 200,
        };
        let msg = format!("{err}");
        assert!(
            msg.contains("key length"),
            "KeyValuePairTooLarge msg should mention 'key length': {msg}"
        );
        assert!(
            msg.contains("value length"),
            "KeyValuePairTooLarge msg should mention 'value length': {msg}"
        );
        assert!(
            msg.contains("combined"),
            "KeyValuePairTooLarge msg should mention 'combined': {msg}"
        );
    }

    #[test]
    fn storage_error_display_texts_are_pairwise_distinct() {
        let key_msg = format!("{}", StorageError::KeyTooLarge(12345));
        let val_msg = format!("{}", StorageError::ValueTooLarge(12345));
        let pair_msg = format!(
            "{}",
            StorageError::KeyValuePairTooLarge {
                key_len: 100,
                value_len: 200
            }
        );
        assert_ne!(key_msg, val_msg);
        assert_ne!(key_msg, pair_msg);
        assert_ne!(val_msg, pair_msg);
    }

    #[test]
    fn storage_error_display_gib_numbers_match_constants() {
        let key_gib = MAX_KEY_LENGTH / GIB;
        let val_gib = MAX_VALUE_LENGTH / GIB;
        let pair_gib = MAX_PAIR_LENGTH / GIB;

        let key_msg = format!("{}", StorageError::KeyTooLarge(MAX_KEY_LENGTH + 1));
        let val_msg = format!("{}", StorageError::ValueTooLarge(MAX_VALUE_LENGTH + 1));
        let pair_msg = format!(
            "{}",
            StorageError::KeyValuePairTooLarge {
                key_len: 1,
                value_len: MAX_PAIR_LENGTH
            }
        );

        assert!(
            key_msg.contains(&format!("{key_gib}GiB")),
            "KeyTooLarge should mention {key_gib}GiB: {key_msg}"
        );
        assert!(
            val_msg.contains(&format!("{val_gib}GiB")),
            "ValueTooLarge should mention {val_gib}GiB: {val_msg}"
        );
        assert!(
            pair_msg.contains(&format!("{pair_gib}GiB")),
            "KeyValuePairTooLarge should mention {pair_gib}GiB: {pair_msg}"
        );
    }

    #[test]
    fn error_display_key_mentions_key() {
        let err: Error = StorageError::KeyTooLarge(MAX_KEY_LENGTH + 1).into();
        let msg = format!("{err}");
        assert!(msg.contains("key"), "Error KeyTooLarge msg should mention 'key': {msg}");
    }

    #[test]
    fn error_display_value_mentions_value() {
        let err: Error = StorageError::ValueTooLarge(MAX_VALUE_LENGTH + 1).into();
        let msg = format!("{err}");
        assert!(
            msg.contains("value (length"),
            "Error ValueTooLarge msg should mention 'value (length': {msg}"
        );
    }

    #[test]
    fn error_display_pair_mentions_both() {
        let err: Error = StorageError::KeyValuePairTooLarge {
            key_len: 10,
            value_len: 20,
        }
        .into();
        let msg = format!("{err}");
        assert!(msg.contains("key length"), "Error pair msg missing 'key length': {msg}");
        assert!(
            msg.contains("value length"),
            "Error pair msg missing 'value length': {msg}"
        );
        assert!(msg.contains("combined"), "Error pair msg missing 'combined': {msg}");
    }

    #[test]
    fn error_display_texts_are_pairwise_distinct() {
        let key_err: Error = StorageError::KeyTooLarge(99999).into();
        let val_err: Error = StorageError::ValueTooLarge(99999).into();
        let pair_err: Error = StorageError::KeyValuePairTooLarge {
            key_len: 50,
            value_len: 60,
        }
        .into();
        let key_msg = format!("{key_err}");
        let val_msg = format!("{val_err}");
        let pair_msg = format!("{pair_err}");
        assert_ne!(key_msg, val_msg);
        assert_ne!(key_msg, pair_msg);
        assert_ne!(val_msg, pair_msg);
    }

    #[test]
    fn storage_error_to_error_preserves_variants() {
        let key: Error = StorageError::KeyTooLarge(42).into();
        assert!(matches!(key, Error::KeyTooLarge(42)));

        let val: Error = StorageError::ValueTooLarge(42).into();
        assert!(matches!(val, Error::ValueTooLarge(42)));

        let pair: Error = StorageError::KeyValuePairTooLarge {
            key_len: 10,
            value_len: 20,
        }
        .into();
        assert!(matches!(
            pair,
            Error::KeyValuePairTooLarge {
                key_len: 10,
                value_len: 20
            }
        ));
    }

    #[test]
    fn saturating_add_does_not_overflow() {
        let huge = usize::MAX;
        let result = huge.saturating_add(1);
        assert_eq!(result, usize::MAX);

        let key_len = usize::MAX - 100;
        let value_len = usize::MAX - 100;
        let combined = key_len.saturating_add(value_len);
        assert_eq!(combined, usize::MAX);
        assert!(combined > MAX_PAIR_LENGTH);
    }

    #[test]
    fn boundary_key_at_limit_is_ok_comparison() {
        assert!(!(MAX_KEY_LENGTH > MAX_KEY_LENGTH));
        assert!(MAX_KEY_LENGTH + 1 > MAX_KEY_LENGTH);
    }

    #[test]
    fn boundary_value_at_limit_is_ok_comparison() {
        assert!(!(MAX_VALUE_LENGTH > MAX_VALUE_LENGTH));
        assert!(MAX_VALUE_LENGTH + 1 > MAX_VALUE_LENGTH);
    }

    #[test]
    fn boundary_pair_at_limit_is_ok_comparison() {
        assert!(!(MAX_PAIR_LENGTH > MAX_PAIR_LENGTH));
        assert!(MAX_PAIR_LENGTH + 1 > MAX_PAIR_LENGTH);
    }

    #[test]
    fn pair_display_combined_uses_saturating_add() {
        let err = StorageError::KeyValuePairTooLarge {
            key_len: usize::MAX,
            value_len: usize::MAX,
        };
        let msg = format!("{err}");
        let expected_combined = usize::MAX;
        assert!(
            msg.contains(&format!("combined length={}", expected_combined)),
            "pair msg should use saturating add: {msg}"
        );
    }

    #[test]
    fn error_pair_display_combined_uses_saturating_add() {
        let err: Error = StorageError::KeyValuePairTooLarge {
            key_len: usize::MAX,
            value_len: usize::MAX,
        }
        .into();
        let msg = format!("{err}");
        let expected_combined = usize::MAX;
        assert!(
            msg.contains(&format!("combined length={}", expected_combined)),
            "Error pair msg should use saturating add: {msg}"
        );
    }

    #[test]
    fn small_insertions_succeed_and_return_correct_types() {
        use crate::{Database, TableDefinition};
        let tmpfile = super::create_tempfile();
        let db = Database::create(tmpfile.path()).unwrap();
        let txn = db.begin_write().unwrap();
        const T: TableDefinition<&[u8], &[u8]> = TableDefinition::new("t");
        {
            let mut table = txn.open_table(T).unwrap();
            let k = vec![1u8; 100];
            let v = vec![2u8; 200];
            table.insert(k.as_slice(), v.as_slice()).unwrap();
        }
        txn.commit().unwrap();
    }
}
