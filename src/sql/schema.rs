use super::engine::Transaction;
use super::parser::format_ident;
use super::types::{DataType, Value};
use crate::error::{Error, Result};

use serde_derive::{Deserialize, Serialize};
use std::fmt::{self, Display};

/// The catalog stores schema information
pub trait Catalog {
    /// Creates a new table
    fn create_table(&mut self, table: Table) -> Result<()>;
    /// Deletes an existing table, or errors if it does not exist
    fn delete_table(&mut self, table: &str) -> Result<()>;
    /// Reads a table, if it exists
    fn read_table(&self, table: &str) -> Result<Option<Table>>;
    /// Iterates over all tables
    fn scan_tables(&self) -> Result<Tables>;

    /// Reads a table, and errors if it does not exist
    fn must_read_table(&self, table: &str) -> Result<Table> {
        self.read_table(table)?
            .ok_or_else(|| Error::Value(format!("Table {} does not exist", table)))
    }

    /// Returns all references to a table, as table,column pairs.
    fn table_references(&self, table: &str, with_self: bool) -> Result<Vec<(String, Vec<String>)>> {
        Ok(self
            .scan_tables()?
            .filter(|t| with_self || t.name != table)
            .map(|t| {
                (
                    t.name,
                    t.columns
                        .iter()
                        .filter(|c| c.references.as_deref() == Some(table))
                        .map(|c| c.name.clone())
                        .collect::<Vec<_>>(),
                )
            })
            .filter(|(_, cs)| !cs.is_empty())
            .collect())
    }
}

/// A table scan iterator
pub type Tables = Box<dyn DoubleEndedIterator<Item = Table> + Send>;

/// A table schema
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub struct Table {
    pub name: String,
    pub columns: Vec<Column>,
}

impl Table {
    /// Creates a new table schema
    pub fn new(name: String, columns: Vec<Column>) -> Result<Self> {
        let table = Self { name, columns };
        Ok(table)
    }

    /// Fetches a column by name
    pub fn get_column(&self, name: &str) -> Result<&Column> {
        self.columns.iter().find(|c| c.name == name).ok_or_else(|| {
            Error::Value(format!("Column {} not found in table {}", name, self.name))
        })
    }

    /// Fetches a column index by name
    pub fn get_column_index(&self, name: &str) -> Result<usize> {
        self.columns.iter().position(|c| c.name == name).ok_or_else(|| {
            Error::Value(format!("Column {} not found in table {}", name, self.name))
        })
    }

    /// Returns the primary key column of the table
    pub fn get_primary_key(&self) -> Result<&Column> {
        self.columns
            .iter()
            .find(|c| c.primary_key)
            .ok_or_else(|| Error::Value(format!("Primary key not found in table {}", self.name)))
    }

    /// Returns the primary key value of a row
    pub fn get_row_key(&self, row: &[Value]) -> Result<Value> {
        row.get(
            self.columns
                .iter()
                .position(|c| c.primary_key)
                .ok_or_else(|| Error::Value("Primary key not found".into()))?,
        )
        .cloned()
        .ok_or_else(|| Error::Value("Primary key value not found for row".into()))
    }

    /// Validates the table schema
    pub fn validate(&self, txn: &mut dyn Transaction) -> Result<()> {
        if self.columns.is_empty() {
            return Err(Error::Value(format!("Table {} has no columns", self.name)));
        }
        match self.columns.iter().filter(|c| c.primary_key).count() {
            1 => {}
            0 => return Err(Error::Value(format!("No primary key in table {}", self.name))),
            _ => return Err(Error::Value(format!("Multiple primary keys in table {}", self.name))),
        };
        for column in &self.columns {
            column.validate(self, txn)?;
        }
        Ok(())
    }

    /// Validates a row
    pub fn validate_row(&self, row: &[Value], txn: &mut dyn Transaction) -> Result<()> {
        if row.len() != self.columns.len() {
            return Err(Error::Value(format!("Invalid row size for table {}", self.name)));
        }
        let pk = self.get_row_key(row)?;
        for (column, value) in self.columns.iter().zip(row.iter()) {
            column.validate_value(self, &pk, value, txn)?;
        }
        Ok(())
    }
}

impl Display for Table {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CREATE TABLE {} (\n{}\n)",
            format_ident(&self.name),
            self.columns.iter().map(|c| format!("  {}", c)).collect::<Vec<String>>().join(",\n")
        )
    }
}

/// A table column schema
///
/// 定义为 meta 中的 column.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub struct Column {
    /// Column name
    pub name: String,
    /// Column datatype
    pub datatype: DataType,
    /// Whether the column is a primary key
    pub primary_key: bool,
    /// Whether the column allows null values
    pub nullable: bool,
    /// The default value of the column
    pub default: Option<Value>,
    /// Whether the column should only take unique values
    pub unique: bool,
    /// The table which is referenced by this foreign key
    pub references: Option<String>,
    /// Whether the column should be indexed
    pub index: bool,
}

impl Column {
    /// Validates the column schema
    pub fn validate(&self, table: &Table, txn: &mut dyn Transaction) -> Result<()> {
        // Validate primary key
        if self.primary_key && self.nullable {
            return Err(Error::Value(format!("Primary key {} cannot be nullable", self.name)));
        }
        if self.primary_key && !self.unique {
            return Err(Error::Value(format!("Primary key {} must be unique", self.name)));
        }

        // Validate default value
        if let Some(default) = &self.default {
            if let Some(datatype) = default.datatype() {
                if datatype != self.datatype {
                    return Err(Error::Value(format!(
                        "Default value for column {} has datatype {}, must be {}",
                        self.name, datatype, self.datatype
                    )));
                }
            } else if !self.nullable {
                return Err(Error::Value(format!(
                    "Can't use NULL as default value for non-nullable column {}",
                    self.name
                )));
            }
        } else if self.nullable {
            return Err(Error::Value(format!(
                "Nullable column {} must have a default value",
                self.name
            )));
        }

        // Validate references
        if let Some(reference) = &self.references {
            let target = if reference == &table.name {
                table.clone()
            } else if let Some(table) = txn.read_table(reference)? {
                table
            } else {
                return Err(Error::Value(format!(
                    "Table {} referenced by column {} does not exist",
                    reference, self.name
                )));
            };
            if self.datatype != target.get_primary_key()?.datatype {
                return Err(Error::Value(format!(
                    "Can't reference {} primary key of table {} from {} column {}",
                    target.get_primary_key()?.datatype,
                    target.name,
                    self.datatype,
                    self.name
                )));
            }
        }

        Ok(())
    }

    /// Validates a column value
    ///
    /// Column 对输入每列的值做检查的时候使用. self 是 table schema, value 是写入数据, pk 是数据对应的 row.
    pub fn validate_value(
        &self,
        table: &Table,
        pk: &Value,
        value: &Value,
        txn: &mut dyn Transaction,
    ) -> Result<()> {
        // Validate datatype
        match value.datatype() {
            None if self.nullable => Ok(()),
            None => Err(Error::Value(format!("NULL value not allowed for column {}", self.name))),
            // 这个地方没有 cast, 直接做检查.
            Some(ref datatype) if datatype != &self.datatype => Err(Error::Value(format!(
                "Invalid datatype {} for {} column {}",
                datatype, self.datatype, self.name
            ))),
            _ => Ok(()),
        }?;

        // Validate value
        match value {
            Value::String(s) if s.len() > 1024 => {
                Err(Error::Value("Strings cannot be more than 1024 bytes".into()))
            }
            _ => Ok(()),
        }?;

        // Validate outgoing references
        // 寻找 foreign key.
        if let Some(target) = &self.references {
            // 看看 value 是什么类型, 做一些 dyn 分发.
            match value {
                // NULL 的话, 啥都不干
                Value::Null => Ok(()),
                Value::Float(f) if f.is_nan() => Ok(()),
                // TODO(maple): 这个地方没搞懂, target == &table.name 是什么意思, self-ref?
                v if target == &table.name && v == pk => Ok(()),
                // txn 读取 v, 读取 (Column, value). 如果是 none 即错误.
                v if txn.read(target, v)?.is_none() => Err(Error::Value(format!(
                    "Referenced primary key {} in table {} does not exist",
                    v, target,
                ))),
                _ => Ok(()),
            }?;
        }

        // Validate uniqueness constraints
        //
        // 检查非 unique 的内容
        if self.unique && !self.primary_key && value != &Value::Null {
            let index = table.get_column_index(&self.name)?;
            // 我日, 走了一个 full scan?
            // TODO(maple): maybe add an expression for value checking.
            let mut scan = txn.scan(&table.name, None)?;
            while let Some(row) = scan.next().transpose()? {
                if row.get(index).unwrap_or(&Value::Null) == value
                    && &table.get_row_key(&row)? != pk
                {
                    return Err(Error::Value(format!(
                        "Unique value {} already exists for column {}",
                        value, self.name
                    )));
                }
            }
        }

        Ok(())
    }
}

impl Display for Column {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut sql = format_ident(&self.name);
        sql += &format!(" {}", self.datatype);
        if self.primary_key {
            sql += " PRIMARY KEY";
        }
        if !self.nullable && !self.primary_key {
            sql += " NOT NULL";
        }
        if let Some(default) = &self.default {
            sql += &format!(" DEFAULT {}", default);
        }
        if self.unique && !self.primary_key {
            sql += " UNIQUE";
        }
        if let Some(reference) = &self.references {
            sql += &format!(" REFERENCES {}", reference);
        }
        if self.index {
            sql += " INDEX";
        }
        write!(f, "{}", sql)
    }
}
