pub mod engine;
pub mod execution;
pub mod parser;
pub mod plan;
pub mod schema;
pub mod types;

#[cfg(test)]
mod test {
    use crate::sql;
    use crate::sql::engine::kv::Transaction;
    use crate::sql::engine::Transaction as SQLEngineTransactionTrait;
    use crate::sql::plan;
    use crate::sql::schema::{Catalog, Column, Table};
    use crate::sql::types::Value;
    use crate::storage::kv::mvcc::Transaction as MvccEngineTransaction;
    use crate::storage::kv::{Test as KvTest, MVCC};
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;

    fn mock_catalog() -> Box<Transaction> {
        let test_kv_engine = KvTest::new();
        let mvcc_engine = MVCC::new(Box::new(test_kv_engine));
        let engine_txn = mvcc_engine.begin().unwrap();
        let txn = Transaction::new(engine_txn);
        Box::new(txn)
    }

    #[test]
    fn test_basic_sql() {
        // "SELECT a, b FROM table WHERE a > 10" will failed, because table is a keyword.
        let mut p =
            parser::Parser::new("SELECT sum(a), b FROM t WHERE a > 0 GROUP BY b HAVING b > 10");
        let statement_result = p.parse();
        println!("{:?}", statement_result);
        // cast to Statement
        let statement = statement_result.unwrap();

        let mut mocked_catalog_engine = mock_catalog();
        let column_vec = vec![
            schema::Column {
                name: "a".into(),
                datatype: sql::types::DataType::Integer,
                primary_key: true,
                nullable: false,
                default: None,
                unique: true,
                index: false,
                references: None,
            },
            schema::Column {
                name: "b".into(),
                datatype: sql::types::DataType::String,
                primary_key: false,
                nullable: false,
                default: None,
                unique: false,
                index: false,
                references: None,
            },
        ];
        let table = Table::new("t".to_string(), column_vec).unwrap();
        mocked_catalog_engine.create_table(table).unwrap();

        let mut mock_planner = plan::Planner::new(&mut *mocked_catalog_engine);
        let planner_result = mock_planner.build(statement);

        println!("{:?}", planner_result);
    }
}
