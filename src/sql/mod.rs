pub mod engine;
pub mod execution;
pub mod parser;
pub mod plan;
pub mod schema;
pub mod types;

#[cfg(test)]
mod test {
    use super::*;
    use crate::sql;
    use crate::sql::engine::kv::Transaction;
    use crate::sql::engine::Transaction as SQLEngineTransactionTrait;
    use crate::sql::plan;
    use crate::sql::schema::{Catalog, Column, Table};
    use crate::sql::types::Value;
    use crate::storage::kv::mvcc::Transaction as MvccEngineTransaction;
    use crate::storage::kv::{Test as KvTest, MVCC};

    fn mock_catalog() -> Box<Transaction> {
        let test_kv_engine = KvTest::new();
        let mvcc_engine = MVCC::new(Box::new(test_kv_engine));
        let engine_txn = mvcc_engine.begin().unwrap();
        let txn = Transaction::new(engine_txn);
        Box::new(txn)
    }

    /// {table:t, schema: <a: int, b: string, c: float>, pk(a)}
    fn test_table1() -> Table {
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
            schema::Column {
                name: "c".into(),
                datatype: sql::types::DataType::Float,
                primary_key: false,
                nullable: false,
                default: None,
                unique: false,
                index: false,
                references: None,
            },
        ];
        Table::new("t".to_string(), column_vec).unwrap()
    }

    /// {table:f, schema: <a: int, k: string>, pk(k)}
    fn test_table2() -> Table {
        let column_vec = vec![
            schema::Column {
                name: "a".into(),
                datatype: sql::types::DataType::Integer,
                primary_key: false,
                nullable: false,
                default: None,
                unique: true,
                index: true,
                references: None,
            },
            schema::Column {
                name: "k".into(),
                datatype: sql::types::DataType::String,
                primary_key: true,
                nullable: false,
                default: None,
                unique: true,
                index: true,
                references: None,
            },
        ];
        Table::new("f".to_string(), column_vec).unwrap()
    }

    fn test_any_sql(sql: &str, txn: &mut Transaction) {
        // "SELECT a, b FROM table WHERE a > 10" will failed, because table is a keyword.
        let mut p = parser::Parser::new(sql);
        let statement_result = p.parse();
        println!("{:?}", statement_result);
        // cast to Statement
        let statement = statement_result.unwrap();

        let mut mock_planner = plan::Planner::new(&mut *txn);
        let planner_result = mock_planner.build(statement);

        // Print physical structure
        println!("{:?}", planner_result);
        let plan = planner_result.unwrap();
        // Print logical structure
        println!("{}", plan);

        let _ = plan.execute(txn);
    }

    #[test]
    fn test_literal() {
        let mut mocked_catalog_engine = mock_catalog();

        test_any_sql("SELECT 123, 12.3, NULL, 'mio', true, false", &mut mocked_catalog_engine);
    }

    #[test]
    fn test_literal_and_type() {
        let mut mocked_catalog_engine = mock_catalog();

        let table = test_table1();
        mocked_catalog_engine.create_table(table).unwrap();

        test_any_sql(
            "SELECT * from t WHERE c < 12 OR c = 15 OR a > 12.5",
            &mut mocked_catalog_engine,
        );
    }

    #[test]
    fn test_select_column() {
        let mut mocked_catalog_engine = mock_catalog();

        let table = test_table1();
        mocked_catalog_engine.create_table(table).unwrap();

        test_any_sql(
            "SELECT * from t WHERE c < 12 OR c = 15 OR a > 12.5",
            &mut mocked_catalog_engine,
        );
    }

    #[test]
    fn test_basic_sql_group_by() {
        let mut mocked_catalog_engine = mock_catalog();

        let table = test_table1();
        mocked_catalog_engine.create_table(table).unwrap();

        test_any_sql(
            "SELECT sum(a), b FROM t WHERE a > 0 GROUP BY b HAVING b > 10",
            &mut mocked_catalog_engine,
        );
    }

    #[test]
    fn test_basic_sql_group_by_expr() {
        let mut mocked_catalog_engine = mock_catalog();

        let table = test_table1();
        mocked_catalog_engine.create_table(table).unwrap();

        // "SELECT sum(a), b FROM t WHERE a > 0 GROUP BY (b + 2)" will failed
        test_any_sql(
            "SELECT sum(a), b + 2 FROM t WHERE a > 0 GROUP BY (b + 2)",
            &mut mocked_catalog_engine,
        );
    }

    #[test]
    fn test_basic_sql_group_by_expr2() {
        let mut mocked_catalog_engine = mock_catalog();

        let table = test_table1();
        mocked_catalog_engine.create_table(table).unwrap();

        test_any_sql(
            "SELECT COUNT(*) FROM t WHERE a > 0 GROUP BY b / 100",
            &mut mocked_catalog_engine,
        );
    }

    #[test]
    fn test_basic_sql_group_by_expr3() {
        let mut mocked_catalog_engine = mock_catalog();

        let table = test_table1();
        mocked_catalog_engine.create_table(table).unwrap();

        test_any_sql(
            "SELECT b as t_b, COUNT(a) FROM t WHERE a > 0 GROUP BY t_b HAVING COUNT(a) > COUNT(c)",
            &mut mocked_catalog_engine,
        );
    }

    #[test]
    fn test_basic_sql_join() {
        let mut mocked_catalog_engine = mock_catalog();

        let table = test_table1();
        mocked_catalog_engine.create_table(table).unwrap();

        let table = test_table2();
        mocked_catalog_engine.create_table(table).unwrap();

        // oops, seems toydb not supports using.
        // test_any_sql(
        //     "SELECT f.a, b, k FROM t JOIN f USING a",
        //     &mut mocked_catalog_engine,
        // );

        test_any_sql("SELECT t.a, b, k FROM t JOIN f ON t.a = f.a", &mut mocked_catalog_engine);
    }

    #[test]
    fn test_unqualified_name() {
        let mut mocked_catalog_engine = mock_catalog();

        let table = test_table1();
        mocked_catalog_engine.create_table(table).unwrap();

        // 我试了下, '...' 才能表述字符串字面量, SQL 语法是这样的吗？
        test_any_sql("SELECT a FROM t WHERE a > 0 and b = 'dodo'", &mut mocked_catalog_engine);
    }

    #[test]
    fn test_table_and_field_alias() {
        let mut mocked_catalog_engine = mock_catalog();

        let table = test_table1();
        mocked_catalog_engine.create_table(table).unwrap();

        test_any_sql(
            "SELECT t_alias.a * 2, (a - 114514) as a_alias FROM t as t_alias WHERE a > 0 and b = 'dodo'",
            &mut mocked_catalog_engine,
        );
    }

    /* subquery is not supported in toydb
    #[test]
    fn test_subquery() {
        let mut mocked_catalog_engine = mock_catalog();

        let table = test_table1();
        mocked_catalog_engine.create_table(table).unwrap();

        test_any_sql(
            "SELECT * FROM (SELECT t.a, b, k FROM t JOIN f ON t.a = f.a)",
            &mut mocked_catalog_engine,
        );
    }
    */
}
