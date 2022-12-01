use super::super::parser::ast;
use super::super::schema::{Catalog, Column, Table};
use super::super::types::{Expression, Value};
use super::{Aggregate, Direction, Node, Plan};
use crate::error::{Error, Result};

use log::debug;
use std::collections::{HashMap, HashSet};
use std::mem::replace;

/// A query plan builder.
pub struct Planner<'a, C: Catalog> {
    catalog: &'a mut C,
}

impl<'a, C: Catalog> Planner<'a, C> {
    /// Creates a new planner.
    pub fn new(catalog: &'a mut C) -> Self {
        Self { catalog }
    }

    /// Builds a plan for an AST statement.
    pub fn build(&mut self, statement: ast::Statement) -> Result<Plan> {
        Ok(Plan(self.build_statement(statement)?))
    }

    /// Builds a plan node for a statement.
    fn build_statement(&self, statement: ast::Statement) -> Result<Node> {
        Ok(match statement {
            // Transaction control and explain statements should have been handled by session.
            ast::Statement::Begin { .. } | ast::Statement::Commit | ast::Statement::Rollback => {
                return Err(Error::Internal(format!(
                    "Unexpected transaction statement {:?}",
                    statement
                )))
            }

            // 还没支持 EXPLAIN 呢.
            ast::Statement::Explain(_) => {
                return Err(Error::Internal("Unexpected explain statement".into()))
            }

            // DDL statements (schema changes).
            //
            // 直接 parse 出的结构就是 Column, 算是 AST 和内部一体化了.
            ast::Statement::CreateTable { name, columns } => Node::CreateTable {
                schema: Table::new(
                    name,
                    columns
                        .into_iter()
                        .map(|c| {
                            let nullable = c.nullable.unwrap_or(!c.primary_key);
                            let default = match c.default {
                                Some(expr) => Some(self.evaluate_constant(expr)?),
                                None if nullable => Some(Value::Null),
                                None => None,
                            };
                            Ok(Column {
                                name: c.name,
                                datatype: c.datatype,
                                primary_key: c.primary_key,
                                nullable,
                                default,
                                index: c.index && !c.primary_key,
                                unique: c.unique || c.primary_key,
                                references: c.references,
                            })
                        })
                        .collect::<Result<_>>()?,
                )?,
            },

            ast::Statement::DropTable(table) => Node::DropTable { table },

            // DML statements (mutations).
            ast::Statement::Delete { table, r#where } => {
                let scope = &mut Scope::from_table(self.catalog.must_read_table(&table)?)?;
                Node::Delete {
                    table: table.clone(),
                    source: Box::new(Node::Scan {
                        table,
                        alias: None,
                        filter: r#where.map(|e| self.build_expression(scope, e)).transpose()?,
                    }),
                }
            }

            // Insert 直接插入对应的表达式
            //
            // 然后递归从 build statements 到 build expression
            ast::Statement::Insert { table, columns, values } => Node::Insert {
                table,
                columns: columns.unwrap_or_else(Vec::new),
                expressions: values
                    .into_iter()
                    .map(|exprs| {
                        exprs
                            .into_iter()
                            .map(|expr| self.build_expression(&mut Scope::constant(), expr))
                            .collect::<Result<_>>()
                    })
                    .collect::<Result<_>>()?,
            },

            ast::Statement::Update { table, set, r#where } => {
                let scope = &mut Scope::from_table(self.catalog.must_read_table(&table)?)?;
                Node::Update {
                    table: table.clone(),
                    source: Box::new(Node::Scan {
                        table,
                        alias: None,
                        filter: r#where.map(|e| self.build_expression(scope, e)).transpose()?,
                    }),
                    expressions: set
                        .into_iter()
                        .map(|(c, e)| {
                            Ok((
                                scope.resolve(None, &c)?,
                                Some(c),
                                self.build_expression(scope, e)?,
                            ))
                        })
                        .collect::<Result<_>>()?,
                }
            }

            // Queries.
            //
            // Select 的结构比较复杂, 要当成多个部分处理.
            ast::Statement::Select {
                mut select,
                from,
                r#where,
                group_by,
                mut having,
                mut order,
                offset,
                limit,
            } => {
                // 产生一个空白 Scope 来做 binding
                let scope = &mut Scope::new();

                // Build FROM clause.
                //
                // 先 Build from 子句, 这里成分是 TableRef, 如果有多个相当于这些做 CrossJoin.
                // (这些是怎么丢给 optimizer 的?)
                let mut node = if !from.is_empty() {
                    self.build_from_clause(scope, from)?
                } else if select.is_empty() {
                    return Err(Error::Value("Can't select * without a table".into()));
                } else {
                    Node::Nothing
                };

                // Build WHERE clause.
                //
                // Where 会用到 from 的 naming, source 设置为上一个 node.
                if let Some(expr) = r#where {
                    node = Node::Filter {
                        source: Box::new(node),
                        predicate: self.build_expression(scope, expr)?,
                    };
                };

                // Build SELECT clause.
                //
                // 处理 select.
                let mut hidden = 0;
                // 如果 select 是 empty, 这里相当于全部输出.
                if !select.is_empty() {
                    // 处理 Select 之前, 选择哪些 columns 的问题

                    // Inject hidden SELECT columns for fields and aggregates used in ORDER BY and
                    // HAVING expressions but not present in existing SELECT output. These will be
                    // removed again by a later projection.
                    //
                    // 尝试插入 hidden.
                    if let Some(ref mut expr) = having {
                        hidden += self.inject_hidden(expr, &mut select)?;
                    }
                    for (expr, _) in order.iter_mut() {
                        hidden += self.inject_hidden(expr, &mut select)?;
                    }

                    // Extract any aggregate functions and GROUP BY expressions, replacing them with
                    // Column placeholders. Aggregations are handled by evaluating group expressions
                    // and aggregate function arguments in a pre-projection, passing the results
                    // to an aggregation node, and then evaluating the final SELECT expressions
                    // in the post-projection. For example:
                    //
                    // SELECT (MAX(rating * 100) - MIN(rating * 100)) / 100
                    // FROM movies
                    // GROUP BY released - 2000
                    //
                    // Results in the following nodes:
                    //
                    // - Projection: rating * 100, rating * 100, released - 2000
                    // - Aggregation: max(#0), min(#1) group by #2
                    // - Projection: (#0 - #1) / 100
                    //
                    // GROUP BY 会需要插入 having columns, 这个地方需要筛选出上面需要的列, 这里会转成:
                    // * GROUP BY 的列 + 上面需要的列(select 里面的)
                    // * Agg Operator
                    // * 上层再走个 Projection

                    // 这里先抽出 agg, 即 HAVING, ORDERING 和 SELECT 出的字段, 把 Agg(expr) 等转化为 expr on columnRef,
                    // 函数抽在 aggregates 中.
                    let aggregates = self.extract_aggregates(&mut select)?;
                    // 抽出 Group By 的 Columns.
                    let groups = self.extract_groups(&mut select, group_by, aggregates.len())?;
                    // 构建 Agg PlanNode, 这部分会抽掉 groups 和 aggregates
                    if !aggregates.is_empty() || !groups.is_empty() {
                        node = self.build_aggregation(scope, node, groups, aggregates)?;
                    }

                    // Build the remaining non-aggregate projection.
                    // 再嵌套一层 Projection, 这里会从 Group By 和 Agg 中抽出 naming.
                    let expressions: Vec<(Expression, Option<String>)> = select
                        .into_iter()
                        .map(|(e, l)| Ok((self.build_expression(scope, e)?, l)))
                        .collect::<Result<_>>()?;
                    scope.project(&expressions)?;
                    node = Node::Projection { source: Box::new(node), expressions };
                };

                // Build HAVING clause.
                if let Some(expr) = having {
                    node = Node::Filter {
                        source: Box::new(node),
                        predicate: self.build_expression(scope, expr)?,
                    };
                };

                // Build ORDER clause.
                if !order.is_empty() {
                    node = Node::Order {
                        source: Box::new(node),
                        orders: order
                            .into_iter()
                            .map(|(e, o)| {
                                Ok((
                                    self.build_expression(scope, e)?,
                                    match o {
                                        ast::Order::Ascending => Direction::Ascending,
                                        ast::Order::Descending => Direction::Descending,
                                    },
                                ))
                            })
                            .collect::<Result<_>>()?,
                    };
                }

                // Build OFFSET clause.
                if let Some(expr) = offset {
                    node = Node::Offset {
                        source: Box::new(node),
                        offset: match self.evaluate_constant(expr)? {
                            Value::Integer(i) if i >= 0 => Ok(i as u64),
                            v => Err(Error::Value(format!("Invalid offset {}", v))),
                        }?,
                    }
                }

                // Build LIMIT clause.
                // 插入一个 LIMIT 子句.
                if let Some(expr) = limit {
                    node = Node::Limit {
                        source: Box::new(node),
                        limit: match self.evaluate_constant(expr)? {
                            Value::Integer(i) if i >= 0 => Ok(i as u64),
                            v => Err(Error::Value(format!("Invalid limit {}", v))),
                        }?,
                    }
                }

                // Remove any hidden columns.
                // Hidden 应该不对外输出, 需要移除掉最终的冗余.
                if hidden > 0 {
                    node = Node::Projection {
                        source: Box::new(node),
                        expressions: (0..(scope.len() - hidden))
                            .map(|i| (Expression::Field(i, None), None))
                            .collect(),
                    }
                }

                debug!("node is {}", node);
                node
            }
        })
    }

    /// Builds a FROM clause consisting of several items. Each item is either a single table or a
    /// join of an arbitrary number of tables. All of the items are joined, since e.g. 'SELECT * FROM
    /// a, b' is an implicit join of a and b.
    ///
    /// (这个地方好像没考虑子查询) 从 FROM 中构建, 然后把所有子句构建成 NLJ(CrossJoin).
    /// Scope 等于多个 scope 做 merge
    fn build_from_clause(&self, scope: &mut Scope, from: Vec<ast::FromItem>) -> Result<Node> {
        // TODO(maple): 为什么每个子表都是去 base_scope 来构建?
        let base_scope = scope.clone();
        let mut items = from.into_iter();
        // 对第一个 from item 递归构建, 单独构建子表(子表之间应该没有 dependency?)
        // 子表的信息
        let mut node = match items.next() {
            Some(item) => self.build_from_item(scope, item)?,
            None => return Err(Error::Value("No from items given".into())),
        };
        for item in items {
            // 单独构建一个 scope, 然后构建右表 scope.
            let mut right_scope = base_scope.clone();
            let right = self.build_from_item(&mut right_scope, item)?;
            // 递归做一次 NLJ, 构成一个表, 然后 scope 去 merge 右表.
            node = Node::NestedLoopJoin {
                left: Box::new(node),
                left_size: scope.len(),
                right: Box::new(right),
                predicate: None,
                outer: false,
            };
            scope.merge(right_scope)?;
        }
        Ok(node)
    }

    /// Builds FROM items, which can either be a single table or a chained join of multiple tables,
    /// e.g. 'SELECT * FROM a LEFT JOIN b ON b.a_id = a.id'. Any tables will be stored in
    /// self.tables keyed by their query name (i.e. alias if given, otherwise name). The table can
    /// only be referenced by the query name (so if alias is given, cannot reference by name).
    ///
    /// 如果给了 alias, 那就不能 ref-by-name 了, 否则会根据 table name 来构建.
    ///
    /// 构建顺序: Table 直接拿 table 构建一组 name.
    /// Join:
    /// * 左 build, 这个时候字段应该等同于左侧的数量
    /// * 右 build, 这个时候字段数量等于左 + 右
    /// * build expr
    /// * 这里会把 right-join 转成 left-join, 然后插入一个 Projection
    fn build_from_item(&self, scope: &mut Scope, item: ast::FromItem) -> Result<Node> {
        Ok(match item {
            ast::FromItem::Table { name, alias } => {
                // 这个表是根据名字 -- 映射来构建的, 然后还会尝试构建一个 alias
                scope.add_table(
                    alias.clone().unwrap_or_else(|| name.clone()),
                    self.catalog.must_read_table(&name)?,
                )?;
                // 这里直接产生的是一个 TableScan 节点.
                Node::Scan { table: name, alias, filter: None }
            }

            ast::FromItem::Join { left, right, r#type, predicate } => {
                // Right outer joins are built as a left outer join with an additional projection
                // to swap the resulting columns.
                //
                // Right Join 强行处理成一致的内容.
                let (left, right) = match r#type {
                    ast::JoinType::Right => (right, left),
                    _ => (left, right),
                };
                let left = Box::new(self.build_from_item(scope, *left)?);
                let left_size = scope.len();
                let right = Box::new(self.build_from_item(scope, *right)?);
                // 用 scope 来编 expression.
                let predicate = predicate.map(|e| self.build_expression(scope, e)).transpose()?;
                // CROSS JOIN 和 inner 具体区别呢？
                let outer = match r#type {
                    ast::JoinType::Cross | ast::JoinType::Inner => false,
                    ast::JoinType::Left | ast::JoinType::Right => true,
                };
                // TODO(maple): 这个地方为什么可以不带 JoinKey 呢?
                let mut node = Node::NestedLoopJoin { left, left_size, right, predicate, outer };
                // 如果是 right-join, 插入一个 Project Operator.
                if matches!(r#type, ast::JoinType::Right) {
                    // 把 field 字段名构建成 ref-expr.
                    // 这个地方感觉处理的不是很优雅, 就是构建了一组 project, 把节点 ref 的逆转成(右部分-左部分).
                    let expressions = (left_size..scope.len())
                        .chain(0..left_size)
                        .map(|i| Ok((Expression::Field(i, scope.get_label(i)?), None)))
                        .collect::<Result<Vec<_>>>()?;
                    scope.project(&expressions)?;
                    node = Node::Projection { source: Box::new(node), expressions }
                }
                node
            }
        })
    }

    /// Builds an aggregation node. All aggregate parameters and GROUP BY expressions are evaluated
    /// in a pre-projection, whose results are fed into an Aggregate node. This node computes the
    /// aggregates for the given groups, passing the group values through directly.
    ///
    /// source: 下层的 node, 返回的结构参照 Scope.
    fn build_aggregation(
        &self,
        scope: &mut Scope,
        source: Node,
        groups: Vec<(ast::Expression, Option<String>)>,
        aggregations: Vec<(Aggregate, ast::Expression)>,
    ) -> Result<Node> {
        let mut aggregates = Vec::new();
        let mut expressions = Vec::new();
        /*
         * 下面两组顺序先后没关系, 但决定了 `expressions` 中的顺序, 先 Selection 中的, 后
         * GROUP BY 中的, 和 column-ref 中的逻辑是对应的.
         */
        // 先 build aggregators ( AVG(a+2), SUM(b) 等等 )
        for (aggregate, expr) in aggregations {
            aggregates.push(aggregate);
            expressions.push((self.build_expression(scope, expr)?, None));
        }
        // 再 build GROUP BY.
        for (expr, label) in groups {
            expressions.push((self.build_expression(scope, expr)?, label));
        }
        scope.project(
            &expressions
                .iter()
                .cloned()
                .enumerate()
                .map(|(i, (e, l))| {
                    if i < aggregates.len() {
                        // We pass null values here since we don't want field references to hit
                        // the fields in scope before the aggregation.
                        (Expression::Constant(Value::Null), None)
                    } else {
                        (e, l)
                    }
                })
                .collect::<Vec<_>>(),
        )?;
        // Agg + Project + origin.
        let node = Node::Aggregation {
            source: Box::new(Node::Projection { source: Box::new(source), expressions }),
            aggregates,
        };
        Ok(node)
    }

    /// Extracts aggregate functions from an AST expression tree. This finds the aggregate
    /// function calls, replaces them with ast::Expression::Column(i), maps the aggregate functions
    /// to aggregates, and returns them along with their argument expressions.
    ///
    /// 这里相当于从 Select 里面抽出 expr. exprs 都是裸的 ast::expr.
    /// SELECT max(a) ... 抽成
    /// Select ColumnRef() + Aggr Expressions, 然后返回返回 (aggMethod, agg expr), 比如 (MAX, a + 2).
    fn extract_aggregates(
        &self,
        exprs: &mut [(ast::Expression, Option<String>)],
    ) -> Result<Vec<(Aggregate, ast::Expression)>> {
        let mut aggregates = Vec::new();
        for (expr, _) in exprs {
            // 对 expr 中的表达式, 丢进里面的 lambda 来 transform
            // 这里会把 functionCall 的 ast 抽出来, 看看是否是 avg/min/max, 然后提出函数
            expr.transform_mut(
                &mut |mut e| match &mut e {
                    // 只检查函数成员为 1 的 fn-call
                    ast::Expression::Function(f, args) if args.len() == 1 => {
                        if let Some(aggregate) = self.aggregate_from_name(f) {
                            // 添加对应的 agg 项, 然后 transform
                            aggregates.push((aggregate, args.remove(0)));
                            // 指向 aggregates 的长度
                            // 这个地方很诡异, 是 aggregates 的长度, **暗示了这里会把 aggr 前置**
                            Ok(ast::Expression::Column(aggregates.len() - 1))
                        } else {
                            Ok(e)
                        }
                    }
                    _ => Ok(e),
                },
                &mut |e| Ok(e),
            )?;
        }
        // agg 内部不能有 agg.
        for (_, expr) in &aggregates {
            if self.is_aggregate(expr) {
                return Err(Error::Value("Aggregate functions can't be nested".into()));
            }
        }
        Ok(aggregates)
    }

    /// Extracts group by expressions, and replaces them with column references with the given
    /// offset. These can be either an arbitrary expression, a reference to a SELECT column, or the
    /// same expression as a SELECT column. The following are all valid:
    ///
    /// SELECT released / 100 AS century, COUNT(*) FROM movies GROUP BY century
    /// SELECT released / 100, COUNT(*) FROM movies GROUP BY released / 100
    /// SELECT COUNT(*) FROM movies GROUP BY released / 100
    ///
    /// GroupBy column 替换为 GROUP BY Expr. 返回 (原 expr, alias).
    fn extract_groups(
        &self,
        // exprs: 输入的表达式列表, min-max 被转成 Column.
        exprs: &mut Vec<(ast::Expression, Option<String>)>,
        // Group by 的列表
        group_by: Vec<ast::Expression>,
        offset: usize,
    ) -> Result<Vec<(ast::Expression, Option<String>)>> {
        // (ast::Expr, expr)
        let mut groups = Vec::with_capacity(exprs.len());
        for g in group_by {
            // Look for references to SELECT columns with AS labels
            // 如果是 GROUP BY col-name, 这个地方会强制找到 expr 中对应的对象.
            // SELECT a as alias ... GROUP BY a, 把 selection 中的 a 替换为对 group 的 column-ref,
            // 然后 group 表达为 (a, alias-for-a).
            // TODO(maple): 这个地方感觉匹配不到是有问题的吧, 如果是 field-ref
            if let ast::Expression::Field(None, label) = &g {
                if let Some(i) = exprs.iter().position(|(_, l)| l.as_deref() == Some(label)) {
                    groups.push((
                        replace(&mut exprs[i].0, ast::Expression::Column(offset + groups.len())),
                        exprs[i].1.clone(),
                    ));
                    continue;
                }
            }
            // Look for expressions exactly equal to the group expression
            //
            // 如果是非 field 的 expr, 或者是 SELECT a + 2 ... GROUP BY (a+2), 就加入 selection-list, 谈后把成员指向它.
            // 作为 ((a + 2), alias)
            if let Some(i) = exprs.iter().position(|(e, _)| e == &g) {
                groups.push((
                    replace(&mut exprs[i].0, ast::Expression::Column(offset + groups.len())),
                    exprs[i].1.clone(),
                ));
                continue;
            }
            // Otherwise, just use the group expression directly
            //
            // 直接使用这个表达式, 无 alias.
            groups.push((g, None))
        }
        // Make sure no group expressions contain Column references, which would be placed here
        // during extract_aggregates().
        for (expr, _) in &groups {
            if self.is_aggregate(expr) {
                return Err(Error::Value("Group expression cannot contain aggregates".into()));
            }
        }
        Ok(groups)
    }

    /// Injects hidden expressions into SELECT expressions. This is used for ORDER BY and HAVING, in
    /// order to apply these to fields or aggregates that are not present in the SELECT output, e.g.
    /// to order on a column that is not selected. This is done by replacing the relevant parts of
    /// the given expression with Column references to either existing columns or new, hidden
    /// columns in the select expressions. Returns the number of hidden columns added.
    ///
    /// ORDER BY 和 HAVING 的子表达式为 expr, 把 expr 注入 select.
    fn inject_hidden(
        &self,
        expr: &mut ast::Expression,
        select: &mut Vec<(ast::Expression, Option<String>)>,
    ) -> Result<usize> {
        // Replace any identical expressions or label references with column references.
        for (i, (sexpr, label)) in select.iter().enumerate() {
            // 如果是 select 里面的成员. (eg. select a + 2 ... GROUP BY a + 2 ORDER BY a + 2, 已经存在于 list 中)
            // 这里的情况是 expr 正好相等
            if expr == sexpr {
                *expr = ast::Expression::Column(i);
                continue;
            }
            // 否则, 去 visit + transmute 树的成员
            if let Some(label) = label {
                expr.transform_mut(
                    &mut |e| match e {
                        ast::Expression::Field(None, ref l) if l == label => {
                            Ok(ast::Expression::Column(i))
                        }
                        e => Ok(e),
                    },
                    &mut |e| Ok(e),
                )?;
            }
        }
        // Any remaining aggregate functions and field references must be extracted as hidden
        // columns.
        let mut hidden = 0;
        // 这个地方相当于递归访问, 直到出现匹配的 ast::Expression::Field, 遇到以后, 会加入 select 列表,
        // 然后让 Column 指向它.
        // TODO(maple): 这个地方如果是已经遇到的 Field 会怎么样? eg: SELECT a, SUM(*) GROUP BY (a+2)
        expr.transform_mut(
            &mut |e| match &e {
                ast::Expression::Function(f, a) if self.aggregate_from_name(f).is_some() => {
                    if let ast::Expression::Column(c) = a[0] {
                        if self.is_aggregate(&select[c].0) {
                            return Err(Error::Value(
                                "Aggregate function cannot reference aggregate".into(),
                            ));
                        }
                    }
                    select.push((e, None));
                    hidden += 1;
                    Ok(ast::Expression::Column(select.len() - 1))
                }
                ast::Expression::Field(_, _) => {
                    select.push((e, None));
                    hidden += 1;
                    Ok(ast::Expression::Column(select.len() - 1))
                }
                _ => Ok(e),
            },
            &mut |e| Ok(e),
        )?;
        Ok(hidden)
    }

    /// Returns the aggregate corresponding to the given aggregate function name.
    fn aggregate_from_name(&self, name: &str) -> Option<Aggregate> {
        match name {
            "avg" => Some(Aggregate::Average),
            "count" => Some(Aggregate::Count),
            "max" => Some(Aggregate::Max),
            "min" => Some(Aggregate::Min),
            "sum" => Some(Aggregate::Sum),
            _ => None,
        }
    }

    /// Checks whether a given expression is an aggregate expression.
    fn is_aggregate(&self, expr: &ast::Expression) -> bool {
        expr.contains(&|e| match e {
            ast::Expression::Function(f, _) => self.aggregate_from_name(f).is_some(),
            _ => false,
        })
    }

    /// Builds an expression from an AST expression
    ///
    /// Build 阶段都是去拿已有的.
    fn build_expression(&self, scope: &mut Scope, expr: ast::Expression) -> Result<Expression> {
        use Expression::*;
        Ok(match expr {
            ast::Expression::Literal(l) => Constant(match l {
                ast::Literal::Null => Value::Null,
                ast::Literal::Boolean(b) => Value::Boolean(b),
                ast::Literal::Integer(i) => Value::Integer(i),
                ast::Literal::Float(f) => Value::Float(f),
                ast::Literal::String(s) => Value::String(s),
            }),
            // FROM COLUMN(1), 选取后面的列
            ast::Expression::Column(i) => Field(i, scope.get_label(i)?),
            ast::Expression::Field(table, name) => {
                // 通过 resolve 接口拿到 field-id.
                Field(scope.resolve(table.as_deref(), &name)?, Some((table, name)))
            }
            ast::Expression::Function(name, _) => {
                return Err(Error::Value(format!("Unknown function {}", name,)))
            }
            ast::Expression::Operation(op) => match op {
                // Logical operators
                ast::Operation::And(lhs, rhs) => And(
                    self.build_expression(scope, *lhs)?.into(),
                    self.build_expression(scope, *rhs)?.into(),
                ),
                ast::Operation::Not(expr) => Not(self.build_expression(scope, *expr)?.into()),
                ast::Operation::Or(lhs, rhs) => Or(
                    self.build_expression(scope, *lhs)?.into(),
                    self.build_expression(scope, *rhs)?.into(),
                ),

                // Comparison operators
                ast::Operation::Equal(lhs, rhs) => Equal(
                    self.build_expression(scope, *lhs)?.into(),
                    self.build_expression(scope, *rhs)?.into(),
                ),
                ast::Operation::GreaterThan(lhs, rhs) => GreaterThan(
                    self.build_expression(scope, *lhs)?.into(),
                    self.build_expression(scope, *rhs)?.into(),
                ),
                ast::Operation::GreaterThanOrEqual(lhs, rhs) => Or(
                    GreaterThan(
                        self.build_expression(scope, *lhs.clone())?.into(),
                        self.build_expression(scope, *rhs.clone())?.into(),
                    )
                    .into(),
                    Equal(
                        self.build_expression(scope, *lhs)?.into(),
                        self.build_expression(scope, *rhs)?.into(),
                    )
                    .into(),
                ),
                ast::Operation::IsNull(expr) => IsNull(self.build_expression(scope, *expr)?.into()),
                ast::Operation::LessThan(lhs, rhs) => LessThan(
                    self.build_expression(scope, *lhs)?.into(),
                    self.build_expression(scope, *rhs)?.into(),
                ),
                ast::Operation::LessThanOrEqual(lhs, rhs) => Or(
                    LessThan(
                        self.build_expression(scope, *lhs.clone())?.into(),
                        self.build_expression(scope, *rhs.clone())?.into(),
                    )
                    .into(),
                    Equal(
                        self.build_expression(scope, *lhs)?.into(),
                        self.build_expression(scope, *rhs)?.into(),
                    )
                    .into(),
                ),
                ast::Operation::Like(lhs, rhs) => Like(
                    self.build_expression(scope, *lhs)?.into(),
                    self.build_expression(scope, *rhs)?.into(),
                ),
                ast::Operation::NotEqual(lhs, rhs) => Not(Equal(
                    self.build_expression(scope, *lhs)?.into(),
                    self.build_expression(scope, *rhs)?.into(),
                )
                .into()),

                // Mathematical operators
                ast::Operation::Assert(expr) => Assert(self.build_expression(scope, *expr)?.into()),
                ast::Operation::Add(lhs, rhs) => Add(
                    self.build_expression(scope, *lhs)?.into(),
                    self.build_expression(scope, *rhs)?.into(),
                ),
                ast::Operation::Divide(lhs, rhs) => Divide(
                    self.build_expression(scope, *lhs)?.into(),
                    self.build_expression(scope, *rhs)?.into(),
                ),
                ast::Operation::Exponentiate(lhs, rhs) => Exponentiate(
                    self.build_expression(scope, *lhs)?.into(),
                    self.build_expression(scope, *rhs)?.into(),
                ),
                ast::Operation::Factorial(expr) => {
                    Factorial(self.build_expression(scope, *expr)?.into())
                }
                ast::Operation::Modulo(lhs, rhs) => Modulo(
                    self.build_expression(scope, *lhs)?.into(),
                    self.build_expression(scope, *rhs)?.into(),
                ),
                ast::Operation::Multiply(lhs, rhs) => Multiply(
                    self.build_expression(scope, *lhs)?.into(),
                    self.build_expression(scope, *rhs)?.into(),
                ),
                ast::Operation::Negate(expr) => Negate(self.build_expression(scope, *expr)?.into()),
                ast::Operation::Subtract(lhs, rhs) => Subtract(
                    self.build_expression(scope, *lhs)?.into(),
                    self.build_expression(scope, *rhs)?.into(),
                ),
            },
        })
    }

    /// Builds and evaluates a constant AST expression.
    fn evaluate_constant(&self, expr: ast::Expression) -> Result<Value> {
        self.build_expression(&mut Scope::constant(), expr)?.evaluate(None)
    }
}

/// Manages names available to expressions and executors, and maps them onto columns/fields.
///
/// Scope 包含一组有序的 columns. 同时包含限定符, 表示这些东西的归属.
#[derive(Clone, Debug)]
pub struct Scope {
    // If true, the scope is constant and cannot contain any variables.
    constant: bool,
    // Currently visible tables, by query name (i.e. alias or actual name).
    //
    // Table 的名称集合, 构建了 alias / table_name -> Table 的映射
    // TODO(maple): 这个 Table 是怎么构建出来的? 具体 binding 吗?
    tables: HashMap<String, Table>,
    // Column labels, if any (qualified by table name when available)
    //
    // columns 是表的 source-of-truth, 下面三个 hash 都是根据这玩意造出来的.
    columns: Vec<(Option<String>, Option<String>)>,
    // Qualified names to column indexes.
    //
    // qualified: 有 table-name 和 field-name 的, 这个地方 field-name 可能是一个 label.
    qualified: HashMap<(String, String), usize>,
    // Unqualified names to column indexes, if unique.
    unqualified: HashMap<String, usize>,
    // Unqualified ambiguous names.
    // TODO(maple): ambiguous 的 name 是怎么 solving 的?
    ambiguous: HashSet<String>,
}

impl Scope {
    /// Creates a new, empty scope.
    fn new() -> Self {
        Self {
            constant: false,
            tables: HashMap::new(),
            columns: Vec::new(),
            qualified: HashMap::new(),
            unqualified: HashMap::new(),
            ambiguous: HashSet::new(),
        }
    }

    /// Creates a constant scope.
    fn constant() -> Self {
        let mut scope = Self::new();
        scope.constant = true;
        scope
    }

    /// Creates a scope from a table.
    fn from_table(table: Table) -> Result<Self> {
        let mut scope = Self::new();
        scope.add_table(table.name.clone(), table)?;
        Ok(scope)
    }

    /// Adds a column to the scope.
    #[allow(clippy::map_entry)]
    fn add_column(&mut self, table: Option<String>, label: Option<String>) {
        if let Some(l) = label.clone() {
            // 如果有表, 则作为 qualified name
            if let Some(t) = table.clone() {
                self.qualified.insert((t, l.clone()), self.columns.len());
            }
            // 否则, push-back 到 unqualified 或者 ambiguous
            if !self.ambiguous.contains(&l) {
                if !self.unqualified.contains_key(&l) {
                    self.unqualified.insert(l, self.columns.len());
                } else {
                    self.unqualified.remove(&l);
                    self.ambiguous.insert(l);
                }
            }
        }
        // 无论如何都会 push columns.
        self.columns.push((table, label));
    }

    /// Adds a table to the scope.
    ///
    /// label 相当于 alias / table name 二选一
    fn add_table(&mut self, label: String, table: Table) -> Result<()> {
        if self.constant {
            return Err(Error::Internal("Can't modify constant scope".into()));
        }
        if self.tables.contains_key(&label) {
            return Err(Error::Value(format!("Duplicate table name {}", label)));
        }
        // 这个地方加入 table 里所有的 column, 形式都是 table name - column
        for column in &table.columns {
            self.add_column(Some(label.clone()), Some(column.name.clone()));
        }
        self.tables.insert(label, table);
        Ok(())
    }

    /// Fetches a column from the scope by index.
    fn get_column(&self, index: usize) -> Result<(Option<String>, Option<String>)> {
        if self.constant {
            return Err(Error::Value(format!(
                "Expression must be constant, found column {}",
                index
            )));
        }
        self.columns
            .get(index)
            .cloned()
            .ok_or_else(|| Error::Value(format!("Column index {} not found", index)))
    }

    /// Fetches a column label by index, if any.
    fn get_label(&self, index: usize) -> Result<Option<(Option<String>, String)>> {
        Ok(match self.get_column(index)? {
            (table, Some(name)) => Some((table, name)),
            _ => None,
        })
    }

    /// Merges two scopes, by appending the given scope to self.
    ///
    /// 叫 merge, 其实不如叫 append-to-right, 添加到 Scope 的最右侧.
    fn merge(&mut self, scope: Scope) -> Result<()> {
        if self.constant {
            return Err(Error::Internal("Can't modify constant scope".into()));
        }
        // 查看 Table 和自身 Table 有没有冲突
        // TODO(maple):有冲突会怎么半, 比如 a JOIN b, a JOIN c
        for (label, table) in scope.tables {
            if self.tables.contains_key(&label) {
                return Err(Error::Value(format!("Duplicate table name {}", label)));
            }
            self.tables.insert(label, table);
        }
        for (table, label) in scope.columns {
            self.add_column(table, label);
        }
        Ok(())
    }

    /// Resolves a name, optionally qualified by a table name.
    ///
    /// Resolve 去拿到对应的 name.
    fn resolve(&self, table: Option<&str>, name: &str) -> Result<usize> {
        if self.constant {
            return Err(Error::Value(format!(
                "Expression must be constant, found field {}",
                if let Some(table) = table { format!("{}.{}", table, name) } else { name.into() }
            )));
        }
        if let Some(table) = table {
            if !self.tables.contains_key(table) {
                return Err(Error::Value(format!("Unknown table {}", table)));
            }
            self.qualified
                .get(&(table.into(), name.into()))
                .copied()
                .ok_or_else(|| Error::Value(format!("Unknown field {}.{}", table, name)))
        } else if self.ambiguous.contains(name) {
            // TODO(maple): ambiguous 为什么是在这个阶段发现的?
            Err(Error::Value(format!("Ambiguous field {}", name)))
        } else {
            // 在 unqualified 里面找 name.
            // (虚拟生成的字段是否是 unqualified?)
            // 一点想法: 感觉 put 的时候用 Option 本身不是一个好主意, 我现在都分不清什么时候会 put unqualifed 了.
            self.unqualified
                .get(name)
                .copied()
                .ok_or_else(|| Error::Value(format!("Unknown field {}", name)))
        }
    }

    /// Number of columns in the current scope.
    fn len(&self) -> usize {
        self.columns.len()
    }

    /// Projects the scope. This takes a set of expressions and labels in the current scope,
    /// and returns a new scope for the projection.
    ///
    /// Project 会产生一个新的 expr, 但是会调换顺序.
    fn project(&mut self, projection: &[(Expression, Option<String>)]) -> Result<()> {
        if self.constant {
            return Err(Error::Internal("Can't modify constant scope".into()));
        }
        let mut new = Self::new();
        new.tables = self.tables.clone();
        for (expr, label) in projection {
            match (expr, label) {
                // 添加了一个没有 table name 的名称.
                (_, Some(label)) => new.add_column(None, Some(label.clone())),
                (Expression::Field(_, Some((Some(table), name))), _) => {
                    new.add_column(Some(table.clone()), Some(name.clone()))
                }
                (Expression::Field(_, Some((None, name))), _) => {
                    if let Some(i) = self.unqualified.get(name) {
                        let (table, name) = self.columns[*i].clone();
                        new.add_column(table, name);
                    }
                }
                (Expression::Field(i, None), _) => {
                    let (table, label) = self.columns.get(*i).cloned().unwrap_or((None, None));
                    new.add_column(table, label)
                }
                _ => new.add_column(None, None),
            }
        }
        *self = new;
        Ok(())
    }
}
