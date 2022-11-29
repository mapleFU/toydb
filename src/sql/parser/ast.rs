use super::super::types::DataType;
use crate::error::Result;

use std::collections::BTreeMap;
use std::mem::replace;

/// Statements
#[derive(Clone, Debug, PartialEq)]
#[allow(clippy::large_enum_variant)]
pub enum Statement {
    Begin {
        readonly: bool,
        version: Option<u64>,
    },
    Commit,
    Rollback,
    Explain(Box<Statement>),

    CreateTable {
        name: String,
        columns: Vec<Column>,
    },
    DropTable(String),

    Delete {
        table: String,
        r#where: Option<Expression>,
    },
    Insert {
        table: String,
        columns: Option<Vec<String>>,
        values: Vec<Vec<Expression>>,
    },
    Update {
        table: String,
        set: BTreeMap<String, Expression>,
        r#where: Option<Expression>,
    },

    Select {
        /// Select expr as alias, 后面是个 Optional 的 alias...
        /// 讲道理, 我就是不太喜欢这种用 Tuple 而不是 fieldName 的...
        select: Vec<(Expression, Option<String>)>,
        from: Vec<FromItem>,
        r#where: Option<Expression>,
        group_by: Vec<Expression>,
        having: Option<Expression>,
        order: Vec<(Expression, Order)>,
        offset: Option<Expression>,
        limit: Option<Expression>,
    },
}

/// A FROM item
#[derive(Clone, Debug, PartialEq)]
pub enum FromItem {
    Table {
        name: String,
        alias: Option<String>,
    },
    // JoinType 考虑数种, 递归进行定义.
    Join {
        left: Box<FromItem>,
        right: Box<FromItem>,
        r#type: JoinType,
        // Predict 产生对应的逻辑
        predicate: Option<Expression>,
    },
}

/// A JOIN type
#[derive(Clone, Debug, PartialEq)]
pub enum JoinType {
    Cross,
    Inner,
    Left,
    Right,
}

/// A column
#[derive(Clone, Debug, PartialEq)]
pub struct Column {
    pub name: String,
    pub datatype: DataType,
    pub primary_key: bool,
    pub nullable: Option<bool>,
    pub default: Option<Expression>,
    pub unique: bool,
    pub index: bool,
    pub references: Option<String>,
}

/// Sort orders
#[derive(Clone, Debug, PartialEq)]
pub enum Order {
    Ascending,
    Descending,
}

/// Expressions
#[derive(Clone, Debug, PartialEq)]
pub enum Expression {
    /// TableName, Name(alias or field-name)
    Field(Option<String>, String),
    /// 内部实现, 对某个 Column 的引用.
    Column(usize), // only used during plan building to break off expression subtrees
    /// bool 之类的 literal, 这个地方感觉没有进行类型推断和类型提升
    /// 目测是解析出是什么类型就当成什么类型处理了.
    Literal(Literal),
    /// AVG(expr1, expr2...)
    Function(String, Vec<Expression>),
    /// 非 fnCall 形式的操作, 这里会把 LIKE 之类的都归为 Operation 了.
    Operation(Operation),
}

impl From<Literal> for Expression {
    fn from(literal: Literal) -> Self {
        Self::Literal(literal)
    }
}

impl From<Operation> for Expression {
    fn from(op: Operation) -> Self {
        Self::Operation(op)
    }
}

/// Literals
#[derive(Clone, Debug, PartialEq)]
pub enum Literal {
    Null,
    Boolean(bool),
    Integer(i64),
    Float(f64),
    String(String),
}

/// Operations (done by operators)
#[derive(Clone, Debug, PartialEq)]
pub enum Operation {
    // Logical operators
    And(Box<Expression>, Box<Expression>),
    Not(Box<Expression>),
    Or(Box<Expression>, Box<Expression>),

    // Comparison operators
    Equal(Box<Expression>, Box<Expression>),
    GreaterThan(Box<Expression>, Box<Expression>),
    GreaterThanOrEqual(Box<Expression>, Box<Expression>),
    IsNull(Box<Expression>),
    LessThan(Box<Expression>, Box<Expression>),
    LessThanOrEqual(Box<Expression>, Box<Expression>),
    NotEqual(Box<Expression>, Box<Expression>),

    // Mathematical operators
    Add(Box<Expression>, Box<Expression>),
    Assert(Box<Expression>),
    Divide(Box<Expression>, Box<Expression>),
    Exponentiate(Box<Expression>, Box<Expression>),
    Factorial(Box<Expression>),
    Modulo(Box<Expression>, Box<Expression>),
    Multiply(Box<Expression>, Box<Expression>),
    Negate(Box<Expression>),
    Subtract(Box<Expression>, Box<Expression>),

    // String operators
    Like(Box<Expression>, Box<Expression>),
}

impl Expression {
    /// Walks the expression tree while calling a closure. Returns true as soon as the closure
    /// returns true. This is the inverse of walk().
    pub fn contains<F: Fn(&Expression) -> bool>(&self, visitor: &F) -> bool {
        !self.walk(&|e| !visitor(e))
    }

    /// Replaces the expression with result of the closure. Helper function for transform().
    fn replace_with<F: FnMut(Self) -> Result<Self>>(&mut self, mut f: F) -> Result<()> {
        // Temporarily replace expression with a null value, in case closure panics. May consider
        // replace_with crate if this hampers performance.
        let expr = replace(self, Expression::Literal(Literal::Null));
        *self = f(expr)?;
        Ok(())
    }

    /// Transforms the expression tree by applying a closure before and after descending.
    ///
    /// 这个其实有点 visit tree 的感觉, 这个代码其实是有点蛋疼的. 遍历方式大概是:
    /// * before(self)
    /// * transmute 所有子节点, 对于 Literal, Field, Column 不做任何检查. 这里递归对 Operation / Function 使用
    ///   transmute. (这些感觉都是动态的，感觉 Visitor 模式代码写的爽些?)
    /// * after(self)
    pub fn transform<B, A>(mut self, before: &mut B, after: &mut A) -> Result<Self>
    where
        B: FnMut(Self) -> Result<Self>,
        A: FnMut(Self) -> Result<Self>,
    {
        use Operation::*;
        self = before(self)?;
        match &mut self {
            Self::Operation(Add(lhs, rhs))
            | Self::Operation(And(lhs, rhs))
            | Self::Operation(Divide(lhs, rhs))
            | Self::Operation(Equal(lhs, rhs))
            | Self::Operation(Exponentiate(lhs, rhs))
            | Self::Operation(GreaterThan(lhs, rhs))
            | Self::Operation(GreaterThanOrEqual(lhs, rhs))
            | Self::Operation(LessThan(lhs, rhs))
            | Self::Operation(LessThanOrEqual(lhs, rhs))
            | Self::Operation(Like(lhs, rhs))
            | Self::Operation(Modulo(lhs, rhs))
            | Self::Operation(Multiply(lhs, rhs))
            | Self::Operation(NotEqual(lhs, rhs))
            | Self::Operation(Or(lhs, rhs))
            | Self::Operation(Subtract(lhs, rhs)) => {
                Self::replace_with(lhs, |e| e.transform(before, after))?;
                Self::replace_with(rhs, |e| e.transform(before, after))?;
            }

            Self::Operation(Assert(expr))
            | Self::Operation(Factorial(expr))
            | Self::Operation(IsNull(expr))
            | Self::Operation(Negate(expr))
            | Self::Operation(Not(expr)) => {
                Self::replace_with(expr, |e| e.transform(before, after))?
            }

            Self::Function(_, exprs) => {
                for expr in exprs {
                    Self::replace_with(expr, |e| e.transform(before, after))?;
                }
            }

            Self::Literal(_) | Self::Field(_, _) | Self::Column(_) => {}
        };
        after(self)
    }

    /// Transforms an expression using a mutable reference.
    pub fn transform_mut<B, A>(&mut self, before: &mut B, after: &mut A) -> Result<()>
    where
        B: FnMut(Self) -> Result<Self>,
        A: FnMut(Self) -> Result<Self>,
    {
        self.replace_with(|e| e.transform(before, after))
    }

    /// Walks the expression tree, calling a closure for every node. Halts if closure returns false.
    ///
    /// walk 则是 visitor 的另一种形式了.
    pub fn walk<F: Fn(&Expression) -> bool>(&self, visitor: &F) -> bool {
        use Operation::*;
        visitor(self)
            && match self {
                Self::Operation(Add(lhs, rhs))
                | Self::Operation(And(lhs, rhs))
                | Self::Operation(Divide(lhs, rhs))
                | Self::Operation(Equal(lhs, rhs))
                | Self::Operation(Exponentiate(lhs, rhs))
                | Self::Operation(GreaterThan(lhs, rhs))
                | Self::Operation(GreaterThanOrEqual(lhs, rhs))
                | Self::Operation(LessThan(lhs, rhs))
                | Self::Operation(LessThanOrEqual(lhs, rhs))
                | Self::Operation(Like(lhs, rhs))
                | Self::Operation(Modulo(lhs, rhs))
                | Self::Operation(Multiply(lhs, rhs))
                | Self::Operation(NotEqual(lhs, rhs))
                | Self::Operation(Or(lhs, rhs))
                | Self::Operation(Subtract(lhs, rhs)) => lhs.walk(visitor) && rhs.walk(visitor),

                Self::Operation(Assert(expr))
                | Self::Operation(Factorial(expr))
                | Self::Operation(IsNull(expr))
                | Self::Operation(Negate(expr))
                | Self::Operation(Not(expr)) => expr.walk(visitor),

                Self::Function(_, exprs) => {
                    for expr in exprs {
                        if !expr.walk(visitor) {
                            return false;
                        }
                    }
                    true
                }

                Self::Literal(_) | Self::Field(_, _) | Self::Column(_) => true,
            }
    }
}
