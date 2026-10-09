//! Rust コード生成モジュール
//!
//! 型推論結果から Rust コードを生成する。

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::io::{self, Write};

use crate::ast::{AssertKind, AssignOp, BinOp, BlockItem, CompoundStmt, Declaration, DeclSpecs, DerivedDecl, Expr, ExprKind, ForInit, FunctionDef, Initializer, ParamDecl, Stmt, TypeSpec};

use crate::intern::InternedStr;
use crate::enum_dict::EnumDict;
use crate::infer_api::InferResult;
use crate::intern::StringInterner;
use crate::macro_infer::{MacroInferContext, MacroInferInfo, MacroParam, ParseResult};
use crate::rust_decl::RustDeclDict;
use crate::syn_codegen::normalize_parens;
use crate::unified_type::UnifiedType;
use crate::sexp::SexpPrinter;

static IS_WINDOWS_TARGET: std::sync::LazyLock<bool> = std::sync::LazyLock::new(is_windows_target);

/// Check to use instead of #[cfg(windows)].
/// cfg checks host, not target platform, so cross-compile would be wrong.
/// If the environment is not found, the hosts platform will be used as fallback.
fn is_windows_target() -> bool {
    std::env::var_os("CARGO_CFG_TARGET_OS").map_or(cfg!(windows), |target_os| target_os == "windows")
}

/// bindings.rs から抽出した codegen 用情報
#[derive(Debug, Default, Clone)]
pub struct BindingsInfo {
    /// 配列型の extern static 変数名の集合
    pub static_arrays: HashSet<String>,
    /// extern static 変数名 → 型文字列（"[T; N]" や "T" 等）
    /// 配列の要素型抽出に使う。`static_array_element_type` も参照。
    pub static_types: HashMap<String, String>,
    /// ビットフィールドのメソッド名集合（構造体名 → メソッド名セット）
    pub bitfield_methods: HashMap<String, HashSet<String>>,
}

impl BindingsInfo {
    /// RustDeclDict から BindingsInfo を構築
    pub fn from_rust_decl_dict(dict: &RustDeclDict) -> Self {
        Self {
            static_arrays: dict.static_arrays.clone(),
            static_types: dict.static_types.clone(),
            bitfield_methods: dict.bitfield_methods.clone(),
        }
    }

    /// `static_arrays` に含まれる名前の **配列要素型** を返す。
    /// `static_types[name]` が `"[T; N]"` 形式なら T を取り出す。
    /// 抽出できなかったら None（呼出側でフォールバック）。
    pub fn static_array_element_type(&self, name: &str) -> Option<String> {
        let ty = self.static_types.get(name)?;
        let s = ty.trim();
        let s = s.strip_prefix('[')?;
        let s = s.strip_suffix(']')?;
        let semi = s.rfind(';')?;
        Some(s[..semi].trim().to_string())
    }
}

/// libc crate から提供される関数名のリスト
/// codegen がそのまま関数呼び出しとして出力する関数のみ
/// （`__builtin_expect` 等の codegen 変換済み関数は含めない）
const LIBC_FUNCTIONS: &[&str] = &[
    "strcmp", "strlen", "strncmp", "strcpy", "strncpy",
    "memset", "memchr", "memcpy", "memmove",
];

/// 既知の libc 関数の引数型 (bindings.rs に登録されない関数を補完)。
///
/// 呼出側は `as *mut c_char` で渡しがちだが libc は `*mut c_void` を要求する。
/// ハードコードした型を `get_callee_param_type_extended` 経由で引くと、
/// 自動 as-cast (cast_arg_syn_if_needed の void-pointer 分岐) に乗る。
fn libc_fn_param_type(func_name: &str, arg_index: usize) -> Option<UnifiedType> {
    match (func_name, arg_index) {
        // memset(void *s, int c, size_t n)
        ("memset", 0) => Some(UnifiedType::from_rust_str("*mut c_void")),
        ("memset", 1) => Some(UnifiedType::from_rust_str("c_int")),
        ("memset", 2) => Some(UnifiedType::from_rust_str("usize")),
        // memcpy(void *dest, const void *src, size_t n)
        ("memcpy", 0) | ("memmove", 0) => Some(UnifiedType::from_rust_str("*mut c_void")),
        ("memcpy", 1) | ("memmove", 1) => Some(UnifiedType::from_rust_str("*const c_void")),
        ("memcpy", 2) | ("memmove", 2) => Some(UnifiedType::from_rust_str("usize")),
        // memchr(const void *s, int c, size_t n)
        ("memchr", 0) => Some(UnifiedType::from_rust_str("*const c_void")),
        ("memchr", 1) => Some(UnifiedType::from_rust_str("c_int")),
        ("memchr", 2) => Some(UnifiedType::from_rust_str("usize")),
        // memcmp(const void *s1, const void *s2, size_t n)
        ("memcmp", 0) | ("memcmp", 1) => Some(UnifiedType::from_rust_str("*const c_void")),
        ("memcmp", 2) => Some(UnifiedType::from_rust_str("usize")),
        // strcmp / strncmp: *const c_char
        ("strcmp", 0) | ("strcmp", 1) | ("strncmp", 0) | ("strncmp", 1) =>
            Some(UnifiedType::from_rust_str("*const c_char")),
        ("strncmp", 2) => Some(UnifiedType::from_rust_str("usize")),
        // strlen: *const c_char
        ("strlen", 0) => Some(UnifiedType::from_rust_str("*const c_char")),
        // strcpy/strncpy: dest *mut, src *const c_char
        ("strcpy", 0) | ("strncpy", 0) => Some(UnifiedType::from_rust_str("*mut c_char")),
        ("strcpy", 1) | ("strncpy", 1) => Some(UnifiedType::from_rust_str("*const c_char")),
        ("strncpy", 2) => Some(UnifiedType::from_rust_str("usize")),
        _ => None,
    }
}

/// コード生成時に解決可能なシンボルの集合
///
/// bindings.rs、マクロ辞書、inline 関数辞書、ビルトイン関数等から
/// 既知のシンボル名を収集する。コード生成時に `ExprKind::Ident` が
/// この集合に含まれない場合、未解決シンボルとして検出する。
pub struct KnownSymbols {
    names: HashSet<String>,
}

impl KnownSymbols {
    /// InferResult から既知シンボル集合を構築
    pub fn new(result: &InferResult, interner: &StringInterner) -> Self {
        let mut names = HashSet::new();

        // bindings.rs の関数名
        if let Some(ref dict) = result.rust_decl_dict {
            for name in dict.fns.keys() {
                names.insert(name.clone());
            }
            for name in dict.consts.keys() {
                names.insert(name.clone());
            }
            for name in dict.types.keys() {
                names.insert(name.clone());
            }
            for name in dict.structs.keys() {
                names.insert(name.clone());
            }
            for name in &dict.enums {
                names.insert(name.clone());
            }
            for name in &dict.statics {
                names.insert(name.clone());
            }
            for name in &dict.static_arrays {
                names.insert(name.clone());
            }
        }

        // マクロ名（関数呼び出しとして保持されるもの）
        for (name_id, info) in &result.infer_ctx.macros {
            let name_str = interner.get(*name_id);
            // 関数マクロのみ既知とする（オブジェクトマクロは除外）
            // オブジェクトマクロ名（例: `n`, `s`, `c`）を登録すると、
            // ジェネリック誤検出で残ったパラメータ参照が既知扱いになってしまう
            if info.has_body && info.is_function {
                names.insert(name_str.to_string());
            }
        }

        // inline 関数名
        for (name_id, _) in result.inline_fn_dict.iter() {
            let name_str = interner.get(*name_id);
            names.insert(name_str.to_string());
        }

        // ビルトイン関数（codegen が変換・除去するもの）
        let builtins = [
            "__builtin_expect",
            "__builtin_offsetof",
            "offsetof",
            "__builtin_types_compatible_p",
            "__builtin_constant_p",
            "__builtin_choose_expr",
            "__builtin_unreachable",
            "__builtin_trap",
            "__builtin_assume",
            "__builtin_bswap16",
            "__builtin_bswap32",
            "__builtin_bswap64",
            "__builtin_popcount",
            "__builtin_clz",
            "__builtin_ctz",
            "pthread_mutex_lock",
            "pthread_mutex_unlock",
            "pthread_rwlock_rdlock",
            "pthread_rwlock_wrlock",
            "pthread_rwlock_unlock",
            "pthread_getspecific",
            "pthread_cond_wait",
            "pthread_cond_signal",
            "getenv",
            "ASSERT_IS_LITERAL",
            "ASSERT_IS_PTR",
            "ASSERT_NOT_PTR",
        ];
        for name in builtins {
            names.insert(name.to_string());
        }

        // libc 関数（use libc::{...} で利用可能になる）
        for name in LIBC_FUNCTIONS {
            names.insert(name.to_string());
        }

        // 自動生成 static const 配列名（`static_array_emitter.rs` 由来）
        // 注: struct/typedef alias 名は `generate()` 側で実際に出力できた名前のみ
        // 後から `insert()` する。事前にここで全部入れると未生成の型を参照する
        // コードを「既知」とみなして compile error を起こす可能性がある。
        for (name_id, _) in result.global_const_dict.iter() {
            names.insert(interner.get(*name_id).to_string());
        }

        // Rust プリミティブ / 標準識別子
        let rust_primitives = [
            "true", "false", "std", "crate", "self", "super",
            "null_mut", "null",
            "PerlInterpreter", "my_perl",
            // 出力ヘッダで `type X = Y;` 定義しているもの
            // (`generate_use_statements` 参照)
            "size_t", "ssize_t", "SSize_t",
            // std::ffi 由来（use 文で import 済み）
            "c_void", "c_char", "c_uchar", "c_int", "c_uint",
            "c_long", "c_ulong", "c_short", "c_ushort",
        ];
        for name in rust_primitives {
            names.insert(name.to_string());
        }

        Self { names }
    }

    /// シンボル名が既知かどうかチェック
    fn contains(&self, name: &str) -> bool {
        self.names.contains(name)
    }

    /// 既知シンボルとして名前を追加
    pub fn insert(&mut self, name: String) {
        self.names.insert(name);
    }
}

/// Rust の予約語リスト（strict keywords + reserved keywords）
/// 注: true/false はリテラルなので含めない
const RUST_KEYWORDS: &[&str] = &[
    // Strict keywords (true/false は除外 - リテラルなのでエスケープ不要)
    "as", "async", "await", "break", "const", "continue", "crate", "dyn",
    "else", "enum", "extern", "fn", "for", "if", "impl", "in",
    "let", "loop", "match", "mod", "move", "mut", "pub", "ref", "return",
    "self", "Self", "static", "struct", "super", "trait", "type",
    "unsafe", "use", "where", "while",
    // Reserved keywords
    "abstract", "become", "box", "do", "final", "gen", "macro", "override",
    "priv", "try", "typeof", "unsized", "virtual", "yield",
];

/// 識別子を Rust コードに変換
///
/// - Rust の予約語は r# を付ける
/// - C のプリプロセッサマクロは Rust の同等品に変換
fn escape_rust_keyword(name: &str) -> String {
    match name {
        // C プリプロセッサマクロ → Rust マクロ
        "__FILE__" => "file!()".to_string(),
        "__LINE__" => "line!()".to_string(),
        // Rust 予約語はエスケープ
        _ if RUST_KEYWORDS.contains(&name) => format!("r#{}", name),
        // その他はそのまま
        _ => name.to_string(),
    }
}

/// 単語境界を考慮した文字列置換
///
/// 型パラメータ名の置換時に、部分文字列一致を避けるために使用。
/// 例: "XV" を "T" に置換するとき、"XPVNV" は変更しない。
fn replace_word(s: &str, word: &str, replacement: &str) -> String {
    if word.is_empty() {
        return s.to_string();
    }
    let mut result = String::with_capacity(s.len());
    let mut start = 0;
    let bytes = s.as_bytes();
    let word_bytes = word.as_bytes();
    while let Some(pos) = s[start..].find(word) {
        let abs_pos = start + pos;
        // 前方の境界チェック
        let before_ok = abs_pos == 0 || !is_ident_char(bytes[abs_pos - 1]);
        // 後方の境界チェック
        let after_pos = abs_pos + word.len();
        let after_ok = after_pos >= bytes.len() || !is_ident_char(bytes[after_pos]);

        if before_ok && after_ok {
            result.push_str(&s[start..abs_pos]);
            result.push_str(replacement);
            start = after_pos;
        } else {
            result.push_str(&s[start..abs_pos + word_bytes.len()]);
            start = abs_pos + word_bytes.len();
        }
    }
    result.push_str(&s[start..]);
    result
}

/// 識別子を構成する文字かどうか
fn is_ident_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// 二項演算子を Rust 形式に変換
fn bin_op_to_rust(op: BinOp) -> &'static str {
    match op {
        BinOp::Add => "+",
        BinOp::Sub => "-",
        BinOp::Mul => "*",
        BinOp::Div => "/",
        BinOp::Mod => "%",
        BinOp::BitAnd => "&",
        BinOp::BitOr => "|",
        BinOp::BitXor => "^",
        BinOp::Shl => "<<",
        BinOp::Shr => ">>",
        BinOp::Lt => "<",
        BinOp::Gt => ">",
        BinOp::Le => "<=",
        BinOp::Ge => ">=",
        BinOp::Eq => "==",
        BinOp::Ne => "!=",
        BinOp::LogAnd => "&&",
        BinOp::LogOr => "||",
    }
}

/// 代入演算子を Rust 形式に変換
fn assign_op_to_rust(op: AssignOp) -> &'static str {
    match op {
        AssignOp::Assign => "=",
        AssignOp::MulAssign => "*=",
        AssignOp::DivAssign => "/=",
        AssignOp::ModAssign => "%=",
        AssignOp::AddAssign => "+=",
        AssignOp::SubAssign => "-=",
        AssignOp::ShlAssign => "<<=",
        AssignOp::ShrAssign => ">>=",
        AssignOp::AndAssign => "&=",
        AssignOp::XorAssign => "^=",
        AssignOp::OrAssign => "|=",
    }
}

/// 文字をエスケープ
fn escape_char(c: u8) -> String {
    match c {
        b'\'' => "\\'".to_string(),
        b'\\' => "\\\\".to_string(),
        b'\n' => "\\n".to_string(),
        b'\r' => "\\r".to_string(),
        b'\t' => "\\t".to_string(),
        c if c.is_ascii_graphic() || c == b' ' => (c as char).to_string(),
        c => format!("\\x{:02x}", c),
    }
}

/// 文字列をエスケープ
fn escape_string(s: &[u8]) -> String {
    s.iter().map(|&c| escape_char(c)).collect()
}

/// C の `(void)` 単独パラメータ = 引数なし、を判定する。
///
/// K&R 方式との互換のため、C では明示的に `void` を単一パラメータとして
/// 書くことで「引数なし」を宣言する慣習がある (例: `int foo(void)`)。
/// Rust には対応する概念がなく、そのまま `_: ()` に訳すと呼出側との
/// 食い違いが起きるため、**パラメータなし** として生成する。
fn is_void_only_param_list(params: &[ParamDecl]) -> bool {
    if params.len() != 1 {
        return false;
    }
    let p = &params[0];
    // 名前付きでない（無名引数）こと、ポインタ派生していないこと、
    // かつ単独の TypeSpec::Void であること。
    let declarator_is_trivial = match &p.declarator {
        None => true,
        Some(d) => d.name.is_none() && d.derived.is_empty(),
    };
    let specs_is_void = p.specs.type_specs.len() == 1
        && matches!(p.specs.type_specs[0], TypeSpec::Void);
    declarator_is_trivial && specs_is_void
}

/// 式がゼロ定数かどうかを判定
fn is_zero_constant(expr: &Expr) -> bool {
    match &expr.kind {
        ExprKind::IntLit(0) => true,
        ExprKind::UIntLit(0) => true,
        _ => false,
    }
}

/// 式が bool として扱える形式かどうかを判定
///
/// キャスト `(expr as bool)` を含む場合も true を返す
pub fn is_boolean_expr(expr: &Expr) -> bool {
    match &expr.kind {
        ExprKind::Binary { op, .. } => matches!(op,
            BinOp::Lt | BinOp::Gt | BinOp::Le | BinOp::Ge |
            BinOp::Eq | BinOp::Ne | BinOp::LogAnd | BinOp::LogOr
        ),
        // (expr as bool) も bool を返す
        ExprKind::Cast { type_name, .. } => {
            // TypeSpec が Bool かチェック
            type_name.specs.type_specs.iter().any(|ts| {
                matches!(ts, TypeSpec::Bool)
            })
        }
        // LogNot は常に bool を返す（if 式として生成される）
        ExprKind::LogNot(_) => true,
        // 三項演算子: 両 arm が bool なら全体も bool。
        // <=5.34 の cBOOL は `((c) ? (bool)1 : (bool)0)` で、arm は
        // (bool) キャストにより bool として生成される。ここで ternary を
        // 非 bool と誤分類すると外側に更に `!= 0` が付き、Rust では
        // `bool != {integer}` (E0308) になる (5.36+ は cBOOL が
        // `((bool)(c))` の単純キャストになったため露出しない)
        ExprKind::Conditional { then_expr, else_expr, .. } => {
            is_boolean_expr(then_expr) && is_boolean_expr(else_expr)
        }
        _ => false,
    }
}


/// is_boolean_expr の再帰版: __builtin_expect(cond, val) を透過する
fn is_boolean_expr_recursive(expr: &Expr, interner: &StringInterner) -> bool {
    if is_boolean_expr(expr) {
        return true;
    }
    match &expr.kind {
        ExprKind::Call { func, args } => {
            if let ExprKind::Ident(name) = &func.kind {
                if interner.get(*name) == "__builtin_expect" && !args.is_empty() {
                    return is_boolean_expr_recursive(&args[0], interner);
                }
            }
        }
        // Cast to bool: 内側が bool なら全体も bool
        ExprKind::Cast { type_name, expr: inner } => {
            if type_name.specs.type_specs.iter().any(|ts| matches!(ts, TypeSpec::Bool)) {
                return true;
            }
            return is_boolean_expr_recursive(inner, interner);
        }
        _ => {}
    }
    false
}

/// コンテキスト付き bool 式判定: 呼び出し先マクロ/外部関数の戻り値型も考慮
pub fn is_boolean_expr_with_context(
    expr: &Expr,
    bool_return_macros: &HashSet<InternedStr>,
    bool_return_externals: &HashSet<InternedStr>,
) -> bool {
    if is_boolean_expr(expr) {
        return true;
    }
    match &expr.kind {
        ExprKind::Call { func, .. } => {
            if let ExprKind::Ident(name) = &func.kind {
                return bool_return_macros.contains(name)
                    || bool_return_externals.contains(name);
            }
        }
        ExprKind::MacroCall { name, .. } => {
            return bool_return_macros.contains(name)
                || bool_return_externals.contains(name);
        }
        _ => {}
    }
    false
}

/// TypeRepr がポインタ型かどうか判定
fn is_type_repr_pointer(ty: &crate::type_repr::TypeRepr) -> bool {
    use crate::type_repr::TypeRepr;
    match ty {
        TypeRepr::CType { derived, .. } => {
            derived.iter().any(|d| matches!(d, crate::type_repr::CDerivedType::Pointer { .. }))
        }
        TypeRepr::RustType { repr, .. } => {
            matches!(repr, crate::type_repr::RustTypeRepr::Pointer { .. })
        }
        TypeRepr::Inferred(inferred) => {
            inferred.resolved_type()
                .map(|r| is_type_repr_pointer(r))
                .unwrap_or(false)
        }
    }
}

/// 自家生成マクロの param に対する全制約のうち、Tier が最も高い
/// (=数値が小さい) 非 void TypeRepr のクローンを返す。
/// `param.expr_id()` および `param_to_exprs` から得た全 ExprId を走査する。
fn best_constraint_for_macro_param(
    info: &MacroInferInfo,
    param: &MacroParam,
) -> Option<crate::type_repr::TypeRepr> {
    let mut best: Option<(&crate::type_repr::TypeRepr, u8)> = None;

    // param_constraints (マクロ自身の apidoc パラメータ宣言など) を先に走査する:
    // tier 同点のとき「自身の宣言」が「呼び出し先由来の制約」に勝つように
    // (厳密未満 `<` の比較なので先着が優先される)。
    if let Some(constraints) = info.type_env.get_param_constraints(param.name) {
        for c in constraints {
            if c.ty.is_void() { continue; }
            let tier = c.ty.confidence_tier();
            if best.is_none() || tier < best.unwrap().1 {
                best = Some((&c.ty, tier));
            }
        }
    }

    let mut all_expr_ids: Vec<crate::ast::ExprId> = info
        .type_env
        .param_to_exprs
        .get(&param.name)
        .map(|ids| ids.iter().cloned().collect())
        .unwrap_or_default();
    all_expr_ids.push(param.expr_id());

    for expr_id in &all_expr_ids {
        if let Some(constraints) = info.type_env.expr_constraints.get(expr_id) {
            for c in constraints {
                if c.ty.is_void() { continue; }
                let tier = c.ty.confidence_tier();
                if best.is_none() || tier < best.unwrap().1 {
                    best = Some((&c.ty, tier));
                }
            }
        }
    }
    best.map(|(t, _)| t.clone())
}

/// `Expr` を再帰的に走査し、`ExprKind::Ident(name)` のうち `subs` に
/// マッチするものを `subs[name]` のクローンで置換する。
///
/// 用途: 自家生成マクロの本体式に対し、`(param_name → arg_expr)` の
/// 対応で alpha 置換を行う（C プリプロセッサの token 置換相当）。
/// `&MACRO(args)` を `&<inlined_body>` に展開するために使う。
fn substitute_idents(expr: &mut Expr, subs: &HashMap<InternedStr, &Expr>) {
    if let ExprKind::Ident(name) = &expr.kind {
        if let Some(replacement) = subs.get(name) {
            *expr = (*replacement).clone();
            return;
        }
    }
    match &mut expr.kind {
        ExprKind::Ident(_)
        | ExprKind::IntLit(_)
        | ExprKind::UIntLit(_)
        | ExprKind::FloatLit(_)
        | ExprKind::CharLit(_)
        | ExprKind::StringLit(_)
        | ExprKind::SizeofType(_)
        | ExprKind::Alignof(_) => {}
        ExprKind::Index { expr: e, index } => {
            substitute_idents(e, subs);
            substitute_idents(index, subs);
        }
        ExprKind::Call { func, args } => {
            substitute_idents(func, subs);
            for arg in args {
                substitute_idents(arg, subs);
            }
        }
        ExprKind::Member { expr: e, .. }
        | ExprKind::PtrMember { expr: e, .. }
        | ExprKind::PostInc(e)
        | ExprKind::PostDec(e)
        | ExprKind::PreInc(e)
        | ExprKind::PreDec(e)
        | ExprKind::AddrOf(e)
        | ExprKind::Deref(e)
        | ExprKind::UnaryPlus(e)
        | ExprKind::UnaryMinus(e)
        | ExprKind::BitNot(e)
        | ExprKind::LogNot(e)
        | ExprKind::Sizeof(e)
        | ExprKind::Cast { expr: e, .. } => substitute_idents(e, subs),
        ExprKind::Binary { lhs, rhs, .. }
        | ExprKind::Assign { lhs, rhs, .. }
        | ExprKind::Comma { lhs, rhs } => {
            substitute_idents(lhs, subs);
            substitute_idents(rhs, subs);
        }
        ExprKind::Conditional { cond, then_expr, else_expr } => {
            substitute_idents(cond, subs);
            substitute_idents(then_expr, subs);
            substitute_idents(else_expr, subs);
        }
        ExprKind::Assert { condition, .. } => substitute_idents(condition, subs),
        ExprKind::MacroCall { args, expanded, .. } => {
            for arg in args {
                substitute_idents(arg, subs);
            }
            substitute_idents(expanded, subs);
        }
        ExprKind::BuiltinCall { args, .. } => {
            for arg in args {
                if let crate::ast::BuiltinArg::Expr(e) = arg {
                    substitute_idents(e, subs);
                }
            }
        }
        ExprKind::CompoundLit { init, .. } => {
            for item in init {
                if let crate::ast::Initializer::Expr(e) = &mut item.init {
                    substitute_idents(e, subs);
                }
            }
        }
        // StmtExpr 内部は文を含むので alpha 置換は当面非対応（保守的）
        ExprKind::StmtExpr(_) => {}
    }
}

/// 式が「文の位置で値を捨てると `-D warnings` を踏む」形をしているか判定する。
/// 主な用途: C カンマ `(A, B)` の LHS を `let _ = A;` で包むかの判定。
///
/// 該当パターン（wrap が必要）:
/// - trailing-expr 付き block (`{ stmt; expr }`): `unused unary / logical op` の温床
/// - 論理演算 `&&` / `||` (`unused logical operation`)
/// - 単項式 `*p` `-x` `!b` 等 (`unused unary operation`)
///
/// 該当しないパターン（wrap 不要）:
/// - 関数呼出 / メソッド呼出: 大半の FFI 関数は `()` を返し must_use ではない。
///   `must_use` 戻り値（`size_of_val` 等）は `(void)cast` 経路で別途処理される
/// - assert!() や `{ stmt; }` のように `()` を返すと明らかなもの
/// - 単純な代入 `x = y` (Rust では `()` 型)
fn expr_yields_value_for_stmt_use(expr: &syn::Expr) -> bool {
    match expr {
        syn::Expr::Block(b) => {
            matches!(b.block.stmts.last(), Some(syn::Stmt::Expr(_, None)))
        }
        syn::Expr::Binary(b) => matches!(
            b.op,
            syn::BinOp::And(_) | syn::BinOp::Or(_)
        ),
        syn::Expr::Unary(_) => true,
        // パーレン内は中身を見る（`(a || b)` 等）
        syn::Expr::Paren(p) => expr_yields_value_for_stmt_use(&p.expr),
        _ => false,
    }
}

/// unsigned 型へのキャスト式かどうか判定
/// 例: "(x as usize)", "(x as u32)"
fn is_unsigned_cast_expr(expr_str: &str) -> bool {
    if let Some(pos) = expr_str.rfind(" as ") {
        let after = &expr_str[pos + 4..].trim_end_matches(')');
        matches!(*after, "usize" | "u8" | "u16" | "u32" | "u64" | "u128" | "c_uint" | "c_ulong" | "c_ulonglong")
    } else {
        false
    }
}

/// 指定された型名が unsigned 整数型（プリミティブ + 既知エイリアス）か判定。
/// 真なら `{ty}::MAX` のような associated const が使える。
fn is_unsigned_integer_target(ty: &str) -> bool {
    matches!(ty,
        "u8" | "u16" | "u32" | "u64" | "u128" | "usize" |
        "U8" | "U16" | "U32" | "U64" |
        "UV" | "STRLEN" | "Size_t" | "size_t" | "PERL_UINTMAX_T" |
        "c_uchar" | "c_ushort" | "c_uint" | "c_ulong" | "c_ulonglong"
    )
}


/// 式文字列の最外レベルの不要な括弧を除去する。
/// "(expr)" → "expr" （先頭の '(' と末尾の ')' が対応する場合のみ）
/// ただしブロック式 "({...})" は除去しない（ `{...} op expr` が構文エラーになるため）
fn strip_outer_parens(s: &str) -> &str {
    let s = s.trim();
    if s.len() < 2 || !s.starts_with('(') || !s.ends_with(')') {
        return s;
    }
    // 先頭の '(' と末尾の ')' が対応するかチェック
    let inner = &s[1..s.len() - 1];
    // ブロック式 ({...}) は strip しない
    if inner.trim_start().starts_with('{') {
        return s;
    }
    let mut depth = 0i32;
    for ch in inner.chars() {
        match ch {
            '(' | '{' | '[' => depth += 1,
            ')' | '}' | ']' => {
                depth -= 1;
                if depth < 0 {
                    // 内部で閉じ括弧が余る → 先頭と末尾は非対応
                    return s;
                }
            }
            _ => {}
        }
    }
    if depth == 0 {
        inner
    } else {
        s
    }
}

/// 式が NULL リテラル（整数 0 または (void*)0 のような Cast）かどうか判定
/// assert(expr || !"message") パターンの RHS からメッセージ文字列を抽出
fn extract_assert_message(expr: &Expr) -> Option<String> {
    if let ExprKind::LogNot(inner) = &expr.kind {
        if let ExprKind::StringLit(bytes) = &inner.kind {
            return Some(String::from_utf8_lossy(bytes).into_owned());
        }
    }
    None
}

/// assert 条件が `real_cond || !"message"` パターンかどうかを分解する
fn decompose_assert_with_message(condition: &Expr) -> Option<(&Expr, String)> {
    if let ExprKind::Binary { op: BinOp::LogOr, lhs, rhs } = &condition.kind {
        if let Some(msg) = extract_assert_message(rhs) {
            return Some((lhs, msg));
        }
    }
    None
}

/// Perl の SV サブタイプ（GV, HV, AV, CV, IO 等）から SV へのポインタキャストかどうかを判定
fn is_sv_subtype_cast(from: &UnifiedType, to: &UnifiedType) -> bool {
    // inner 型を取得 (Named 以外は "c_void" 扱いにして void ポインタ互換を
    // 拾う)。`Pointer { inner: Void }` は to_rust_string で c_void になる。
    let inner_name = |ut: &UnifiedType| -> Option<String> {
        match ut.inner_type()? {
            UnifiedType::Named(name) => Some(name.clone()),
            UnifiedType::Void => Some("c_void".to_string()),
            _ => None,
        }
    };
    let from_name = match inner_name(from) {
        Some(n) => n,
        None => return false,
    };
    let to_name = match inner_name(to) {
        Some(n) => n,
        None => return false,
    };
    // SV サブタイプのリスト (bindings.rs の構造体名は小文字版を出すケースが
    // 多い: `pub type SV = sv;` などの typedef エイリアス関係のため、両方を
    // 対象にする)
    const SV_SUBTYPES: &[&str] = &[
        "GV", "HV", "AV", "CV", "IO", "p5rx", "REGEXP",
        "gv", "hv", "av", "cv", "io", "regexp",
        // `typedef AV PAD;` (pad.h).
        //PADLIST/PADNAMELIST/PADNAME are separate structs, not SVs, so they must not be listed here.
        "PAD",
    ];
    let sv_like = |n: &str| n == "SV" || n == "sv";
    // SV ↔ サブタイプ（双方向）
    (SV_SUBTYPES.contains(&from_name.as_str()) && sv_like(&to_name))
        || (sv_like(&from_name) && SV_SUBTYPES.contains(&to_name.as_str()))
        // サブタイプ ↔ サブタイプ (GV → CV など)
        || (SV_SUBTYPES.contains(&from_name.as_str())
            && SV_SUBTYPES.contains(&to_name.as_str()))
        // c_void ↔ 任意のポインタ
        || to_name == "c_void"
        || from_name == "c_void"
}

fn is_null_literal(expr: &Expr) -> bool {
    match &expr.kind {
        ExprKind::IntLit(0) => true,
        ExprKind::Cast { expr: inner, .. } => is_null_literal(inner),
        _ => false,
    }
}

/// ポインタ型に対応する null ポインタ式を生成
fn null_ptr_expr(return_type: &UnifiedType) -> String {
    if return_type.is_const_pointer() {
        "std::ptr::null()".to_string()
    } else {
        "std::ptr::null_mut()".to_string()
    }
}

/// 型文字列を正規化して整数型なら canonical Rust primitive に変換
fn normalize_integer_type(ty: &str) -> Option<&'static str> {
    match ty {
        "u8" | "U8" | "c_uchar" => Some("u8"),
        "u16" | "U16" | "c_ushort" => Some("u16"),
        "u32" | "U32" | "c_uint" => Some("u32"),
        "u64" | "U64" | "UV" | "c_ulonglong"
            | "PERL_UINTMAX_T" => Some("u64"),
        "i8" | "I8" | "c_schar" | "c_char" => Some("i8"),
        "i16" | "I16" | "c_short" => Some("i16"),
        "i32" | "I32" | "c_int" => Some("i32"),
        "i64" | "I64" | "IV" | "c_longlong" => Some("i64"),
        "usize" | "STRLEN" => Some("usize"),
        "isize" | "SSize_t" | "ssize_t" | "PADOFFSET" => Some("isize"),
        // Perl 固有の整数 typedef。Stack_off_t は 5.32+ で I32、それ以前は
        // IV (i64) 相当だが、bindings.rs の型に合わせて i32 として扱う。
        "Stack_off_t" => Some("i32"),
        "c_long" => {
            Some(if *IS_WINDOWS_TARGET {
                "i32"
            } else {
                "i64"
            })
        },
        "c_ulong" => {
            Some(if *IS_WINDOWS_TARGET {
                "u32"
            } else {
                "u64"
            })
        },
        _ => None,
    }
}

/// 64-bit プラットフォームで i64/isize, u64/usize を同一視して比較
fn integer_types_compatible(a: &str, b: &str) -> bool {
    if a == b { return true; }
    matches!((a, b),
        ("i64", "isize") | ("isize", "i64") |
        ("u64", "usize") | ("usize", "u64")
    )
}

/// 整数型の幅ランク (昇格順序判定用)
/// returns (is_signed, width_rank)
fn integer_type_rank(ty: &str) -> Option<(bool, u8)> {
    match normalize_integer_type(ty)? {
        "u8" => Some((false, 1)), "i8" => Some((true, 1)),
        "u16" => Some((false, 2)), "i16" => Some((true, 2)),
        "u32" => Some((false, 4)), "i32" => Some((true, 4)),
        "u64" => Some((false, 8)), "i64" => Some((true, 8)),
        "usize" => Some((false, 8)), "isize" => Some((true, 8)),
        _ => None,
    }
}

/// 二項ビット演算で C の整数昇格に従い広い方の型を返す
/// 同一正規化型なら None（キャスト不要）
fn wider_integer_type(a: &str, b: &str) -> Option<&'static str> {
    let na = normalize_integer_type(a)?;
    let nb = normalize_integer_type(b)?;
    if na == nb { return None; }
    let (a_signed, a_rank) = integer_type_rank(a)?;
    let (_b_signed, b_rank) = integer_type_rank(b)?;
    if a_rank == b_rank {
        // 同一幅: unsigned が勝つ (C規格 6.3.1.8)
        Some(if a_signed { nb } else { na })
    } else if a_rank > b_rank {
        Some(na)
    } else {
        Some(nb)
    }
}

/// 2 つのポインタ型の inner が Rust として「同じ型」か判定する
/// （const/mut は無視）。
///
/// `*mut c_char` と `*mut i8`、`*const c_int` と `*const i32` のように、
/// C 別名と Rust プリミティブの対応関係を吸収する。
/// 二重ポインタ `**const T` vs `**mut T` にも再帰的に対応する。
pub fn pointer_inner_compatible(a: &UnifiedType, b: &UnifiedType) -> bool {
    let a_inner = match a { UnifiedType::Pointer { inner, .. } => inner.as_ref(), _ => return false };
    let b_inner = match b { UnifiedType::Pointer { inner, .. } => inner.as_ref(), _ => return false };
    type_inner_compatible(a_inner, b_inner)
}

/// `pointer_inner_compatible` の再帰ヘルパ。ポインタ以外の inner も扱える。
fn type_inner_compatible(a: &UnifiedType, b: &UnifiedType) -> bool {
    // 両方ポインタなら再帰
    if a.is_pointer() && b.is_pointer() {
        return pointer_inner_compatible(a, b);
    }
    // 整数型は normalize_integer_type で正規化して比較
    let a_s = a.to_rust_string();
    let b_s = b.to_rust_string();
    if let (Some(na), Some(nb)) = (normalize_integer_type(&a_s), normalize_integer_type(&b_s)) {
        return na == nb;
    }
    // void は双方で吸収
    if a.is_void() && b.is_void() { return true; }
    // それ以外は文字列一致で判定
    a_s == b_s
}

/// 両方ポインタで、inner は compatible だが const/mut が違う場合に true。
pub fn pointer_const_differs(a: &UnifiedType, b: &UnifiedType) -> bool {
    if !(a.is_pointer() && b.is_pointer()) { return false; }
    if a.is_const_pointer() == b.is_const_pointer() { return false; }
    pointer_inner_compatible(a, b)
}

/// マクロ本体を走査し、`&mut param` や代入先として使用されるパラメータを検出する
/// ポインタパラメータが *mut である必要があるかを判定する。
/// callee_const_params: 呼び出し先マクロで *const に確定したパラメータ情報
///   key = マクロ名(InternedStr), value = const パラメータの引数位置集合
pub fn collect_must_mut_pointer_params(
    parse_result: &ParseResult,
    params: &[MacroParam],
    callee_const_params: &HashMap<InternedStr, HashSet<usize>>,
) -> HashSet<InternedStr> {
    let param_names: HashSet<InternedStr> = params.iter().map(|p| p.name).collect();
    let mut result = HashSet::new();
    match parse_result {
        ParseResult::Expression(expr) => {
            collect_must_mut_from_expr(expr, &param_names, callee_const_params, &mut result);
        }
        ParseResult::Statement(items) => {
            for item in items {
                if let BlockItem::Stmt(stmt) = item {
                    collect_must_mut_from_stmt(stmt, &param_names, callee_const_params, &mut result);
                }
            }
        }
        ParseResult::Unparseable(_) => {}
    }
    result
}

pub fn collect_must_mut_from_stmt(
    stmt: &Stmt,
    params: &HashSet<InternedStr>,
    callee_const: &HashMap<InternedStr, HashSet<usize>>,
    result: &mut HashSet<InternedStr>,
) {
    match stmt {
        Stmt::Expr(Some(expr), _) | Stmt::Return(Some(expr), _) => {
            collect_must_mut_from_expr(expr, params, callee_const, result);
        }
        Stmt::Compound(compound) => {
            for item in &compound.items {
                match item {
                    BlockItem::Stmt(s) => collect_must_mut_from_stmt(s, params, callee_const, result),
                    BlockItem::Decl(decl) => {
                        for init_decl in &decl.declarators {
                            if let Some(Initializer::Expr(init)) = &init_decl.init {
                                collect_must_mut_from_expr(init, params, callee_const, result);
                            }
                        }
                    }
                }
            }
        }
        Stmt::If { cond, then_stmt, else_stmt, .. } => {
            collect_must_mut_from_expr(cond, params, callee_const, result);
            collect_must_mut_from_stmt(then_stmt, params, callee_const, result);
            if let Some(e) = else_stmt {
                collect_must_mut_from_stmt(e, params, callee_const, result);
            }
        }
        Stmt::While { cond, body, .. } | Stmt::DoWhile { cond, body, .. } => {
            collect_must_mut_from_expr(cond, params, callee_const, result);
            collect_must_mut_from_stmt(body, params, callee_const, result);
        }
        Stmt::For { init, cond, step, body, .. } => {
            if let Some(ForInit::Expr(e)) = init { collect_must_mut_from_expr(e, params, callee_const, result); }
            if let Some(e) = cond { collect_must_mut_from_expr(e, params, callee_const, result); }
            if let Some(e) = step { collect_must_mut_from_expr(e, params, callee_const, result); }
            collect_must_mut_from_stmt(body, params, callee_const, result);
        }
        Stmt::Switch { expr, body, .. } => {
            collect_must_mut_from_expr(expr, params, callee_const, result);
            collect_must_mut_from_stmt(body, params, callee_const, result);
        }
        _ => {}
    }
}

pub fn collect_must_mut_from_expr(
    expr: &Expr,
    params: &HashSet<InternedStr>,
    callee_const: &HashMap<InternedStr, HashSet<usize>>,
    result: &mut HashSet<InternedStr>,
) {
    match &expr.kind {
        // *param = expr, param->field = expr → param must be *mut
        ExprKind::Assign { lhs, rhs, .. } => {
            mark_lvalue_mut(lhs, params, result);
            collect_must_mut_from_expr(lhs, params, callee_const, result);
            collect_must_mut_from_expr(rhs, params, callee_const, result);
        }
        // ++(*param), (*param)++ 等
        ExprKind::PreInc(inner) | ExprKind::PreDec(inner) |
        ExprKind::PostInc(inner) | ExprKind::PostDec(inner) => {
            mark_lvalue_mut(inner, params, result);
            collect_must_mut_from_expr(inner, params, callee_const, result);
        }
        // func(param) — 呼び出し先の引数 mutability をチェック
        ExprKind::Call { func, args } => {
            // 呼び出し先マクロの const 情報をチェック
            if let ExprKind::Ident(func_name) = &func.kind {
                let const_arg_positions = callee_const.get(func_name);
                for (i, arg) in args.iter().enumerate() {
                    if let ExprKind::Ident(arg_name) = &arg.kind {
                        if params.contains(arg_name) {
                            // 呼び出し先の i 番目が const なら mut 不要
                            let is_const_at_callee = const_arg_positions
                                .map_or(false, |positions| positions.contains(&i));
                            if !is_const_at_callee {
                                // 呼び出し先が const でない（or 情報なし）→ mut 必要
                                result.insert(*arg_name);
                            }
                        }
                    }
                    collect_must_mut_from_expr(arg, params, callee_const, result);
                }
            } else {
                for arg in args {
                    collect_must_mut_from_expr(arg, params, callee_const, result);
                }
            }
            collect_must_mut_from_expr(func, params, callee_const, result);
        }
        // 再帰
        ExprKind::Binary { lhs, rhs, .. } | ExprKind::Comma { lhs, rhs } => {
            collect_must_mut_from_expr(lhs, params, callee_const, result);
            collect_must_mut_from_expr(rhs, params, callee_const, result);
        }
        ExprKind::Conditional { cond, then_expr, else_expr } => {
            collect_must_mut_from_expr(cond, params, callee_const, result);
            collect_must_mut_from_expr(then_expr, params, callee_const, result);
            collect_must_mut_from_expr(else_expr, params, callee_const, result);
        }
        // MacroCall(name, args) — 呼び出し先マクロの引数 mutability をチェック
        ExprKind::MacroCall { name, args, expanded, .. } => {
            let const_arg_positions = callee_const.get(name);
            for (i, arg) in args.iter().enumerate() {
                if let ExprKind::Ident(arg_name) = &arg.kind {
                    if params.contains(arg_name) {
                        let is_const_at_callee = const_arg_positions
                            .map_or(false, |positions| positions.contains(&i));
                        if !is_const_at_callee {
                            result.insert(*arg_name);
                        }
                    }
                }
                collect_must_mut_from_expr(arg, params, callee_const, result);
            }
            collect_must_mut_from_expr(expanded, params, callee_const, result);
        }
        ExprKind::Deref(inner) | ExprKind::UnaryMinus(inner) | ExprKind::BitNot(inner) |
        ExprKind::LogNot(inner) | ExprKind::AddrOf(inner) |
        ExprKind::Cast { expr: inner, .. } => {
            collect_must_mut_from_expr(inner, params, callee_const, result);
        }
        ExprKind::Member { expr: inner, .. } | ExprKind::PtrMember { expr: inner, .. } => {
            collect_must_mut_from_expr(inner, params, callee_const, result);
        }
        ExprKind::Sizeof(inner) => {
            collect_must_mut_from_expr(inner, params, callee_const, result);
        }
        ExprKind::Assert { condition, .. } => {
            collect_must_mut_from_expr(condition, params, callee_const, result);
        }
        ExprKind::StmtExpr(compound) => {
            for item in &compound.items {
                match item {
                    BlockItem::Stmt(s) => collect_must_mut_from_stmt(s, params, callee_const, result),
                    BlockItem::Decl(decl) => {
                        for init_decl in &decl.declarators {
                            if let Some(Initializer::Expr(init)) = &init_decl.init {
                                collect_must_mut_from_expr(init, params, callee_const, result);
                            }
                        }
                    }
                }
            }
        }
        _ => {}
    }
}

/// 代入先の式に含まれるパラメータを must-mut としてマークする
pub fn mark_lvalue_mut(expr: &Expr, params: &HashSet<InternedStr>, result: &mut HashSet<InternedStr>) {
    match &expr.kind {
        // *param = ... → param must be *mut
        ExprKind::Deref(inner) => {
            if let ExprKind::Ident(name) = &inner.kind {
                if params.contains(name) {
                    result.insert(*name);
                }
            }
            // (*param).field の場合も再帰的にチェック
            mark_lvalue_mut(inner, params, result);
        }
        // param->field = ... → param must be *mut
        ExprKind::PtrMember { expr: inner, .. } => {
            if let ExprKind::Ident(name) = &inner.kind {
                if params.contains(name) {
                    result.insert(*name);
                }
            }
            mark_lvalue_mut(inner, params, result);
        }
        // (*param).field = ... → param must be *mut
        ExprKind::Member { expr: inner, .. } => {
            mark_lvalue_mut(inner, params, result);
        }
        // (SomeType*)param → キャスト先が *mut ならパラメータも mut
        ExprKind::Cast { expr: inner, type_name } => {
            if let ExprKind::Ident(name) = &inner.kind {
                if params.contains(name) {
                    // キャスト先がポインタで non-const なら mut 必要
                    let has_non_const_ptr = type_name.declarator.as_ref()
                        .map(|d| d.derived.iter().any(|dd| {
                            matches!(dd, crate::ast::DerivedDecl::Pointer(q) if !q.is_const)
                        }))
                        .unwrap_or(false);
                    if has_non_const_ptr {
                        result.insert(*name);
                    }
                }
            }
            mark_lvalue_mut(inner, params, result);
        }
        // MacroCall(name, expanded) → expanded 形式で lvalue を再帰チェック
        ExprKind::MacroCall { expanded, args, .. } => {
            mark_lvalue_mut(expanded, params, result);
            // 引数にパラメータが直接渡されている場合もチェック
            for arg in args {
                mark_lvalue_mut(arg, params, result);
            }
        }
        // Call の lvalue 使用: func(param) が lvalue として使われる場合
        // マクロ関数の呼び出し結果が lvalue なら、引数パラメータは *mut 必要
        ExprKind::Call { args, .. } => {
            for arg in args {
                if let ExprKind::Ident(name) = &arg.kind {
                    if params.contains(name) {
                        result.insert(*name);
                    }
                }
                mark_lvalue_mut(arg, params, result);
            }
        }
        _ => {}
    }
}

// (旧 collect_mut_params 系は Phase 2 の crate::local_usage へ移設した。
//  GH #16/#22/#23 — Pass Separation Rule)

/// 構造体フィールド名 → 型の逆引きマップを構築
/// 全構造体で同名フィールドの型が一致する場合のみ含む
/// 型文字列が関数ポインタ形式（裸 fn または `Option<...fn(...)>`）かを判定する。
///
/// `quote::ToTokens` 経由の出力は `fn (` のようにスペースが入ることがあるため、
/// `fn(` と `fn (` の両方を許容する。
fn type_str_is_fn_pointer(ty_str: &str) -> bool {
    ty_str.contains("fn(") || ty_str.contains("fn (")
}

fn build_field_type_map(dict: Option<&RustDeclDict>) -> HashMap<String, UnifiedType> {
    let mut map: HashMap<String, UnifiedType> = HashMap::new();
    let mut conflicts: HashSet<String> = HashSet::new();
    if let Some(dict) = dict {
        for st in dict.structs.values() {
            for field in &st.fields {
                if conflicts.contains(&field.name) {
                    continue;
                }
                match map.entry(field.name.clone()) {
                    std::collections::hash_map::Entry::Vacant(e) => {
                        e.insert(field.uty.clone());
                    }
                    std::collections::hash_map::Entry::Occupied(e) => {
                        if e.get() != &field.uty {
                            conflicts.insert(field.name.clone());
                            e.remove();
                        }
                    }
                }
            }
        }
        // bitfield アクセサ getter の戻り値型を統合。
        // C ソースの `(*o).op_type` （Member 式）は意味的に「op_type フィールド」
        // 相当だが、bindings.rs では bitfield アクセサ getter (`pub fn op_type(&self) -> U16`)
        // として現れる。型推論では同名のフィールドアクセスと同じ扱いで OK。
        for ((_struct, method), ret_ty) in &dict.bitfield_method_types {
            if conflicts.contains(method) {
                continue;
            }
            let uty = UnifiedType::from_rust_str(ret_ty);
            match map.entry(method.clone()) {
                std::collections::hash_map::Entry::Vacant(e) => {
                    e.insert(uty);
                }
                std::collections::hash_map::Entry::Occupied(e) => {
                    if e.get() != &uty {
                        conflicts.insert(method.clone());
                        e.remove();
                    }
                }
            }
        }
    }
    map
}

/// コード生成の設定
#[derive(Debug, Clone)]
pub struct CodegenConfig {
    /// inline 関数を出力するか
    pub emit_inline_fns: bool,
    /// マクロを出力するか
    pub emit_macros: bool,
    /// コメントにソース位置を含めるか
    pub include_source_location: bool,
    /// ヘッダーに出力する use 文
    /// 空の場合はデフォルトの use 文を出力
    pub use_statements: Vec<String>,
    /// AST ダンプ対象関数名（デバッグ用）
    pub dump_ast_for: Option<String>,
    /// 型推論ダンプ対象関数名（デバッグ用）
    pub dump_types_for: Option<String>,
}

impl Default for CodegenConfig {
    fn default() -> Self {
        Self {
            emit_inline_fns: true,
            emit_macros: true,
            include_source_location: true,
            use_statements: Vec::new(),
            dump_ast_for: None,
            dump_types_for: None,
        }
    }
}

impl CodegenConfig {
    /// デフォルトの use 文を取得
    ///
    /// 生成コードで使用される C 型をインポートする。
    /// `size_t` などは Rust 組み込み型のエイリアスとして定義。
    pub fn default_use_statements() -> Vec<String> {
        vec![
            // 生成コードで実際に参照されない type alias / import が出ても
            // CI の `-D warnings` で落ちないよう dead_code / unused_imports を allow。
            "#[allow(unused_imports)] use std::ffi::{c_void, c_char, c_uchar, c_int, c_uint, c_long, c_ulong, c_short, c_ushort, c_longlong, c_ulonglong}".to_string(),
            "#[allow(non_camel_case_types, dead_code)] type size_t = usize".to_string(),
            "#[allow(non_camel_case_types, dead_code)] type ssize_t = isize".to_string(),
            "#[allow(non_camel_case_types, dead_code)] type SSize_t = isize".to_string(),
        ]
    }

    /// use 文を設定
    pub fn with_use_statements(mut self, statements: Vec<String>) -> Self {
        self.use_statements = statements;
        self
    }

    /// use 文を追加
    pub fn add_use_statement(mut self, statement: impl Into<String>) -> Self {
        self.use_statements.push(statement.into());
        self
    }
}

/// 生成ステータス
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenerateStatus {
    /// 正常生成
    Success,
    /// パース失敗（S式をコメント出力）
    ParseFailed,
    /// 型推論不完全（型付S式をコメント出力）
    TypeIncomplete,
    /// 利用不可関数を呼び出す（コメント出力）
    CallsUnavailable,
    /// goto を含む（生成対象から除外）
    ContainsGoto,
    /// ジェネリクス型パラメータを含む（Rust の as T キャスト不可）
    GenericUnsupported,
    /// スキップ（対象外）
    Skip,
}

/// 文が goto を含むか再帰的に検査
fn stmt_contains_goto(stmt: &Stmt) -> bool {
    match stmt {
        Stmt::Goto(_, _) => true,
        Stmt::Compound(cs) => block_items_contain_goto(&cs.items),
        Stmt::If { then_stmt, else_stmt, .. } => {
            stmt_contains_goto(then_stmt)
                || else_stmt.as_ref().is_some_and(|s| stmt_contains_goto(s))
        }
        Stmt::Switch { body, .. }
        | Stmt::While { body, .. }
        | Stmt::For { body, .. } => stmt_contains_goto(body),
        Stmt::DoWhile { body, .. } => stmt_contains_goto(body),
        Stmt::Label { stmt, .. } => stmt_contains_goto(stmt),
        Stmt::Case { stmt, .. } | Stmt::Default { stmt, .. } => stmt_contains_goto(stmt),
        _ => false,
    }
}

/// ブロック項目リストが goto を含むか検査
fn block_items_contain_goto(items: &[BlockItem]) -> bool {
    items.iter().any(|item| match item {
        BlockItem::Stmt(stmt) => stmt_contains_goto(stmt),
        BlockItem::Decl(_) => false,
    })
}

/// 文がトップレベルで `break;` を含むか検査
/// (ネストされた loop/switch 内の break は除外)。
fn stmt_contains_top_level_break(stmt: &Stmt) -> bool {
    match stmt {
        Stmt::Break(_) => true,
        Stmt::Compound(cs) => cs.items.iter().any(|item| match item {
            BlockItem::Stmt(s) => stmt_contains_top_level_break(s),
            BlockItem::Decl(_) => false,
        }),
        Stmt::If { then_stmt, else_stmt, .. } => {
            stmt_contains_top_level_break(then_stmt)
                || else_stmt.as_ref().is_some_and(|s| stmt_contains_top_level_break(s))
        }
        Stmt::Label { stmt, .. } => stmt_contains_top_level_break(stmt),
        // ネストした loop/switch 内の break は「外側ループ」に関係ないので
        // 再帰しない
        _ => false,
    }
}

/// コード生成統計
#[derive(Debug, Clone, Default)]
pub struct CodegenStats {
    /// 正常生成されたマクロ数
    pub macros_success: usize,
    /// パース失敗マクロ数
    pub macros_parse_failed: usize,
    /// 型推論失敗マクロ数
    pub macros_type_incomplete: usize,
    /// 利用不可関数呼び出しマクロ数
    pub macros_calls_unavailable: usize,
    /// カスケード依存でコメントアウトされたマクロ数
    pub macros_cascade_unavailable: usize,
    /// ジェネリクス未対応マクロ数
    pub macros_generic_unsupported: usize,
    /// 未解決シンボルを含むマクロ数
    pub macros_unresolved_names: usize,
    /// 正常生成された inline 関数数
    pub inline_fns_success: usize,
    /// 型推論失敗 inline 関数数
    pub inline_fns_type_incomplete: usize,
    /// 未解決シンボルを含む inline 関数数
    pub inline_fns_unresolved_names: usize,
    /// カスケード依存でコメントアウトされた inline 関数数
    pub inline_fns_cascade_unavailable: usize,
    /// goto を含む inline 関数数
    pub inline_fns_contains_goto: usize,
}

/// codegen の関数別結果。emit された名前と、されなかった名前 → 理由。
///
/// `CodegenDriver` が出力 (Pass 3) の各マーカー分岐で記録する。
/// `--require-codegen-list` の検証 (`check_required`) に使う。
/// BTree 系なのは違反レポートの列挙順を決定化するため。
#[derive(Debug, Clone, Default)]
pub struct CodegenReport {
    /// 実関数として emit された名前
    pub emitted: BTreeSet<String>,
    /// emit されなかった名前 → 理由 (マーカー種別 + 詳細)
    pub skipped: BTreeMap<String, String>,
}

impl CodegenReport {
    fn record_emit(&mut self, name: impl Into<String>) {
        self.emitted.insert(name.into());
    }

    fn record_skip(&mut self, name: impl Into<String>, reason: impl Into<String>) {
        self.skipped.insert(name.into(), reason.into());
    }

    /// required のうち emit されなかったものを理由付きで返す。
    /// 全て emit 済みなら Ok。emitted にも skipped にも無い名前は
    /// codegen 対象に現れなかったもの (apidoc 宣言なし、ヘッダに定義なし等)。
    pub fn check_required(&self, required: &[String]) -> Result<(), RequireCodegenError> {
        let mut violations = Vec::new();
        for name in required {
            if self.emitted.contains(name) {
                continue;
            }
            let reason = match self.skipped.get(name) {
                Some(r) => r.clone(),
                None => "not a codegen target (no apidoc declaration or not defined in headers)"
                    .to_string(),
            };
            violations.push(RequireViolation {
                name: name.clone(),
                reason,
            });
        }
        if violations.is_empty() {
            Ok(())
        } else {
            Err(RequireCodegenError { violations })
        }
    }
}

/// `--require-codegen-list` 違反 (必須名が emit されなかった)
#[derive(Debug, Clone)]
pub struct RequireCodegenError {
    pub violations: Vec<RequireViolation>,
}

/// 1 件の違反: 名前と、emit されなかった理由
#[derive(Debug, Clone)]
pub struct RequireViolation {
    pub name: String,
    pub reason: String,
}

impl std::fmt::Display for RequireCodegenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "require-codegen-list violation: {} function(s) not generated:",
            self.violations.len()
        )?;
        for v in &self.violations {
            writeln!(f, "  {} - {}", v.name, v.reason)?;
        }
        Ok(())
    }
}

impl std::error::Error for RequireCodegenError {}

/// 一つの関数の生成結果
#[derive(Debug, Clone)]
pub struct GeneratedCode {
    /// 生成されたコード
    pub code: String,
    /// 不完全マーカーの数
    pub incomplete_count: usize,
    /// 検出された未解決シンボル名（重複なし、出現順）
    pub unresolved_names: Vec<String>,
    /// 使用された libc 関数名
    pub used_libc_fns: HashSet<String>,
    /// codegen で検出されたエラー（コメントアウトの理由）
    pub codegen_errors: Vec<String>,
}

impl GeneratedCode {
    /// 生成が完全かどうか（不完全マーカーがないか）
    pub fn is_complete(&self) -> bool {
        self.incomplete_count == 0
    }

    /// 未解決シンボルがあるかどうか
    pub fn has_unresolved_names(&self) -> bool {
        !self.unresolved_names.is_empty()
    }
}

/// 単一関数を生成するためのコード生成器（使い捨て）
///
/// 各関数の生成ごとにフレッシュなインスタンスを作成して使用する。
/// 生成中に不完全マーカーが出力された回数をカウントし、
/// 生成完了時に `GeneratedCode` として結果を返す。
pub struct RustCodegen<'a> {
    interner: &'a StringInterner,
    /// Enum バリアント辞書（パターンマッチ用）
    enum_dict: &'a EnumDict,
    /// マクロ推論コンテキスト（THX マクロ呼び出し判定用）
    macro_ctx: &'a MacroInferContext,
    /// bindings.rs から抽出した情報
    bindings_info: BindingsInfo,
    /// 内部バッファ（生成結果を蓄積）
    buffer: String,
    /// 不完全マーカーの生成回数
    incomplete_count: usize,
    /// 現在生成中のマクロの型パラメータマップ
    /// 仮引数名(InternedStr) → ジェネリック名("T", "U", ...)
    current_type_param_map: HashMap<InternedStr, String>,
    /// 現在生成中のマクロのリテラル文字列パラメータ名の集合
    current_literal_string_params: HashSet<InternedStr>,
    /// 現在生成中の関数の戻り値型
    current_return_type: Option<UnifiedType>,
    /// Call式のlvalue展開時に使用するパラメータ置換テーブル
    /// マクロ仮引数名 → 実引数のRust文字列
    param_substitutions: HashMap<InternedStr, String>,
    /// 現在生成中の関数のパラメータ型情報
    /// パラメータ名 → 型
    current_param_types: HashMap<InternedStr, UnifiedType>,
    /// 既知シンボル集合への参照（未解決シンボル検出用）
    known_symbols: &'a KnownSymbols,
    /// 現在の関数のローカルスコープ（パラメータ名 + ローカル変数名）
    current_local_names: HashSet<InternedStr>,
    /// 検出された未解決シンボル名（重複なし、出現順）
    unresolved_names: Vec<String>,
    /// 使用された libc 関数名
    used_libc_fns: HashSet<String>,
    /// Rust 宣言辞書への参照（関数パラメータ型参照用）
    rust_decl_dict: Option<&'a RustDeclDict>,
    /// inline 関数辞書への参照（戻り値型/引数型判定用）
    inline_fn_dict: Option<&'a crate::inline_fn::InlineFnDict>,
    /// FieldsDict への参照（共通フィールドマクロの canonical type 参照用）
    fields_dict: Option<&'a crate::fields_dict::FieldsDict>,
    /// 構造体フィールド名 → 型の逆引きマップ
    field_type_map: HashMap<String, UnifiedType>,
    /// AST ダンプ対象関数名（デバッグ用）
    dump_ast_for: Option<String>,
    /// 型推論ダンプ対象関数名（デバッグ用）
    dump_types_for: Option<String>,
    /// const ポインタに変換可能なパラメータの引数位置集合
    const_pointer_positions: HashSet<usize>,
    /// 再代入されるローカル変数名の集合（let mut 判定用）
    mut_local_names: HashSet<InternedStr>,
    /// Phase 2 の名前使用解析 (local_usage)。zeroed 初期化判定などで参照
    current_local_usage: Option<crate::local_usage::LocalUsageAnalysis>,
    /// 本体で未使用のパラメータ名（`_` prefix を付ける。GH #22）
    unused_param_names: HashSet<InternedStr>,
    /// codegen で検出されたエラー
    codegen_errors: Vec<String>,
    /// このマクロが bool を返すと判定されたか
    is_bool_return: bool,
    /// codegen で bool を返すと判定されたマクロの集合（呼び出し先の bool 判定用）
    bool_return_macros: HashSet<InternedStr>,
    /// 対象 perl が threaded build か（false なら my_perl 注入を抑止）
    perl_threaded: bool,
}

/// コード生成全体を管理する構造体
///
/// 実際の出力先（Write）を保持し、生成の成功/失敗に応じて
/// 適切な形式で出力する。
pub struct CodegenDriver<'a, W: Write> {
    writer: W,
    interner: &'a StringInterner,
    /// Enum バリアント辞書（パターンマッチ用）
    enum_dict: &'a EnumDict,
    /// マクロ推論コンテキスト（THX マクロ呼び出し判定用）
    macro_ctx: &'a MacroInferContext,
    /// bindings.rs から抽出した情報
    bindings_info: BindingsInfo,
    config: CodegenConfig,
    stats: CodegenStats,
    /// 関数別の生成結果 (--require-codegen-list 検証用)
    report: CodegenReport,
    /// 生成されたコード全体で使用された libc 関数名
    used_libc_fns: HashSet<String>,
    /// 正常生成された inline 関数名（クロスドメインカスケード検出用）
    successfully_generated_inlines: HashSet<InternedStr>,
    /// 生成可能と予測されるマクロの集合（inline→macro カスケード検出用）
    generatable_macros: HashSet<InternedStr>,
    /// const ポインタに変換可能なマクロパラメータ: マクロ名 → const パラメータの引数位置集合
    const_pointer_params: HashMap<InternedStr, HashSet<usize>>,
    /// bool を返すと判定されたマクロの集合
    bool_return_macros: HashSet<InternedStr>,
    /// 対象 perl が threaded build か（false なら my_perl 注入を抑止）
    perl_threaded: bool,
}

impl<'a> RustCodegen<'a> {
    /// 新しい単一関数用コード生成器を作成
    pub fn new(
        interner: &'a StringInterner,
        enum_dict: &'a EnumDict,
        macro_ctx: &'a MacroInferContext,
        bindings_info: BindingsInfo,
        known_symbols: &'a KnownSymbols,
        rust_decl_dict: Option<&'a RustDeclDict>,
        inline_fn_dict: Option<&'a crate::inline_fn::InlineFnDict>,
    ) -> Self {
        Self {
            interner,
            enum_dict,
            macro_ctx,
            bindings_info,
            buffer: String::new(),
            incomplete_count: 0,
            current_type_param_map: HashMap::new(),
            current_literal_string_params: HashSet::new(),
            current_return_type: None,
            param_substitutions: HashMap::new(),
            current_param_types: HashMap::new(),
            known_symbols,
            current_local_names: HashSet::new(),
            unresolved_names: Vec::new(),
            used_libc_fns: HashSet::new(),
            rust_decl_dict,
            inline_fn_dict,
            fields_dict: None,
            field_type_map: build_field_type_map(rust_decl_dict),
            dump_ast_for: None,
            dump_types_for: None,
            const_pointer_positions: HashSet::new(),
            is_bool_return: false,
            bool_return_macros: HashSet::new(),
            mut_local_names: HashSet::new(),
            current_local_usage: None,
            unused_param_names: HashSet::new(),
            codegen_errors: Vec::new(),
            // デフォルトは threaded（後方互換）。Driver 経由で
            // with_perl_threaded() で上書きされる。
            perl_threaded: true,
        }
    }

    /// 対象 perl が threaded かを設定
    pub fn with_perl_threaded(mut self, threaded: bool) -> Self {
        self.perl_threaded = threaded;
        self
    }

    /// AST ダンプ対象関数名を設定（デバッグ用）
    pub fn with_dump_ast_for(mut self, name: Option<String>) -> Self {
        self.dump_ast_for = name;
        self
    }

    pub fn with_dump_types_for(mut self, name: Option<String>) -> Self {
        self.dump_types_for = name;
        self
    }

    /// FieldsDict への参照を設定
    pub fn with_fields_dict(mut self, dict: &'a crate::fields_dict::FieldsDict) -> Self {
        self.fields_dict = Some(dict);
        // bindings.rs に無い自動生成 struct (body_details 等) の
        // フィールド型を field_type_map にマージし、Member 式の型推論で
        // 利用できるようにする。bindings.rs 側で既に登録されている
        // フィールドは優先する。
        for (_name, def) in dict.iter_struct_defs() {
            for m in &def.members {
                let member_name = self.interner.get(m.name).to_string();
                if self.field_type_map.contains_key(&member_name) {
                    continue;
                }
                let rust_ty = m.type_repr.to_rust_string(self.interner);
                self.field_type_map
                    .insert(member_name, UnifiedType::from_rust_str(&rust_ty));
            }
        }
        self
    }

    /// const ポインタ位置を設定
    pub fn with_const_pointer_positions(mut self, positions: HashSet<usize>) -> Self {
        self.const_pointer_positions = positions;
        self
    }

    /// bool 戻り値フラグと bool マクロ集合を設定
    pub fn with_bool_return(mut self, is_bool: bool, bool_macros: HashSet<InternedStr>) -> Self {
        self.is_bool_return = is_bool;
        self.bool_return_macros = bool_macros;
        self
    }

    /// 指定された関数名が AST ダンプ対象かどうかを判定し、対象なら AST をコメントとして出力
    /// 型推論結果を stderr にダンプ（デバッグ用）
    fn dump_type_info(&self, name_str: &str, info: &MacroInferInfo, params_str: &str, return_type: &str) {
        eprintln!("=== Type dump for {} ===", name_str);
        // パラメータ型
        for (i, p) in info.params.iter().enumerate() {
            let pname = self.interner.get(p.name);
            let is_const = info.const_pointer_positions.contains(&i);
            // 全制約を表示
            let expr_ids: Vec<_> = info.type_env.param_to_exprs
                .get(&p.name)
                .map(|ids| ids.iter().cloned().collect())
                .unwrap_or_default();
            let mut all_ids = expr_ids;
            all_ids.push(p.expr_id());
            eprintln!("  param[{}] {} (const_position={})", i, pname, is_const);
            for eid in &all_ids {
                if let Some(constraints) = info.type_env.expr_constraints.get(eid) {
                    for c in constraints {
                        eprintln!("    constraint: tier={} rust={} context={} source={:?}",
                            c.ty.confidence_tier(),
                            c.ty.to_rust_string(self.interner),
                            c.context,
                            match &c.ty {
                                crate::type_repr::TypeRepr::CType { source, .. } => format!("{:?}", source),
                                crate::type_repr::TypeRepr::RustType { source, .. } => format!("{:?}", source),
                                crate::type_repr::TypeRepr::Inferred(i) => format!("Inferred({:?})", std::mem::discriminant(i)),
                            }
                        );
                    }
                }
            }
        }
        eprintln!("  params_str: {}", params_str);
        // 戻り値型
        eprintln!("  return_type: {}", return_type);
        eprintln!("  is_bool_return: {}", info.is_bool_return);
        if let Some(ty) = info.get_return_type() {
            eprintln!("  return TypeRepr: tier={} rust={}", ty.confidence_tier(), ty.to_rust_string(self.interner));
        }
        // return_constraints (apidoc 由来)
        if !info.type_env.return_constraints.is_empty() {
            eprintln!("  return_constraints:");
            for c in &info.type_env.return_constraints {
                eprintln!("    tier={} rust={} context={}", c.ty.confidence_tier(), c.ty.to_rust_string(self.interner), c.context);
            }
        }
        // ルート式の全制約
        if let ParseResult::Expression(ref expr) = info.parse_result {
            if let Some(constraints) = info.type_env.expr_constraints.get(&expr.id) {
                eprintln!("  root expr constraints:");
                for c in constraints {
                    eprintln!("    tier={} rust={} context={}",
                        c.ty.confidence_tier(),
                        c.ty.to_rust_string(self.interner),
                        c.context,
                    );
                }
            }
        }
        eprintln!("=== End type dump ===");
    }

    fn dump_ast_comment_for_expr(&mut self, name_str: &str, parse_result: &ParseResult) {
        if self.dump_ast_for.as_deref() != Some(name_str) {
            return;
        }
        let sexp = match parse_result {
            ParseResult::Expression(expr) => {
                let mut buf = Vec::new();
                let mut printer = SexpPrinter::new(&mut buf, self.interner);
                let _ = printer.print_expr(expr);
                String::from_utf8_lossy(&buf).into_owned()
            }
            ParseResult::Statement(block_items) => {
                let mut buf = Vec::new();
                let mut printer = SexpPrinter::new(&mut buf, self.interner);
                for item in block_items {
                    if let BlockItem::Stmt(stmt) = item {
                        let _ = printer.print_stmt(stmt);
                    } else if let BlockItem::Decl(decl) = item {
                        let _ = printer.print_declaration(decl);
                    }
                }
                String::from_utf8_lossy(&buf).into_owned()
            }
            ParseResult::Unparseable(msg) => {
                format!("(unparseable: {})", msg.as_deref().unwrap_or("unknown"))
            }
        };
        self.writeln(&format!("// [AST dump for {}]", name_str));
        for line in sexp.lines() {
            self.writeln(&format!("// {}", line));
        }
    }

    /// 指定された関数名が AST ダンプ対象かどうかを判定し、対象なら CompoundStmt をコメントとして出力
    fn dump_ast_comment_for_body(&mut self, name_str: &str, body: &CompoundStmt) {
        if self.dump_ast_for.as_deref() != Some(name_str) {
            return;
        }
        let mut buf = Vec::new();
        let mut printer = SexpPrinter::new(&mut buf, self.interner);
        for item in &body.items {
            match item {
                BlockItem::Stmt(stmt) => { let _ = printer.print_stmt(stmt); }
                BlockItem::Decl(decl) => { let _ = printer.print_declaration(decl); }
            }
        }
        let sexp = String::from_utf8_lossy(&buf).into_owned();
        self.writeln(&format!("// [AST dump for {}]", name_str));
        for line in sexp.lines() {
            self.writeln(&format!("// {}", line));
        }
    }

    /// Call 式が lvalue マクロ呼び出しなら、展開済み lvalue を syn::Expr で返す。
    /// パラメータ置換マップは依然 `String` だが、本体の AST 走査は
    /// `build_syn_expr` を経由するため `expr_to_rust*` への依存はない。
    fn try_expand_call_as_lvalue_syn(&mut self, func: &Expr, args: &[Expr],
                                     info: Option<&MacroInferInfo>) -> Option<syn::Expr> {
        if let ExprKind::Ident(name) = &func.kind {
            if self.should_emit_as_macro_call(*name) {
                if let Some(macro_info) = self.macro_ctx.macros.get(name) {
                    if let ParseResult::Expression(body) = &macro_info.parse_result {
                        let body = body.clone();
                        let saved_params = std::mem::take(&mut self.param_substitutions);
                        for (i, param) in macro_info.params.iter().enumerate() {
                            if let Some(arg) = args.get(i) {
                                let arg_syn = self.build_syn_expr(arg, info);
                                let arg_str = crate::syn_codegen::expr_to_string(&arg_syn);
                                self.param_substitutions.insert(param.name, arg_str);
                            }
                        }
                        let body_syn = self.build_syn_expr(&body, info);
                        self.param_substitutions = saved_params;
                        return Some(body_syn);
                    }
                }
            }
        }
        None
    }


    /// ポインタ式をbool条件に変換するラッパー（inline関数用）— 統一版に委譲
    fn wrap_as_bool_condition_inline(&self, expr: &Expr, expr_str: &str) -> String {
        self.wrap_as_bool_condition(expr, expr_str, None)
    }

    /// 式の型を推定（inline 関数用）— 統一版に委譲
    fn infer_expr_type_inline(&self, expr: &Expr) -> Option<UnifiedType> {
        self.infer_expr_type_unified(expr, None)
    }

    /// TypeName から型文字列を取得（読み取り専用、整数型を正しく解決する版）
    fn type_name_to_type_str_readonly(&self, type_name: &crate::ast::TypeName) -> String {
        // ポインタの個数をカウント
        let pointer_count = type_name.declarator.as_ref()
            .map(|d| d.derived.iter().filter(|dd| matches!(dd, crate::ast::DerivedDecl::Pointer(_))).count())
            .unwrap_or(0);
        // const チェック:
        // C の "const T *p" → pointee は const → Rust: *const T
        //   この場合 const は specs.qualifiers.is_const にある
        // C の "T * const p" → ポインタ自体が const (再代入不可) → Rust: *mut T
        //   こ���場合 const は Pointer(qualifiers.is_const) にある → Rust の *const ではない
        let is_const_ptr = pointer_count == 1 && type_name.specs.qualifiers.is_const;
        // 基本型を取得
        let base = self.base_type_str_readonly(&type_name.specs.type_specs);
        // ポインタをラップ
        let mut result = base;
        for _ in 0..pointer_count {
            let prefix = if is_const_ptr { "*const " } else { "*mut " };
            result = format!("{}{}", prefix, result);
        }
        result
    }

    /// TypeSpec リストから基本型名を取得する（読み取り専用）
    fn base_type_str_readonly(&self, type_specs: &[TypeSpec]) -> String {
        // typedef 名を優先
        for spec in type_specs {
            if let TypeSpec::TypedefName(name) = spec {
                return self.interner.get(*name).to_string();
            }
        }
        // struct/union/enum 名
        for spec in type_specs {
            match spec {
                TypeSpec::Struct(s) | TypeSpec::Union(s) => {
                    if let Some(n) = &s.name {
                        return self.interner.get(*n).to_string();
                    }
                }
                TypeSpec::Enum(e) => {
                    if let Some(n) = &e.name {
                        return self.interner.get(*n).to_string();
                    }
                }
                _ => {}
            }
        }
        let mut is_void = false;
        let mut is_char = false;
        let mut is_int = false;
        let mut is_short = false;
        let mut is_long = 0usize;
        let mut is_unsigned = false;
        for spec in type_specs {
            match spec {
                TypeSpec::Void => is_void = true,
                TypeSpec::Char => is_char = true,
                TypeSpec::Int => is_int = true,
                TypeSpec::Short => is_short = true,
                TypeSpec::Long => is_long += 1,
                TypeSpec::Unsigned => is_unsigned = true,
                TypeSpec::Signed => {}
                TypeSpec::Bool => return "bool".to_string(),
                _ => {}
            }
        }
        if is_void { return "c_void".to_string(); }
        if is_char { return if is_unsigned { "c_uchar".to_string() } else { "c_char".to_string() }; }
        if is_short { return if is_unsigned { "c_ushort".to_string() } else { "c_short".to_string() }; }
        if is_long >= 2 { return if is_unsigned { "c_ulonglong".to_string() } else { "c_longlong".to_string() }; }
        if is_long == 1 { return if is_unsigned { "c_ulong".to_string() } else { "c_long".to_string() }; }
        if is_int || is_unsigned { return if is_unsigned { "c_uint".to_string() } else { "c_int".to_string() }; }
        "c_int".to_string()
    }

    /// 式の型を推定（macro 関数用）— 統一版に委譲
    fn infer_expr_type(&self, expr: &Expr, info: &MacroInferInfo) -> Option<UnifiedType> {
        self.infer_expr_type_unified(expr, Some(info))
    }

    // ================================================================
    // 統一型推論 (macro/inline 共通)
    // ================================================================

    /// 式の型を推定（macro/inline 統一版）
    ///
    /// `info` が Some の場合は TypeEnv (Tier ベース) も参照する（マクロ用）。
    /// None の場合は current_param_types のみ参照する（inline 用）。
    fn infer_expr_type_unified(&self, expr: &Expr, info: Option<&MacroInferInfo>) -> Option<UnifiedType> {
        match &expr.kind {
            ExprKind::Ident(name) => {
                // current_param_types を最優先（Tier ベースで決定済み）
                if let Some(ut) = self.current_param_types.get(name) {
                    return Some(ut.clone());
                }
                // TypeEnv 参照（マクロ用: Tier ベースで最良の制約を選択）
                if let Some(info) = info {
                    // param_to_exprs 経由の expr_constraints
                    if let Some(expr_ids) = info.type_env.param_to_exprs.get(name) {
                        let mut best: Option<(UnifiedType, u8)> = None;
                        for expr_id in expr_ids {
                            if let Some(constraints) = info.type_env.expr_constraints.get(expr_id) {
                                for c in constraints {
                                    if c.ty.is_void() { continue; }
                                    let tier = c.ty.confidence_tier();
                                    if best.is_none() || tier < best.as_ref().unwrap().1 {
                                        best = Some((UnifiedType::from_rust_str(&c.ty.to_rust_string(self.interner)), tier));
                                    }
                                }
                            }
                        }
                        if let Some((ut, _)) = best {
                            return Some(ut);
                        }
                    }
                    // param_constraints（フォールバック、Tier ベース）
                    if let Some(constraints) = info.type_env.param_constraints.get(name) {
                        let mut best: Option<(UnifiedType, u8)> = None;
                        for c in constraints {
                            if c.ty.is_void() { continue; }
                            let tier = c.ty.confidence_tier();
                            if best.is_none() || tier < best.as_ref().unwrap().1 {
                                best = Some((UnifiedType::from_rust_str(&c.ty.to_rust_string(self.interner)), tier));
                            }
                        }
                        if let Some((ut, _)) = best {
                            return Some(ut);
                        }
                    }
                }
                // 定数の型
                if let Some(dict) = self.rust_decl_dict {
                    let name_str = self.interner.get(*name);
                    if let Some(c) = dict.consts.get(name_str) {
                        return Some(c.uty.clone());
                    }
                    // extern static 変数の型 (PL_utf8skip など)
                    if let Some(ty_str) = dict.static_types.get(name_str) {
                        return Some(UnifiedType::from_rust_str(ty_str));
                    }
                }
                // enum バリアント（C 側 EnumDict 由来）。bindings.rs では nominal
                // 型の variant として現れるため、属する enum 名を Named 型として
                // 返す。比較・ビット演算等での enum→int キャスト挿入で利用。
                if let Some(enum_name) = self.enum_dict.get_enum_for_variant(*name) {
                    let enum_str = self.interner.get(enum_name).to_string();
                    return Some(UnifiedType::Named(enum_str));
                }
                None
            }
            ExprKind::Cast { type_name, .. } => {
                Some(UnifiedType::from_rust_str(&self.type_name_to_type_str_readonly(type_name)))
            }
            ExprKind::Member { expr: base, member } | ExprKind::PtrMember { expr: base, member } => {
                // まず base 式の型から struct を特定し、fields_dict で
                // メンバ型を正確に引く (同名フィールドが別 struct に存在する
                // 場合の誤解決を避ける)。見つからなければ field_type_map に
                // フォールバック。
                let resolve_struct_name = |this: &Self| -> Option<String> {
                    // マクロ推論の type_env を優先
                    if let Some(info_ref) = info {
                        if let Some(constraints) = info_ref.type_env.expr_constraints.get(&base.id) {
                            if let Some(base_ty) = constraints.first().map(|c| &c.ty) {
                                let sn = if matches!(&expr.kind, ExprKind::PtrMember { .. }) {
                                    base_ty.pointee_name()
                                } else {
                                    base_ty.type_name()
                                };
                                if let Some(n) = sn {
                                    return Some(this.interner.get(n).to_string());
                                }
                            }
                        }
                    }
                    // inline 関数のローカル / パラメータ: UnifiedType 経由で
                    // 推論して Named を取り出す
                    let base_ut = this.infer_expr_type_unified(base, info)?;
                    let target = if matches!(&expr.kind, ExprKind::PtrMember { .. }) {
                        base_ut.inner_type()?
                    } else {
                        &base_ut
                    };
                    if let UnifiedType::Named(n) = target {
                        return Some(n.clone());
                    }
                    None
                };

                let member_str = self.interner.get(*member);
                // fields_dict の C ソース由来型を第一優先 (同名フィールドが
                // 別 struct に存在するケースの誤解決を避けるため struct 名で
                // 限定)。ただし C 側で `[T; SVt_LAST]` のようにサイズが定数式で
                // 未解決の場合は `*mut T` になり配列情報を失うので、
                // field_type_map (bindings.rs 由来の Array) の方を優先する
                // フォールバックを組み合わせる。
                let fd_ty: Option<UnifiedType> = self.fields_dict.and_then(|fd| {
                    let struct_name_str = resolve_struct_name(self)?;
                    let struct_name = self.interner.lookup(&struct_name_str)?;
                    let member_ty = fd.member_type(struct_name, *member)?;
                    let rust_ty = member_ty.to_rust_string(self.interner);
                    if rust_ty.contains("/* fn */") {
                        return None;
                    }
                    Some(UnifiedType::from_rust_str(&rust_ty))
                });
                let ftm_ty = self.field_type_map.get(member_str).cloned();
                // field_type_map が配列で fields_dict がポインタなら ftm を採用。
                // (SVt_LAST のような unresolved サイズで fields_dict が配列情報を失うケース)
                match (&fd_ty, &ftm_ty) {
                    (Some(fd_ut), Some(ftm_ut))
                        if fd_ut.is_pointer() && ftm_ut.to_rust_string().starts_with('[') =>
                    {
                        return ftm_ty;
                    }
                    _ => {}
                }
                fd_ty.or(ftm_ty)
            }
            ExprKind::Deref(inner) => {
                let inner_ut = self.infer_expr_type_unified(inner, info)?;
                inner_ut.inner_type().cloned()
            }
            ExprKind::Index { expr: base, .. } => {
                // C の `a[i]` は `*(a + i)` と等価。base が配列 / ポインタ
                // 型なら要素型を返す。
                let base_ut = self.infer_expr_type_unified(base, info)?;
                base_ut.inner_type().cloned()
            }
            ExprKind::Binary { op, lhs, rhs } => {
                match op {
                    BinOp::Shl | BinOp::Shr => self.infer_expr_type_unified(lhs, info),
                    BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor => {
                        let lt = self.infer_expr_type_unified(lhs, info);
                        let rt = self.infer_expr_type_unified(rhs, info);
                        match (&lt, &rt) {
                            (Some(l), Some(r)) => {
                                let ls = l.to_rust_string();
                                let rs = r.to_rust_string();
                                wider_integer_type(&ls, &rs)
                                    .map(|w| UnifiedType::from_rust_str(w))
                                    .or(lt)
                            }
                            (Some(_), None) => lt,
                            (None, Some(_)) => rt,
                            _ => None,
                        }
                    }
                    BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Gt
                    | BinOp::Le | BinOp::Ge | BinOp::LogAnd | BinOp::LogOr => {
                        Some(UnifiedType::Bool)
                    }
                    _ => {
                        // C のポインタ減算（同型ポインタ同士）は ptrdiff_t/isize を返す。
                        // `offset_from` メソッド呼び出しに変換されるので isize 扱い。
                        if *op == BinOp::Sub {
                            let lp = self.is_pointer_expr_unified(lhs, info)
                                || self.infer_expr_type_unified(lhs, info).is_some_and(|ut| ut.is_pointer());
                            let rp = self.is_pointer_expr_unified(rhs, info)
                                || self.infer_expr_type_unified(rhs, info).is_some_and(|ut| ut.is_pointer());
                            if lp && rp {
                                return Some(UnifiedType::Named("isize".to_string()));
                            }
                        }
                        let lt = self.infer_expr_type_unified(lhs, info);
                        if lt.is_some() { return lt; }
                        self.infer_expr_type_unified(rhs, info)
                    }
                }
            }
            ExprKind::BitNot(inner) | ExprKind::UnaryMinus(inner) => self.infer_expr_type_unified(inner, info),
            ExprKind::CharLit(_) => Some(UnifiedType::from_rust_str("i8")),
            ExprKind::UIntLit(_) => Some(UnifiedType::Int { signed: false, size: crate::unified_type::IntSize::LongLong }),
            // Rust の std::mem::size_of / size_of_val は `usize` を返す。
            // 旧実装は Int{false, Long} を返していたが、これは to_rust_string
            // で "c_ulong" になり usize と混同できず `u64 * usize` 等の
            // E0277 を起こした。明示的に Named("usize") にして区別する。
            ExprKind::Sizeof(_) | ExprKind::SizeofType(_) => Some(UnifiedType::Named("usize".to_string())),
            ExprKind::Call { func, .. } => {
                // メソッド呼び出し: offset/wrapping_add 等はレシーバと同じ型
                if let ExprKind::Member { expr: receiver, member, .. } = &func.kind {
                    let method_name = self.interner.get(*member);
                    if matches!(method_name, "offset" | "wrapping_add" | "wrapping_sub" | "wrapping_offset") {
                        return self.infer_expr_type_unified(receiver, info);
                    }
                }
                if let ExprKind::Ident(name) = &func.kind {
                    let func_name = self.interner.get(*name);
                    if let Some(ret_ut) = self.get_callee_return_type(func_name) {
                        return Some(ret_ut.clone());
                    }
                    // 自家生成マクロの戻り値型
                    if let Some(macro_info) = self.macro_ctx.macros.get(name) {
                        if let Some(ty) = macro_info.get_return_type() {
                            return Some(UnifiedType::from_rust_str(&ty.to_rust_string(self.interner)));
                        }
                    }
                    // inline 関数の戻り値型 (自家生成 Perl_utf8_hop 等)
                    if let Some(dict) = self.inline_fn_dict {
                        if let Some(func_def) = dict.get(*name) {
                            // 戻り値型を文字列組立 (読取専用 API を使用)
                            let base = self.base_type_str_readonly(&func_def.specs.type_specs);
                            let mut result = base;
                            let pointer_count = func_def.declarator.derived.iter()
                                .take_while(|d| !matches!(d, crate::ast::DerivedDecl::Function(_)))
                                .filter(|d| matches!(d, crate::ast::DerivedDecl::Pointer(_)))
                                .count();
                            let is_const_ptr = pointer_count == 1
                                && func_def.specs.qualifiers.is_const;
                            for _ in 0..pointer_count {
                                let prefix = if is_const_ptr { "*const " } else { "*mut " };
                                result = format!("{}{}", prefix, result);
                            }
                            return Some(UnifiedType::from_rust_str(&result));
                        }
                    }
                }
                None
            }
            ExprKind::MacroCall { name, expanded, .. } => {
                if let Some(macro_info) = self.macro_ctx.macros.get(name) {
                    if let Some(ty) = macro_info.get_return_type() {
                        return Some(UnifiedType::from_rust_str(&ty.to_rust_string(self.interner)));
                    }
                }
                self.infer_expr_type_unified(expanded, info)
            }
            ExprKind::Conditional { then_expr, else_expr, .. } => {
                if is_null_literal(then_expr) {
                    return self.infer_expr_type_unified(else_expr, info);
                }
                if is_null_literal(else_expr) {
                    return self.infer_expr_type_unified(then_expr, info);
                }
                let tt = self.infer_expr_type_unified(then_expr, info);
                let et = self.infer_expr_type_unified(else_expr, info);
                match (&tt, &et) {
                    (Some(t), Some(e)) if t.is_void_pointer() && e.is_concrete_pointer() => et,
                    (Some(t), Some(e)) if e.is_void_pointer() && t.is_concrete_pointer() => tt,
                    (Some(_), _) => tt,
                    (None, _) => et,
                }
            }
            _ => None,
        }
    }

    /// 式がポインタ型かどうかを推定（macro/inline 統一版）
    fn is_pointer_expr_unified(&self, expr: &Expr, info: Option<&MacroInferInfo>) -> bool {
        match &expr.kind {
            ExprKind::Ident(name) => {
                if let Some(ut) = self.current_param_types.get(name) {
                    return ut.is_pointer();
                }
                if let Some(info) = info {
                    if let Some(constraints) = info.type_env.param_constraints.get(name) {
                        for c in constraints {
                            if is_type_repr_pointer(&c.ty) {
                                return true;
                            }
                        }
                    }
                    if let Some(expr_ids) = info.type_env.param_to_exprs.get(name) {
                        for expr_id in expr_ids {
                            if let Some(constraints) = info.type_env.expr_constraints.get(expr_id) {
                                for c in constraints {
                                    if is_type_repr_pointer(&c.ty) {
                                        return true;
                                    }
                                }
                            }
                        }
                    }
                }
                false
            }
            ExprKind::Cast { type_name, .. } => {
                type_name.declarator.as_ref()
                    .map(|d| d.derived.iter().any(|dd| matches!(dd, crate::ast::DerivedDecl::Pointer { .. })))
                    .unwrap_or(false)
            }
            ExprKind::AddrOf(_) => true,
            ExprKind::Member { member, .. } | ExprKind::PtrMember { member, .. } => {
                let member_str = self.interner.get(*member);
                self.field_type_map.get(member_str).is_some_and(|ut| ut.is_pointer())
            }
            ExprKind::Deref(inner) => {
                if let Some(ut) = self.infer_expr_type_unified(inner, info) {
                    if let Some(derefed) = ut.inner_type() {
                        return derefed.is_pointer();
                    }
                }
                false
            }
            ExprKind::Call { func, .. } | ExprKind::MacroCall { expanded: func, .. } => {
                let check_func = match &expr.kind {
                    ExprKind::MacroCall { name, .. } => {
                        if let Some(callee) = self.macro_ctx.macros.get(name) {
                            for c in &callee.type_env.return_constraints {
                                if is_type_repr_pointer(&c.ty) { return true; }
                            }
                        }
                        func
                    }
                    _ => func,
                };
                if let ExprKind::Ident(name) = &check_func.kind {
                    if let Some(callee) = self.macro_ctx.macros.get(name) {
                        for c in &callee.type_env.return_constraints {
                            if is_type_repr_pointer(&c.ty) { return true; }
                        }
                    }
                    if let Some(ret_ut) = self.get_callee_return_type(self.interner.get(*name)) {
                        return ret_ut.is_pointer();
                    }
                }
                false
            }
            ExprKind::Binary { op, lhs, rhs } => {
                match op {
                    BinOp::Add => self.is_pointer_expr_unified(lhs, info) || self.is_pointer_expr_unified(rhs, info),
                    BinOp::Sub => self.is_pointer_expr_unified(lhs, info) && !self.is_pointer_expr_unified(rhs, info),
                    _ => false,
                }
            }
            _ => false,
        }
    }

    /// 式の型が `Option<fn(...)>` / `Option<unsafe extern "C" fn(...)>`
    /// または同型の typedef エイリアスかを判定する。
    ///
    /// bindgen 由来の関数ポインタフィールドは `Option<fn(...)>` で包まれる。
    /// この形式は整数 0 との `==`/`!=` 比較で `expected Option<...>` エラーを
    /// 起こすため、呼出側で `.is_some()`/`.is_none()` に変換する。
    fn is_option_fn_pointer_expr(&self, expr: &Expr, info: Option<&MacroInferInfo>) -> bool {
        // Member/PtrMember の直接フォールバック: infer が None を返す
        // （inline 関数のローカル変数など）場合に備えて、field_type_map で
        // フィールド名から直接 fn pointer 型か確認する。
        if let ExprKind::Member { member, .. } | ExprKind::PtrMember { member, .. } = &expr.kind {
            let member_str = self.interner.get(*member);
            if let Some(ut) = self.field_type_map.get(member_str) {
                let ty_str = ut.to_rust_string();
                if type_str_is_fn_pointer(&ty_str) {
                    return true;
                }
                if let Some(dict) = self.rust_decl_dict {
                    if let Some(alias) = dict.types.get(&ty_str) {
                        if type_str_is_fn_pointer(&alias.ty) {
                            return true;
                        }
                    }
                }
            }
        }
        let Some(ut) = self.infer_expr_type_unified(expr, info) else { return false };
        let ty_str = ut.to_rust_string();
        if type_str_is_fn_pointer(&ty_str) {
            return true;
        }
        // typedef エイリアス経由: dict.types で展開して fn 形式か確認
        if let Some(dict) = self.rust_decl_dict {
            if let Some(alias) = dict.types.get(&ty_str) {
                if type_str_is_fn_pointer(&alias.ty) {
                    return true;
                }
            }
        }
        false
    }

    /// ポインタ式をbool条件に変換するラッパー（macro/inline 統一版）
    fn wrap_as_bool_condition(&self, expr: &Expr, expr_str: &str, info: Option<&MacroInferInfo>) -> String {
        if self.is_bool_expr_with_dict(expr) {
            return expr_str.to_string();
        }
        // パラメータが bool 型なら != 0 不要
        if let ExprKind::Ident(name) = &expr.kind {
            if let Some(ut) = self.current_param_types.get(name) {
                if ut.is_bool() {
                    return expr_str.to_string();
                }
            }
        }
        // フィールドが bool 型なら != 0 不要
        if let ExprKind::Member { member, .. } | ExprKind::PtrMember { member, .. } = &expr.kind {
            let member_str = self.interner.get(*member);
            if let Some(ut) = self.field_type_map.get(member_str) {
                if ut.is_bool() {
                    return expr_str.to_string();
                }
            }
        }
        // __builtin_expect(cond, val) → cond の型をチェック
        if let ExprKind::Call { func, args, .. } = &expr.kind {
            if let ExprKind::Ident(name) = &func.kind {
                if self.interner.get(*name) == "__builtin_expect" && !args.is_empty() {
                    return self.wrap_as_bool_condition(&args[0], expr_str, info);
                }
            }
        }
        if expr_str.ends_with(" as bool)") || expr_str.ends_with("!= 0)") || expr_str.ends_with(".is_null()") {
            return expr_str.to_string();
        }
        // Option<fn(...)> は `!= 0` で比較できず、真偽判定は .is_some()。
        // field_type_map 経由でフィールド名から検出する。
        if self.is_option_fn_pointer_expr(expr, info) {
            return format!("{}.is_some()", expr_str);
        }
        if self.is_pointer_expr_unified(expr, info)
            || self.infer_expr_type_unified(expr, info).is_some_and(|ut| ut.is_pointer()) {
            // `!{expr}.is_null()` 直付けだと expr が Cast の場合に
            // `x as T.is_null()` となり E0001 系 (cast cannot be followed
            // by method call)。必ず paren で包んで method chain の対象を
            // 明確化する。
            return format!("!({}).is_null()", expr_str);
        }
        format!("({} != 0)", strip_outer_parens(expr_str))
    }

    /// 呼び出し先が THX マクロで、my_perl が不足しているかチェック
    ///
    /// 以下の条件を満たす場合に true を返す：
    /// 1. 呼び出し先が MacroInferContext.macros に存在する
    /// 2. その MacroInferInfo.is_thx_dependent が true
    /// 3. 実引数が仮引数より1つ少ない（my_perl が不足）
    fn needs_my_perl_for_call(&self, func_name: crate::InternedStr, actual_arg_count: usize) -> bool {
        // 非 threaded perl では my_perl 引数自体が存在しないので必ず false。
        // is_thx_dependent は既に Phase 2 で false になっているはずだが、
        // 防御的にここでも mode をチェック。
        if !self.perl_threaded {
            return false;
        }
        if let Some(callee_info) = self.macro_ctx.macros.get(&func_name) {
            if callee_info.is_thx_dependent {
                // THX マクロの期待引数数 = params.len() + 1 (my_perl)
                let expected_count = callee_info.params.len() + 1;
                // 実引数が1つ少ない場合、my_perl が必要
                return actual_arg_count + 1 == expected_count;
            }
        }
        false
    }

    /// 式が既知の static 配列名かどうかをチェック
    fn is_static_array_expr(&self, expr: &Expr) -> bool {
        if let ExprKind::Ident(name) = &expr.kind {
            let name_str = self.interner.get(*name);
            self.bindings_info.static_arrays.contains(name_str)
        } else {
            false
        }
    }

    /// 式が「ポインタに減衰させて `.as_ptr()` を付けるべき配列値」か判定。
    /// Ident (static 配列)、struct フィールドの配列型、どちらも対象。
    ///
    /// ただし flexible-array フィールド（`char data[]` 等）は
    /// `maybe_decay_flex_array` が `&raw const field as *mut T` に既に
    /// 変換済みであるため、ここでは配列扱いしない。そうしないと
    /// `ptr.as_ptr()` の二重適用が発生する。
    fn is_array_like_expr(&self, expr: &Expr, info: Option<&MacroInferInfo>) -> bool {
        if self.is_static_array_expr(expr) {
            return true;
        }
        // flex array のメンバアクセスは除外（既に pointer にデケイ済）
        if let ExprKind::Member { expr: base, member }
              | ExprKind::PtrMember { expr: base, member } = &expr.kind
        {
            if let (Some(fd), Some(info)) = (self.fields_dict, info) {
                if let Some(constraints) = info.type_env.expr_constraints.get(&base.id) {
                    if let Some(base_type) = constraints.first().map(|c| &c.ty) {
                        let struct_name = match &expr.kind {
                            ExprKind::PtrMember { .. } => base_type.pointee_name(),
                            _ => base_type.type_name(),
                        };
                        if let Some(sn) = struct_name {
                            if fd.is_flexible_array_field(sn, *member) {
                                return false;
                            }
                        }
                    }
                }
            }
        }
        if let Some(ut) = self.infer_expr_type_unified(expr, info) {
            let s = ut.to_rust_string();
            if s.starts_with('[') && s.contains(';') {
                return true;
            }
        }
        false
    }

    /// フィールド名がビットフィールドメソッドかどうかをチェック
    ///
    /// 構造体名が不明な場合は、全構造体のビットフィールドメソッド名を検索
    fn is_bitfield_method(&self, member_name: &str) -> bool {
        self.bindings_info.bitfield_methods.values()
            .any(|methods| methods.contains(member_name))
    }

    /// 呼び出し先マクロのジェネリック型パラメータ情報を取得
    fn get_callee_generic_params(&self, func_name: InternedStr) -> Option<&HashMap<i32, String>> {
        let callee_info = self.macro_ctx.macros.get(&func_name)?;
        if callee_info.generic_type_params.is_empty() {
            return None;
        }
        if callee_info.generic_type_params.keys().any(|&k| k >= 0) {
            Some(&callee_info.generic_type_params)
        } else {
            None
        }
    }

    /// キャスト先の型が enum 型かどうか判定
    fn is_enum_cast_target(&self, type_name: &crate::ast::TypeName) -> bool {
        for spec in &type_name.specs.type_specs {
            match spec {
                TypeSpec::TypedefName(name) => return self.enum_dict.is_target_enum(*name),
                TypeSpec::Enum(_) => return true,
                _ => {}
            }
        }
        false
    }

    /// 呼び出し先マクロの特定の引数位置がリテラル文字列パラメータかチェック
    fn callee_expects_literal_string(&self, func_name: InternedStr, arg_index: usize) -> bool {
        if let Some(callee_info) = self.macro_ctx.macros.get(&func_name) {
            return callee_info.literal_string_params.contains(&arg_index);
        }
        false
    }

    /// 呼び出し先関数の指定引数位置の型を取得
    fn get_callee_param_type(&self, func_name: &str, arg_index: usize) -> Option<&UnifiedType> {
        self.rust_decl_dict?.fns.get(func_name).and_then(|f| {
            f.params.get(arg_index).map(|p| &p.uty)
        })
    }

    /// 呼び出し先関数のパラメータ型を取得（inline 関数/マクロ関数もフォールバック）
    fn get_callee_param_type_extended(&mut self, func_name: &str, arg_index: usize) -> Option<UnifiedType> {
        // 0. 既知 libc 関数のハードコード型 (bindings.rs に無く、
        //    呼出側が `*mut c_char` を渡すが実際は `*mut c_void` が期待される
        //    ケースで as-cast を自動挿入するため)。
        if let Some(ut) = libc_fn_param_type(func_name, arg_index) {
            return Some(ut);
        }
        // 1. bindings.rs
        if let Some(ut) = self.get_callee_param_type(func_name, arg_index) {
            return Some(ut.clone());
        }
        if let Some(interned) = self.interner.lookup(func_name) {
            // 2. inline 関数
            if let Some(dict) = self.inline_fn_dict {
                if let Some(func_def) = dict.get(interned) {
                    for d in &func_def.declarator.derived {
                        if let DerivedDecl::Function(param_list) = d {
                            if let Some(param) = param_list.params.get(arg_index) {
                                let ty = self.param_type_only(param);
                                return Some(UnifiedType::from_rust_str(&ty));
                            }
                            break;
                        }
                    }
                }
            }
            // 3. 自家生成マクロ関数の type_env からパラメータ型を取得
            if let Some(macro_info) = self.macro_ctx.macros.get(&interned) {
                // THX 依存の場合、arg_index 0 は my_perl なのでスキップ。
                // 非 threaded では my_perl が存在しないので素通り。
                let macro_param_idx = if self.perl_threaded && macro_info.is_thx_dependent {
                    if arg_index == 0 {
                        return Some(UnifiedType::from_rust_str("*mut PerlInterpreter"));
                    }
                    arg_index - 1
                } else {
                    arg_index
                };
                if let Some(param) = macro_info.params.get(macro_param_idx) {
                    // best-tier 制約 (param.expr_id() + param_to_exprs 全 ExprId 走査)
                    // 宣言型 (`get_param_type`) と同じ選択方式に揃え、
                    // 呼出側のキャスト判定が真の宣言と一致するようにする。
                    if let Some(mut ty) = best_constraint_for_macro_param(macro_info, param) {
                        // const/mut も callee 側の最終宣言と一致させる
                        // (get_param_type と同じロジック: const_pointer_positions に含まれていれば const)
                        let should_be_const = macro_info
                            .const_pointer_positions
                            .contains(&macro_param_idx);
                        if should_be_const {
                            ty.make_outer_pointer_const();
                        } else if ty.has_outer_pointer() {
                            ty.make_outer_pointer_mut();
                        }
                        let rust_ty = ty.to_rust_string(self.interner);
                        return Some(UnifiedType::from_rust_str(&rust_ty));
                    }
                }
            }
        }
        None
    }

    /// 呼び出し先関数の指定引数位置が bool 型かどうか判定（自家生成マクロも参照）
    fn callee_param_is_bool(&self, func_name: &str, arg_index: usize) -> bool {
        // 1. bindings.rs の関数
        if let Some(param_ut) = self.get_callee_param_type(func_name, arg_index) {
            return param_ut.is_bool();
        }
        // 2. 自家生成マクロ関数のパラメータ型
        if let Some(interned) = self.interner.lookup(func_name) {
            if let Some(macro_info) = self.macro_ctx.macros.get(&interned) {
                // THX マクロは my_perl が自動挿入されるのでオフセットを引く。
                // 非 threaded では my_perl 注入が無いのでオフセット 0。
                let macro_arg_index = if self.perl_threaded
                    && macro_info.is_thx_dependent
                    && arg_index > 0
                {
                    arg_index - 1
                } else {
                    arg_index
                };
                if let Some(param) = macro_info.params.get(macro_arg_index) {
                    // type_env からパラメータの型制約を取得
                    if let Some(expr_ids) = macro_info.type_env.param_to_exprs.get(&param.name) {
                        for expr_id in expr_ids {
                            if let Some(constraints) = macro_info.type_env.expr_constraints.get(expr_id) {
                                for c in constraints {
                                    let rust_ty = c.ty.to_rust_string(self.interner);
                                    if rust_ty == "bool" {
                                        return true;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // 3. 自家生成 inline 関数のパラメータ型
            if let Some(dict) = self.inline_fn_dict {
                if let Some(func_def) = dict.get(interned) {
                    for d in &func_def.declarator.derived {
                        if let DerivedDecl::Function(param_list) = d {
                            if let Some(param) = param_list.params.get(arg_index) {
                                let has_bool = param.specs.type_specs.iter().any(|ts| matches!(ts, TypeSpec::Bool));
                                let has_pointer = param.declarator.as_ref().map_or(false, |decl| {
                                    decl.derived.iter().any(|d| matches!(d, DerivedDecl::Pointer(_)))
                                });
                                if has_bool && !has_pointer {
                                    return true;
                                }
                            }
                            break;
                        }
                    }
                }
            }
        }
        false
    }

    /// 呼び出し先関数の戻り値型を取得
    fn get_callee_return_type(&self, func_name: &str) -> Option<&UnifiedType> {
        self.rust_decl_dict?.fns.get(func_name).and_then(|f| {
            f.uret_ty.as_ref()
        })
    }

    /// `UnifiedType` が bindings.rs 上の Rust enum 型を指すかを判定。
    /// （C 側では int 互換だが、Rust 側は nominal 型なので、整数演算には
    /// `as <int>` キャストが必要になる）
    fn is_rust_enum_type(&self, ut: &UnifiedType) -> bool {
        if let UnifiedType::Named(name) = ut {
            if let Some(dict) = self.rust_decl_dict {
                return dict.enums.contains(name.as_str());
            }
        }
        false
    }

    /// 式が bool を返すかどうかを判定（関数の戻り値型も考慮）
    fn is_bool_expr_with_dict(&self, expr: &Expr) -> bool {
        if is_boolean_expr_recursive(expr, self.interner) {
            return true;
        }
        // 配列インデックス: 配列要素型が bool なら bool
        if let ExprKind::Index { expr: base, .. } = &expr.kind {
            if let ExprKind::Ident(name) = &base.kind {
                if let Some(dict) = self.rust_decl_dict {
                    let name_str = self.interner.get(*name);
                    if let Some(c) = dict.consts.get(name_str) {
                        if let Some(inner) = c.uty.inner_type() {
                            if inner.is_bool() {
                                return true;
                            }
                        }
                    }
                    // static 配列: 名前が PL_valid_types_* なら bool 配列
                    if dict.statics.contains(name_str) && name_str.starts_with("PL_valid_types_") {
                        return true;
                    }
                }
            }
        }
        // パラメータが bool 型
        if let ExprKind::Ident(name) = &expr.kind {
            if let Some(ut) = self.current_param_types.get(name) {
                if ut.is_bool() {
                    return true;
                }
            }
        }
        // フィールドが bool 型
        if let ExprKind::Member { member, .. } | ExprKind::PtrMember { member, .. } = &expr.kind {
            let member_str = self.interner.get(*member);
            if let Some(ut) = self.field_type_map.get(member_str) {
                if ut.is_bool() {
                    return true;
                }
            }
        }
        // 関数呼び出しの戻り値型が bool かチェック
        if let ExprKind::Call { func, .. } = &expr.kind {
            if let ExprKind::Ident(name) = &func.kind {
                // 0. codegen で bool と判定されたマクロ
                if self.bool_return_macros.contains(name) {
                    return true;
                }
                let func_name = self.interner.get(*name);
                // 1. bindings.rs の関数
                if let Some(ret_ut) = self.get_callee_return_type(func_name) {
                    return ret_ut.is_bool();
                }
                // 2. 自家生成マクロ関数の戻り値型
                if let Some(macro_info) = self.macro_ctx.macros.get(name) {
                    if let Some(ty) = macro_info.get_return_type() {
                        let rust_ty = ty.to_rust_string(self.interner);
                        return rust_ty == "bool";
                    }
                }
                // 3. 自家生成 inline 関数の戻り値型
                if let Some(dict) = self.inline_fn_dict {
                    if let Some(func_def) = dict.get(*name) {
                        let has_bool_return = func_def.specs.type_specs.iter().any(|ts| matches!(ts, TypeSpec::Bool));
                        let has_return_pointer = func_def.declarator.derived.iter().any(|d| matches!(d, DerivedDecl::Pointer(_)));
                        if has_bool_return && !has_return_pointer {
                            return true;
                        }
                    }
                }
            }
        }
        // MacroCall の場合も bool_return_macros をチェック
        if let ExprKind::MacroCall { name, .. } = &expr.kind {
            if self.bool_return_macros.contains(name) {
                return true;
            }
        }
        false
    }

    /// 式の最内部にある literal_string_param の Ident を探す
    /// identity builtin（ASSERT_IS_LITERAL 等）を透過して中身をチェック
    fn find_literal_string_ident<'b>(&self, expr: &'b Expr) -> Option<&'b InternedStr> {
        match &expr.kind {
            ExprKind::Ident(name) if self.current_literal_string_params.contains(name) => {
                Some(name)
            }
            ExprKind::Call { func, args } if args.len() == 1 => {
                // identity builtin (ASSERT_IS_LITERAL, ASSERT_IS_PTR, ASSERT_NOT_PTR)
                if let ExprKind::Ident(fname) = &func.kind {
                    let func_name = self.interner.get(*fname);
                    if func_name == "ASSERT_IS_LITERAL"
                        || func_name == "ASSERT_IS_PTR"
                        || func_name == "ASSERT_NOT_PTR"
                    {
                        return self.find_literal_string_ident(&args[0]);
                    }
                }
                None
            }
            _ => None,
        }
    }


    /// マクロ呼び出し形式で出力すべきかを判定
    ///
    /// 以下の場合にマクロ呼び出し形式で出力する：
    /// - MacroInferContext に登録されており、生成対象である
    ///
    /// そうでない場合は展開形式で出力する。
    fn should_emit_as_macro_call(&self, name: crate::InternedStr) -> bool {
        // MacroInferContext にマクロ情報があり、パース可能なら生成対象
        if let Some(info) = self.macro_ctx.macros.get(&name) {
            // パース成功し、利用不可関数を呼ばないマクロのみ
            return info.is_parseable() && !info.calls_unavailable;
        }
        false
    }

    /// 不完全マーカー: 型が不明
    fn unknown_marker(&mut self) -> &'static str {
        self.incomplete_count += 1;
        "/* unknown */"
    }

    /// 不完全マーカー: TODO
    fn todo_marker(&mut self, msg: &str) -> String {
        self.incomplete_count += 1;
        format!("/* TODO: {} */", msg)
    }

    /// 不完全マーカー: 型
    fn type_marker(&mut self) -> &'static str {
        self.incomplete_count += 1;
        "/* type */"
    }

    /// Declaration から変数名を収集して current_local_names に追加
    /// ローカル変数の型も current_param_types に登録（ポインタ検出用）
    fn collect_decl_names(&mut self, decl: &Declaration) {
        let base_type = self.decl_specs_to_rust(&decl.specs);
        for init_decl in &decl.declarators {
            if let Some(name) = init_decl.declarator.name {
                self.current_local_names.insert(name);
                let ty = self.apply_derived_to_type(&base_type, &init_decl.declarator.derived);
                self.current_param_types.insert(name, UnifiedType::from_rust_str(&ty));
            }
        }
    }

    /// 関数本体全体を再帰的に走査して、ネストした compound 内・StmtExpr 内・
    /// for ループ初期化部の Declaration を全て `current_local_names` に追加する。
    /// scope 厳密性は犠牲にして「同一名はどのスコープでも解決済み」と扱う簡易版。
    /// `STMT_START { let v = ...; ... } STMT_END` 展開で生まれる block-local
    /// 変数を未解決と誤検出しないようにする。
    fn collect_local_names_recursive(&mut self, body: &CompoundStmt) {
        for item in &body.items {
            self.collect_local_names_from_block_item(item);
        }
    }

    fn collect_local_names_from_block_item(&mut self, item: &BlockItem) {
        match item {
            BlockItem::Decl(d) => self.collect_decl_names(d),
            BlockItem::Stmt(s) => self.collect_local_names_from_stmt(s),
        }
    }

    fn collect_local_names_from_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Compound(c) => self.collect_local_names_recursive(c),
            Stmt::If { then_stmt, else_stmt, cond, .. } => {
                self.collect_local_names_from_expr(cond);
                self.collect_local_names_from_stmt(then_stmt);
                if let Some(es) = else_stmt {
                    self.collect_local_names_from_stmt(es);
                }
            }
            Stmt::Switch { body, expr, .. } => {
                self.collect_local_names_from_expr(expr);
                self.collect_local_names_from_stmt(body);
            }
            Stmt::While { body, cond, .. } => {
                self.collect_local_names_from_expr(cond);
                self.collect_local_names_from_stmt(body);
            }
            Stmt::DoWhile { body, cond, .. } => {
                self.collect_local_names_from_stmt(body);
                self.collect_local_names_from_expr(cond);
            }
            Stmt::For { init, cond, step, body, .. } => {
                if let Some(i) = init {
                    match i {
                        ForInit::Decl(d) => self.collect_decl_names(d),
                        ForInit::Expr(e) => self.collect_local_names_from_expr(e),
                    }
                }
                if let Some(c) = cond { self.collect_local_names_from_expr(c); }
                if let Some(s) = step { self.collect_local_names_from_expr(s); }
                self.collect_local_names_from_stmt(body);
            }
            Stmt::Expr(Some(e), _) => self.collect_local_names_from_expr(e),
            Stmt::Return(Some(e), _) => self.collect_local_names_from_expr(e),
            Stmt::Label { stmt, .. } | Stmt::Case { stmt, .. } | Stmt::Default { stmt, .. } => {
                self.collect_local_names_from_stmt(stmt);
            }
            _ => {}
        }
    }

    fn collect_local_names_from_expr(&mut self, expr: &Expr) {
        if let ExprKind::StmtExpr(c) = &expr.kind {
            self.collect_local_names_recursive(c);
        }
        // 他の式ノードに含まれる StmtExpr は普通存在しない（C 文法上）が、
        // 念のため簡易的に再帰しておく
        match &expr.kind {
            ExprKind::Binary { lhs, rhs, .. }
            | ExprKind::Assign { lhs, rhs, .. }
            | ExprKind::Comma { lhs, rhs } => {
                self.collect_local_names_from_expr(lhs);
                self.collect_local_names_from_expr(rhs);
            }
            ExprKind::Conditional { cond, then_expr, else_expr } => {
                self.collect_local_names_from_expr(cond);
                self.collect_local_names_from_expr(then_expr);
                self.collect_local_names_from_expr(else_expr);
            }
            ExprKind::Cast { expr: e, .. }
            | ExprKind::AddrOf(e) | ExprKind::Deref(e)
            | ExprKind::UnaryPlus(e) | ExprKind::UnaryMinus(e)
            | ExprKind::BitNot(e) | ExprKind::LogNot(e)
            | ExprKind::PreInc(e) | ExprKind::PreDec(e)
            | ExprKind::PostInc(e) | ExprKind::PostDec(e)
            | ExprKind::Sizeof(e)
            | ExprKind::Member { expr: e, .. } | ExprKind::PtrMember { expr: e, .. } => {
                self.collect_local_names_from_expr(e);
            }
            ExprKind::Call { func, args } => {
                self.collect_local_names_from_expr(func);
                for a in args { self.collect_local_names_from_expr(a); }
            }
            ExprKind::Index { expr: e, index } => {
                self.collect_local_names_from_expr(e);
                self.collect_local_names_from_expr(index);
            }
            _ => {}
        }
    }

    /// Declaration からローカル変数の型のみ収集して current_param_types に追加
    /// (current_local_names には追加しない — 未解決シンボル検出に影響しないように)
    fn collect_decl_types(&mut self, decl: &Declaration) {
        let base_type = self.decl_specs_to_rust(&decl.specs);
        for init_decl in &decl.declarators {
            if let Some(name) = init_decl.declarator.name {
                let ty = self.apply_derived_to_type(&base_type, &init_decl.declarator.derived);
                self.current_param_types.insert(name, UnifiedType::from_rust_str(&ty));
            }
        }
    }

    /// バッファに行を書き込み
    fn writeln(&mut self, s: &str) {
        self.buffer.push_str(s);
        self.buffer.push('\n');
    }

    /// 生成結果を取得（self を消費）
    fn into_generated_code(self) -> GeneratedCode {
        GeneratedCode {
            code: self.buffer,
            incomplete_count: self.incomplete_count,
            unresolved_names: self.unresolved_names,
            used_libc_fns: self.used_libc_fns,
            codegen_errors: self.codegen_errors,
        }
    }

    /// マクロ関数を生成（self を消費）
    pub fn generate_macro(mut self, info: &MacroInferInfo) -> GeneratedCode {
        let name_str = self.interner.get(info.name);

        // ローカルスコープ: マクロのパラメータ名を登録
        for p in &info.params {
            self.current_local_names.insert(p.name);
        }

        // 型パラメータマップを構築
        self.current_type_param_map = info.generic_type_params.iter()
            .filter(|(idx, _)| **idx >= 0)
            .filter_map(|(idx, generic_name)| {
                info.params.get(*idx as usize).map(|p| (p.name, generic_name.clone()))
            })
            .collect();

        // 型パラメータになったパラメータは通常パラメータとしては存在しないので
        // current_local_names から除外する（ジェネリック誤検出時に unresolved 検出するため）
        for (name, _) in &self.current_type_param_map {
            self.current_local_names.remove(name);
        }

        // リテラル文字列パラメータの名前集合を構築
        self.current_literal_string_params = info.literal_string_params.iter()
            .filter_map(|&idx| info.params.get(idx).map(|p| p.name))
            .collect();

        // ジェネリック句を生成
        let generic_clause = self.build_generic_clause(info);

        // パラメータリストを構築（型情報付き）
        // type/cast パラメータは値引数ではないので除外
        let params_with_types = self.build_param_list(info);

        // 戻り値の型を取得（current_return_type にもセット）
        let return_type = self.get_return_type(info);
        self.current_return_type = Some(UnifiedType::from_rust_str(&return_type));

        // 型推論ダンプ
        if self.dump_types_for.as_deref() == Some(name_str) {
            self.dump_type_info(name_str, info, &params_with_types, &return_type);
        }

        // THX 依存の場合は my_perl パラメータを追加。
        // 非 threaded perl では my_perl 自体が存在しないので注入しない。
        let thx_param = if self.perl_threaded && info.is_thx_dependent {
            "my_perl: *mut PerlInterpreter"
        } else {
            ""
        };

        // 関数シグネチャ
        let params_str = if thx_param.is_empty() {
            params_with_types.clone()
        } else if params_with_types.is_empty() {
            thx_param.to_string()
        } else {
            format!("{}, {}", thx_param, params_with_types)
        };

        // AST ダンプ（デバッグ用）
        self.dump_ast_comment_for_expr(name_str, &info.parse_result);

        // ドキュメントコメント
        let thx_info = if info.is_thx_dependent { " [THX]" } else { "" };
        let generic_info = if !generic_clause.is_empty() { " [generic]" } else { "" };
        self.writeln(&format!("/// {}{}{} - macro function", name_str, thx_info, generic_info));
        self.writeln("#[inline]");
        self.writeln("#[allow(unsafe_op_in_unsafe_fn)]");

        // 関数定義（ジェネリック句付き）
        self.writeln(&format!("pub unsafe fn {}{}({}) -> {} {{", name_str, generic_clause, params_str, return_type));

        // unsafe 操作（関数呼び出し or デリファレンス）を含む場合のみ unsafe ブロックを生成
        let needs_unsafe = info.has_unsafe_ops();
        let body_indent = if needs_unsafe { "        " } else { "    " };

        if needs_unsafe {
            self.writeln("    unsafe {");
        }

        match &info.parse_result {
            ParseResult::Expression(expr) => {
                let type_hint = self.current_return_type.as_ref().map(|ut| ut.to_rust_string());
                let mut syn_expr = self.build_syn_expr_with_type_hint(expr, Some(info), type_hint.as_deref());

                if self.current_return_type.as_ref().is_some_and(|ut| ut.is_void()) {
                    let s = normalize_parens(&crate::syn_codegen::expr_to_string(&syn_expr));
                    self.writeln(&format!("{}{};", body_indent, s));
                } else if self.current_return_type.as_ref().is_some_and(|ut| ut.is_bool())
                    && !self.is_bool_expr_with_dict(expr)
                    && !crate::syn_codegen::is_bool_syn_expr(&syn_expr) {
                    if self.is_pointer_expr_unified(expr, Some(info))
                        || self.infer_expr_type_unified(expr, Some(info)).is_some_and(|ut| ut.is_pointer()) {
                        let s = normalize_parens(&crate::syn_codegen::expr_to_string(&syn_expr));
                        // `!x.is_null()` 直付けだと x が cast の場合に
                        // `x as T.is_null()` と解釈される。必ず括弧で包む。
                        self.writeln(&format!("{}!({}).is_null()", body_indent, s));
                    } else {
                        syn_expr = crate::syn_codegen::wrap_as_bool(syn_expr);
                        let s = normalize_parens(&crate::syn_codegen::expr_to_string(&syn_expr));
                        self.writeln(&format!("{}{}", body_indent, s));
                    }
                } else {
                    syn_expr = self.cast_return_syn_expr_if_needed(expr, Some(info), syn_expr);
                    let s = normalize_parens(&crate::syn_codegen::expr_to_string(&syn_expr));
                    self.writeln(&format!("{}{}", body_indent, s));
                }
            }
            ParseResult::Statement(block_items) => {
                for item in block_items {
                    if let BlockItem::Stmt(stmt) = item {
                        let rust_stmt = self.stmt_to_rust(stmt, info);
                        if rust_stmt.is_empty() {
                            continue; // 空文 (空マクロ消去跡)
                        }
                        self.writeln(&format!("{}{}", body_indent, rust_stmt));
                    }
                }
            }
            ParseResult::Unparseable(_) => {
                self.writeln(&format!("{}unimplemented!()", body_indent));
            }
        }

        if needs_unsafe {
            self.writeln("    }");
        }

        self.writeln("}");
        self.writeln("");

        self.into_generated_code()
    }

    /// ジェネリック句を生成（例: "<T>" or "<T, U>"）
    fn build_generic_clause(&self, info: &MacroInferInfo) -> String {
        if info.generic_type_params.is_empty() {
            return String::new();
        }

        // 型パラメータを収集（重複排除、ソート）
        let mut params: Vec<&String> = info.generic_type_params.values().collect();
        params.sort();
        params.dedup();

        format!("<{}>", params.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", "))
    }

    /// パラメータリストを構築（型情報付き）
    /// type/cast パラメータは型パラメータなので値引数からは除外する
    /// 副作用: 各パラメータの型を current_param_types に登録し、
    /// Phase 2 の名前使用解析を self (mut_local_names / unused_param_names /
    /// current_local_usage) に反映する
    fn build_param_list(&mut self, info: &MacroInferInfo) -> String {
        // Phase 2 (analyze_all_macros) で計算済みの解析を読む。
        // テスト等で未設定の場合のみその場で計算する
        let usage = info.local_usage.clone().unwrap_or_else(|| {
            crate::local_usage::analyze_macro(&info.parse_result, &info.params)
        });
        let mut_params = usage.mut_names();
        self.mut_local_names = mut_params.clone();
        self.unused_param_names = info.params.iter()
            .map(|p| p.name)
            .filter(|n| usage.is_unused(*n))
            .collect();
        self.current_local_usage = Some(usage);
        let mut parts = Vec::new();
        for (i, p) in info.params.iter().enumerate() {
            if info.generic_type_params.contains_key(&(i as i32)) {
                continue;
            }
            // 未使用パラメータは `_` prefix (unused_variables 対策、GH #22)。
            // 本体に参照が無いことが保証されるので束縛名だけ変えれば良い
            let name = if self.unused_param_names.contains(&p.name) {
                format!("_{}", self.interner.get(p.name))
            } else {
                escape_rust_keyword(self.interner.get(p.name))
            };
            let ty = self.get_param_type(p, info, i);
            // current_param_types に登録（bool 判定等で使用）
            self.current_param_types.insert(p.name, UnifiedType::from_rust_str(&ty));
            let mut_prefix = if mut_params.contains(&p.name) { "mut " } else { "" };
            parts.push(format!("{}{}: {}", mut_prefix, name, ty));
        }
        parts.join(", ")
    }

    /// パラメータの型を取得
    fn get_param_type(&mut self, param: &MacroParam, info: &MacroInferInfo, param_index: usize) -> String {
        // ジェネリック型パラメータかチェック
        if let Some(generic_name) = info.generic_type_params.get(&(param_index as i32)) {
            return generic_name.clone();
        }

        // リテラル文字列パラメータかチェック（apidoc の "..." 引数）
        if info.literal_string_params.contains(&param_index) {
            return "&str".to_string();
        }

        let should_be_const = self.const_pointer_positions.contains(&param_index);

        if let Some(mut ty) = best_constraint_for_macro_param(info, param) {
            if should_be_const {
                ty.make_outer_pointer_const();
            } else if ty.has_outer_pointer() {
                // must-mut: Phase 2 で *mut と確定 → *const になっていたら *mut に戻す
                ty.make_outer_pointer_mut();
            }
            return self.type_repr_to_rust(&ty);
        }

        self.unknown_marker().to_string()
    }

    /// 戻り値の型を取得
    ///
    /// ジェネリック戻り値型を優先し、なければ
    /// MacroInferInfo::get_return_type() を使用して、
    /// return_constraints（apidoc由来）を expr_constraints より優先する
    fn get_return_type(&mut self, info: &MacroInferInfo) -> String {
        // ジェネリック戻り値型かチェック（-1 = return type）
        if let Some(generic_name) = info.generic_type_params.get(&-1) {
            return generic_name.clone();
        }

        // 依存順パスで bool と判定されていればそのまま bool を返す
        if self.is_bool_return {
            return "bool".to_string();
        }

        match &info.parse_result {
            ParseResult::Expression(expr) => {
                if let Some(ty) = info.get_return_type() {
                    let mut ty_str = self.type_repr_to_rust(ty);
                    if ty_str != "()" {
                        // 式の推論型がポインタで const なら戻り値も const に
                        if ty_str.contains("*mut") {
                            if let Some(expr_ut) = self.infer_expr_type(expr, info) {
                                if expr_ut.is_const_pointer() {
                                    ty_str = ty_str.replace("*mut", "*const");
                                }
                            }
                        }
                        return ty_str;
                    }
                    // "()" が返された場合: 式の実際の型を確認
                    // 式が本当に void を返す(void 関数呼び出し等)なら "()" で正しい
                    // そうでなければ型推論の誤りなのでフォールバック
                    if let Some(ut) = self.infer_expr_type(expr, info) {
                        let s = ut.to_rust_string();
                        if s != "()" {
                            return s;
                        }
                    }
                    // 式型推論でも "()" → void で正しい
                    return ty_str;
                }
                self.unknown_marker().to_string()
            }
            ParseResult::Statement(_) => "()".to_string(),
            ParseResult::Unparseable(_) => "()".to_string(),
        }
    }

    /// TypeRepr を Rust 型文字列に変換
    ///
    /// 戻り値に `/*` が含まれていたら不完全型としてカウントする
    fn type_repr_to_rust(&mut self, ty: &crate::type_repr::TypeRepr) -> String {
        let result = ty.to_rust_string(self.interner);
        let result = self.substitute_type_params(&result);
        if result.contains("/*") {
            self.incomplete_count += 1;
        }
        result
    }

    /// 型文字列中の型パラメータ名を generic 名に置換
    fn substitute_type_params(&self, type_str: &str) -> String {
        if self.current_type_param_map.is_empty() {
            return type_str.to_string();
        }
        let mut result = type_str.to_string();
        for (param_name, generic_name) in &self.current_type_param_map {
            let name_str = self.interner.get(*param_name);
            // 単語境界を考慮した置換（部分文字列一致を避ける）
            result = replace_word(&result, name_str, generic_name);
        }
        result
    }

    /// MUTABLE_PTR パターンを検出
    ///
    /// `({ void *p_ = (expr); p_; })` のような構造を検出し、
    /// 初期化子の式を返す。
    fn detect_mutable_ptr_pattern<'b>(&self, compound: &'b CompoundStmt) -> Option<&'b Expr> {
        // 2つの要素: 宣言 + 式文
        if compound.items.len() != 2 {
            return None;
        }

        // 最初の要素: 宣言
        let decl = match &compound.items[0] {
            BlockItem::Decl(d) => d,
            _ => return None,
        };

        // 宣言子が1つで、初期化子がある
        if decl.declarators.len() != 1 {
            return None;
        }
        let init_decl = &decl.declarators[0];
        let declared_name = init_decl.declarator.name?;
        let init = init_decl.init.as_ref()?;

        // 初期化子は式
        let init_expr = match init {
            Initializer::Expr(e) => e.as_ref(),
            _ => return None,
        };

        // 2番目の要素: 式文で、宣言した変数を参照
        let last_expr = match &compound.items[1] {
            BlockItem::Stmt(Stmt::Expr(Some(e), _)) => e,
            _ => return None,
        };

        // 最後の式が宣言した変数への参照
        if let ExprKind::Ident(name) = &last_expr.kind {
            if *name == declared_name {
                return Some(init_expr);
            }
        }

        None
    }

    // ================================================================
    // syn::Expr ベースの式構築 (Step 1+)
    // ================================================================

    /// flexible array member の field access を `&raw const place as *mut T`
    /// に変換する。配列名→ポインタ decay の C 慣用句を Rust で正しく表現する。
    ///
    /// 検出条件: base 式の TypeRepr から typedef/struct 名を取り、
    /// `FieldsDict::flexible_array_element` が要素型を返した場合のみ変換。
    /// それ以外は `access` をそのまま返す。
    fn maybe_decay_flex_array(
        &self,
        access: syn::Expr,
        base: &Expr,
        member: InternedStr,
        info: Option<&MacroInferInfo>,
        is_ptr_member: bool,
    ) -> syn::Expr {
        let Some(fd) = self.fields_dict else { return access; };
        let Some(info) = info else { return access; };
        let Some(constraints) = info.type_env.expr_constraints.get(&base.id) else {
            return access;
        };
        let Some(base_type) = constraints.first().map(|c| &c.ty) else {
            return access;
        };
        let struct_name = if is_ptr_member {
            base_type.pointee_name()
        } else {
            base_type.type_name()
        };
        let Some(struct_name) = struct_name else { return access; };
        let Some(elem) = fd.flexible_array_element(struct_name, member) else {
            return access;
        };
        // `&raw const access as *mut <element>` を生成
        let elem_str = elem.to_rust_string(self.interner);
        let target_ty_str = format!("*mut {}", elem_str);
        let raw_const = syn::Expr::RawAddr(syn::ExprRawAddr {
            attrs: vec![],
            and_token: Default::default(),
            raw: Default::default(),
            mutability: syn::PointerMutability::Const(Default::default()),
            expr: Box::new(access),
        });
        crate::syn_codegen::insert_cast(raw_const, crate::syn_codegen::parse_type(&target_ty_str))
    }

    /// `&MACRO(args)` の AddrOf inner が「自家生成マクロへの Call」で、
    /// 呼び出し先が **Expression body**（lvalue chain として展開可能）なら、
    /// 本体式を引数で alpha 置換した `Expr` を返す。
    ///
    /// 用途: `&GvSV(gv)` のような C 慣用句で、Rust 側の wrap 関数が
    /// temporary を返すため `&raw mut <fn_call>` が E0745 になる問題の回避。
    /// マクロ本体を inline 展開して `&raw mut (*GvGP(gv)).gp_sv` のような
    /// 真の place 式に変換する。
    fn try_inline_call_for_addrof(&self, inner: &Expr) -> Option<Expr> {
        let (callee_id, args) = match &inner.kind {
            ExprKind::Call { func, args } => match &func.kind {
                ExprKind::Ident(name) => (*name, args),
                _ => return None,
            },
            _ => return None,
        };

        let callee_info = self.macro_ctx.macros.get(&callee_id)?;
        let body = match &callee_info.parse_result {
            ParseResult::Expression(e) => e,
            _ => return None,
        };

        // THX 依存だと第一引数が my_perl で arg と zip がずれる → 当面非対応
        if callee_info.is_thx_dependent {
            return None;
        }
        if callee_info.params.len() != args.len() {
            return None;
        }

        // param_name → arg_expr の置換マップを構築
        let mut subs: HashMap<InternedStr, &Expr> = HashMap::new();
        for (param, arg) in callee_info.params.iter().zip(args.iter()) {
            subs.insert(param.name, arg);
        }

        let mut inlined = (**body).clone();
        substitute_idents(&mut inlined, &subs);
        Some(inlined)
    }

    /// C AST の式を syn::Expr に変換する（macro/inline 統一）
    ///
    /// `info` が Some の場合はマクロ用、None の場合は inline 用。
    /// 未対応の ExprKind は expr_to_rust_ctx のフォールバック文字列を syn::parse_str で変換する。
    fn build_syn_expr(&mut self, expr: &Expr, info: Option<&MacroInferInfo>) -> syn::Expr {
        use crate::syn_codegen::*;

        match &expr.kind {
            ExprKind::Ident(name) => {
                // lvalue展開時のパラメータ置換
                if let Some(subst) = self.param_substitutions.get(name) {
                    // 置換文字列をパース
                    return syn::parse_str(subst).unwrap_or_else(|_| int_lit(0));
                }
                let name_str = self.interner.get(*name);
                // libc 関数の使用を記録
                if LIBC_FUNCTIONS.contains(&name_str) {
                    self.used_libc_fns.insert(name_str.to_string());
                }
                // 未解決シンボルチェック
                if !self.current_local_names.contains(name)
                    && !self.enum_dict.is_enum_variant(*name)
                    && !self.known_symbols.contains(name_str)
                {
                    let s = name_str.to_string();
                    if !self.unresolved_names.contains(&s) {
                        self.unresolved_names.push(s);
                    }
                }
                // true/false は Rust の bool リテラルとして出力（r#true 回避）
                if name_str == "true" || name_str == "false" {
                    return syn::Expr::Lit(syn::ExprLit {
                        attrs: vec![],
                        lit: syn::Lit::Bool(syn::LitBool {
                            value: name_str == "true",
                            span: proc_macro2::Span::call_site(),
                        }),
                    });
                }
                let escaped = escape_rust_keyword(name_str);
                // extern static 配列はポインタとして使われるため raw pointer に減衰させる。
                //
                // Rust 2024 edition では `static mut` への暗黙の shared reference が
                // hard error 化された (`creating a shared reference to mutable static`)。
                // `.as_ptr()` は内部で `&self -> *const T` の参照経由を行うため、
                // `static mut PL_fold_locale: [u8; N]` のような mutable static に対して
                // 直接呼ぶと当該 lint を踏む。
                //
                // - `(&raw const NAME[0])` は const 評価で `[N=0]` の OOB を踏む
                //   (bindgen が不明サイズの extern static を `[T; 0]` で出すため)。
                // - `(&raw const NAME) as *const u8` は型情報を捨てるため `.offset(i)`
                //   が byte 単位になり、要素配列で計算が壊れる。
                // - 正解は **`(&raw const NAME) as *const T`**。bindings から要素型 T
                //   を取り出して埋め込む。取り出せなかったら u8 にフォールバック。
                if self.bindings_info.static_arrays.contains(name_str) {
                    let elem = self.bindings_info.static_array_element_type(name_str)
                        .unwrap_or_else(|| "u8".to_string());
                    return syn::parse_str(&format!(
                        "((&raw const {}) as *const {})",
                        escaped, elem
                    ))
                    .unwrap_or_else(|_| int_lit(0));
                }
                syn::Expr::Path(syn::ExprPath {
                    attrs: vec![],
                    qself: None,
                    path: ident(&escaped).into(),
                })
            }
            ExprKind::IntLit(n) => int_lit(*n),
            ExprKind::UIntLit(n) => {
                let lit = syn::LitInt::new(&format!("{}u64", n), proc_macro2::Span::call_site());
                syn::Expr::Lit(syn::ExprLit { attrs: vec![], lit: syn::Lit::Int(lit) })
            }
            ExprKind::FloatLit(f) => {
                let lit = syn::LitFloat::new(&format!("{}", f), proc_macro2::Span::call_site());
                syn::Expr::Lit(syn::ExprLit { attrs: vec![], lit: syn::Lit::Float(lit) })
            }
            ExprKind::CharLit(c) => {
                let s = if c.is_ascii() {
                    format!("b'{}' as i8", escape_char(*c))
                } else {
                    format!("0x{:02x}u8 as i8", c)
                };
                syn::parse_str(&s).unwrap_or_else(|_| int_lit(0))
            }
            ExprKind::StringLit(s) => {
                syn::parse_str(&format!("c\"{}\"", escape_string(s)))
                    .unwrap_or_else(|_| int_lit(0))
            }
            ExprKind::Deref(inner) => {
                let e = self.build_syn_expr(inner, info);
                deref(e)
            }
            ExprKind::AddrOf(inner) => {
                // C の `&MACRO(args)` パターン: マクロが lvalue を返すことを
                // 期待する C コード。Rust では MACRO が unsafe fn として
                // wrap されるため戻り値が temporary になり `&raw mut <call>`
                // が E0745 を起こす。callee の本体式を inline 展開して
                // lvalue 性を回復する。
                if let Some(inlined) = self.try_inline_call_for_addrof(inner) {
                    let e = self.build_syn_expr(&inlined, info);
                    return addr_of_mut(e);
                }
                let e = self.build_syn_expr(inner, info);
                addr_of_mut(e)
            }
            ExprKind::UnaryPlus(inner) => {
                self.build_syn_expr(inner, info)
            }
            ExprKind::BitNot(inner) => {
                let e = self.build_syn_expr(inner, info);
                syn::Expr::Unary(syn::ExprUnary {
                    attrs: vec![],
                    op: syn::UnOp::Not(Default::default()),
                    expr: Box::new(e),
                })
            }
            ExprKind::Member { expr: base, member } => {
                let e = self.build_syn_expr(base, info);
                let m = self.interner.get(*member);
                if self.is_bitfield_method(m) {
                    // bitfield → メソッド呼び出し
                    syn::Expr::MethodCall(syn::ExprMethodCall {
                        attrs: vec![],
                        receiver: Box::new(e),
                        dot_token: Default::default(),
                        method: ident(m),
                        turbofish: None,
                        paren_token: Default::default(),
                        args: syn::punctuated::Punctuated::new(),
                    })
                } else {
                    let access = field_access(e, m);
                    self.maybe_decay_flex_array(access, base, *member, info, /*is_ptr_member=*/false)
                }
            }
            ExprKind::PtrMember { expr: base, member } => {
                let e = self.build_syn_expr(base, info);
                let m = self.interner.get(*member);
                let derefed = deref(e);
                if self.is_bitfield_method(m) {
                    syn::Expr::MethodCall(syn::ExprMethodCall {
                        attrs: vec![],
                        receiver: Box::new(derefed),
                        dot_token: Default::default(),
                        method: ident(m),
                        turbofish: None,
                        paren_token: Default::default(),
                        args: syn::punctuated::Punctuated::new(),
                    })
                } else {
                    let access = field_access(derefed, m);
                    self.maybe_decay_flex_array(access, base, *member, info, /*is_ptr_member=*/true)
                }
            }
            ExprKind::Comma { lhs, rhs } => {
                let l = self.build_syn_expr(lhs, info);
                let r = self.build_syn_expr(rhs, info);
                // C カンマ `(A, B)` の翻訳。LHS の値は捨てるが、A が
                // assignment-as-expression の場合 codegen は `{ stmt; trailing }`
                // のような trailing 値付きブロックを作る（例: `*fp |= X` →
                // `{ *fp |= X; *fp }`）。これを文の位置で `;` 付きに置くと
                // Rust の trailing 単項式・論理演算・must_use 戻り値が discarded で
                // CI の `-D warnings` を踏む。
                //
                // 条件付き `let _ = A;` wrap: A が値を返すと判明している
                // パターン（trailing-expr 付き block / 論理演算 / 単項 / 関数呼出）
                // のみ wrap する。`{ assert!(...); }` のように常に `()` を返す
                // 形式まで wrap すると、既存の生成コードを不必要に汚すため除外。
                if expr_yields_value_for_stmt_use(&l) {
                    let l_str = expr_to_string(&l);
                    let r_str = expr_to_string(&r);
                    syn::parse_str(&format!("{{ let _ = {}; {} }}", l_str, r_str))
                        .unwrap_or_else(|_| int_lit(0))
                } else {
                    syn::Expr::Block(syn::ExprBlock {
                        attrs: vec![],
                        label: None,
                        block: syn::Block {
                            brace_token: Default::default(),
                            stmts: vec![
                                syn::Stmt::Expr(l, Some(Default::default())),
                                syn::Stmt::Expr(r, None),
                            ],
                        },
                    })
                }
            }
            ExprKind::UnaryMinus(inner) => {
                let e = self.build_syn_expr(inner, info);
                // unsigned 型のキャスト結果に対する負号は wrapping_neg を使用
                let e_str = expr_to_string(&e);
                if is_unsigned_cast_expr(&e_str) {
                    return syn::parse_str(&format!("({}).wrapping_neg()", e_str.trim_start_matches('-')))
                        .unwrap_or_else(|_| int_lit(0));
                }
                if let Some(ut) = self.infer_expr_type_unified(inner, info) {
                    let ts = ut.to_rust_string();
                    if matches!(normalize_integer_type(&ts), Some("usize" | "u8" | "u16" | "u32" | "u64")) {
                        self.codegen_errors.push(format!("cannot negate unsigned type: -({}: {})", e_str, ts));
                    }
                }
                syn::Expr::Unary(syn::ExprUnary {
                    attrs: vec![],
                    op: syn::UnOp::Neg(Default::default()),
                    expr: Box::new(e),
                })
            }
            ExprKind::LogNot(inner) => {
                // C の `!"literal"` は 0 (false)。文字列リテラルは常に非 NULL。
                if matches!(&inner.kind, ExprKind::StringLit(_)) {
                    return syn::parse_str("false").unwrap_or_else(|_| int_lit(0));
                }
                let e = self.build_syn_expr(inner, info);
                if !self.is_bool_expr_with_dict(inner)
                    && (self.is_pointer_expr_unified(inner, info)
                        || self.infer_expr_type_unified(inner, info).is_some_and(|ut| ut.is_pointer()))
                {
                    // C の `!ptr` は「ptr が NULL か」そのもの。`.is_null()` が
                    // 否定を内包するので、外側に ! を重ねてはならない
                    // (かつて二重否定になっており、SvPVXtrue 等の null ガードが
                    // 反転していた)。
                    return syn::Expr::MethodCall(syn::ExprMethodCall {
                        attrs: vec![],
                        receiver: Box::new(e),
                        dot_token: Default::default(),
                        method: ident("is_null"),
                        turbofish: None,
                        paren_token: Default::default(),
                        args: syn::punctuated::Punctuated::new(),
                    });
                }
                let bool_e = if self.is_bool_expr_with_dict(inner) {
                    e
                } else {
                    // 整数 → != 0 して bool に変換、否定は外側 ! が担当
                    wrap_as_bool(e)
                };
                syn::Expr::Unary(syn::ExprUnary {
                    attrs: vec![],
                    op: syn::UnOp::Not(Default::default()),
                    expr: Box::new(bool_e),
                })
            }
            ExprKind::Cast { type_name, expr: inner } => {
                let t = self.type_name_to_rust(type_name);
                // C の慣用表現 `(unsigned)-1` は最大値を意味する。
                // `-1 as T` では E0600 (usize に対する単項 - は不可) になるため、
                // `T::MAX` に置換する。`-N` 一般は今は扱わず、`-1` のみ対応。
                if is_unsigned_integer_target(&t) {
                    if let ExprKind::UnaryMinus(minus_inner) = &inner.kind {
                        if matches!(&minus_inner.kind,
                            ExprKind::IntLit(1) | ExprKind::UIntLit(1))
                        {
                            return syn::parse_str(&format!("{}::MAX", t))
                                .unwrap_or_else(|_| int_lit(0));
                        }
                    }
                }
                // C の const-cast パターン `(T*)&place`:
                // place が `*const` ポインタ deref 経由（例: `(c)->field` で
                // `c: *const PERL_CONTEXT`）だと `&raw mut place` が E0596
                // になるため、`&raw const place as *mut T` で const を剥がす。
                // place が元から mut でも結果は等価（raw pointer の as cast は
                // mut/const を問わない）。
                if (t.starts_with("*mut ") || t.starts_with("*const "))
                    && matches!(&inner.kind, ExprKind::AddrOf(_))
                {
                    if let ExprKind::AddrOf(addrof_inner) = &inner.kind {
                        // AddrOf の inner が「callee マクロ呼び出し」なら inline 展開
                        let inlined_owned;
                        let inner_to_build: &Expr =
                            if let Some(inlined) = self.try_inline_call_for_addrof(addrof_inner) {
                                inlined_owned = inlined;
                                &inlined_owned
                            } else {
                                addrof_inner
                            };
                        let inner_e = self.build_syn_expr(inner_to_build, info);
                        let raw_const = syn::Expr::RawAddr(syn::ExprRawAddr {
                            attrs: vec![],
                            and_token: Default::default(),
                            raw: Default::default(),
                            mutability: syn::PointerMutability::Const(Default::default()),
                            expr: Box::new(inner_e),
                        });
                        return insert_cast(raw_const, parse_type(&t));
                    }
                }
                let e = self.build_syn_expr(inner, info);
                if t == "()" {
                    // void キャスト → 式の値を `let _` で捨てる。
                    // 単純な `{ expr; }` だと `must_use` 関数の戻り値（size_of_val 等）、
                    // 論理演算 (`a || b`)、単項式 (`*ptr`) を捨てたときに
                    // unused_must_use / unused_logical_op / unused_unary lint で
                    // CI の `-D warnings` が落ちる。`let _ = expr;` ならどんな式でも
                    // 「使った」扱いになり、副作用ある式（assignment, fn call）でも安全。
                    let e_str = expr_to_string(&e);
                    return syn::parse_str(&format!("{{ let _ = {}; }}", e_str))
                        .unwrap_or_else(|_| int_lit(0));
                }
                if t == "bool" {
                    // bool キャスト
                    if self.is_bool_expr_with_dict(inner) {
                        return e;
                    }
                    if self.is_pointer_expr_unified(inner, info)
                        || self.infer_expr_type_unified(inner, info).is_some_and(|ut| ut.is_pointer()) {
                        // ポインタ → !ptr.is_null()
                        let is_null = syn::Expr::MethodCall(syn::ExprMethodCall {
                            attrs: vec![],
                            receiver: Box::new(e),
                            dot_token: Default::default(),
                            method: ident("is_null"),
                            turbofish: None,
                            paren_token: Default::default(),
                            args: syn::punctuated::Punctuated::new(),
                        });
                        return syn::Expr::Unary(syn::ExprUnary {
                            attrs: vec![],
                            op: syn::UnOp::Not(Default::default()),
                            expr: Box::new(is_null),
                        });
                    }
                    return wrap_as_bool(e);
                }
                if self.is_enum_cast_target(type_name) {
                    // enum キャスト → transmute
                    let e_str = expr_to_string(&e);
                    return syn::parse_str(&format!("std::mem::transmute::<_, {}>({})", t, e_str))
                        .unwrap_or_else(|_| int_lit(0));
                }
                insert_cast(e, parse_type(&t))
            }
            ExprKind::Sizeof(inner) => {
                // sizeof(literal_string_param) → param.len() + 1
                if let ExprKind::Ident(name) = &inner.kind {
                    if self.current_literal_string_params.contains(name) {
                        let param = escape_rust_keyword(self.interner.get(*name));
                        return syn::parse_str(&format!("({}.len() + 1)", param))
                            .unwrap_or_else(|_| int_lit(0));
                    }
                }
                let e = self.build_syn_expr(inner, info);
                let e_str = expr_to_string(&e);
                // `&` 単項はフィールド/関数呼出より低優先なため、
                // `&len + 1` は `(&len) + 1` と解釈される。
                // 常に `&(expr)` と paren で包み、normalize_parens で
                // 不要な括弧を除去する。
                syn::parse_str(&format!("std::mem::size_of_val(&({}))", e_str))
                    .unwrap_or_else(|_| int_lit(0))
            }
            ExprKind::SizeofType(type_name) => {
                let t = self.type_name_to_rust(type_name);
                syn::parse_str(&format!("std::mem::size_of::<{}>()", t))
                    .unwrap_or_else(|_| int_lit(0))
            }
            ExprKind::Index { expr: base, index } => {
                use crate::syn_codegen::*;
                let i = self.build_syn_expr(index, info);
                let i_isize = cast_syn_expr(i, "isize");
                let base_expr: syn::Expr = if self.is_array_like_expr(base, info) {
                    // 配列ベースを raw pointer に減衰させる。
                    // - 通常: `array.as_ptr()`
                    // - extern static (mutable static 含む): `((&raw const NAME) as *const T)`
                    //   (Rust 2024 で `static mut` への shared reference が hard error 化。
                    //   `.as_ptr()` 経由の暗黙参照を避ける。要素型 T は bindings から取得)
                    let raw_ptr_expr: syn::Expr = if let ExprKind::Ident(n) = &base.kind {
                        let name_str = self.interner.get(*n);
                        let escaped = escape_rust_keyword(name_str);
                        if self.bindings_info.static_arrays.contains(name_str) {
                            let elem = self.bindings_info
                                .static_array_element_type(name_str)
                                .unwrap_or_else(|| "u8".to_string());
                            syn::parse_str(&format!(
                                "((&raw const {}) as *const {})",
                                escaped, elem
                            ))
                            .unwrap_or_else(|_| int_lit(0))
                        } else {
                            let id = ident_expr(escaped.as_str());
                            method_call(id, "as_ptr", vec![])
                        }
                    } else {
                        let b = self.build_syn_expr(base, info);
                        method_call(b, "as_ptr", vec![])
                    };
                    // 要素型を可能な限り抽出して `*mut ELEM` にキャスト。
                    // `as_ptr()` / `&raw const` どちらも `*const T` だが、後段で
                    // `&mut a[i]` を取るときに E0596 を避けるため mut へ寄せる。
                    let elem_str = self.infer_expr_type_unified(base, info)
                        .and_then(|ut| ut.inner_type().cloned())
                        .map(|inner| inner.to_rust_string());
                    if let Some(elem) = elem_str {
                        cast_syn_expr(raw_ptr_expr, &format!("*mut {}", elem))
                    } else {
                        raw_ptr_expr
                    }
                } else {
                    self.build_syn_expr(base, info)
                };
                let offset_call = method_call(base_expr, "offset", vec![i_isize]);
                deref(offset_call)
            }
            ExprKind::Conditional { cond, then_expr, else_expr } => {
                let c = self.build_syn_expr(cond, info);
                let c_str = expr_to_string(&c);
                let cond_str = self.wrap_as_bool_condition(cond, &c_str, info);
                let cond_syn: syn::Expr = syn::parse_str(&cond_str).unwrap_or(c);

                let type_hint = self.current_return_type.as_ref().map(|ut| ut.to_rust_string());
                let tt = self.infer_expr_type_unified(then_expr, info);
                let et = self.infer_expr_type_unified(else_expr, info);

                // Fix 3: type_hint ベースの null→null_ptr, 0/1→bool 変換
                if let Some(ref hint) = type_hint {
                    let hint_ut = UnifiedType::from_rust_str(hint);
                    if hint_ut.is_pointer() {
                        // type_hint (current_return_type) はマクロ全体の戻り値で、
                        // call 引数内の入れ子三項にも渡ってくる。反対 arm の実推論型が
                        // 非ポインタ (整数フラグ等) なら、この 0 はフラグの 0 であって
                        // NULL ではない (例: 5.44 hv_fetch の
                        // `HV_FETCH_JUST_SV | (lval ? HV_FETCH_LVALUE : 0)`)。
                        // その場合は null 化せず arm ベースのフォールバックに任せる
                        if is_null_literal(else_expr)
                            && tt.as_ref().map_or(true, |t| t.is_pointer())
                        {
                            let t = self.build_syn_expr(then_expr, info);
                            let e: syn::Expr = syn::parse_str(&null_ptr_expr(&hint_ut))
                                .unwrap_or_else(|_| int_lit(0));
                            return if_else(cond_syn, t, e);
                        }
                        if is_null_literal(then_expr)
                            && et.as_ref().map_or(true, |t| t.is_pointer())
                        {
                            let t: syn::Expr = syn::parse_str(&null_ptr_expr(&hint_ut))
                                .unwrap_or_else(|_| int_lit(0));
                            let e = self.build_syn_expr(else_expr, info);
                            return if_else(cond_syn, t, e);
                        }
                    }
                    // 整数型の type_hint + 片方が null リテラル → null を 0 に。
                    // C `cond ? i : 0` の 0 がマクロ展開で NULL になっているケース。
                    if normalize_integer_type(hint).is_some() {
                        if is_null_literal(else_expr) {
                            let t = self.build_syn_expr(then_expr, info);
                            return if_else(cond_syn, t, int_lit(0));
                        }
                        if is_null_literal(then_expr) {
                            let e = self.build_syn_expr(else_expr, info);
                            return if_else(cond_syn, int_lit(0), e);
                        }
                    }
                    if hint_ut.is_bool() {
                        let then_syn = match &then_expr.kind {
                            ExprKind::IntLit(0) => syn::parse_str("false").unwrap(),
                            ExprKind::IntLit(1) => syn::parse_str("true").unwrap(),
                            _ => self.build_syn_expr(then_expr, info),
                        };
                        let else_syn = match &else_expr.kind {
                            ExprKind::IntLit(0) => syn::parse_str("false").unwrap(),
                            ExprKind::IntLit(1) => syn::parse_str("true").unwrap(),
                            _ => self.build_syn_expr(else_expr, info),
                        };
                        return if_else(cond_syn, then_syn, else_syn);
                    }
                }

                // null リテラル分岐の型推論（type_hint がない場合のフォールバック）
                let then_syn = if is_null_literal(then_expr) {
                    if let Some(ref eut) = et {
                        if eut.is_pointer() {
                            syn::parse_str(&null_ptr_expr(eut)).unwrap_or_else(|_| int_lit(0))
                        } else { self.build_syn_expr(then_expr, info) }
                    } else { self.build_syn_expr(then_expr, info) }
                } else { self.build_syn_expr(then_expr, info) };

                let else_syn = if is_null_literal(else_expr) {
                    if let Some(ref tut) = tt {
                        if tut.is_pointer() {
                            syn::parse_str(&null_ptr_expr(tut)).unwrap_or_else(|_| int_lit(0))
                        } else { self.build_syn_expr(else_expr, info) }
                    } else { self.build_syn_expr(else_expr, info) }
                } else { self.build_syn_expr(else_expr, info) };

                // Fix 2: wider_integer_type キャスト
                if let (Some(tut), Some(eut)) = (&tt, &et) {
                    let ts = tut.to_rust_string();
                    let es = eut.to_rust_string();
                    if let (Some(tn), Some(en)) = (normalize_integer_type(&ts), normalize_integer_type(&es)) {
                        if tn != en {
                            if let Some(wider) = wider_integer_type(&ts, &es) {
                                let (then_final, else_final) = if normalize_integer_type(&ts) != Some(wider) {
                                    (cast_syn_expr(then_syn, wider), else_syn)
                                } else {
                                    (then_syn, cast_syn_expr(else_syn, wider))
                                };
                                return if_else(cond_syn, then_final, else_final);
                            }
                        }
                    }
                    // SV サブタイプ間: 両枝を共通型 (type_hint 優先、なければ then 型)
                    // に揃えて if/else 型整合を確保する。
                    if tut.is_pointer() && eut.is_pointer() && ts != es
                        && is_sv_subtype_cast(tut, eut)
                    {
                        // type_hint が void (`()` や `c_void`) の場合は cast 対象に
                        // しない (return 型が void の文脈で分岐値は捨てられる)。
                        let target_str = type_hint.as_deref()
                            .filter(|t| *t != "()" && !UnifiedType::from_rust_str(t).is_void())
                            .unwrap_or(&ts)
                            .to_string();
                        return if_else(
                            cond_syn,
                            cast_syn_expr(then_syn, &target_str),
                            cast_syn_expr(else_syn, &target_str),
                        );
                    }
                }

                if_else(cond_syn, then_syn, else_syn)
            }
            ExprKind::Binary { op, lhs, rhs } => {
                // sizeof(literal_string_param) - 1 → param.len()
                if *op == BinOp::Sub {
                    if let ExprKind::Sizeof(inner) = &lhs.kind {
                        if let ExprKind::Ident(name) = &inner.kind {
                            if self.current_literal_string_params.contains(name) {
                                if let ExprKind::IntLit(1) = &rhs.kind {
                                    let param = escape_rust_keyword(self.interner.get(*name));
                                    return syn::parse_str(&format!("{}.len()", param))
                                        .unwrap_or_else(|_| int_lit(0));
                                }
                            }
                        }
                    }
                }

                // C 文字列リテラル vs 0 比較 → 常に非 NULL (c"..." は &CStr)
                //   `c"..." == 0` → false / `c"..." != 0` → true
                if matches!(op, BinOp::Eq | BinOp::Ne) {
                    let s_eq = |s: &ExprKind| matches!(s, ExprKind::StringLit(_));
                    let lhs_is_str = s_eq(&lhs.kind);
                    let rhs_is_str = s_eq(&rhs.kind);
                    if (lhs_is_str && is_null_literal(rhs))
                       || (rhs_is_str && is_null_literal(lhs))
                    {
                        return syn::parse_str(
                            if *op == BinOp::Eq { "false" } else { "true" }
                        ).unwrap_or_else(|_| int_lit(0));
                    }
                }
                // Option<fn(...)> == 0 / != 0 → .is_none() / .is_some()
                // bindgen の関数ポインタフィールドは Option で包まれるため、
                // ポインタ null 比較より先に Option 判定を行う。
                if matches!(op, BinOp::Eq | BinOp::Ne) {
                    let opt_lhs = (is_null_literal(rhs) || matches!(&rhs.kind, ExprKind::IntLit(0)))
                        && self.is_option_fn_pointer_expr(lhs, info);
                    let opt_rhs = (is_null_literal(lhs) || matches!(&lhs.kind, ExprKind::IntLit(0)))
                        && self.is_option_fn_pointer_expr(rhs, info);
                    if opt_lhs || opt_rhs {
                        let receiver_expr = if opt_lhs { lhs } else { rhs };
                        let r = self.build_syn_expr(receiver_expr, info);
                        let method = if *op == BinOp::Eq { "is_none" } else { "is_some" };
                        return syn::Expr::MethodCall(syn::ExprMethodCall {
                            attrs: vec![], receiver: Box::new(r), dot_token: Default::default(),
                            method: ident(method), turbofish: None,
                            paren_token: Default::default(), args: syn::punctuated::Punctuated::new(),
                        });
                    }
                }
                // ポインタ == 0 / != 0 → .is_null()
                if matches!(op, BinOp::Eq | BinOp::Ne) {
                    if is_null_literal(rhs) {
                        if self.is_pointer_expr_unified(lhs, info)
                            || self.infer_expr_type_unified(lhs, info).is_some_and(|ut| ut.is_pointer()) {
                            let l = self.build_syn_expr(lhs, info);
                            let is_null = syn::Expr::MethodCall(syn::ExprMethodCall {
                                attrs: vec![], receiver: Box::new(l), dot_token: Default::default(),
                                method: ident("is_null"), turbofish: None,
                                paren_token: Default::default(), args: syn::punctuated::Punctuated::new(),
                            });
                            return if *op == BinOp::Eq { is_null } else {
                                syn::Expr::Unary(syn::ExprUnary {
                                    attrs: vec![], op: syn::UnOp::Not(Default::default()),
                                    expr: Box::new(is_null),
                                })
                            };
                        }
                    }
                    if is_null_literal(lhs) {
                        if self.is_pointer_expr_unified(rhs, info)
                            || self.infer_expr_type_unified(rhs, info).is_some_and(|ut| ut.is_pointer()) {
                            let r = self.build_syn_expr(rhs, info);
                            let is_null = syn::Expr::MethodCall(syn::ExprMethodCall {
                                attrs: vec![], receiver: Box::new(r), dot_token: Default::default(),
                                method: ident("is_null"), turbofish: None,
                                paren_token: Default::default(), args: syn::punctuated::Punctuated::new(),
                            });
                            return if *op == BinOp::Eq { is_null } else {
                                syn::Expr::Unary(syn::ExprUnary {
                                    attrs: vec![], op: syn::UnOp::Not(Default::default()),
                                    expr: Box::new(is_null),
                                })
                            };
                        }
                    }
                    // bool_expr != 0 → bool_expr, bool_expr == 0 → !bool_expr
                    if self.is_bool_expr_with_dict(lhs) {
                        match (&rhs.kind, op) {
                            (ExprKind::IntLit(0), BinOp::Ne) | (ExprKind::IntLit(1), BinOp::Eq) => {
                                return self.build_syn_expr(lhs, info);
                            }
                            (ExprKind::IntLit(0), BinOp::Eq) | (ExprKind::IntLit(1), BinOp::Ne) => {
                                let l = self.build_syn_expr(lhs, info);
                                return syn::Expr::Unary(syn::ExprUnary {
                                    attrs: vec![], op: syn::UnOp::Not(Default::default()),
                                    expr: Box::new(l),
                                });
                            }
                            _ => {}
                        }
                    }
                    if self.is_bool_expr_with_dict(rhs) {
                        match (&lhs.kind, op) {
                            (ExprKind::IntLit(0), BinOp::Ne) | (ExprKind::IntLit(1), BinOp::Eq) => {
                                return self.build_syn_expr(rhs, info);
                            }
                            (ExprKind::IntLit(0), BinOp::Eq) | (ExprKind::IntLit(1), BinOp::Ne) => {
                                let r = self.build_syn_expr(rhs, info);
                                return syn::Expr::Unary(syn::ExprUnary {
                                    attrs: vec![], op: syn::UnOp::Not(Default::default()),
                                    expr: Box::new(r),
                                });
                            }
                            _ => {}
                        }
                    }
                }

                // ポインタ ± 整数 → .offset()
                if matches!(op, BinOp::Add | BinOp::Sub) {
                    // static 配列の Ident (`bodies_by_type` 等) は build_syn_expr
                    // が既に `.as_ptr()` を付加するため、`l_arr` には含めない。
                    // ここで再度 `.as_ptr()` を被せると *const T に対して呼ばれて
                    // E0599 になる。非 Ident の「フィールド直値が配列型」に限定。
                    let l_arr = !self.is_static_array_expr(lhs)
                        && self.is_array_like_expr(lhs, info);
                    let r_arr = !self.is_static_array_expr(rhs)
                        && self.is_array_like_expr(rhs, info);
                    let lp = l_arr
                        || self.is_static_array_expr(lhs)
                        || self.is_pointer_expr_unified(lhs, info)
                        || self.infer_expr_type_unified(lhs, info).is_some_and(|ut| ut.is_pointer());
                    let rp = r_arr
                        || self.is_static_array_expr(rhs)
                        || self.is_pointer_expr_unified(rhs, info)
                        || self.infer_expr_type_unified(rhs, info).is_some_and(|ut| ut.is_pointer());
                    // 配列値に対して `(arr.as_ptr() as *mut ELEM)` で mut pointer を
                    // 作るヘルパ。l_arr は field-access の配列（build_syn_expr が
                    // `.as_ptr()` を付けない）、is_static_array は Ident の static
                    // 配列（build_syn_expr が既に `.as_ptr()` を付ける）。どちらの
                    // ケースでも結果が `*const T` にならないよう `*mut T` へ cast
                    // する。
                    let l_is_static_arr = self.is_static_array_expr(lhs);
                    let r_is_static_arr = self.is_static_array_expr(rhs);
                    let cast_to_mut = |this: &mut Self, expr: syn::Expr, arr_expr: &Expr,
                                        needs_as_ptr: bool| -> syn::Expr {
                        let elem = this.infer_expr_type_unified(arr_expr, info)
                            .and_then(|ut| ut.inner_type().cloned())
                            .map(|u| u.to_rust_string());
                        let base = if needs_as_ptr {
                            crate::syn_codegen::method_call(expr, "as_ptr", vec![])
                        } else {
                            expr
                        };
                        if let Some(e) = elem {
                            crate::syn_codegen::cast_syn_expr(base, &format!("*mut {}", e))
                        } else {
                            base
                        }
                    };
                    if lp && !rp {
                        let l = self.build_syn_expr(lhs, info);
                        let l = if l_arr {
                            cast_to_mut(self, l, lhs, /*needs_as_ptr=*/true)
                        } else if l_is_static_arr {
                            cast_to_mut(self, l, lhs, /*needs_as_ptr=*/false)
                        } else {
                            l
                        };
                        let r = self.build_syn_expr(rhs, info);
                        let r_isize = crate::syn_codegen::cast_syn_expr(r, "isize");
                        let arg = if *op == BinOp::Add { r_isize } else {
                            syn::Expr::Unary(syn::ExprUnary {
                                attrs: vec![],
                                op: syn::UnOp::Neg(Default::default()),
                                expr: Box::new(r_isize),
                            })
                        };
                        return crate::syn_codegen::method_call(l, "offset", vec![arg]);
                    }
                    if rp && !lp && *op == BinOp::Add {
                        let l = self.build_syn_expr(lhs, info);
                        let r = self.build_syn_expr(rhs, info);
                        let r = if r_arr {
                            cast_to_mut(self, r, rhs, /*needs_as_ptr=*/true)
                        } else if r_is_static_arr {
                            cast_to_mut(self, r, rhs, /*needs_as_ptr=*/false)
                        } else {
                            r
                        };
                        let l_isize = crate::syn_codegen::cast_syn_expr(l, "isize");
                        return crate::syn_codegen::method_call(r, "offset", vec![l_isize]);
                    }
                    if lp && rp && *op == BinOp::Sub {
                        let l = self.build_syn_expr(lhs, info);
                        let r = self.build_syn_expr(rhs, info);
                        return crate::syn_codegen::method_call(l, "offset_from", vec![r]);
                    }
                }

                // float vs int literal → float に変換
                if matches!(&rhs.kind, ExprKind::IntLit(_)) {
                    if let Some(lut) = self.infer_expr_type_unified(lhs, info) {
                        if lut.is_float() {
                            if let ExprKind::IntLit(v) = &rhs.kind {
                                let l = self.build_syn_expr(lhs, info);
                                let l_str = expr_to_string(&l);
                                return syn::parse_str(&format!("{} {} {}.0", l_str, bin_op_to_rust(*op), v))
                                    .unwrap_or_else(|_| int_lit(0));
                            }
                        }
                    }
                }
                if matches!(&lhs.kind, ExprKind::IntLit(_)) {
                    if let Some(rut) = self.infer_expr_type_unified(rhs, info) {
                        if rut.is_float() {
                            if let ExprKind::IntLit(v) = &lhs.kind {
                                let r = self.build_syn_expr(rhs, info);
                                let r_str = expr_to_string(&r);
                                return syn::parse_str(&format!("{}.0 {} {}", v, bin_op_to_rust(*op), r_str))
                                    .unwrap_or_else(|_| int_lit(0));
                            }
                        }
                    }
                }

                // ポインタ比較で const/mut が違う場合の cast。
                // `*mut T < *const T` は Rust で E0308（types differ in mutability）。
                // rhs を lhs の型に合わせる（wrapper の呼出規約で lhs が mut、
                // callee 由来の rhs が const になりやすい）。
                if matches!(op, BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge | BinOp::Eq | BinOp::Ne) {
                    if let (Some(lut), Some(rut)) =
                        (self.infer_expr_type_unified(lhs, info),
                         self.infer_expr_type_unified(rhs, info))
                    {
                        if lut.is_pointer() && rut.is_pointer()
                            && pointer_inner_compatible(&lut, &rut)
                            && (lut.is_const_pointer() != rut.is_const_pointer()
                                || lut.to_rust_string() != rut.to_rust_string())
                        {
                            let l = self.build_syn_expr(lhs, info);
                            let r = self.build_syn_expr(rhs, info);
                            let target = lut.to_rust_string();
                            return syn::Expr::Binary(syn::ExprBinary {
                                attrs: vec![], left: Box::new(l),
                                op: crate::syn_codegen::to_syn_binop(*op),
                                right: Box::new(crate::syn_codegen::cast_syn_expr(r, &target)),
                            });
                        }
                    }
                }

                let l = self.build_syn_expr(lhs, info);
                let r = self.build_syn_expr(rhs, info);

                // 論理演算子: bool 変換
                if matches!(op, BinOp::LogAnd | BinOp::LogOr) {
                    let l_str = expr_to_string(&l);
                    let r_str = expr_to_string(&r);
                    let l_bool = self.wrap_as_bool_condition(lhs, &l_str, info);
                    let r_bool = self.wrap_as_bool_condition(rhs, &r_str, info);
                    let l_syn: syn::Expr = syn::parse_str(&l_bool).unwrap_or(l);
                    let r_syn: syn::Expr = syn::parse_str(&r_bool).unwrap_or(r);
                    return syn::Expr::Binary(syn::ExprBinary {
                        attrs: vec![], left: Box::new(l_syn),
                        op: crate::syn_codegen::to_syn_binop(*op),
                        right: Box::new(r_syn),
                    });
                }

                // 型キャスト挿入（整数幅、bool→int、float↔int、enum→int）
                let lt = self.infer_expr_type_unified(lhs, info);
                let rt = self.infer_expr_type_unified(rhs, info);

                // 片側だけ判定可能でも enum は必ず integer にキャストする。
                // `cs << 7` のように rhs が IntLit Binary で型推論 None の
                // ケースを拾う（enum 単独で Shl/BitOr 等が定義されていない）。
                let make_binary_op = |left: syn::Expr, right: syn::Expr| -> syn::Expr {
                    syn::Expr::Binary(syn::ExprBinary {
                        attrs: vec![], left: Box::new(left),
                        op: crate::syn_codegen::to_syn_binop(*op),
                        right: Box::new(right),
                    })
                };
                match (&lt, &rt) {
                    (Some(lut), None) if self.is_rust_enum_type(lut) => {
                        return make_binary_op(cast_syn_expr(l, "u32"), r);
                    }
                    (None, Some(rut)) if self.is_rust_enum_type(rut) => {
                        return make_binary_op(l, cast_syn_expr(r, "u32"));
                    }
                    _ => {}
                }

                if let (Some(lut), Some(rut)) = (&lt, &rt) {
                    let make_binary = |left: syn::Expr, right: syn::Expr| -> syn::Expr {
                        syn::Expr::Binary(syn::ExprBinary {
                            attrs: vec![], left: Box::new(left),
                            op: crate::syn_codegen::to_syn_binop(*op),
                            right: Box::new(right),
                        })
                    };
                    // enum → integer キャスト（C 側は int 互換だが Rust enum は
                    // nominal 型なので比較・ビット演算で `as <int>` が必要）
                    let l_is_enum = self.is_rust_enum_type(lut);
                    let r_is_enum = self.is_rust_enum_type(rut);
                    if l_is_enum && !r_is_enum {
                        let rs = rut.to_rust_string();
                        let target = normalize_integer_type(&rs).unwrap_or("u32");
                        return make_binary(cast_syn_expr(l, target), r);
                    }
                    if r_is_enum && !l_is_enum {
                        let ls = lut.to_rust_string();
                        let target = normalize_integer_type(&ls).unwrap_or("u32");
                        return make_binary(l, cast_syn_expr(r, target));
                    }
                    // bool → integer キャスト
                    if rut.is_bool() {
                        let ls = lut.to_rust_string();
                        if let Some(nl) = normalize_integer_type(&ls) {
                            return make_binary(l, cast_syn_expr(r, nl));
                        }
                    }
                    if lut.is_bool() {
                        let rs = rut.to_rust_string();
                        if let Some(nr) = normalize_integer_type(&rs) {
                            return make_binary(cast_syn_expr(l, nr), r);
                        }
                    }
                    // float ↔ integer キャスト
                    if lut.is_float() && !rut.is_float() {
                        let ls = lut.to_rust_string();
                        let float_ty = if ls == "c_float" || ls == "f32" { "f32" } else { "f64" };
                        return make_binary(l, cast_syn_expr(r, float_ty));
                    }
                    if rut.is_float() && !lut.is_float() {
                        let rs = rut.to_rust_string();
                        let float_ty = if rs == "c_float" || rs == "f32" { "f32" } else { "f64" };
                        return make_binary(cast_syn_expr(l, float_ty), r);
                    }
                    // 整数幅不一致 → wider type にキャスト
                    let ls = lut.to_rust_string();
                    let rs = rut.to_rust_string();
                    if let Some(wider) = wider_integer_type(&ls, &rs) {
                        if normalize_integer_type(&ls) != Some(wider) {
                            return make_binary(cast_syn_expr(l, wider), r);
                        } else {
                            return make_binary(l, cast_syn_expr(r, wider));
                        }
                    }
                }
                // float (片方のみ型判明)
                {
                    let make_binary = |left: syn::Expr, right: syn::Expr| -> syn::Expr {
                        syn::Expr::Binary(syn::ExprBinary {
                            attrs: vec![], left: Box::new(left),
                            op: crate::syn_codegen::to_syn_binop(*op),
                            right: Box::new(right),
                        })
                    };
                    match (&lt, &rt) {
                        (Some(lut), None) if lut.is_float() => {
                            let ls = lut.to_rust_string();
                            let float_ty = if ls == "c_float" || ls == "f32" { "f32" } else { "f64" };
                            return make_binary(l, cast_syn_expr(r, float_ty));
                        }
                        (None, Some(rut)) if rut.is_float() => {
                            let rs = rut.to_rust_string();
                            let float_ty = if rs == "c_float" || rs == "f32" { "f32" } else { "f64" };
                            return make_binary(cast_syn_expr(l, float_ty), r);
                        }
                        _ => {}
                    }
                    // ビット演算で片方のみ型判明
                    if matches!(op, BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor) {
                        match (&lt, &rt) {
                            (Some(lut), None) => {
                                let ls = lut.to_rust_string();
                                if let Some(nl) = normalize_integer_type(&ls) {
                                    return make_binary(l, cast_syn_expr(r, nl));
                                }
                            }
                            (None, Some(rut)) => {
                                let rs = rut.to_rust_string();
                                if let Some(nr) = normalize_integer_type(&rs) {
                                    return make_binary(cast_syn_expr(l, nr), r);
                                }
                            }
                            _ => {}
                        }
                    }
                }
                // 基本の二項演算
                syn::Expr::Binary(syn::ExprBinary {
                    attrs: vec![],
                    left: Box::new(l),
                    op: crate::syn_codegen::to_syn_binop(*op),
                    right: Box::new(r),
                })
            }
            ExprKind::Call { func, args } => {
                // 共通フィールドマクロが定義する関数ポインタフィールドの呼び出しを検出。
                // `(*x).xcv_root_u.xcv_xsub(args)` のような Member/PtrMember 経由の Call で、
                // canonical type が fn ポインタなら `<field>.unwrap_unchecked()(args)` 形式で出力。
                if let Some(syn_call) = self.try_build_common_macro_fn_call(func, args, info) {
                    return syn_call;
                }
                // builtin 関数の特殊処理
                if let ExprKind::Ident(name) = &func.kind {
                    let func_name = self.interner.get(*name);
                    // __builtin_expect → 引数を透過
                    if func_name == "__builtin_expect" && !args.is_empty() {
                        return self.build_syn_expr(&args[0], info);
                    }
                    // __builtin_unreachable → unreachable_unchecked
                    if func_name == "__builtin_unreachable" {
                        return syn::parse_str("std::hint::unreachable_unchecked()").unwrap();
                    }
                    // __builtin_ctz/clz → trailing_zeros/leading_zeros
                    if (func_name == "__builtin_ctz" || func_name == "__builtin_ctzl") && args.len() == 1 {
                        let arg = self.build_syn_expr(&args[0], info);
                        return syn::Expr::MethodCall(syn::ExprMethodCall {
                            attrs: vec![], receiver: Box::new(arg), dot_token: Default::default(),
                            method: ident("trailing_zeros"), turbofish: None,
                            paren_token: Default::default(), args: syn::punctuated::Punctuated::new(),
                        });
                    }
                    if (func_name == "__builtin_clz" || func_name == "__builtin_clzl") && args.len() == 1 {
                        let arg = self.build_syn_expr(&args[0], info);
                        return syn::Expr::MethodCall(syn::ExprMethodCall {
                            attrs: vec![], receiver: Box::new(arg), dot_token: Default::default(),
                            method: ident("leading_zeros"), turbofish: None,
                            paren_token: Default::default(), args: syn::punctuated::Punctuated::new(),
                        });
                    }
                    // ASSERT_IS_LITERAL 等 → 引数を透過
                    if matches!(func_name, "ASSERT_IS_LITERAL" | "ASSERT_IS_PTR" | "ASSERT_NOT_PTR")
                        && args.len() == 1
                    {
                        return self.build_syn_expr(&args[0], info);
                    }
                    // offsetof → std::mem::offset_of!
                    //
                    // args[0] は本来 BuiltinCall arm で TypeName として処理される。
                    // ここに来るのは offsetof が通常の関数呼び出しとしてパースされた
                    // ケースで、args[0] は型名を表す Ident になる。build_syn_expr の
                    // Ident arm は known_symbols にない名前を unresolved_names に
                    // 登録してしまい、cascade 連鎖で依存先まで TYPE_INCOMPLETE 化する。
                    // 型名の Ident は名前そのものを使えば十分なので bypass する。
                    if matches!(func_name, "offsetof" | "__builtin_offsetof") && args.len() == 2 {
                        let type_name_str = if let ExprKind::Ident(name) = &args[0].kind {
                            escape_rust_keyword(self.interner.get(*name)).to_string()
                        } else {
                            let s = self.build_syn_expr(&args[0], info);
                            expr_to_string(&s)
                        };
                        if let Some(field_path) = self.expr_to_field_path(&args[1]) {
                            return syn::parse_str(&format!("std::mem::offset_of!({}, {})", type_name_str, field_path))
                                .unwrap_or_else(|_| int_lit(0));
                        }
                    }
                }

                // 関数名を構築
                let f_syn = self.build_syn_expr(func, info);
                let f_str = expr_to_string(&f_syn);

                let callee_name = if let ExprKind::Ident(name) = &func.kind { Some(*name) } else { None };
                let needs_my_perl = callee_name
                    .map(|name| self.needs_my_perl_for_call(name, args.len()))
                    .unwrap_or(false);

                // ジェネリック型パラメータのチェック
                let callee_generics = callee_name
                    .and_then(|name| self.get_callee_generic_params(name).cloned());

                if let Some(ref generics) = callee_generics {
                    // turbofish 構文 — 文字列ベースで構築（型引数のため）
                    let mut type_args = Vec::new();
                    let mut value_args: Vec<String> = if needs_my_perl {
                        vec!["my_perl".to_string()]
                    } else { vec![] };
                    let mut value_idx = if needs_my_perl { 1usize } else { 0 };
                    for (i, arg) in args.iter().enumerate() {
                        if generics.contains_key(&(i as i32)) {
                            let syn_arg = self.build_syn_expr(arg, info);
                            type_args.push(normalize_parens(&expr_to_string(&syn_arg)));
                        } else {
                            value_args.push(self.build_arg_string_unified(arg, info, callee_name, value_idx));
                            value_idx += 1;
                        }
                    }
                    return syn::parse_str(&format!("{}::<{}>({})", f_str, type_args.join(", "), value_args.join(", ")))
                        .unwrap_or_else(|_| int_lit(0));
                }

                // 通常の関数呼び出し — 引数を処理（統一版）
                let mut arg_strs: Vec<String> = if needs_my_perl {
                    vec!["my_perl".to_string()]
                } else { vec![] };
                let arg_offset = if needs_my_perl { 1usize } else { 0 };
                for (i, arg) in args.iter().enumerate() {
                    arg_strs.push(self.build_arg_string_unified(arg, info, callee_name, i + arg_offset));
                }
                syn::parse_str(&format!("{}({})", f_str, arg_strs.join(", ")))
                    .unwrap_or_else(|_| int_lit(0))
            }
            ExprKind::MacroCall { name, args, expanded, .. } => {
                if self.should_emit_as_macro_call(*name) {
                    let name_str = escape_rust_keyword(self.interner.get(*name));
                    let needs_my_perl = self.needs_my_perl_for_call(*name, args.len());
                    let mut a: Vec<String> = if needs_my_perl {
                        vec!["my_perl".to_string()]
                    } else { vec![] };
                    for arg in args {
                        let arg_str = expr_to_string(&self.build_syn_expr(arg, info));
                        a.push(normalize_parens(&arg_str));
                    }
                    syn::parse_str(&format!("{}({})", name_str, a.join(", ")))
                        .unwrap_or_else(|_| int_lit(0))
                } else {
                    self.build_syn_expr(expanded, info)
                }
            }
            ExprKind::BuiltinCall { name, args } => {
                let func_name = self.interner.get(*name);
                if matches!(func_name, "offsetof" | "__builtin_offsetof" | "STRUCT_OFFSET")
                    && args.len() == 2
                {
                    let type_str = match &args[0] {
                        crate::ast::BuiltinArg::TypeName(tn) => self.type_name_to_rust(tn),
                        crate::ast::BuiltinArg::Expr(e) => {
                            let s = self.build_syn_expr(e, info);
                            expr_to_string(&s)
                        }
                    };
                    let field_expr = match &args[1] {
                        crate::ast::BuiltinArg::Expr(e) => self.expr_to_field_path(e),
                        _ => None,
                    };
                    if let Some(fp) = field_expr {
                        return syn::parse_str(&format!("std::mem::offset_of!({}, {})", type_str, fp))
                            .unwrap_or_else(|_| int_lit(0));
                    }
                }
                // フォールバック: 通常の関数呼び出し
                let a: Vec<String> = args.iter().map(|arg| match arg {
                    crate::ast::BuiltinArg::Expr(e) => {
                        let s = self.build_syn_expr(e, info);
                        expr_to_string(&s)
                    }
                    crate::ast::BuiltinArg::TypeName(tn) => self.type_name_to_rust(tn),
                }).collect();
                syn::parse_str(&format!("{}({})", func_name, a.join(", ")))
                    .unwrap_or_else(|_| int_lit(0))
            }
            ExprKind::Assign { op, lhs, rhs } => {
                self.build_assign_syn_expr(*op, lhs, rhs, info)
            }
            ExprKind::PreInc(inner) => {
                self.build_inc_dec_syn_expr(inner, info, /*is_inc=*/ true, /*is_post=*/ false)
            }
            ExprKind::PreDec(inner) => {
                self.build_inc_dec_syn_expr(inner, info, /*is_inc=*/ false, /*is_post=*/ false)
            }
            ExprKind::PostInc(inner) => {
                self.build_inc_dec_syn_expr(inner, info, /*is_inc=*/ true, /*is_post=*/ true)
            }
            ExprKind::PostDec(inner) => {
                self.build_inc_dec_syn_expr(inner, info, /*is_inc=*/ false, /*is_post=*/ true)
            }
            ExprKind::Assert { kind, condition } => {
                let assert_str = if let Some((real_cond, msg)) = decompose_assert_with_message(condition) {
                    let c = self.build_syn_expr(real_cond, info);
                    let c_str = expr_to_string(&c);
                    let cond_str = self.wrap_as_bool_condition(real_cond, &c_str, info);
                    format!("assert!({}, \"{}\")", normalize_parens(&cond_str), msg)
                } else {
                    let c = self.build_syn_expr(condition, info);
                    let c_str = expr_to_string(&c);
                    if is_boolean_expr(condition) || self.is_bool_expr_with_dict(condition) {
                        format!("assert!({})", normalize_parens(&c_str))
                    } else if self.is_pointer_expr_unified(condition, info)
                        || self.infer_expr_type_unified(condition, info).is_some_and(|ut| ut.is_pointer()) {
                        // `.is_null()` をくっつける際は受け手側を括弧で囲む。
                        // c_str が `*x` のような単項式の場合 `!*x.is_null()` が
                        // `!*(x.is_null())` と解釈されるため。
                        format!("assert!(!({}).is_null())", c_str)
                    } else {
                        format!("assert!({} != 0)", normalize_parens(&c_str))
                    }
                };
                let result = match kind {
                    AssertKind::Assert => assert_str,
                    AssertKind::AssertUnderscore => format!("{{ {}; }}", assert_str),
                };
                syn::parse_str(&result).unwrap_or_else(|_| int_lit(0))
            }
            ExprKind::StmtExpr(compound) => {
                // MUTABLE_PTR パターン → 内部式に変換
                if let Some(init_expr) = self.detect_mutable_ptr_pattern(compound) {
                    return self.build_syn_expr(init_expr, info);
                }
                // 通常の statement expression: Rust のブロック式として出力。
                // 旧パスの実装と同じ部分文字列構築 + syn::parse_str だが、
                // 内部の式は build_expr_string (build_syn_expr 経由) を通る。
                let mut parts: Vec<String> = Vec::new();
                for item in &compound.items {
                    match item {
                        BlockItem::Stmt(Stmt::Expr(Some(e), _)) => {
                            parts.push(self.build_expr_string(e, info));
                        }
                        BlockItem::Stmt(stmt) => {
                            let s = match info {
                                Some(info) => self.stmt_to_rust(stmt, info),
                                None => self.stmt_to_rust_inline(stmt, ""),
                            };
                            if !s.is_empty() {
                                // 空文を push すると join で `;;` になる
                                parts.push(s);
                            }
                        }
                        BlockItem::Decl(decl) => {
                            self.collect_decl_types(decl);
                            let decl_str = self.decl_to_rust_let(decl, "");
                            for line in decl_str.lines() {
                                let trimmed = line.trim();
                                if !trimmed.is_empty() {
                                    parts.push(trimmed.strip_suffix(';').unwrap_or(trimmed).to_string());
                                }
                            }
                        }
                    }
                }
                let block_str = if parts.is_empty() {
                    "{ }".to_string()
                } else if parts.len() == 1 {
                    parts.pop().unwrap()
                } else {
                    let last = parts.pop().unwrap();
                    let stmts = parts.join("; ");
                    format!("{{ {}; {} }}", stmts, last)
                };
                syn::parse_str(&block_str).unwrap_or_else(|_| int_lit(0))
            }
            ExprKind::Alignof(ty) => {
                let ty_str = self.type_name_to_rust(ty);
                syn::parse_str(&format!("std::mem::align_of::<{}>()", ty_str))
                    .unwrap_or_else(|_| int_lit(0))
            }
            // CompoundLit / 未対応バリアント: 0 を返してエラーを記録。
            // 旧パスでも `/* TODO */` マーカーで生成コードが壊れる挙動だったため、
            // 体感的には同等。実用上 macro_bindings.rs にはほぼ現れない。
            _ => {
                self.codegen_errors.push(format!(
                    "unhandled ExprKind in syn codegen: {:?}",
                    std::mem::discriminant(&expr.kind)
                ));
                int_lit(0)
            }
        }
    }

    /// 関数引数を文字列に変換（macro/inline 統一版）
    ///
    /// expr_to_rust_arg 相当の処理を build_syn_expr ベースで行う。
    fn build_arg_string_unified(&mut self, arg: &Expr, info: Option<&MacroInferInfo>,
                                  callee: Option<InternedStr>, arg_index: usize) -> String {
        // macro: literal string パラメータ変換
        if info.is_some() {
            if let Some(name) = self.find_literal_string_ident(arg) {
                if let Some(callee_name) = callee {
                    if self.callee_expects_literal_string(callee_name, arg_index) {
                        return escape_rust_keyword(self.interner.get(*name));
                    }
                }
                let param = escape_rust_keyword(self.interner.get(*name));
                // callee の期待型が分かればそれに cast する
                // (memCHRs → libc::memchr の *const c_void 等。GH #15)。
                // 不明時は従来どおり *const c_char
                if let Some(callee_name) = callee {
                    let func_name = self.interner.get(callee_name).to_string();
                    if let Some(expected) = self.get_callee_param_type_extended(&func_name, arg_index) {
                        if expected.is_pointer() {
                            return format!("{}.as_ptr() as {}", param, expected.to_rust_string());
                        }
                    }
                }
                return format!("{}.as_ptr() as *const c_char", param);
            }
        }
        // null pointer 変換
        if is_null_literal(arg) {
            if let Some(callee_name) = callee {
                let func_name = self.interner.get(callee_name).to_string();
                if let Some(expected_ut) = self.get_callee_param_type_extended(&func_name, arg_index) {
                    if expected_ut.is_pointer() {
                        return null_ptr_expr(&expected_ut);
                    }
                }
            }
        }
        // bool パラメータ変換
        if let Some(callee_name) = callee {
            let func_name = self.interner.get(callee_name);
            if self.callee_param_is_bool(func_name, arg_index) {
                match &arg.kind {
                    ExprKind::IntLit(0) => return "false".to_string(),
                    ExprKind::IntLit(1) => return "true".to_string(),
                    _ => {}
                }
            }
        }
        // 式の生成
        let mut syn_expr = self.build_syn_expr(arg, info);
        // 整数幅キャスト / SV subtype キャストを syn レベルで挿入
        // （文字列 "as type" の優先順位崩壊を防止）
        if let Some(callee_name) = callee {
            let func_name = self.interner.get(callee_name).to_string();
            if let Some(expected_ut) = self.get_callee_param_type_extended(&func_name, arg_index) {
                let actual_ut = self.infer_expr_type_unified(arg, info);
                let actual_ty = actual_ut.as_ref().map(|ut| ut.to_rust_string());
                let expected_ty = expected_ut.to_rust_string();
                // C 文字列リテラル (`c"..."`, &CStr 型) を callee の `*const c_char` 等
                // に変換する際に `.as_ptr()` を付加。これを忘れると
                // `expected *const i8, found &CStr` エラーになる。
                if matches!(&arg.kind, ExprKind::StringLit(_))
                    && expected_ut.is_pointer()
                {
                    syn_expr = crate::syn_codegen::method_call(syn_expr, "as_ptr", vec![]);
                }
                syn_expr = self.cast_arg_syn_if_needed(syn_expr, actual_ty.as_deref(), &expected_ty);
            }
        }
        normalize_parens(&crate::syn_codegen::expr_to_string(&syn_expr))
    }

    /// 引数のキャスト挿入を syn::Expr レベルで実施。
    /// 旧パスの `cast_integer_arg_if_needed` (文字列ベース) と論理は同等だが、
    /// `as` 演算子の優先順位崩壊を起こさない。
    fn cast_arg_syn_if_needed(&self, arg_expr: syn::Expr,
                              actual_ty: Option<&str>, expected_ty: &str) -> syn::Expr {
        use crate::syn_codegen::cast_syn_expr;
        if let Some(actual) = actual_ty {
            // enum → integer キャスト（actual が Rust enum で expected が整数型）
            let actual_ut = UnifiedType::from_rust_str(actual);
            if self.is_rust_enum_type(&actual_ut) {
                if let Some(target) = normalize_integer_type(expected_ty) {
                    return cast_syn_expr(arg_expr, target);
                }
            }
            let na = normalize_integer_type(actual);
            let ne = normalize_integer_type(expected_ty);
            if let (Some(a), Some(e)) = (na, ne) {
                // Rust never converts between isize/i64 or usize/u64 implicitly,
                // so any difference needs a cast.
                if a != e {
                    return cast_syn_expr(arg_expr, e);
                }
                return arg_expr;
            }
            // ポインタ型のサブタイプ変換 (e.g., *mut GV → *mut SV)
            if actual != expected_ty {
                let actual_ut = UnifiedType::from_rust_str(actual);
                let expected_ut = UnifiedType::from_rust_str(expected_ty);
                if actual_ut.is_pointer() && expected_ut.is_pointer()
                    && is_sv_subtype_cast(&actual_ut, &expected_ut) {
                    let cast_ty = if actual.contains("*const") {
                        expected_ty.replace("*mut", "*const")
                    } else {
                        expected_ty.to_string()
                    };
                    return cast_syn_expr(arg_expr, &cast_ty);
                }
                // `*mut c_void` 受け取り: libc の memset/memcpy/memcmp 等は
                // `*mut c_void` / `*const c_void` を取るが、呼出側は
                // `*mut c_char` や `*mut T` を渡すことが多い。raw pointer
                // の as-cast はいつでも合法なので、void pointer への変換を
                // 汎用的に許可する。
                if actual_ut.is_pointer() && expected_ut.is_pointer()
                    && expected_ut.is_void_pointer()
                {
                    return cast_syn_expr(arg_expr, expected_ty);
                }
                // const/mut のみ違い、または inner 別名違い（c_char≡i8 等）
                // の自動キャスト。Rust は単一レベルの `*mut T → *const T` を
                // 自動強制するため、その場合だけ cast を挿入しない。
                // ネストした二重ポインタの const 違い（例: `*mut *mut T`
                // → `*mut *const T`）は強制されないので cast が必要。
                if pointer_inner_compatible(&actual_ut, &expected_ut) {
                    let top_mut_to_const = !actual_ut.is_const_pointer()
                        && expected_ut.is_const_pointer();
                    let top_const_same = actual_ut.is_const_pointer()
                        == expected_ut.is_const_pointer();
                    // 内側まで完全一致（文字列レベル）か
                    let inner_exact_match = actual_ut.inner_type()
                        .zip(expected_ut.inner_type())
                        .is_some_and(|(a, b)| a.to_rust_string() == b.to_rust_string());
                    let auto_coerces = (top_mut_to_const || top_const_same)
                        && inner_exact_match;
                    if !auto_coerces {
                        return cast_syn_expr(arg_expr, expected_ty);
                    }
                }
            }
            return arg_expr;
        }
        // actual 不明 + expected が SV ポインタ → 関数呼び出し風なら as キャスト
        let expected_ut = UnifiedType::from_rust_str(expected_ty);
        if expected_ut.is_pointer() {
            if let Some(inner) = expected_ut.inner_type() {
                if let UnifiedType::Named(name) = inner {
                    let n = name.as_str();
                    if matches!(n, "SV" | "GV" | "HV" | "AV" | "CV" | "IO") {
                        if matches!(&arg_expr, syn::Expr::Call(_) | syn::Expr::MethodCall(_)) {
                            return cast_syn_expr(arg_expr, expected_ty);
                        }
                    }
                }
            }
        }
        arg_expr
    }

    /// 関数ポインタフィールド呼び出しを検出し、
    /// `<receiver>.<field>.unwrap_unchecked()(args)` 形式の `syn::Expr` を構築する。
    ///
    /// `_XPVCV_COMMON` の `xcv_xsub` や `PerlInterpreter` の `Ilockhook` のような
    /// `Option<unsafe extern "C" fn(...)>` フィールドを呼び出すパターンに対応する。
    /// 検出条件:
    ///
    /// 1. callee が `Member` または `PtrMember` アクセスである
    /// 2. そのフィールドが「関数ポインタ」と判定できる
    ///    - 第一に共通フィールドマクロ（`_XPVCV_COMMON` 等）由来の canonical
    ///      type で判定（C ソース由来）
    ///    - フォールバックとして bindings.rs 由来の `field_type_map` で判定
    ///      （文字列ヒューリスティク: `fn(` を含む）
    ///
    /// 当てはまらなければ `None` を返し、呼び出し側は通常の Call 生成パスへ進む。
    fn try_build_common_macro_fn_call(
        &mut self,
        func: &Expr,
        args: &[Expr],
        info: Option<&MacroInferInfo>,
    ) -> Option<syn::Expr> {
        use crate::syn_codegen::*;

        // C: `(*s->fp)(args)` is the same call as `s->fp(args)`;
        //dereferencing a function pointer is a no-op (e.g. PerlEnv_putenv under PERL_IMPLICIT_SYS).
        // PERL_IMPLICITY_SYS is windows only
        let func = match &func.kind {
            ExprKind::Deref(inner) => inner.as_ref(),
            _ => func
        };

        let member_id = match &func.kind {
            ExprKind::Member { member, .. } | ExprKind::PtrMember { member, .. } => *member,
            _ => return None,
        };

        // 第一: 共通フィールドマクロ由来の canonical type で判定
        let mut is_fn_ptr = self
            .fields_dict
            .and_then(|d| d.canonical_field(member_id).map(|(_, f)| f.is_fn_pointer))
            .unwrap_or(false);

        // フォールバック: bindings.rs 側の型を見て fn(...) 形式なら fn ポインタとみなす。
        // bindgen 出力では Option<unsafe extern "C" fn(...)> 形式になるため
        // "fn(" を含むかでヒューリスティック判定。フィールドの直接の型が
        // typedef（例: `share_proc_t`）の場合は RustDeclDict.types で解決する。
        if !is_fn_ptr {
            let field_name = self.interner.get(member_id);
            if let Some(ut) = self.field_type_map.get(field_name) {
                let ty_str = ut.to_rust_string();
                if type_str_is_fn_pointer(&ty_str) {
                    is_fn_ptr = true;
                } else if let Some(dict) = self.rust_decl_dict {
                    if let Some(alias) = dict.types.get(&ty_str) {
                        if type_str_is_fn_pointer(&alias.ty) {
                            is_fn_ptr = true;
                        }
                    }
                }
            }
        }

        if !is_fn_ptr {
            return None;
        }

        // <receiver>.<field> までを syn::Expr で組む
        let field_access = self.build_syn_expr(func, info);
        // .unwrap_unchecked() を挟む（bindgen の Option<fn> をはがす）
        let callee = method_call(field_access, "unwrap_unchecked", vec![]);

        // 引数: 既存ヘルパで構築（callee 名なし → 型ベース cast は無効）
        let mut punctuated = syn::punctuated::Punctuated::new();
        for (i, arg) in args.iter().enumerate() {
            let s = self.build_arg_string_unified(arg, info, None, i);
            let parsed = syn::parse_str(&s).unwrap_or_else(|_| int_lit(0));
            punctuated.push(parsed);
        }

        Some(syn::Expr::Call(syn::ExprCall {
            attrs: vec![],
            func: Box::new(callee),
            paren_token: Default::default(),
            args: punctuated,
        }))
    }

    /// lvalue 用の syn::Expr を構築（MacroCall/Call の展開対応）
    fn build_lvalue_syn_expr(&mut self, expr: &Expr, info: Option<&MacroInferInfo>) -> syn::Expr {
        if let ExprKind::MacroCall { expanded, .. } = &expr.kind {
            return self.build_syn_expr(expanded, info);
        }
        if let ExprKind::Call { func, args } = &expr.kind {
            if let Some(expanded) = self.try_expand_call_as_lvalue_syn(func, args, info) {
                return expanded;
            }
            let syn_expr = self.build_syn_expr(expr, info);
            let s = crate::syn_codegen::expr_to_string(&syn_expr);
            self.codegen_errors.push(format!("invalid lvalue: {} cannot be assigned to", s));
            return syn_expr;
        }
        self.build_syn_expr(expr, info)
    }

    /// `Pre/PostInc/Dec` の「lv を 1 つ増減する」代入式と lvalue を構築。
    /// 値が要らない文位置では pre/post の区別が消えるため、step だけを使う。
    fn build_inc_dec_step_expr(&mut self, inner: &Expr, info: Option<&MacroInferInfo>,
                               is_inc: bool) -> (syn::Expr, syn::Expr) {
        use crate::syn_codegen::*;
        let lv = self.build_lvalue_syn_expr(inner, info);
        let is_ptr = self.is_pointer_expr_unified(inner, info)
            || self.infer_expr_type_unified(inner, info).is_some_and(|ut| ut.is_pointer());
        let step: syn::Expr = if is_ptr {
            // lv = lv.wrapping_add(1)  または wrapping_sub
            let method = if is_inc { "wrapping_add" } else { "wrapping_sub" };
            let call = method_call(lv.clone(), method, vec![int_lit(1)]);
            assign_expr(lv.clone(), call)
        } else {
            let op = if is_inc {
                syn::BinOp::AddAssign(Default::default())
            } else {
                syn::BinOp::SubAssign(Default::default())
            };
            assign_op_expr(lv.clone(), op, int_lit(1))
        };
        (step, lv)
    }

    /// `Pre/PostInc/Dec` を syn::Expr で構築。
    /// 旧パス `expr_to_rust_ctx` の対応 arm（rust_codegen.rs の PreInc 〜 PostDec）と同等。
    fn build_inc_dec_syn_expr(&mut self, inner: &Expr, info: Option<&MacroInferInfo>,
                              is_inc: bool, is_post: bool) -> syn::Expr {
        use crate::syn_codegen::*;
        let (step, lv) = self.build_inc_dec_step_expr(inner, info, is_inc);
        let step_stmt = semi_stmt(step);
        if is_post {
            // { let _t = lv; <step>; _t }
            let save = let_stmt("_t", lv.clone());
            block_with_value(vec![save, step_stmt], ident_expr("_t"))
        } else {
            // { <step>; lv }
            block_with_value(vec![step_stmt], lv)
        }
    }

    /// 文位置の `Pre/PostInc/Dec` を step 代入文だけで出力する。
    /// block_with_value 形 `{ *lv -= 1; *lv }` を文として出すと末尾の
    /// 単項式が unused_must_use に触れるため (Perl_cx_popformat 等)。
    fn build_inc_dec_stmt(&mut self, inner: &Expr, info: Option<&MacroInferInfo>,
                          is_inc: bool, indent: &str) -> String {
        let (step, _lv) = self.build_inc_dec_step_expr(inner, info, is_inc);
        let s = normalize_parens(&crate::syn_codegen::expr_to_string(&step));
        format!("{}{};", indent, s)
    }

    /// 式を文位置で出力する共通経路。代入・増減式は値を返さない文形式に
    /// 落とす (stmt_to_rust / stmt_to_rust_inline / for の step 部で共用)。
    fn expr_stmt_string(&mut self, expr: &Expr, info: Option<&MacroInferInfo>, indent: &str) -> String {
        match &expr.kind {
            ExprKind::Assign { op, lhs, rhs } => {
                self.build_assign_stmt(op, lhs, rhs, indent, info)
            }
            ExprKind::PreInc(inner) | ExprKind::PostInc(inner) => {
                self.build_inc_dec_stmt(inner, info, true, indent)
            }
            ExprKind::PreDec(inner) | ExprKind::PostDec(inner) => {
                self.build_inc_dec_stmt(inner, info, false, indent)
            }
            _ => format!("{}{};", indent, self.build_expr_string(expr, info)),
        }
    }

    /// `Assign` を syn::Expr で構築。
    /// 旧パス `expr_to_rust_ctx` の Assign arm（rust_codegen.rs:4380 周辺）と同等。
    fn build_assign_syn_expr(&mut self, op: AssignOp, lhs: &Expr, rhs: &Expr,
                             info: Option<&MacroInferInfo>) -> syn::Expr {
        use crate::syn_codegen::*;
        let l = self.build_lvalue_syn_expr(lhs, info);
        let lhs_ut = self.infer_expr_type_unified(lhs, info);

        // RHS の構築（null リテラル特別扱い + プレーン Assign の整数幅キャスト）
        let r: syn::Expr = if is_null_literal(rhs) && op == AssignOp::Assign {
            match &lhs_ut {
                Some(lut) if lut.is_pointer() => {
                    if lut.is_const_pointer() {
                        syn::parse_str("std::ptr::null()").unwrap_or_else(|_| int_lit(0))
                    } else {
                        syn::parse_str("std::ptr::null_mut()").unwrap_or_else(|_| int_lit(0))
                    }
                }
                Some(_) => int_lit(0),
                None => syn::parse_str("std::ptr::null_mut()").unwrap_or_else(|_| int_lit(0)),
            }
        } else {
            let mut r_expr = self.build_syn_expr(rhs, info);
            // プレーン Assign のみ: LHS 型に合わせて整数幅キャスト挿入
            if op == AssignOp::Assign {
                // 整数リテラル 0/1 を float LHS に代入 → `0.0` / `1.0` に変換
                if let Some(ref lut) = lhs_ut {
                    if lut.is_float() {
                        if let ExprKind::IntLit(n) = &rhs.kind {
                            r_expr = syn::parse_str(&format!("{}.0", n))
                                .unwrap_or_else(|_| int_lit(0));
                        }
                    }
                }
                if let Some(ref lut) = lhs_ut {
                    if let Some(rut) = self.infer_expr_type_unified(rhs, info) {
                        let ls = lut.to_rust_string();
                        let rs = rut.to_rust_string();
                        if let (Some(nl), Some(nr)) = (
                            normalize_integer_type(&ls),
                            normalize_integer_type(&rs),
                        ) {
                            if !integer_types_compatible(nl, nr) {
                                r_expr = cast_syn_expr(r_expr, nl);
                            }
                        }
                        // ポインタ const/mut 不一致 → LHS 型に明示キャスト
                        else if lut.is_pointer() && rut.is_pointer()
                            && lut.is_const_pointer() != rut.is_const_pointer()
                        {
                            r_expr = cast_syn_expr(r_expr, &ls);
                        }
                        // SV サブタイプ / void ポインタ間の変換
                        else if lut.is_pointer() && rut.is_pointer()
                            && ls != rs
                            && is_sv_subtype_cast(&rut, lut)
                        {
                            r_expr = cast_syn_expr(r_expr, &ls);
                        }
                    }
                }
            }
            r_expr
        };

        // ビットフィールドセッター: LHS が `.getter()` 形式の MethodCall
        // （bitfield_methods 由来）だった場合、`set_getter(val)` 呼び出しに
        // 書き換える。`a.f() = v` は E0070 (invalid lvalue) になるため。
        if op == AssignOp::Assign {
            if let syn::Expr::MethodCall(mc) = &l {
                if mc.args.is_empty() {
                    let method_name = mc.method.to_string();
                    if self.is_bitfield_method(&method_name) {
                        let setter_name = format!("set_{}", method_name);
                        // setter が期待する型 (bitfield getter の戻り値型)
                        // に RHS を cast する。bool や違う整数幅で渡される
                        // ケースに対応 (`set_op_moresib(bool)` 等の E0308)。
                        let arg_val = if let Some(dict) = self.rust_decl_dict {
                            // bitfield_method_types: (struct, method) → ret_ty
                            // struct 名が不明なケースは任意の一致で決める
                            let ret_ty = dict.bitfield_method_types.iter()
                                .find(|((_, m), _)| m == &method_name)
                                .map(|(_, ty)| ty.clone());
                            if let Some(ty) = ret_ty {
                                cast_syn_expr(r, &ty)
                            } else {
                                r
                            }
                        } else {
                            r
                        };
                        let setter_call = method_call(
                            (*mc.receiver).clone(),
                            &setter_name,
                            vec![arg_val],
                        );
                        let stmt = semi_stmt(setter_call);
                        return block_with_value(vec![stmt], l);
                    }
                }
            }
        }

        // op に応じた文を構築し、ブロックでラップ
        let stmt: syn::Stmt = match op {
            AssignOp::Assign => semi_stmt(assign_expr(l.clone(), r)),
            AssignOp::AddAssign | AssignOp::SubAssign => {
                let is_ptr = self.is_pointer_expr_unified(lhs, info)
                    || lhs_ut.as_ref().is_some_and(|ut| ut.is_pointer());
                if is_ptr {
                    // lv = lv.wrapping_add(r as usize);
                    let method = if op == AssignOp::AddAssign { "wrapping_add" } else { "wrapping_sub" };
                    let r_usize = cast_syn_expr(r, "usize");
                    let call = method_call(l.clone(), method, vec![r_usize]);
                    semi_stmt(assign_expr(l.clone(), call))
                } else {
                    let syn_op = c_assign_op_to_syn_compound(op).unwrap();
                    semi_stmt(assign_op_expr(l.clone(), syn_op, r))
                }
            }
            AssignOp::AndAssign | AssignOp::OrAssign | AssignOp::XorAssign => {
                // 整数幅が異なる場合 RHS を LHS 型にキャスト。
                // RHS が enum (Named) の場合も LHS 整数型へキャストする
                // (Rust enum は nominal 型で `|=` 等が直接働かない)。
                let lt = &lhs_ut;
                let rt = self.infer_expr_type_unified(rhs, info);
                let r_final = {
                    let mut casted = false;
                    let mut ret = r;
                    if let (Some(lut), Some(rut)) = (lt, &rt) {
                        let ls = lut.to_rust_string();
                        let rs = rut.to_rust_string();
                        let nl = normalize_integer_type(&ls);
                        let nr = normalize_integer_type(&rs);
                        if nl.is_some() && nr.is_some() && nl != nr {
                            ret = cast_syn_expr(ret, nl.unwrap());
                            casted = true;
                        }
                        // RHS が enum 型で LHS が整数 → `r as <lhs-int>`
                        if !casted && self.is_rust_enum_type(rut) && nl.is_some() {
                            ret = cast_syn_expr(ret, nl.unwrap());
                            casted = true;
                        }
                    }
                    if !casted {
                        if let (Some(lut), None) = (lt, &rt) {
                            let ls = lut.to_rust_string();
                            if let Some(nl) = normalize_integer_type(&ls) {
                                ret = cast_syn_expr(ret, nl);
                            }
                        }
                    }
                    ret
                };
                let syn_op = c_assign_op_to_syn_compound(op).unwrap();
                semi_stmt(assign_op_expr(l.clone(), syn_op, r_final))
            }
            _ => {
                let syn_op = c_assign_op_to_syn_compound(op).unwrap();
                semi_stmt(assign_op_expr(l.clone(), syn_op, r))
            }
        };
        block_with_value(vec![stmt], l)
    }

    /// lvalue 用の文字列を構築（`build_lvalue_syn_expr` の文字列化版）
    fn build_lvalue_string(&mut self, expr: &Expr, info: Option<&MacroInferInfo>) -> String {
        let syn_expr = self.build_lvalue_syn_expr(expr, info);
        normalize_parens(&crate::syn_codegen::expr_to_string(&syn_expr))
    }

    /// build_syn_expr + type_hint 適用（generate_macro のトップレベル用）
    fn build_syn_expr_with_type_hint(&mut self, expr: &Expr, info: Option<&MacroInferInfo>,
                                       type_hint: Option<&str>) -> syn::Expr {
        use crate::syn_codegen::*;
        if let Some(ty) = type_hint {
            let ut = UnifiedType::from_rust_str(ty);
            if ut.is_pointer() && is_null_literal(expr) {
                return syn::parse_str(&null_ptr_expr(&ut)).unwrap_or_else(|_| int_lit(0));
            }
            if ut.is_bool() {
                match &expr.kind {
                    ExprKind::IntLit(0) => return syn::parse_str("false").unwrap(),
                    ExprKind::IntLit(1) => return syn::parse_str("true").unwrap(),
                    _ => {}
                }
            }
        }
        self.build_syn_expr(expr, info)
    }

    // ================================================================
    // 統一文変換 (macro/inline 共通)
    // ================================================================

    /// return 文を syn::Expr ベースで構築（macro/inline 統一）
    ///
    /// 整数幅キャストや bool 変換を syn::Expr レベルで挿入することで、
    /// 文字列ベースの `(rhs as ty)` フォーマットが `normalize_parens` で
    /// 括弧を剥がされて優先順位が崩れる問題（`a & b as u8` 等）を回避する。
    fn build_return_stmt(&mut self, expr: &Expr, indent: &str, info: Option<&MacroInferInfo>) -> String {
        use crate::syn_codegen::*;
        if let Some(ref rt) = self.current_return_type {
            if rt.is_pointer() && is_null_literal(expr) {
                return format!("{}return {};", indent, null_ptr_expr(rt));
            }
            if rt.is_bool() {
                match &expr.kind {
                    ExprKind::IntLit(0) => return format!("{}return false;", indent),
                    ExprKind::IntLit(1) => return format!("{}return true;", indent),
                    _ => {
                        let mut syn_expr = self.build_syn_expr(expr, info);
                        if !self.is_bool_expr_with_dict(expr) && !is_bool_syn_expr(&syn_expr) {
                            syn_expr = wrap_as_bool(syn_expr);
                        }
                        let s = normalize_parens(&expr_to_string(&syn_expr));
                        return format!("{}return {};", indent, s);
                    }
                }
            }
        }
        let mut syn_expr = self.build_syn_expr(expr, info);
        syn_expr = self.cast_return_syn_expr_if_needed(expr, info, syn_expr);
        let s = normalize_parens(&expr_to_string(&syn_expr));
        format!("{}return {};", indent, s)
    }

    /// 返り値式の型キャストを syn::Expr レベルで挿入（必要なら）。
    /// `cast_return_expr_if_needed_unified` の syn 版。
    fn cast_return_syn_expr_if_needed(&self, expr: &Expr, info: Option<&MacroInferInfo>,
                                      syn_expr: syn::Expr) -> syn::Expr {
        let Some(ret_ut) = &self.current_return_type else { return syn_expr };
        let Some(expr_ut) = self.infer_expr_type_unified(expr, info) else { return syn_expr };
        let ret_s = ret_ut.to_rust_string();
        let expr_s = expr_ut.to_rust_string();
        // enum → integer 戻り値キャスト
        if self.is_rust_enum_type(&expr_ut) {
            if let Some(nr) = normalize_integer_type(&ret_s) {
                return crate::syn_codegen::cast_syn_expr(syn_expr, nr);
            }
        }
        if let (Some(nr), Some(ne)) = (normalize_integer_type(&ret_s), normalize_integer_type(&expr_s)) {
            // Rust never converts between isize/i64 or usize/u64 implicitly,
            // so any difference needs a cast.
            if nr != ne {
                return crate::syn_codegen::cast_syn_expr(syn_expr, nr);
            }
        }
        // ポインタ戻り値: inner が Rust 上 compatible（c_char≡i8 等）だが文字列
        // 表現が違う場合は return 型に合わせて as cast を入れる。
        // const/mut 違いも同時に吸収する。
        if ret_ut.is_pointer() && expr_ut.is_pointer() && ret_s != expr_s
            && pointer_inner_compatible(ret_ut, &expr_ut)
        {
            return crate::syn_codegen::cast_syn_expr(syn_expr, &ret_s);
        }
        syn_expr
    }

    /// 代入文を構築（macro/inline 統一、文コンテキスト用）
    ///
    /// LHS (`l`) と RHS (`r`) は両方とも syn::Expr 経由でビルドし、
    /// 整数幅キャストや `wrapping_add` の `as usize` も `cast_syn_expr` で挿入する。
    /// 文字列レベルで `as` を挿入する旧経路は優先順位崩壊を起こすため使わない。
    fn build_assign_stmt(&mut self, op: &AssignOp, lhs: &Expr, rhs: &Expr, indent: &str, info: Option<&MacroInferInfo>) -> String {
        use crate::syn_codegen::*;
        let l = self.build_lvalue_string(lhs, info);
        let lhs_ut = self.infer_expr_type_unified(lhs, info);

        // RHS の syn::Expr を組み立てる（null リテラル特別扱い + 必要に応じてキャスト）
        let r_syn: syn::Expr = if is_null_literal(rhs) && *op == AssignOp::Assign {
            match &lhs_ut {
                Some(lut) if lut.is_pointer() => {
                    let s = if lut.is_const_pointer() { "std::ptr::null()" } else { "std::ptr::null_mut()" };
                    syn::parse_str(s).unwrap_or_else(|_| int_lit(0))
                }
                Some(_) => int_lit(0),
                None => syn::parse_str("std::ptr::null_mut()").unwrap_or_else(|_| int_lit(0)),
            }
        } else {
            let mut r_syn = self.build_syn_expr(rhs, info);
            // 整数幅キャスト
            if *op == AssignOp::Assign {
                // 整数リテラル 0/1 を float LHS に代入 → `0.0` / `1.0`
                if let Some(ref lut) = lhs_ut {
                    if lut.is_float() {
                        if let ExprKind::IntLit(n) = &rhs.kind {
                            r_syn = syn::parse_str(&format!("{}.0", n))
                                .unwrap_or_else(|_| int_lit(0));
                        }
                    }
                }
                if let Some(ref lut) = lhs_ut {
                    if let Some(rut) = self.infer_expr_type_unified(rhs, info) {
                        let ls = lut.to_rust_string();
                        let rs = rut.to_rust_string();
                        // enum → integer 代入キャスト
                        if self.is_rust_enum_type(&rut) {
                            if let Some(nl) = normalize_integer_type(&ls) {
                                r_syn = cast_syn_expr(r_syn, nl);
                            }
                        } else if let (Some(nl), Some(nr)) = (normalize_integer_type(&ls), normalize_integer_type(&rs)) {
                            // 代入では isize/i64 や usize/u64 を区別する（Rust 的に別型）。
                            // integer_types_compatible の緩い判定は演算側で使用。
                            if nl != nr {
                                r_syn = cast_syn_expr(r_syn, nl);
                            }
                        }
                        // ポインタ const/mut 不一致、あるいは inner 別名違い
                        // （c_char≡i8 等の Rust レベル同型）→ LHS 型に明示 cast
                        if pointer_const_differs(lut, &rut) {
                            r_syn = cast_syn_expr(r_syn, &ls);
                        } else if lut.is_pointer() && rut.is_pointer()
                            && ls != rs
                            && pointer_inner_compatible(lut, &rut)
                        {
                            r_syn = cast_syn_expr(r_syn, &ls);
                        }
                        // SV サブタイプ / void ポインタ間の変換
                        // (*sv_).sv_any = <XPVIV *> を受け入れるため)
                        else if lut.is_pointer() && rut.is_pointer()
                            && ls != rs
                            && is_sv_subtype_cast(&rut, lut)
                        {
                            r_syn = cast_syn_expr(r_syn, &ls);
                        }
                    }
                }
            } else if matches!(op, AssignOp::AndAssign | AssignOp::OrAssign | AssignOp::XorAssign) {
                let rt = self.infer_expr_type_unified(rhs, info);
                if let (Some(lut), Some(rut)) = (&lhs_ut, &rt) {
                    let ls = lut.to_rust_string();
                    let rs = rut.to_rust_string();
                    let nl = normalize_integer_type(&ls);
                    let nr = normalize_integer_type(&rs);
                    if nl.is_some() && nr.is_some() && nl != nr {
                        r_syn = cast_syn_expr(r_syn, nl.unwrap());
                    } else if self.is_rust_enum_type(rut) && nl.is_some() {
                        // RHS が enum で LHS が整数 → `r as <lhs-int>`
                        r_syn = cast_syn_expr(r_syn, nl.unwrap());
                    }
                } else if let (Some(lut), None) = (&lhs_ut, &rt) {
                    let ls = lut.to_rust_string();
                    if let Some(nl) = normalize_integer_type(&ls) {
                        r_syn = cast_syn_expr(r_syn, nl);
                    }
                }
            }
            r_syn
        };

        match op {
            AssignOp::Assign => {
                let r = normalize_parens(&expr_to_string(&r_syn));
                format!("{}{} = {};", indent, l, r)
            }
            AssignOp::AddAssign | AssignOp::SubAssign => {
                if self.is_pointer_expr_unified(lhs, info)
                    || lhs_ut.as_ref().is_some_and(|ut| ut.is_pointer()) {
                    let method = if *op == AssignOp::AddAssign { "wrapping_add" } else { "wrapping_sub" };
                    let r_usize = cast_syn_expr(r_syn, "usize");
                    let r = expr_to_string(&r_usize);
                    format!("{}{} = {}.{}({});", indent, l, l, method, r)
                } else {
                    let r = normalize_parens(&expr_to_string(&r_syn));
                    format!("{}{} {} {};", indent, l, assign_op_to_rust(*op), r)
                }
            }
            _ => {
                let r = normalize_parens(&expr_to_string(&r_syn));
                format!("{}{} {} {};", indent, l, assign_op_to_rust(*op), r)
            }
        }
    }

    /// 式を文字列に変換（syn::Expr 経由で生成）
    fn build_expr_string(&mut self, expr: &Expr, info: Option<&MacroInferInfo>) -> String {
        let syn_expr = self.build_syn_expr(expr, info);
        normalize_parens(&crate::syn_codegen::expr_to_string(&syn_expr))
    }

    /// 文を Rust コードに変換（マクロ用）— 統一ヘルパーに委譲
    fn stmt_to_rust(&mut self, stmt: &Stmt, info: &MacroInferInfo) -> String {
        match stmt {
            Stmt::Expr(Some(expr), _) => {
                // 代入式・増減式は inline 経路 (stmt_to_rust_inline) と同様、
                // 値を返さない文形式で出力する。block_with_value 形
                // `{ lv op= r; lv }` を文位置に出すと末尾式が
                // unused_must_use 等の lint に触れるため
                self.expr_stmt_string(expr, Some(info), "")
            }
            // 空文 (空マクロ消去後の残骸) は出力しない。";" を返すと
            // 呼び出し側の join で `;;` (redundant_semicolons) になる
            Stmt::Expr(None, _) => String::new(),
            Stmt::Return(Some(expr), _) => self.build_return_stmt(expr, "", Some(info)),
            Stmt::Return(None, _) => "return;".to_string(),
            _ => self.todo_marker("stmt")
        }
    }

    /// TypeName を Rust 型文字列に変換
    fn type_name_to_rust(&mut self, type_name: &crate::ast::TypeName) -> String {
        // decl_specs_to_rust でベース型を取得（プリミティブ型も正しく変換）
        let base_type = self.decl_specs_to_rust(&type_name.specs);

        // 宣言子からポインタ/配列/関数を適用
        let mut result = if let Some(ref decl) = type_name.declarator {
            self.apply_derived_to_type(&base_type, &decl.derived)
        } else {
            base_type
        };

        // C の const 修飾子（例: const char*）を Rust の *const に反映
        // 最も内側のポインタを *const にする
        if type_name.specs.qualifiers.is_const {
            if let Some(pos) = result.rfind("*mut ") {
                result.replace_range(pos..pos + 5, "*const ");
            }
        }

        result
    }

    /// DeclSpecs を Rust 型文字列に変換
    fn decl_specs_to_rust(&mut self, specs: &DeclSpecs) -> String {
        // typedef 名を優先
        for spec in &specs.type_specs {
            if let TypeSpec::TypedefName(name) = spec {
                // 型パラメータなら generic 名に置換
                if let Some(generic_name) = self.current_type_param_map.get(name) {
                    return generic_name.clone();
                }
                let name_str = self.interner.get(*name).to_string();
                // 未定義型名の検出
                if !self.known_symbols.contains(&name_str) {
                    self.codegen_errors.push(format!("undefined type: {}", name_str));
                }
                return name_str;
            }
        }

        // 基本型をチェック
        let mut is_void = false;
        let mut is_char = false;
        let mut is_int = false;
        let mut is_short = false;
        let mut is_long = 0usize;
        let mut is_unsigned = false;
        let mut is_float = false;
        let mut is_double = false;

        for spec in &specs.type_specs {
            match spec {
                TypeSpec::Void => is_void = true,
                TypeSpec::Char => is_char = true,
                TypeSpec::Int => is_int = true,
                TypeSpec::Short => is_short = true,
                TypeSpec::Long => is_long += 1,
                TypeSpec::Unsigned => is_unsigned = true,
                TypeSpec::Signed => {}
                TypeSpec::Float => is_float = true,
                TypeSpec::Double => is_double = true,
                TypeSpec::Bool => return "bool".to_string(),
                TypeSpec::Struct(spec) => {
                    if let Some(n) = spec.name {
                        return self.interner.get(n).to_string();
                    } else {
                        return self.type_marker().to_string();
                    }
                }
                TypeSpec::Union(spec) => {
                    if let Some(n) = spec.name {
                        return self.interner.get(n).to_string();
                    } else {
                        return self.type_marker().to_string();
                    }
                }
                TypeSpec::Enum(spec) => {
                    if let Some(n) = spec.name {
                        return self.interner.get(n).to_string();
                    } else {
                        return "c_int".to_string();
                    }
                }
                _ => {}
            }
        }

        if is_void {
            return "()".to_string();
        }

        if is_float {
            return "c_float".to_string();
        }

        if is_double {
            return if is_long > 0 { "c_longdouble".to_string() } else { "c_double".to_string() };
        }

        if is_char {
            return if is_unsigned { "c_uchar".to_string() } else { "c_char".to_string() };
        }

        if is_short {
            return if is_unsigned { "c_ushort".to_string() } else { "c_short".to_string() };
        }

        if is_long >= 2 {
            return if is_unsigned { "c_ulonglong".to_string() } else { "c_longlong".to_string() };
        }

        if is_long == 1 {
            return if is_unsigned { "c_ulong".to_string() } else { "c_long".to_string() };
        }

        if is_int || is_unsigned {
            return if is_unsigned { "c_uint".to_string() } else { "c_int".to_string() };
        }

        self.type_marker().to_string()
    }

    /// 派生型を型に適用（関数ポインタを含む完全な処理）
    fn apply_derived_to_type(&mut self, base: &str, derived: &[DerivedDecl]) -> String {
        // Function を探す
        let fn_idx = derived
            .iter()
            .position(|d| matches!(d, DerivedDecl::Function(_)));

        if let Some(idx) = fn_idx {
            if let DerivedDecl::Function(param_list) = &derived[idx] {
                // Function の直前が Pointer なら関数ポインタ
                let is_fn_pointer =
                    idx > 0 && matches!(derived[idx - 1], DerivedDecl::Pointer(_));

                // 戻り値型の派生（Function と fn ptr Pointer を除く）
                let return_end = if is_fn_pointer { idx - 1 } else { idx };
                let return_derived = &derived[..return_end];
                let return_type = self.apply_simple_derived(base, return_derived);

                // パラメータリストを生成（型のみ、名前なし）
                let params: Vec<_> = param_list
                    .params
                    .iter()
                    .map(|p| self.param_type_only(p))
                    .collect();
                let params_str = params.join(", ");

                // 関数型を生成
                let fn_type =
                    format!("unsafe extern \"C\" fn({}) -> {}", params_str, return_type);

                // 関数ポインタの場合は Option でラップ（NULL 許容）
                if is_fn_pointer {
                    return format!("Option<{}>", fn_type);
                }
                return fn_type;
            }
        }

        // 通常の型変換（Function を含まない場合）
        self.apply_simple_derived(base, derived)
    }

    /// 単純な派生型の適用（Pointer と Array のみ）
    fn apply_simple_derived(&self, base: &str, derived: &[DerivedDecl]) -> String {
        self.apply_simple_derived_with_specs_const(base, derived, false)
    }

    /// 派生型を適用（specs の const 情報を考慮）
    fn apply_simple_derived_with_specs_const(&self, base: &str, derived: &[DerivedDecl], specs_is_const: bool) -> String {
        let mut result = base.to_string();
        let mut is_first_pointer = true;
        for d in derived.iter().rev() {
            match d {
                DerivedDecl::Pointer(quals) => {
                    // void ポインタの場合は c_void を使用
                    if result == "()" {
                        result = "c_void".to_string();
                    }
                    // C の "const T *p": specs.is_const が pointee const を表す
                    // C の "T * const p": quals.is_const はポインタ自体の const (Rust では *mut)
                    // ただし "const T * const p" は両方 const → *const
                    let pointee_const = if is_first_pointer {
                        specs_is_const
                    } else {
                        quals.is_const
                    };
                    if pointee_const {
                        result = format!("*const {}", result);
                    } else {
                        result = format!("*mut {}", result);
                    }
                    is_first_pointer = false;
                }
                DerivedDecl::Array(arr) => {
                    // void 配列の場合は c_void を使用
                    if result == "()" {
                        result = "c_void".to_string();
                    }
                    if let Some(ref size_expr) = arr.size {
                        // 定数サイズ配列
                        if let ExprKind::IntLit(n) = &size_expr.kind {
                            result = format!("[{}; {}]", result, n);
                        } else {
                            result = format!("*mut {}", result);
                        }
                    } else {
                        result = format!("*mut {}", result);
                    }
                }
                DerivedDecl::Function(_) => {
                    // この関数では Function は処理しない（apply_derived_to_type で処理）
                }
            }
        }
        result
    }

    /// ParamDecl から型のみを取得（名前なし）
    fn param_type_only(&mut self, param: &ParamDecl) -> String {
        let ty = self.decl_specs_to_rust(&param.specs);
        if let Some(ref declarator) = param.declarator {
            // specs の const 修飾子 (`const U8 *`) をポインタ const として
            // 反映する。apply_derived_to_type は常に specs_is_const=false で
            // 呼び出すため、ここで補正する。
            self.apply_simple_derived_with_specs_const(&ty, &declarator.derived, param.specs.qualifiers.is_const)
        } else {
            ty
        }
    }

    /// inline 関数を生成（self を消費）
    pub fn generate_inline_fn(mut self, name: crate::InternedStr, func_def: &FunctionDef) -> GeneratedCode {
        let name_str = self.interner.get(name);

        // Phase 2 (InlineFnDict::analyze_local_usage) で計算済みの
        // 名前使用解析を読む。テスト等で未設定の場合のみその場で計算する。
        // ネストした compound 内の宣言や Switch/Label も走査済みなので、
        // ループ内再代入の E0384 (GH #16) をここで解消する
        let usage = self.inline_fn_dict
            .and_then(|d| d.local_usage(name))
            .cloned()
            .unwrap_or_else(|| crate::local_usage::analyze_function(func_def));
        let mut_params = usage.mut_names();

        // mut ローカル変数名と未使用パラメータを保存
        // (decl_to_rust_let / param_decl_to_rust で使用)
        self.mut_local_names = mut_params.clone();
        self.unused_param_names = {
            let mut set = HashSet::new();
            for d in &func_def.declarator.derived {
                if let DerivedDecl::Function(param_list) = d {
                    for p in &param_list.params {
                        if let Some(param_name) = p.declarator.as_ref().and_then(|dd| dd.name) {
                            if usage.is_unused(param_name) {
                                set.insert(param_name);
                            }
                        }
                    }
                }
            }
            set
        };
        self.current_local_usage = Some(usage);

        // パラメータリストを取得
        let params_str = self.build_fn_param_list(&func_def.declarator.derived, &mut_params);

        // 戻り値の型を取得（基本型）
        let return_type = self.decl_specs_to_rust(&func_def.specs);

        // declarator の派生型（ポインタなど）を適用（Function を除く）
        // 例: HEK * func(...) の場合、derived = [Pointer, Function]
        //     戻り値型は HEK に Pointer を適用して *mut HEK になる
        let return_derived: Vec<_> = func_def.declarator.derived.iter()
            .filter(|d| !matches!(d, DerivedDecl::Function(_)))
            .cloned()
            .collect();
        let return_type = self.apply_simple_derived_with_specs_const(&return_type, &return_derived, func_def.specs.qualifiers.is_const);
        self.current_return_type = Some(UnifiedType::from_rust_str(&return_type));

        // パラメータの型情報を収集 + ローカルスコープに登録
        for d in &func_def.declarator.derived {
            if let DerivedDecl::Function(param_list) = d {
                for p in &param_list.params {
                    if let Some(ref declarator) = p.declarator {
                        if let Some(param_name) = declarator.name {
                            let ty = self.param_type_only(p);
                            self.current_param_types.insert(param_name, UnifiedType::from_rust_str(&ty));
                            self.current_local_names.insert(param_name);
                        }
                    }
                }
            }
        }

        // 本体のローカル変数宣言もスコープに追加（ネストした compound、
        // StmtExpr、for ループ初期化、すべての block レベルを再帰走査）
        self.collect_local_names_recursive(&func_def.body);

        // THX 依存性を判定（非 threaded では常に false）
        let is_thx_dependent = self.perl_threaded
            && self.is_inline_fn_thx_dependent(&func_def.declarator.derived);
        let thx_info = if is_thx_dependent { " [THX]" } else { "" };

        // AST ダンプ（デバッグ用）
        self.dump_ast_comment_for_body(name_str, &func_def.body);

        // ドキュメントコメント
        self.writeln(&format!("/// {}{} - inline function", name_str, thx_info));
        self.writeln("#[inline]");
        self.writeln("#[allow(unsafe_op_in_unsafe_fn)]");

        // 関数定義
        self.writeln(&format!("pub unsafe fn {}({}) -> {} {{", name_str, params_str, return_type));

        // unsafe 操作（関数呼び出し or デリファレンス）を含む場合のみ unsafe ブロックを生成
        let needs_unsafe = func_def.function_call_count > 0 || func_def.deref_count > 0;

        if needs_unsafe {
            self.writeln("    unsafe {");
            let body_str = self.compound_stmt_to_string(&func_def.body, "        ");
            self.buffer.push_str(&body_str);
            self.writeln("    }");
        } else {
            let body_str = self.compound_stmt_to_string(&func_def.body, "    ");
            self.buffer.push_str(&body_str);
        }

        self.writeln("}");
        self.writeln("");

        self.into_generated_code()
    }

    /// DerivedDecl から関数パラメータリストを構築
    fn build_fn_param_list(&mut self, derived: &[DerivedDecl], mut_params: &HashSet<InternedStr>) -> String {
        for d in derived {
            if let DerivedDecl::Function(param_list) = d {
                // C の `fn(void)` は「引数なし」の同義。単独の void パラメータは
                // Rust の 0 引数に変換する（そうしないと `fn(_: ())` となり呼出側
                // が 0 引数と食い違って E0061 を起こす）。
                if is_void_only_param_list(&param_list.params) {
                    return String::new();
                }
                let params: Vec<_> = param_list.params.iter()
                    .map(|p| self.param_decl_to_rust(p, mut_params))
                    .collect();
                let mut result = params.join(", ");
                if param_list.is_variadic {
                    if !result.is_empty() {
                        result.push_str(", ");
                    }
                    result.push_str("...");
                }
                return result;
            }
        }
        String::new()
    }

    /// inline 関数が THX 依存かどうかを判定
    ///
    /// 最初のパラメータが `my_perl` という名前であれば THX 依存とみなす。
    fn is_inline_fn_thx_dependent(&self, derived: &[DerivedDecl]) -> bool {
        for d in derived {
            if let DerivedDecl::Function(param_list) = d {
                if let Some(first_param) = param_list.params.first() {
                    if let Some(ref declarator) = first_param.declarator {
                        if let Some(name) = declarator.name {
                            let name_str = self.interner.get(name);
                            return name_str == "my_perl";
                        }
                    }
                }
                return false;
            }
        }
        false
    }

    /// ParamDecl を Rust パラメータ宣言に変換
    fn param_decl_to_rust(&mut self, param: &ParamDecl, mut_params: &HashSet<InternedStr>) -> String {
        let param_name_interned = param.declarator
            .as_ref()
            .and_then(|d| d.name);
        let name = match param_name_interned {
            // 未使用パラメータは `_` prefix (unused_variables 対策、GH #22)。
            // 本体に参照が無いことが保証されるので束縛名だけ変えれば良い
            Some(n) if self.unused_param_names.contains(&n) => {
                format!("_{}", self.interner.get(n))
            }
            Some(n) => escape_rust_keyword(self.interner.get(n)),
            None => "_".to_string(),
        };

        let ty = self.decl_specs_to_rust(&param.specs);

        // ポインタ派生型を適用（specs の const を考慮）
        let ty = if let Some(ref declarator) = param.declarator {
            self.apply_simple_derived_with_specs_const(&ty, &declarator.derived, param.specs.qualifiers.is_const)
        } else {
            ty
        };

        // current_param_types に登録（型推論で使用）
        if let Some(n) = param_name_interned {
            self.current_param_types.insert(n, UnifiedType::from_rust_str(&ty));
        }

        let mut_prefix = if param_name_interned.is_some_and(|n| mut_params.contains(&n)) {
            "mut "
        } else {
            ""
        };

        format!("{}{}: {}", mut_prefix, name, ty)
    }

    /// Declaration を Rust の let 宣言に変換
    fn decl_to_rust_let(&mut self, decl: &Declaration, indent: &str) -> String {
        let mut result = String::new();

        // 基本型を取得
        let base_type = self.decl_specs_to_rust(&decl.specs);

        // 各宣言子を処理
        for init_decl in &decl.declarators {
            let name = init_decl.declarator.name
                .map(|n| escape_rust_keyword(self.interner.get(n)))
                .unwrap_or_else(|| "_".to_string());

            // 派生型（ポインタなど）を適用
            let ty = self.apply_derived_to_type(&base_type, &init_decl.declarator.derived);

            // 初期化子
            if let Some(ref init) = init_decl.init {
                match init {
                    Initializer::Expr(expr) => {
                        // 宣言型と式の推論型が異なる整数型なら as キャストを syn レベルで挿入。
                        // 文字列ベースの `({} as {})` は normalize_parens に剥がされて
                        // `a as u32 & b as u8` の優先順位崩壊を起こすため、syn::Expr で
                        // 構築して expr_to_string に括弧の挿入を任せる。
                        let mut init_syn = self.build_syn_expr(expr, None);
                        if let Some(expr_ut) = self.infer_expr_type_inline(expr) {
                            let decl_s = ty.clone();
                            let expr_s = expr_ut.to_rust_string();
                            let nd = normalize_integer_type(&decl_s);
                            let ne = normalize_integer_type(&expr_s);
                            if let (Some(d), Some(e)) = (nd, ne) {
                                // Rust never converts between isize/i64 or usize/u64 implicitly,
                                // so any difference needs a cast.
                                if d != e {
                                    init_syn = crate::syn_codegen::cast_syn_expr(init_syn, d);
                                }
                            }
                        }
                        // ポインタの const/mut 不一致は init 側に明示キャストを挿入する。
                        // 旧実装は宣言型を *const に書換えていたが、それだと以降の
                        // 代入で *mut 型が期待されるマクロ引数などで破綻する。
                        // 例: `let s: *mut U8 = s0;` (s0: *const U8) → `s0 as *mut U8`
                        let decl_is_mut_ptr = ty.contains("*mut ") && !ty.contains("*const ");
                        let decl_is_const_ptr = ty.contains("*const ") && !ty.contains("*mut ");
                        if let Some(expr_ut) = self.infer_expr_type_inline(expr) {
                            if expr_ut.is_pointer() {
                                if decl_is_mut_ptr && expr_ut.is_const_pointer() {
                                    init_syn = crate::syn_codegen::cast_syn_expr(init_syn, &ty);
                                } else if decl_is_const_ptr && !expr_ut.is_const_pointer() {
                                    init_syn = crate::syn_codegen::cast_syn_expr(init_syn, &ty);
                                }
                            }
                        }
                        let init_expr = normalize_parens(&crate::syn_codegen::expr_to_string(&init_syn));
                        // null リテラルの場合は型推論に任せて std::ptr::null_mut()
                        let init_expr = if is_null_literal(expr) && ty.contains("*mut") {
                            "std::ptr::null_mut()".to_string()
                        } else if is_null_literal(expr) && ty.contains("*const") {
                            "std::ptr::null()".to_string()
                        } else {
                            init_expr
                        };
                        let mut_kw = if init_decl.declarator.name.is_some_and(|n| self.mut_local_names.contains(&n)) { "mut " } else { "" };
                        result.push_str(&format!("{}let {}{}: {} = {};\n", indent, mut_kw, name, ty, strip_outer_parens(&init_expr)));
                    }
                    Initializer::List(_) => {
                        // 初期化リストは複雑なので TODO
                        result.push_str(&format!("{}let {}: {} = /* init list */;\n", indent, name, ty));
                    }
                }
            } else {
                // 初期化子なし。Rust の deferred initialization (`let x: T;`)
                // で「宣言 → 後で代入」は合法だが、最初の代入より前に
                // アドレスを取られる out-param パターンは E0381 になるため
                // zeroed 初期化に切り替える (GH #23)
                let mut_kw = if init_decl.declarator.name.is_some_and(|n| self.mut_local_names.contains(&n)) { "mut " } else { "" };
                let needs_zeroed = init_decl.declarator.name.is_some_and(|n| {
                    self.current_local_usage.as_ref().is_some_and(|u| u.needs_zeroed_init(n))
                });
                if needs_zeroed {
                    result.push_str(&format!(
                        "{}let {}{}: {} = std::mem::zeroed(); // C: uninitialized out-param\n",
                        indent, mut_kw, name, ty));
                } else {
                    result.push_str(&format!("{}let {}{}: {}; // uninitialized\n", indent, mut_kw, name, ty));
                }
            }
        }

        result
    }

    /// 複合文を文字列として生成
    fn compound_stmt_to_string(&mut self, stmt: &CompoundStmt, indent: &str) -> String {
        let mut result = String::new();
        for item in &stmt.items {
            match item {
                BlockItem::Decl(decl) => {
                    self.collect_decl_types(decl);
                    result.push_str(&self.decl_to_rust_let(decl, indent));
                }
                BlockItem::Stmt(s) => {
                    let rust_stmt = self.stmt_to_rust_inline(s, indent);
                    if !rust_stmt.is_empty() {
                        result.push_str(&rust_stmt);
                        result.push('\n');
                    }
                }
            }
        }
        result
    }

    /// 文を Rust コードに変換（インライン関数用）
    fn stmt_to_rust_inline(&mut self, stmt: &Stmt, indent: &str) -> String {
        match stmt {
            Stmt::Expr(Some(expr), _) => {
                // 代入式・増減式は値を返さない形式で出力（統一ヘルパー使用）
                self.expr_stmt_string(expr, None, indent)
            }
            Stmt::Expr(None, _) => String::new(),
            Stmt::Return(Some(expr), _) => self.build_return_stmt(expr, indent, None),
            Stmt::Return(None, _) => format!("{}return;", indent),
            Stmt::If { cond, then_stmt, else_stmt, .. } => {
                let cond_str = self.build_expr_string(cond, None);
                // 条件が既に bool なら != 0 を追加しない
                let cond_bool = self.wrap_as_bool_condition_inline(cond, &cond_str);
                let mut result = format!("{}if {} {{\n", indent, normalize_parens(&cond_bool));
                let nested_indent = format!("{}    ", indent);
                result.push_str(&self.stmt_to_rust_inline(then_stmt, &nested_indent));
                result.push_str("\n");
                result.push_str(&format!("{}}}", indent));
                if let Some(else_stmt) = else_stmt {
                    result.push_str(" else {\n");
                    result.push_str(&self.stmt_to_rust_inline(else_stmt, &nested_indent));
                    result.push_str("\n");
                    result.push_str(&format!("{}}}", indent));
                }
                result
            }
            Stmt::Compound(compound) => {
                let mut result = format!("{}{{\n", indent);
                for item in &compound.items {
                    match item {
                        BlockItem::Stmt(s) => {
                            let nested_indent = format!("{}    ", indent);
                            result.push_str(&self.stmt_to_rust_inline(s, &nested_indent));
                            result.push_str("\n");
                        }
                        BlockItem::Decl(decl) => {
                            self.collect_decl_types(decl);
                            let nested_indent = format!("{}    ", indent);
                            result.push_str(&self.decl_to_rust_let(decl, &nested_indent));
                        }
                    }
                }
                result.push_str(&format!("{}}}", indent));
                result
            }
            Stmt::While { cond, body, .. } => {
                let cond_str = self.build_expr_string(cond, None);
                // 条件が既に bool なら != 0 を追加しない。
                // if 側 (:5822) と同様 normalize_parens で外側括弧を除去しないと
                // `while (cond)` が unused_parens になる
                let cond_bool = self.wrap_as_bool_condition_inline(cond, &cond_str);
                let mut result = format!("{}while {} {{\n", indent, normalize_parens(&cond_bool));
                let nested_indent = format!("{}    ", indent);
                result.push_str(&self.stmt_to_rust_inline(body, &nested_indent));
                result.push_str("\n");
                result.push_str(&format!("{}}}", indent));
                result
            }
            Stmt::For { init, cond, step, body, .. } => {
                let mut result = format!("{}{{\n", indent);
                let nested_indent = format!("{}    ", indent);

                // 初期化部分
                if let Some(for_init) = init {
                    match for_init {
                        ForInit::Expr(expr) => {
                            result.push_str(&self.expr_stmt_string(expr, None, &nested_indent));
                            result.push('\n');
                        }
                        ForInit::Decl(decl) => {
                            self.collect_decl_types(decl);
                            result.push_str(&self.decl_to_rust_let(decl, &nested_indent));
                        }
                    }
                }

                // ループ部分
                if let Some(cond_expr) = cond {
                    let cond_str = self.build_expr_string(cond_expr, None);
                    // 条件が既に bool なら != 0 を追加しない (外側括弧除去は While と同様)
                    let cond_bool = self.wrap_as_bool_condition_inline(cond_expr, &cond_str);
                    result.push_str(&format!("{}while {} {{\n", nested_indent, normalize_parens(&cond_bool)));
                } else {
                    result.push_str(&format!("{}loop {{\n", nested_indent));
                }

                let body_indent = format!("{}    ", nested_indent);

                // ループ本体
                result.push_str(&self.stmt_to_rust_inline(body, &body_indent));
                result.push_str("\n");

                // ステップ部分 (i++ 等は値を返さない文形式で)
                if let Some(step_expr) = step {
                    result.push_str(&self.expr_stmt_string(step_expr, None, &body_indent));
                    result.push('\n');
                }

                result.push_str(&format!("{}}}\n", nested_indent));
                result.push_str(&format!("{}}}", indent));
                result
            }
            Stmt::DoWhile { body, cond, .. } => {
                // do { ... } while (0) パターンは一度だけ実行される loop として出力
                // これにより内部の break; が正しく動作する
                if is_zero_constant(cond) {
                    // 内部 body に `break;` が含まれなければ単なる block で十分。
                    // STMT_START/STMT_END 由来なら break なしのケースがほとんど。
                    // その方が `loop {...break;}` の `()`-型による tail 型不一致
                    // (fn が usize 等を返す場合の E0308) を避けられる。
                    if !stmt_contains_top_level_break(body) {
                        let mut result = format!("{}{{\n", indent);
                        let nested_indent = format!("{}    ", indent);
                        result.push_str(&self.stmt_to_rust_inline(body, &nested_indent));
                        result.push_str("\n");
                        result.push_str(&format!("{}}}", indent));
                        return result;
                    }
                    let mut result = format!("{}loop {{\n", indent);
                    let nested_indent = format!("{}    ", indent);
                    result.push_str(&self.stmt_to_rust_inline(body, &nested_indent));
                    result.push_str("\n");
                    result.push_str(&format!("{}    break;\n", indent));
                    result.push_str(&format!("{}}}", indent));
                    return result;
                }

                // 一般的な do-while 文: loop { body; if !cond { break; } }
                let mut result = format!("{}loop {{\n", indent);
                let nested_indent = format!("{}    ", indent);
                result.push_str(&self.stmt_to_rust_inline(body, &nested_indent));
                result.push_str("\n");
                let cond_str = self.build_expr_string(cond, None);
                // bool 式なら !(cond)、そうでなければ cond == 0
                // `!{}` 直付けだと `!a < b` のように優先順位で Lt が bind され、
                // `cannot apply ! to pointer` 等の誤生成になる。
                let break_cond = if is_boolean_expr(cond) {
                    normalize_parens(&format!("!({})", cond_str))
                } else {
                    format!("{} == 0", cond_str)
                };
                result.push_str(&format!("{}    if {} {{ break; }}\n", indent, break_cond));
                result.push_str(&format!("{}}}", indent));
                result
            }
            Stmt::Switch { expr, body, .. } => {
                let expr_str = self.build_expr_string(expr, None);
                let mut result = format!("{}match {} {{\n", indent, expr_str);
                let nested_indent = format!("{}    ", indent);

                // body から Case/Default を収集
                self.collect_switch_cases(body, &nested_indent, &mut result);

                result.push_str(&format!("{}}}", indent));
                result
            }
            Stmt::Case { expr: case_expr, stmt: case_stmt, .. } => {
                // Switch 外で Case が出現した場合（通常は Switch 内で処理される）
                let case_val = self.build_expr_string(case_expr, None);
                let mut result = format!("{}{} => {{\n", indent, case_val);
                let body_indent = format!("{}    ", indent);
                result.push_str(&self.stmt_to_rust_inline(case_stmt, &body_indent));
                result.push_str("\n");
                result.push_str(&format!("{}}}", indent));
                result
            }
            Stmt::Default { stmt: default_stmt, .. } => {
                let mut result = format!("{}_ => {{\n", indent);
                let body_indent = format!("{}    ", indent);
                result.push_str(&self.stmt_to_rust_inline(default_stmt, &body_indent));
                result.push_str("\n");
                result.push_str(&format!("{}}}", indent));
                result
            }
            Stmt::Goto(label, _) => {
                let label_str = self.interner.get(*label);
                format!("{}break '{}; // goto", indent, label_str)
            }
            Stmt::Label { name, stmt: label_stmt, .. } => {
                let label_str = self.interner.get(*name);
                let mut result = format!("{}'{}: {{\n", indent, label_str);
                let nested_indent = format!("{}    ", indent);
                result.push_str(&self.stmt_to_rust_inline(label_stmt, &nested_indent));
                result.push_str("\n");
                result.push_str(&format!("{}}}", indent));
                result
            }
            Stmt::Break(_) => format!("{}break;", indent),
            Stmt::Continue(_) => format!("{}continue;", indent),
            _ => self.todo_marker(&format!("{:?}", std::mem::discriminant(stmt)))
        }
    }

    /// Switch 文の body から Case/Default を収集して match アームを生成
    fn collect_switch_cases(&mut self, stmt: &Stmt, indent: &str, result: &mut String) {
        // パス1: case/default とそれに続く文を収集
        struct SwitchCase {
            patterns: Vec<String>,   // case 式のリスト（複数 case ラベル対応）、空 = default
            body_stmts: Vec<String>,
            is_default: bool,
        }

        let mut cases: Vec<SwitchCase> = Vec::new();
        let body_indent = format!("{}    ", indent);

        // Compound の中身をフラット化して処理
        fn collect_items<'a>(stmt: &'a Stmt, items: &mut Vec<&'a BlockItem>) {
            if let Stmt::Compound(compound) = stmt {
                for item in &compound.items {
                    items.push(item);
                }
            }
        }

        // ネストされた case/default を展開して patterns と最終的な stmt を取得
        fn flatten_case_chain<'a>(stmt: &'a Stmt, patterns: &mut Vec<&'a Expr>) -> (&'a Stmt, bool) {
            match stmt {
                Stmt::Case { expr, stmt: inner_stmt, .. } => {
                    patterns.push(expr);
                    flatten_case_chain(inner_stmt, patterns)
                }
                Stmt::Default { stmt: inner_stmt, .. } => {
                    // default に到達
                    (inner_stmt, true)
                }
                other => (other, false)
            }
        }

        let mut items: Vec<&BlockItem> = Vec::new();
        collect_items(stmt, &mut items);

        for item in items {
            match item {
                BlockItem::Stmt(s) => {
                    match s {
                        Stmt::Case { expr: case_expr, stmt: case_stmt, .. } => {
                            // ネストされた case をフラット化
                            let mut patterns: Vec<&Expr> = vec![case_expr];
                            let (final_stmt, has_default) = flatten_case_chain(case_stmt, &mut patterns);

                            // パターンは enum バリアントをフルパスで出力
                            let pattern_strs: Vec<String> = patterns.iter()
                                .map(|e| self.expr_to_rust_pattern(e))
                                .collect();

                            // final_stmt が Break の場合はスキップ（Rust の match は break 不要）
                            let body_stmts = if matches!(final_stmt, Stmt::Break(_)) {
                                vec![]
                            } else {
                                vec![self.stmt_to_rust_inline(final_stmt, &body_indent)]
                            };
                            cases.push(SwitchCase {
                                patterns: pattern_strs,
                                body_stmts,
                                is_default: has_default,
                            });
                        }
                        Stmt::Default { stmt: default_stmt, .. } => {
                            // ネストされた case をフラット化
                            let mut patterns: Vec<&Expr> = Vec::new();
                            let (final_stmt, _) = flatten_case_chain(default_stmt, &mut patterns);

                            // パターンは enum バリアントをフルパスで出力
                            let pattern_strs: Vec<String> = patterns.iter()
                                .map(|e| self.expr_to_rust_pattern(e))
                                .collect();

                            // final_stmt が Break の場合はスキップ（Rust の match は break 不要）
                            let body_stmts = if matches!(final_stmt, Stmt::Break(_)) {
                                vec![]
                            } else {
                                vec![self.stmt_to_rust_inline(final_stmt, &body_indent)]
                            };
                            cases.push(SwitchCase {
                                patterns: pattern_strs,
                                body_stmts,
                                is_default: true,
                            });
                        }
                        Stmt::Break(_) => {
                            // Rust の match はフォールスルーしないので break は不要
                            // スキップする
                        }
                        other => {
                            // 直前の case に追加
                            if let Some(last) = cases.last_mut() {
                                last.body_stmts.push(self.stmt_to_rust_inline(other, &body_indent));
                            }
                            // case がまだない場合は無視
                        }
                    }
                }
                BlockItem::Decl(decl) => {
                    self.collect_decl_types(decl);
                    // 宣言は直前の case に追加
                    if let Some(last) = cases.last_mut() {
                        last.body_stmts.push(self.decl_to_rust_let(decl, &body_indent));
                    }
                }
            }
        }

        // パス2: 収集した cases から match アームを生成
        let has_default = cases.iter().any(|c| c.is_default);
        for case in &cases {
            let pattern = if case.is_default {
                if case.patterns.is_empty() {
                    "_".to_string()
                } else {
                    // case A: case B: default: ... のパターン
                    format!("{} | _", case.patterns.join(" | "))
                }
            } else {
                case.patterns.join(" | ")
            };

            result.push_str(&format!("{}{} => {{\n", indent, pattern));
            for stmt in &case.body_stmts {
                result.push_str(stmt);
                result.push_str("\n");
            }
            result.push_str(&format!("{}}}\n", indent));
        }
        // C の switch は default が無ければ該当しない値を素通しする。
        // Rust match は網羅性が必要なため、default 無しのケースで
        // `_ => {}` を自動挿入する。
        if !has_default {
            result.push_str(&format!("{}_ => {{}}\n", indent));
        }
    }

    /// offsetof のフィールドパス式をドット区切り文字列に変換
    /// Ident("xnv_u") → "xnv_u"
    /// Member(Ident("xnv_u"), "xnv_nv") → "xnv_u.xnv_nv"
    fn expr_to_field_path(&self, expr: &Expr) -> Option<String> {
        match &expr.kind {
            ExprKind::Ident(name) => {
                Some(self.interner.get(*name).to_string())
            }
            ExprKind::Member { expr: base, member } => {
                let base_path = self.expr_to_field_path(base)?;
                let member_name = self.interner.get(*member);
                Some(format!("{}.{}", base_path, member_name))
            }
            _ => None,
        }
    }


    /// match パターン用の式を Rust に変換
    ///
    /// 通常の式変換と異なり、enum バリアントをフルパスで出力する。
    /// Rust の match パターンでは、単純な識別子は変数束縛として扱われるため、
    /// enum バリアントは `crate::EnumName::VariantName` 形式で出力する必要がある。
    fn expr_to_rust_pattern(&mut self, expr: &Expr) -> String {
        match &expr.kind {
            ExprKind::Ident(name) => {
                // enum バリアントかチェック
                if let Some(enum_name) = self.enum_dict.get_enum_for_variant(*name) {
                    let enum_str = self.interner.get(enum_name);
                    let variant_str = self.interner.get(*name);
                    format!("crate::{}::{}", enum_str, variant_str)
                } else {
                    escape_rust_keyword(self.interner.get(*name))
                }
            }
            // 他の式は通常の変換
            _ => self.build_expr_string(expr, None)
        }
    }
}

impl<'a, W: Write> CodegenDriver<'a, W> {
    /// 新しいコード生成ドライバを作成
    pub fn new(
        writer: W,
        interner: &'a StringInterner,
        enum_dict: &'a EnumDict,
        macro_ctx: &'a MacroInferContext,
        bindings_info: BindingsInfo,
        config: CodegenConfig,
    ) -> Self {
        Self {
            writer,
            interner,
            enum_dict,
            macro_ctx,
            bindings_info,
            config,
            stats: CodegenStats::default(),
            report: CodegenReport::default(),
            used_libc_fns: HashSet::new(),
            successfully_generated_inlines: HashSet::new(),
            generatable_macros: HashSet::new(),
            const_pointer_params: HashMap::new(),
            bool_return_macros: HashSet::new(),
            // デフォルトは threaded（後方互換）。`generate()` で
            // result.perl_build_mode から書き換える。
            perl_threaded: true,
        }
    }

    /// 対象 perl が threaded かを設定
    pub fn with_perl_threaded(mut self, threaded: bool) -> Self {
        self.perl_threaded = threaded;
        self
    }

    /// 統計情報を取得
    pub fn stats(&self) -> &CodegenStats {
        &self.stats
    }

    /// 関数別の生成結果を取得 (generate() 完了後に有効)
    pub fn report(&self) -> &CodegenReport {
        &self.report
    }

    /// 全体を生成
    // デバッグ用: ビルド時のタイムスタンプを埋め込む場合はコメントを外す
    // const BUILD_TIMESTAMP: &'static str = "2025-01-24T17:50:00+09:00";

    pub fn generate(&mut self, result: &InferResult) -> io::Result<()> {
        // 対象 perl の threaded フラグを反映（is_thx_dependent 経路の防御的ガード）
        self.perl_threaded = result.perl_build_mode.is_threaded();

        // 自動生成 struct/typedef を先に決定（known_symbols 構築前）
        // 実際に出力される名前のみ known_symbols に登録するため、emit を先行実施。
        let missing_structs = crate::struct_emitter::emit_missing_structs(
            &result.fields_dict,
            result.rust_decl_dict.as_ref(),
            self.interner,
        );
        let static_arrays = crate::static_array_emitter::emit_static_arrays(
            &result.global_const_dict,
            &result.fields_dict,
            result.rust_decl_dict.as_ref(),
            self.interner,
        );
        // 自動生成した static 配列名を bindings_info.static_arrays / static_types に登録。
        // これにより codegen が `.as_ptr()` 減衰を掛けるべき配列として認識し、
        // 要素型 T も `(&raw const NAME) as *const T` で正しく出せる。
        for n in &static_arrays.emitted_names {
            self.bindings_info.static_arrays.insert(n.clone());
        }
        for (n, t) in &static_arrays.emitted_types {
            self.bindings_info.static_types.insert(n.clone(), t.clone());
        }

        // 既知シンボル集合を構築（未解決シンボル検出用）
        let mut known_symbols = KnownSymbols::new(result, self.interner);
        for n in &missing_structs.emitted_struct_names {
            known_symbols.insert(n.clone());
        }
        for n in &missing_structs.emitted_typedef_names {
            known_symbols.insert(n.clone());
        }

        // 自動生成 struct の bit-field アクセサを bindings_info にマージ。
        // これにより `is_bitfield_method` がそれらの getter 名を認識し、
        // `.name` → `.name()` / `.name = val` → `.set_name(val)` の書換が効く。
        for (struct_name, methods) in &missing_structs.bitfield_methods {
            self.bindings_info
                .bitfield_methods
                .entry(struct_name.clone())
                .or_default()
                .extend(methods.iter().cloned());
        }
        // global_const_dict 由来の static 配列は既に KnownSymbols::new 内で
        // 登録済みなので追加不要

        // ヘッダーコメント
        writeln!(self.writer, "// Auto-generated Rust bindings")?;
        // デバッグ用: タイムスタンプを出力する場合はコメントを外す
        // writeln!(self.writer, "// Generated by libperl-macrogen (build: {})", Self::BUILD_TIMESTAMP)?;
        writeln!(self.writer, "// Generated by libperl-macrogen")?;
        writeln!(self.writer)?;

        // use 文を出力
        self.generate_use_statements()?;

        // target enum のバリアントを import
        self.generate_enum_imports(result)?;

        // 自動生成 struct/union 定義を出力
        if !missing_structs.source.is_empty() {
            self.writer.write_all(missing_structs.source.as_bytes())?;
        }

        // 自動生成 static const 配列を出力（事前算出済み）
        if !static_arrays.source.is_empty() {
            self.writer.write_all(static_arrays.source.as_bytes())?;
        }

        // マクロの生成可能性を事前計算（inline→macro カスケード検出用）
        self.precompute_macro_generability(result, &known_symbols);

        // Phase 2 の解析結果を収集（inline/macro 両方で使用）
        for (&name, info) in &result.infer_ctx.macros {
            if info.is_bool_return {
                self.bool_return_macros.insert(name);
            }
        }

        // inline 関数セクション
        if self.config.emit_inline_fns {
            self.generate_inline_fns(result, &known_symbols)?;
        }

        // マクロセクション
        if self.config.emit_macros {
            self.generate_macros(result, &known_symbols)?;
        }

        // 使用された libc 関数の use 文を出力（rustfmt が先頭に移動する）
        if !self.used_libc_fns.is_empty() {
            let mut fns: Vec<_> = self.used_libc_fns.iter().cloned().collect();
            fns.sort();
            writeln!(self.writer, "use libc::{{{}}};", fns.join(", "))?;
        }

        Ok(())
    }

    /// use 文を生成
    fn generate_use_statements(&mut self) -> io::Result<()> {
        let statements = if self.config.use_statements.is_empty() {
            CodegenConfig::default_use_statements()
        } else {
            self.config.use_statements.clone()
        };

        if !statements.is_empty() {
            for stmt in &statements {
                writeln!(self.writer, "{};", stmt)?;
            }
            writeln!(self.writer)?;
        }

        Ok(())
    }

    /// target enum のバリアントを import
    ///
    /// bindings.rs に存在する enum のみ import する
    fn generate_enum_imports(&mut self, result: &InferResult) -> io::Result<()> {
        let enum_names = result.enum_dict.target_enum_names(self.interner);
        let bindings_enums = result.rust_decl_dict.as_ref().map(|d| &d.enums);

        // bindings.rs に存在する enum のみフィルタリング
        let filtered_names: Vec<_> = enum_names
            .into_iter()
            .filter(|name| {
                bindings_enums.map_or(true, |enums| enums.contains(*name))
            })
            .collect();

        if !filtered_names.is_empty() {
            writeln!(self.writer, "// Enum variant imports")?;
            for name in filtered_names {
                writeln!(self.writer, "#[allow(unused_imports)]")?;
                writeln!(self.writer, "use crate::{}::*;", name)?;
            }
            writeln!(self.writer)?;
        }

        Ok(())
    }

    /// マクロの生成可能性を事前計算
    ///
    /// inline 関数生成前に呼び出し、マクロ→マクロのカスケード検査をシミュレートして
    /// 正常に生成可能なマクロの集合を構築する。
    /// inline→macro のカスケード検出に使用する。
    fn precompute_macro_generability(&mut self, result: &InferResult, known_symbols: &KnownSymbols) {
        // 対象マクロを収集
        let macros: Vec<_> = result.infer_ctx.macros.iter()
            .filter(|(_, info)| self.should_include_macro(info))
            .collect();
        let included_set: HashSet<InternedStr> = macros.iter().map(|(n, _)| **n).collect();

        // 依存順にソート（葉マクロ先頭）
        let sorted_names = self.topological_sort_macros(&macros);

        for name in sorted_names {
            let info = result.infer_ctx.macros.get(&name).unwrap();

            // apidoc_suppressed なマクロは generatable_macros に入れない
            // （Phase 3 の早期 SUPPRESSED 分岐で抑止される。inline→macro の
            //   cascade 検査が `generatable_macros` を参照するので、ここで
            //   除外しないと skip_codegen 対象を呼ぶ inline が誤って success
            //   扱いになる抜け穴ができる）
            if info.apidoc_suppressed {
                continue;
            }

            // カスケード検査: called_functions が生成不可なマクロを含むか
            // 被呼び出し先の判定は is_unavailable_for_codegen() を使う
            // （= apidoc_suppressed 由来の不可も Phase 2 propagation で
            //   calls_unavailable に流れているはずだが、防御的に OR を取る）
            let has_cascade_failure = info.called_functions.iter().any(|called| {
                if included_set.contains(called) {
                    return result.infer_ctx.macros.get(called)
                        .map(|u| u.is_parseable() && !u.is_unavailable_for_codegen())
                        .unwrap_or(false)
                        && !self.generatable_macros.contains(called);
                }
                false
            });
            if has_cascade_failure {
                continue;
            }

            // get_macro_status + trial codegen による判定
            let status = self.get_macro_status(info);
            if status == GenerateStatus::Success {
                // 実際に codegen を試行して完全性を確認
                let codegen = RustCodegen::new(
                    self.interner, self.enum_dict, self.macro_ctx,
                    self.bindings_info.clone(), &known_symbols,
                    result.rust_decl_dict.as_ref(), Some(&result.inline_fn_dict),
                ).with_perl_threaded(self.perl_threaded);
                let generated = codegen.generate_macro(info);
                if generated.is_complete() && !generated.has_unresolved_names() {
                    self.generatable_macros.insert(name);
                }
            }
        }
    }

    /// inline 関数セクションを生成
    ///
    /// 2パス方式:
    /// - Pass 1: 各 inline 関数を生成し、結果を蓄積
    /// - Pass 2: カスケード検査 — 生成成功した関数が失敗した inline 関数や
    ///   マクロを呼び出している場合、CASCADE_UNAVAILABLE に降格
    pub fn generate_inline_fns(&mut self, result: &InferResult, known_symbols: &KnownSymbols) -> io::Result<()> {
        writeln!(self.writer, "// =============================================================================")?;
        writeln!(self.writer, "// Inline Functions")?;
        writeln!(self.writer, "// =============================================================================")?;
        writeln!(self.writer)?;

        // 名前順にソート
        let mut fns: Vec<_> = result.inline_fn_dict.iter()
            .filter(|(_, func_def)| func_def.is_target)
            .collect();
        fns.sort_by_key(|(name, _)| self.interner.get(**name));

        // 対象 inline 関数名の集合
        let inline_set: HashSet<InternedStr> = fns.iter().map(|(n, _)| **n).collect();

        // Pass 1: 各 inline 関数を生成
        enum InlineGenResult {
            CallsUnavailable,
            ContainsGoto,
            UnresolvedNames { code: String, unresolved: Vec<String> },
            CodegenError { code: String, errors: Vec<String> },
            Incomplete { code: String },
            Success { code: String, used_libc: HashSet<String> },
            Suppressed { reason: String },
        }

        let mut gen_results: Vec<(InternedStr, InlineGenResult)> = Vec::new();

        for (name, func_def) in &fns {
            // apidoc skip_codegen 対象（Phase 2 で apidoc_suppressed 済）は
            // 早期に Suppressed。reason 文字列は apidoc_patches から取得。
            if result.inline_fn_dict.is_apidoc_suppressed(**name) {
                let n_str = self.interner.get(**name);
                let reason = result.apidoc_patches.skip_reason(n_str)
                    .unwrap_or("apidoc skip_codegen")
                    .to_string();
                gen_results.push((**name,
                    InlineGenResult::Suppressed { reason }));
                continue;
            }

            // 事前に unavailable と判定された関数はスキップ
            if result.inline_fn_dict.is_calls_unavailable(**name) {
                gen_results.push((**name, InlineGenResult::CallsUnavailable));
                continue;
            }

            if block_items_contain_goto(&func_def.body.items) {
                gen_results.push((**name, InlineGenResult::ContainsGoto));
                continue;
            }

            let codegen = RustCodegen::new(self.interner, self.enum_dict, self.macro_ctx, self.bindings_info.clone(), known_symbols, result.rust_decl_dict.as_ref(), Some(&result.inline_fn_dict))
                .with_perl_threaded(self.perl_threaded)
                .with_dump_ast_for(self.config.dump_ast_for.clone())
                        .with_dump_types_for(self.config.dump_types_for.clone())
                .with_fields_dict(&result.fields_dict)
                .with_bool_return(false, self.bool_return_macros.clone());
            let generated = codegen.generate_inline_fn(**name, func_def);

            if generated.has_unresolved_names() {
                gen_results.push((**name, InlineGenResult::UnresolvedNames {
                    code: generated.code,
                    unresolved: generated.unresolved_names,
                }));
            } else if !generated.codegen_errors.is_empty() {
                gen_results.push((**name, InlineGenResult::CodegenError {
                    code: generated.code,
                    errors: generated.codegen_errors,
                }));
            } else if generated.is_complete() {
                gen_results.push((**name, InlineGenResult::Success {
                    code: generated.code,
                    used_libc: generated.used_libc_fns,
                }));
            } else {
                gen_results.push((**name, InlineGenResult::Incomplete {
                    code: generated.code,
                }));
            }
        }

        // Pass 1.5: successfully_generated_inlines を構築（Success のみ）
        for (name, gen_result) in &gen_results {
            if matches!(gen_result, InlineGenResult::Success { .. }) {
                self.successfully_generated_inlines.insert(*name);
            }
        }

        // Pass 2: カスケード検査 — Success だが失敗した依存先を呼び出す場合は降格
        // InlineFnDict の called_functions を使用（ad-hoc 収集不要）
        // inline→inline と inline→macro の両方を検査
        // 繰り返し伝播（fixpoint）
        let mut changed = true;
        while changed {
            changed = false;
            let current_success = self.successfully_generated_inlines.clone();
            for (name, _) in &gen_results {
                if !current_success.contains(name) {
                    continue;
                }
                if let Some(calls) = result.inline_fn_dict.get_called_functions(*name) {
                    let has_unavailable = calls.iter().any(|called| {
                        // inline→inline: 対象 inline が生成失敗
                        if inline_set.contains(called) && !current_success.contains(called) {
                            return true;
                        }
                        // inline→macro: 対象マクロが生成不可
                        if let Some(macro_info) = result.infer_ctx.macros.get(called) {
                            if macro_info.is_target && self.should_include_macro(macro_info) {
                                if !self.generatable_macros.contains(called) {
                                    return true;
                                }
                            }
                        }
                        false
                    });
                    if has_unavailable {
                        self.successfully_generated_inlines.remove(name);
                        changed = true;
                    }
                }
            }
        }

        // Pass 3: 出力
        for (name, gen_result) in gen_results {
            match gen_result {
                InlineGenResult::CallsUnavailable => {
                    let name_str = self.interner.get(name);
                    let calls = result.inline_fn_dict.get_called_functions(name);
                    // 原因を分類: 純粋に存在しない関数を呼ぶ (absent) と
                    // 推移的に unavailable な依存先を呼ぶ (cascade) を区別する。
                    let absent: Vec<String> = calls
                        .map(|cs| cs.iter()
                            .filter(|c| {
                                let fn_name = self.interner.get(**c);
                                !self.is_function_available(**c, fn_name, result)
                            })
                            .map(|c| self.interner.get(*c).to_string())
                            .collect())
                        .unwrap_or_default();
                    let cascade_deps: Vec<String> = calls
                        .map(|cs| cs.iter()
                            .filter(|c| {
                                let is_unavailable_inline = result.inline_fn_dict.get(**c).is_some()
                                    && result.inline_fn_dict.is_unavailable_for_codegen(**c);
                                let is_unavailable_macro = result.infer_ctx.macros.get(c)
                                    .map(|info| info.is_unavailable_for_codegen())
                                    .unwrap_or(false);
                                is_unavailable_inline || is_unavailable_macro
                            })
                            .map(|c| self.interner.get(*c).to_string())
                            .collect())
                        .unwrap_or_default();
                    // HashSet 由来の列挙順を決定化 (run ごとのコメント揺れ防止)
                    let mut cascade_deps = cascade_deps;
                    cascade_deps.sort();
                    let mut absent = absent;
                    absent.sort();
                    if absent.is_empty() {
                        writeln!(self.writer,
                            "// [CASCADE_UNAVAILABLE] {} - dependency not generated: {}",
                            name_str, cascade_deps.join(", "))?;
                        self.report.record_skip(name_str, format!(
                            "CASCADE_UNAVAILABLE: dependency not generated: {}",
                            cascade_deps.join(", ")));
                    } else {
                        writeln!(self.writer,
                            "// [CALLS_UNAVAILABLE] {} - calls unavailable function(s): {}",
                            name_str, absent.join(", "))?;
                        self.report.record_skip(name_str, format!(
                            "CALLS_UNAVAILABLE: calls unavailable function(s): {}",
                            absent.join(", ")));
                    }
                    writeln!(self.writer)?;
                    self.stats.inline_fns_cascade_unavailable += 1;
                }
                InlineGenResult::ContainsGoto => {
                    let name_str = self.interner.get(name);
                    writeln!(self.writer, "// [CONTAINS_GOTO] {} - excluded (contains goto)", name_str)?;
                    writeln!(self.writer)?;
                    self.report.record_skip(name_str, "CONTAINS_GOTO: excluded (contains goto)");
                    self.stats.inline_fns_contains_goto += 1;
                }
                InlineGenResult::Suppressed { reason } => {
                    let name_str = self.interner.get(name);
                    writeln!(self.writer,
                        "// [CODEGEN_SUPPRESSED] {} - inline function (apidoc patch)",
                        name_str)?;
                    writeln!(self.writer, "// Reason: {}", reason)?;
                    writeln!(self.writer)?;
                    self.report.record_skip(name_str, format!(
                        "CODEGEN_SUPPRESSED (apidoc patch): {}", reason));
                }
                InlineGenResult::UnresolvedNames { code, unresolved } => {
                    let name_str = self.interner.get(name);
                    let mut unresolved = unresolved.clone();
                    unresolved.sort(); // 列挙順の決定化
                    writeln!(self.writer, "// [UNRESOLVED_NAMES] {} - inline function", name_str)?;
                    writeln!(self.writer, "// Unresolved: {}", unresolved.join(", "))?;
                    for line in code.lines() {
                        writeln!(self.writer, "// {}", line)?;
                    }
                    writeln!(self.writer)?;
                    self.report.record_skip(name_str, format!(
                        "UNRESOLVED_NAMES: {}", unresolved.join(", ")));
                    self.stats.inline_fns_unresolved_names += 1;
                }
                InlineGenResult::CodegenError { code, errors } => {
                    let name_str = self.interner.get(name);
                    writeln!(self.writer, "// [CODEGEN_ERROR] {} - inline function", name_str)?;
                    for err in &errors {
                        writeln!(self.writer, "//   {}", err)?;
                    }
                    for line in code.lines() {
                        writeln!(self.writer, "// {}", line)?;
                    }
                    writeln!(self.writer)?;
                    self.report.record_skip(name_str, format!(
                        "CODEGEN_ERROR: {}", errors.join("; ")));
                }
                InlineGenResult::Incomplete { code } => {
                    let name_str = self.interner.get(name);
                    writeln!(self.writer, "// [CODEGEN_INCOMPLETE] {} - inline function", name_str)?;
                    for line in code.lines() {
                        writeln!(self.writer, "// {}", line)?;
                    }
                    writeln!(self.writer)?;
                    self.report.record_skip(name_str, "CODEGEN_INCOMPLETE: type inference incomplete");
                    self.stats.inline_fns_type_incomplete += 1;
                }
                InlineGenResult::Success { code, used_libc } => {
                    if self.successfully_generated_inlines.contains(&name) {
                        // 正常出力
                        write!(self.writer, "{}", code)?;
                        self.used_libc_fns.extend(used_libc.iter().cloned());
                        self.report.record_emit(self.interner.get(name));
                        self.stats.inline_fns_success += 1;
                    } else {
                        // カスケード降格: 呼び出し先の inline 関数が codegen 時に失敗
                        let name_str = self.interner.get(name);
                        let mut unavailable: Vec<String> = result.inline_fn_dict.get_called_functions(name)
                            .map(|calls| calls.iter()
                                .filter(|c| inline_set.contains(c) && !self.successfully_generated_inlines.contains(c))
                                .map(|c| self.interner.get(*c).to_string())
                                .collect())
                            .unwrap_or_default();
                        unavailable.sort(); // 列挙順の決定化
                        writeln!(self.writer, "// [CASCADE_UNAVAILABLE] {} - dependency not generated: {}",
                            name_str, unavailable.join(", "))?;
                        for line in code.lines() {
                            writeln!(self.writer, "// {}", line)?;
                        }
                        writeln!(self.writer)?;
                        self.report.record_skip(name_str, format!(
                            "CASCADE_UNAVAILABLE: dependency not generated: {}",
                            unavailable.join(", ")));
                        self.stats.inline_fns_cascade_unavailable += 1;
                    }
                }
            }
        }

        writeln!(self.writer)?;
        Ok(())
    }

    /// マクロセクションを生成
    pub fn generate_macros(&mut self, result: &InferResult, known_symbols: &KnownSymbols) -> io::Result<()> {
        writeln!(self.writer, "// =============================================================================")?;
        writeln!(self.writer, "// Macro Functions")?;
        writeln!(self.writer, "// =============================================================================")?;
        writeln!(self.writer)?;

        // 対象マクロを収集
        let macros: Vec<_> = result.infer_ctx.macros.iter()
            .filter(|(_, info)| self.should_include_macro(info))
            .collect();
        let included_set: HashSet<InternedStr> = macros.iter().map(|(n, _)| **n).collect();

        // 依存順にソート（葉マクロ先頭）
        let sorted_names = self.topological_sort_macros(&macros);

        // ── Phase 2 の解析結果を参照 ──
        // const/mut, bool 戻り値は Phase 2 (resolve_param_and_return_types) で確定済み
        // MacroInferInfo.const_pointer_positions, is_bool_return を参照する
        let mut callee_const_params: HashMap<InternedStr, HashSet<usize>> = HashMap::new();
        let mut bool_return_macros: HashSet<InternedStr> = HashSet::new();
        for (&name, info) in &result.infer_ctx.macros {
            if !info.const_pointer_positions.is_empty() {
                callee_const_params.insert(name, info.const_pointer_positions.clone());
            }
            if info.is_bool_return {
                bool_return_macros.insert(name);
            }
        }
        self.const_pointer_params = callee_const_params;
        self.bool_return_macros = bool_return_macros;

        // 正常生成されたマクロを追跡
        let mut successfully_generated: HashSet<InternedStr> = HashSet::new();

        for name in sorted_names {
            let info = result.infer_ctx.macros.get(&name).unwrap();

            // ── apidoc skip_codegen 対象なら早期に [CODEGEN_SUPPRESSED] ──
            // Phase 2 (Step 4.4) で `info.apidoc_suppressed` が立っているので
            // それを参照する。reason 文字列は apidoc_patches から取得。
            if info.apidoc_suppressed {
                let name_str_for_patch = self.interner.get(name);
                let reason = result.apidoc_patches.skip_reason(name_str_for_patch)
                    .unwrap_or("apidoc skip_codegen");
                let thx_info = if info.is_thx_dependent { " [THX]" } else { "" };
                writeln!(self.writer,
                    "// [CODEGEN_SUPPRESSED] {}{} - macro function (apidoc patch)",
                    name_str_for_patch, thx_info)?;
                writeln!(self.writer, "// Reason: {}", reason)?;
                writeln!(self.writer)?;
                self.report.record_skip(name_str_for_patch, format!(
                    "CODEGEN_SUPPRESSED (apidoc patch): {}", reason));
                continue;
            }

            // ── apidoc 戻り値型 pair (カンマ式で複数値) も codegen 対象外 ──
            // Phase 2 (Step 4.45) で `info.pair_return` が立っている
            if info.pair_return {
                let name_str = self.interner.get(name);
                writeln!(self.writer,
                    "// [CODEGEN_SUPPRESSED] {} - pair return type (comma expression)",
                    name_str)?;
                writeln!(self.writer,
                    "// Reason: apidoc declares return type `pair`; \
                     not representable as a single-value Rust fn")?;
                writeln!(self.writer)?;
                self.report.record_skip(name_str,
                    "CODEGEN_SUPPRESSED (pair return type)".to_string());
                continue;
            }

            // ── カスケード検査 ──
            // called_functions のうち、生成対象だが生成に失敗した関数があれば
            // カスケード失敗。マクロ→マクロ依存と マクロ→inline 関数依存の両方を検査。
            // （uses ではなく called_functions を使う：uses はインライン展開された
            //   マクロも含むが、called_functions は AST 上の Call 式のみ）
            let unavailable_deps: Vec<String> = info.called_functions.iter()
                .filter(|called| {
                    // Case 1: マクロ→マクロ依存
                    if included_set.contains(called) {
                        return result.infer_ctx.macros.get(called)
                            .map(|u| u.is_parseable() && !u.is_unavailable_for_codegen())
                            .unwrap_or(false)
                            && !successfully_generated.contains(called);
                    }
                    // Case 2: マクロ→inline 関数依存
                    if result.inline_fn_dict.get(**called)
                        .map(|f| f.is_target)
                        .unwrap_or(false)
                    {
                        return !self.successfully_generated_inlines.contains(called);
                    }
                    false
                })
                .map(|called| self.interner.get(*called).to_string())
                .collect();

            if !unavailable_deps.is_empty() {
                self.generate_macro_cascade_unavailable(info, &unavailable_deps)?;
                self.stats.macros_cascade_unavailable += 1;
                continue;
            }

            // ── 既存のステータス判定と生成 ──
            let status = self.get_macro_status(info);
            match status {
                GenerateStatus::Success => {
                    // 新しい RustCodegen を使ってマクロを生成
                    let const_positions = self.const_pointer_params.get(&name)
                        .cloned().unwrap_or_default();
                    let is_bool = self.bool_return_macros.contains(&name);
                    let codegen = RustCodegen::new(self.interner, self.enum_dict, self.macro_ctx, self.bindings_info.clone(), known_symbols, result.rust_decl_dict.as_ref(), Some(&result.inline_fn_dict))
                        .with_perl_threaded(self.perl_threaded)
                        .with_dump_ast_for(self.config.dump_ast_for.clone())
                        .with_dump_types_for(self.config.dump_types_for.clone())
                        .with_fields_dict(&result.fields_dict)
                        .with_const_pointer_positions(const_positions)
                        .with_bool_return(is_bool, self.bool_return_macros.clone());
                    let generated = codegen.generate_macro(info);

                    if generated.has_unresolved_names() {
                        // 未解決シンボルあり：コメントアウトして出力
                        let name_str = self.interner.get(info.name);
                        let thx_info = if info.is_thx_dependent { " [THX]" } else { "" };
                        writeln!(self.writer, "// [UNRESOLVED_NAMES] {}{} - macro function", name_str, thx_info)?;
                        writeln!(self.writer, "// Unresolved: {}", generated.unresolved_names.join(", "))?;
                        for line in generated.code.lines() {
                            writeln!(self.writer, "// {}", line)?;
                        }
                        writeln!(self.writer)?;
                        self.report.record_skip(name_str, format!(
                            "UNRESOLVED_NAMES: {}", generated.unresolved_names.join(", ")));
                        self.stats.macros_unresolved_names += 1;
                    } else if !generated.codegen_errors.is_empty() {
                        // codegen エラー検出：コメントアウトして問題点列挙
                        let name_str = self.interner.get(info.name);
                        let thx_info = if info.is_thx_dependent { " [THX]" } else { "" };
                        writeln!(self.writer, "// [CODEGEN_ERROR] {}{} - macro function", name_str, thx_info)?;
                        for err in &generated.codegen_errors {
                            writeln!(self.writer, "//   {}", err)?;
                        }
                        for line in generated.code.lines() {
                            writeln!(self.writer, "// {}", line)?;
                        }
                        writeln!(self.writer)?;
                        self.report.record_skip(name_str, format!(
                            "CODEGEN_ERROR: {}", generated.codegen_errors.join("; ")));
                    } else if generated.is_complete() {
                        // 完全な生成：そのまま出力
                        write!(self.writer, "{}", generated.code)?;
                        self.used_libc_fns.extend(generated.used_libc_fns.iter().cloned());
                        self.report.record_emit(self.interner.get(info.name));
                        self.stats.macros_success += 1;
                        successfully_generated.insert(name);
                    } else {
                        // 不完全な生成：コメントアウトして出力
                        let name_str = self.interner.get(info.name);
                        let thx_info = if info.is_thx_dependent { " [THX]" } else { "" };
                        writeln!(self.writer, "// [CODEGEN_INCOMPLETE] {}{} - macro function", name_str, thx_info)?;
                        for line in generated.code.lines() {
                            writeln!(self.writer, "// {}", line)?;
                        }
                        writeln!(self.writer)?;
                        self.report.record_skip(name_str, "CODEGEN_INCOMPLETE: type inference incomplete");
                        self.stats.macros_type_incomplete += 1;
                    }
                }
                GenerateStatus::ParseFailed => {
                    self.generate_macro_parse_failed(info)?;
                    self.report.record_skip(self.interner.get(info.name),
                        "PARSE_FAILED: macro body does not parse");
                    self.stats.macros_parse_failed += 1;
                }
                GenerateStatus::TypeIncomplete => {
                    self.generate_macro_type_incomplete(info, result)?;
                    self.report.record_skip(self.interner.get(info.name),
                        "TYPE_INCOMPLETE: parameter/return type could not be inferred");
                    self.stats.macros_type_incomplete += 1;
                }
                GenerateStatus::CallsUnavailable => {
                    // 原因を分類: 「実在しない関数を呼ぶ」vs「推移的に
                    // unavailable な依存先を呼ぶ」。前者なら [CALLS_UNAVAILABLE]、
                    // 後者なら [CASCADE_UNAVAILABLE] を出す。
                    let absent: Vec<&str> = info.called_functions.iter()
                        .filter(|fn_id| {
                            let fn_name = self.interner.get(**fn_id);
                            !self.is_function_available(**fn_id, fn_name, result)
                        })
                        .map(|fn_id| self.interner.get(*fn_id))
                        .collect();
                    if absent.is_empty() {
                        // 純粋に推移依存。unavailable な依存先を列挙
                        let cascade_deps: Vec<String> = info.called_functions.iter()
                            .filter(|fn_id| {
                                if let Some(m) = result.infer_ctx.macros.get(*fn_id) {
                                    return m.is_unavailable_for_codegen();
                                }
                                if result.inline_fn_dict.get(**fn_id).is_some() {
                                    return result.inline_fn_dict
                                        .is_unavailable_for_codegen(**fn_id);
                                }
                                false
                            })
                            .map(|fn_id| self.interner.get(*fn_id).to_string())
                            .collect();
                        self.generate_macro_cascade_unavailable(info, &cascade_deps)?;
                        self.stats.macros_cascade_unavailable += 1;
                    } else {
                        let mut absent_sorted: Vec<String> =
                            absent.iter().map(|s| s.to_string()).collect();
                        absent_sorted.sort();
                        self.generate_macro_calls_unavailable(info, result)?;
                        self.report.record_skip(self.interner.get(info.name), format!(
                            "CALLS_UNAVAILABLE: calls unavailable function(s): {}",
                            absent_sorted.join(", ")));
                        self.stats.macros_calls_unavailable += 1;
                    }
                }
                GenerateStatus::ContainsGoto => {
                    let name_str = self.interner.get(info.name);
                    writeln!(self.writer, "// [CONTAINS_GOTO] {} - excluded (contains goto)", name_str)?;
                    writeln!(self.writer)?;
                    self.report.record_skip(name_str, "CONTAINS_GOTO: excluded (contains goto)");
                }
                GenerateStatus::GenericUnsupported => {
                    let name_str = self.interner.get(info.name);
                    let thx_info = if info.is_thx_dependent { " [THX]" } else { "" };
                    writeln!(self.writer, "// [GENERIC_UNSUPPORTED] {}{} - Rust cannot cast to generic type T", name_str, thx_info)?;
                    // コメントアウトしたコードを出力
                    let const_positions = self.const_pointer_params.get(&name)
                        .cloned().unwrap_or_default();
                    let is_bool = self.bool_return_macros.contains(&name);
                    let codegen = RustCodegen::new(self.interner, self.enum_dict, self.macro_ctx, self.bindings_info.clone(), known_symbols, result.rust_decl_dict.as_ref(), Some(&result.inline_fn_dict))
                        .with_perl_threaded(self.perl_threaded)
                        .with_fields_dict(&result.fields_dict)
                        .with_const_pointer_positions(const_positions)
                        .with_bool_return(is_bool, self.bool_return_macros.clone());
                    let generated = codegen.generate_macro(info);
                    for line in generated.code.lines() {
                        writeln!(self.writer, "// {}", line)?;
                    }
                    writeln!(self.writer)?;
                    self.report.record_skip(self.interner.get(info.name),
                        "GENERIC_UNSUPPORTED: Rust cannot cast to generic type T");
                    self.stats.macros_generic_unsupported += 1;
                }
                GenerateStatus::Skip => {
                    // 出力なし。require 検証用に理由だけ記録する
                    self.report.record_skip(self.interner.get(info.name),
                        "SKIP: excluded from code generation");
                }
            }
        }

        Ok(())
    }

    /// マクロを依存順にソート（葉マクロ先頭、循環はアルファベット順で末尾）
    ///
    /// Kahn's algorithm を使用。`info.uses` の関係から DAG を構築し、
    fn topological_sort_macros(
        &self,
        macros: &[(&InternedStr, &MacroInferInfo)],
    ) -> Vec<InternedStr> {
        use std::collections::VecDeque;

        let macro_set: HashSet<InternedStr> = macros.iter().map(|(n, _)| **n).collect();

        // 入次数マップと逆隣接リスト（dependency → dependents）を構築
        let mut in_degree: HashMap<InternedStr, usize> = HashMap::new();
        let mut dependents: HashMap<InternedStr, Vec<InternedStr>> = HashMap::new();

        for (name, _) in macros {
            in_degree.insert(**name, 0);
        }

        for (name, info) in macros {
            for used in &info.uses {
                if macro_set.contains(used) {
                    // name が used に依存 → used から name への辺
                    *in_degree.entry(**name).or_insert(0) += 1;
                    dependents.entry(*used).or_default().push(**name);
                }
            }
        }

        // 入次数 0 のマクロをキューに投入（アルファベット順で安定化）
        let mut queue: VecDeque<InternedStr> = {
            let mut zeros: Vec<_> = in_degree.iter()
                .filter(|(_, deg)| **deg == 0)
                .map(|(name, _)| *name)
                .collect();
            zeros.sort_by_key(|n| self.interner.get(*n));
            zeros.into_iter().collect()
        };

        let mut result = Vec::with_capacity(macros.len());

        while let Some(name) = queue.pop_front() {
            result.push(name);
            if let Some(deps) = dependents.get(&name) {
                // 依存先の入次数を減算
                let mut newly_ready: Vec<InternedStr> = Vec::new();
                for dep in deps {
                    if let Some(deg) = in_degree.get_mut(dep) {
                        *deg -= 1;
                        if *deg == 0 {
                            newly_ready.push(*dep);
                        }
                    }
                }
                // アルファベット順で安定化
                newly_ready.sort_by_key(|n| self.interner.get(*n));
                for n in newly_ready {
                    queue.push_back(n);
                }
            }
        }

        // 残り（循環メンバー）をアルファベット順で末尾に追加
        if result.len() < macros.len() {
            let result_set: HashSet<_> = result.iter().copied().collect();
            let mut remaining: Vec<_> = macro_set.iter()
                .filter(|n| !result_set.contains(n))
                .copied()
                .collect();
            remaining.sort_by_key(|n| self.interner.get(*n));
            result.extend(remaining);
        }

        result
    }

    /// カスケード依存によるコメントアウト出力
    fn generate_macro_cascade_unavailable(
        &mut self,
        info: &MacroInferInfo,
        unavailable_deps: &[String],
    ) -> io::Result<()> {
        let name_str = self.interner.get(info.name);
        let thx_info = if info.is_thx_dependent { " [THX]" } else { "" };
        let mut deps: Vec<&String> = unavailable_deps.iter().collect();
        deps.sort(); // 列挙順の決定化
        let deps: Vec<&str> = deps.iter().map(|s| s.as_str()).collect();
        writeln!(self.writer,
            "// [CASCADE_UNAVAILABLE] {}{} - dependency not generated: {}",
            name_str, thx_info, deps.join(", "))?;
        writeln!(self.writer)?;
        self.report.record_skip(name_str, format!(
            "CASCADE_UNAVAILABLE: dependency not generated: {}", deps.join(", ")));
        Ok(())
    }

    /// マクロを出力対象にするかどうか
    fn should_include_macro(&self, info: &MacroInferInfo) -> bool {
        // ターゲットでなければスキップ
        if !info.is_target {
            return false;
        }

        // 本体がなければスキップ
        if !info.has_body {
            return false;
        }

        // 関数形式マクロのみ含める（オブジェクトマクロは常にインライン展開される）
        info.is_function
    }

    /// マクロの生成ステータスを判定
    fn get_macro_status(&self, info: &MacroInferInfo) -> GenerateStatus {
        // 利用不可関数を呼び出すマクロは CallsUnavailable
        if info.calls_unavailable {
            return GenerateStatus::CallsUnavailable;
        }

        // ジェネリクス型パラメータを含むマクロは生成不可
        // Rust の as T キャストや T + u32 演算が不可
        if !info.generic_type_params.is_empty() {
            return GenerateStatus::GenericUnsupported;
        }

        match &info.parse_result {
            ParseResult::Unparseable(_) => GenerateStatus::ParseFailed,
            ParseResult::Statement(items) => {
                // goto を含むマクロは除外
                if block_items_contain_goto(items) {
                    return GenerateStatus::ContainsGoto;
                }
                if info.is_fully_confirmed() {
                    GenerateStatus::Success
                } else {
                    GenerateStatus::TypeIncomplete
                }
            }
            ParseResult::Expression(_) => {
                if info.is_fully_confirmed() {
                    GenerateStatus::Success
                } else {
                    GenerateStatus::TypeIncomplete
                }
            }
        }
    }

    /// パース失敗マクロをコメント出力
    fn generate_macro_parse_failed(&mut self, info: &MacroInferInfo) -> io::Result<()> {
        let name_str = self.interner.get(info.name);

        // パラメータリストを構築
        let params_str = if info.is_function {
            let params: Vec<_> = info.params.iter()
                .map(|p| self.interner.get(p.name).to_string())
                .collect();
            format!("({})", params.join(", "))
        } else {
            String::new()
        };

        writeln!(self.writer, "// [PARSE_FAILED] {}{}", name_str, params_str)?;

        // エラーメッセージがあれば出力
        if let ParseResult::Unparseable(Some(err_msg)) = &info.parse_result {
            writeln!(self.writer, "// Error: {}", err_msg)?;
        }

        // TODO: トークン列の S 式出力（Phase 2 で詳細実装）
        writeln!(self.writer, "// (tokens not available in parsed form)")?;
        writeln!(self.writer)?;

        Ok(())
    }

    /// 利用不可関数呼び出しマクロをコメント出力
    fn generate_macro_calls_unavailable(&mut self, info: &MacroInferInfo, result: &InferResult) -> io::Result<()> {
        let name_str = self.interner.get(info.name);

        // パラメータリストを構築
        let params_str = if info.is_function {
            let params: Vec<_> = info.params.iter()
                .map(|p| self.interner.get(p.name).to_string())
                .collect();
            format!("({})", params.join(", "))
        } else {
            String::new()
        };

        // THX 情報
        let thx_info = if info.is_thx_dependent { " [THX]" } else { "" };

        writeln!(self.writer, "// [CALLS_UNAVAILABLE] {}{}{} - calls unavailable function(s)", name_str, params_str, thx_info)?;

        // 利用不可関数を特定して出力
        let unavailable_fns: Vec<_> = info.called_functions.iter()
            .filter(|&fn_id| {
                let fn_name = self.interner.get(*fn_id);
                // bindings.rs にもマクロにも存在しない関数を検出
                !self.is_function_available(*fn_id, fn_name, result)
            })
            .map(|fn_id| self.interner.get(*fn_id))
            .collect();

        if !unavailable_fns.is_empty() {
            let mut unavailable_fns = unavailable_fns;
            unavailable_fns.sort(); // 列挙順の決定化
            writeln!(self.writer, "// Unavailable: {}", unavailable_fns.join(", "))?;
        }

        writeln!(self.writer)?;

        Ok(())
    }

    /// 関数が利用可能かチェック
    fn is_function_available(&self, fn_id: crate::InternedStr, fn_name: &str, result: &InferResult) -> bool {
        // マクロとして存在する場合はOK
        if result.infer_ctx.macros.contains_key(&fn_id) {
            return true;
        }

        // bindings.rs に存在する場合はOK
        if let Some(rust_decl_dict) = &result.rust_decl_dict {
            if rust_decl_dict.fns.contains_key(fn_name) {
                return true;
            }
        }

        // インライン関数として存在する場合はOK
        if result.inline_fn_dict.get(fn_id).is_some() {
            return true;
        }

        // ビルトイン関数リスト（macro_infer.rs と同じ）
        let builtin_fns = [
            "__builtin_expect",
            "__builtin_offsetof",
            "offsetof",
            "__builtin_types_compatible_p",
            "__builtin_constant_p",
            "__builtin_choose_expr",
            "__builtin_unreachable",
            "__builtin_trap",
            "__builtin_assume",
            "__builtin_bswap16",
            "__builtin_bswap32",
            "__builtin_bswap64",
            "__builtin_popcount",
            "__builtin_clz",
            "__builtin_ctz",
            "pthread_mutex_lock",
            "pthread_mutex_unlock",
            "pthread_rwlock_rdlock",
            "pthread_rwlock_wrlock",
            "pthread_rwlock_unlock",
            "memchr",
            "memcpy",
            "memmove",
            "memset",
            "strlen",
            "strcmp",
            "strncmp",
            "strcpy",
            "strncpy",
            "ASSERT_IS_LITERAL",
            "ASSERT_IS_PTR",
            "ASSERT_NOT_PTR",
        ];

        if builtin_fns.contains(&fn_name) {
            return true;
        }

        false
    }

    /// 型推論失敗マクロをコメント出力
    fn generate_macro_type_incomplete(&mut self, info: &MacroInferInfo, result: &InferResult) -> io::Result<()> {
        let name_str = self.interner.get(info.name);

        // パラメータリストを構築
        let params_str = if info.is_function {
            let params: Vec<_> = info.params.iter()
                .map(|p| self.interner.get(p.name).to_string())
                .collect();
            format!("({})", params.join(", "))
        } else {
            String::new()
        };

        writeln!(self.writer, "// [TYPE_INCOMPLETE] {}{}", name_str, params_str)?;

        // 型推論状態を出力
        writeln!(self.writer, "// Args status: {:?}, Return status: {:?}",
            info.args_infer_status, info.return_infer_status)?;

        // 型付 S 式を出力
        writeln!(self.writer, "// Typed S-expression:")?;
        self.write_typed_sexp_comment(info, result)?;

        writeln!(self.writer)?;
        Ok(())
    }

    /// 型付 S 式をコメントとして出力
    fn write_typed_sexp_comment(&mut self, info: &MacroInferInfo, _result: &InferResult) -> io::Result<()> {
        match &info.parse_result {
            ParseResult::Expression(expr) => {
                self.write_expr_sexp_comment(expr, info, "//   ")?;
            }
            ParseResult::Statement(block_items) => {
                for item in block_items {
                    if let BlockItem::Stmt(stmt) = item {
                        self.write_stmt_sexp_comment(stmt, info, "//   ")?;
                    }
                }
            }
            ParseResult::Unparseable(_) => {
                writeln!(self.writer, "//   (unparseable)")?;
            }
        }
        Ok(())
    }

    /// 式の S 式をコメントとして出力
    fn write_expr_sexp_comment(&mut self, expr: &Expr, info: &MacroInferInfo, prefix: &str) -> io::Result<()> {
        // 簡易的な S 式出力
        let mut buf = Vec::new();
        {
            let mut printer = SexpPrinter::new(&mut buf, self.interner);
            let _ = printer.print_expr(expr);
        }

        // 型情報を追加
        let sexp_str = String::from_utf8_lossy(&buf);
        let type_info = self.get_expr_type_info(expr, info);

        // 各行にプレフィックスを付けて出力
        for line in sexp_str.lines() {
            writeln!(self.writer, "{}{}", prefix, line)?;
        }
        if !type_info.is_empty() {
            writeln!(self.writer, "{} :type {}", prefix, type_info)?;
        }

        Ok(())
    }

    /// 文の S 式をコメントとして出力
    fn write_stmt_sexp_comment(&mut self, stmt: &Stmt, _info: &MacroInferInfo, prefix: &str) -> io::Result<()> {
        let mut buf = Vec::new();
        {
            let mut printer = SexpPrinter::new(&mut buf, self.interner);
            let _ = printer.print_stmt(stmt);
        }

        let sexp_str = String::from_utf8_lossy(&buf);
        for line in sexp_str.lines() {
            writeln!(self.writer, "{}{}", prefix, line)?;
        }

        Ok(())
    }

    /// 式の型情報を取得
    fn get_expr_type_info(&self, expr: &Expr, info: &MacroInferInfo) -> String {
        // TypeEnv から型制約を取得
        if let Some(constraints) = info.type_env.expr_constraints.get(&expr.id) {
            if let Some(first) = constraints.first() {
                return first.ty.to_display_string(self.interner);
            }
        }
        "<unknown>".to_string()
    }
}

#[cfg(test)]
mod codegen_report_tests {
    use super::*;

    fn sample_report() -> CodegenReport {
        let mut report = CodegenReport::default();
        report.record_emit("CvFILE");
        report.record_emit("CvROOT");
        report.record_skip("CvSTART", "CASCADE_UNAVAILABLE: dependency not generated: CvROOT");
        report
    }

    #[test]
    fn test_check_required_empty_list_is_noop() {
        let report = sample_report();
        assert!(report.check_required(&[]).is_ok());
    }

    #[test]
    fn test_check_required_emitted_names_pass() {
        let report = sample_report();
        let required = vec!["CvFILE".to_string(), "CvROOT".to_string()];
        assert!(report.check_required(&required).is_ok());
    }

    #[test]
    fn test_check_required_skipped_name_reports_recorded_reason() {
        let report = sample_report();
        let required = vec!["CvSTART".to_string()];
        let err = report.check_required(&required).unwrap_err();
        assert_eq!(err.violations.len(), 1);
        assert_eq!(err.violations[0].name, "CvSTART");
        assert!(err.violations[0].reason.contains("CASCADE_UNAVAILABLE"));
        // Display にも名前と理由が現れる
        let msg = err.to_string();
        assert!(msg.contains("CvSTART"));
        assert!(msg.contains("CASCADE_UNAVAILABLE"));
    }

    #[test]
    fn test_check_required_unknown_name_reports_not_a_target() {
        let report = sample_report();
        let required = vec!["NoSuchFunction".to_string()];
        let err = report.check_required(&required).unwrap_err();
        assert_eq!(err.violations.len(), 1);
        assert!(err.violations[0].reason.contains("not a codegen target"));
    }

    #[test]
    fn test_check_required_collects_all_violations() {
        let report = sample_report();
        let required = vec![
            "CvFILE".to_string(),        // emitted → OK
            "CvSTART".to_string(),       // skipped → 違反
            "NoSuchFunction".to_string(), // 不在 → 違反
        ];
        let err = report.check_required(&required).unwrap_err();
        assert_eq!(err.violations.len(), 2);
    }
}
