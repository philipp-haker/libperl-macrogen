//! Parser integration tests

use std::io::Write;
use tempfile::NamedTempFile;
use libperl_macrogen::{
    DerivedDecl, Expr, ExprKind, ExternalDecl, Initializer, PPConfig, Parser, Preprocessor,
    StorageClass, TypeSpec,
};

/// Helper to parse a source string and return translation unit
fn parse(source: &str) -> Vec<ExternalDecl> {
    let mut file = NamedTempFile::new().unwrap();
    file.write_all(source.as_bytes()).unwrap();
    file.flush().unwrap();

    let config = PPConfig {
        include_paths: vec![],
        predefined: vec![],
        debug_pp: false,
        target_dir: None,
        ..Default::default()
    };

    let mut pp = Preprocessor::new(config);
    pp.add_source_file(file.path()).unwrap();

    let mut parser = Parser::new(&mut pp).unwrap();
    let tu = parser.parse().unwrap();
    tu.decls
}

/// Helper to check if a declaration has typedef storage class
fn is_typedef(decl: &ExternalDecl) -> bool {
    match decl {
        ExternalDecl::Declaration(d) => d.specs.storage == Some(StorageClass::Typedef),
        _ => false,
    }
}

/// Helper to check if a declaration has static storage class
fn is_static(decl: &ExternalDecl) -> bool {
    match decl {
        ExternalDecl::Declaration(d) => d.specs.storage == Some(StorageClass::Static),
        ExternalDecl::FunctionDef(f) => f.specs.storage == Some(StorageClass::Static),
    }
}

/// Helper to check if a declaration has extern storage class
fn is_extern(decl: &ExternalDecl) -> bool {
    match decl {
        ExternalDecl::Declaration(d) => d.specs.storage == Some(StorageClass::Extern),
        ExternalDecl::FunctionDef(f) => f.specs.storage == Some(StorageClass::Extern),
    }
}

/// Helper to check if type specs contain a specific type
fn has_type_spec(decl: &ExternalDecl, check: impl Fn(&TypeSpec) -> bool) -> bool {
    match decl {
        ExternalDecl::Declaration(d) => d.specs.type_specs.iter().any(check),
        ExternalDecl::FunctionDef(f) => f.specs.type_specs.iter().any(check),
    }
}

#[test]
fn test_extension_compound_literal_and_cast() {
    let src = r#"
        typedef float v4 __attribute__((__vector_size__(16)));
        v4 f(void) { return __extension__ (v4){ 0.0f, 0.0f, 0.0f, 0.0f }; }
        long long g(int x) { return __extension__ (long long)x; }
    "#;
    let decls = parse(src);
    assert_eq!(decls.len(), 3);
}

#[test]
fn test_attribute_leading_type_name() {
    let src = r#"
        int f(void) { return sizeof(__attribute__((__vector_size__(16))) int); }
        void g(void) { (__attribute__((__vector_size__(16))) int){4, 1, 2, 3}; }
    "#;
    let decls = parse(src);
    assert_eq!(decls.len(), 2);
}

#[test]
 fn test_builtin_bf16_type() {
       let src = r#"
           typedef __bf16 v8bf __attribute__ ((__vector_size__ (16)));
           __bf16 f(__bf16 x);
       "#;
       let decls = parse(src);
       assert_eq!(decls.len(), 2);
   }

#[test]
fn test_attribute_after_pointer() {
    // e.g. mingw-w64: `extern int *__cdecl _errno(void);` with __cdecl = __attribute__((__cdecl__))
    let decls = parse("extern int * __attribute__((__cdecl__)) _errno(void);");
    assert_eq!(decls.len(), 1);
}

#[test]
fn test_simple_variable() {
    let decls = parse("int x;");
    assert_eq!(decls.len(), 1);
    assert!(has_type_spec(&decls[0], |t| matches!(t, TypeSpec::Int)));
}

#[test]
fn test_multiple_variables() {
    let decls = parse("int x, y, z;");
    assert_eq!(decls.len(), 1);

    if let ExternalDecl::Declaration(d) = &decls[0] {
        assert_eq!(d.declarators.len(), 3);
    } else {
        panic!("Expected declaration");
    }
}

#[test]
fn test_typedef() {
    let decls = parse("typedef int MyInt;");
    assert_eq!(decls.len(), 1);
    assert!(is_typedef(&decls[0]));
}

#[test]
fn test_typedef_usage() {
    let decls = parse("typedef int MyInt;\nMyInt x;");
    assert_eq!(decls.len(), 2);
    assert!(is_typedef(&decls[0]));
}

#[test]
fn test_struct_declaration() {
    let decls = parse("struct Point { int x; int y; };");
    assert_eq!(decls.len(), 1);
    assert!(has_type_spec(&decls[0], |t| matches!(t, TypeSpec::Struct(_))));
}

#[test]
fn test_struct_variable() {
    let decls = parse("struct Point { int x; int y; } p;");
    assert_eq!(decls.len(), 1);

    if let ExternalDecl::Declaration(d) = &decls[0] {
        assert_eq!(d.declarators.len(), 1);
    } else {
        panic!("Expected declaration");
    }
}

#[test]
fn test_union_declaration() {
    let decls = parse("union Value { int i; float f; };");
    assert_eq!(decls.len(), 1);
    assert!(has_type_spec(&decls[0], |t| matches!(t, TypeSpec::Union(_))));
}

#[test]
fn test_enum_declaration() {
    let decls = parse("enum Color { RED, GREEN, BLUE };");
    assert_eq!(decls.len(), 1);
    assert!(has_type_spec(&decls[0], |t| matches!(t, TypeSpec::Enum(_))));
}

#[test]
fn test_enum_with_values() {
    let decls = parse("enum Status { OK = 0, ERROR = 1 };");
    assert_eq!(decls.len(), 1);
}

#[test]
fn test_function_declaration() {
    let decls = parse("int foo(int x, int y);");
    assert_eq!(decls.len(), 1);

    if let ExternalDecl::Declaration(d) = &decls[0] {
        assert_eq!(d.declarators.len(), 1);
    } else {
        panic!("Expected declaration");
    }
}

#[test]
fn test_function_definition() {
    let decls = parse("int foo(int x) { return x + 1; }");
    assert_eq!(decls.len(), 1);
    assert!(matches!(decls[0], ExternalDecl::FunctionDef(_)));
}

#[test]
fn test_static_function() {
    let decls = parse("static int foo(void) { return 0; }");
    assert_eq!(decls.len(), 1);
    assert!(is_static(&decls[0]));
}

#[test]
fn test_extern_variable() {
    let decls = parse("extern int global_var;");
    assert_eq!(decls.len(), 1);
    assert!(is_extern(&decls[0]));
}

#[test]
fn test_const_qualifier() {
    let decls = parse("const int x = 5;");
    assert_eq!(decls.len(), 1);

    if let ExternalDecl::Declaration(d) = &decls[0] {
        assert!(d.specs.qualifiers.is_const);
    } else {
        panic!("Expected declaration");
    }
}

#[test]
fn test_volatile_qualifier() {
    let decls = parse("volatile int x;");
    assert_eq!(decls.len(), 1);

    if let ExternalDecl::Declaration(d) = &decls[0] {
        assert!(d.specs.qualifiers.is_volatile);
    } else {
        panic!("Expected declaration");
    }
}

#[test]
fn test_pointer_declaration() {
    let decls = parse("int *p;");
    assert_eq!(decls.len(), 1);

    if let ExternalDecl::Declaration(d) = &decls[0] {
        assert!(d.declarators[0].declarator.derived.iter().any(|d| matches!(d, DerivedDecl::Pointer(_))));
    } else {
        panic!("Expected declaration");
    }
}

#[test]
fn test_array_declaration() {
    let decls = parse("int arr[10];");
    assert_eq!(decls.len(), 1);

    if let ExternalDecl::Declaration(d) = &decls[0] {
        assert!(d.declarators[0].declarator.derived.iter().any(|d| matches!(d, DerivedDecl::Array(_))));
    } else {
        panic!("Expected declaration");
    }
}

#[test]
fn test_function_pointer() {
    let decls = parse("int (*fp)(int, int);");
    assert_eq!(decls.len(), 1);
}

#[test]
fn test_inline_function() {
    let decls = parse("inline int foo(void) { return 0; }");
    assert_eq!(decls.len(), 1);

    if let ExternalDecl::FunctionDef(f) = &decls[0] {
        assert!(f.specs.is_inline);
    } else {
        panic!("Expected function definition");
    }
}

#[test]
fn test_static_inline() {
    let decls = parse("static inline int foo(void) { return 0; }");
    assert_eq!(decls.len(), 1);
    assert!(is_static(&decls[0]));

    if let ExternalDecl::FunctionDef(f) = &decls[0] {
        assert!(f.specs.is_inline);
    } else {
        panic!("Expected function definition");
    }
}

#[test]
fn test_void_function() {
    let decls = parse("void foo(void);");
    assert_eq!(decls.len(), 1);
    assert!(has_type_spec(&decls[0], |t| matches!(t, TypeSpec::Void)));
}

#[test]
fn test_variadic_function() {
    let decls = parse("int printf(const char *fmt, ...);");
    assert_eq!(decls.len(), 1);
}

#[test]
fn test_long_long() {
    let decls = parse("long long x;");
    assert_eq!(decls.len(), 1);

    if let ExternalDecl::Declaration(d) = &decls[0] {
        let long_count = d.specs.type_specs.iter().filter(|t| matches!(t, TypeSpec::Long)).count();
        assert_eq!(long_count, 2);
    } else {
        panic!("Expected declaration");
    }
}

#[test]
fn test_unsigned_int() {
    let decls = parse("unsigned int x;");
    assert_eq!(decls.len(), 1);
    assert!(has_type_spec(&decls[0], |t| matches!(t, TypeSpec::Unsigned)));
    assert!(has_type_spec(&decls[0], |t| matches!(t, TypeSpec::Int)));
}

#[test]
fn test_short_int() {
    let decls = parse("short int x;");
    assert_eq!(decls.len(), 1);
    assert!(has_type_spec(&decls[0], |t| matches!(t, TypeSpec::Short)));
}

#[test]
fn test_initializer() {
    let decls = parse("int x = 42;");
    assert_eq!(decls.len(), 1);

    if let ExternalDecl::Declaration(d) = &decls[0] {
        assert!(d.declarators[0].init.is_some());
    } else {
        panic!("Expected declaration");
    }
}

#[test]
fn test_array_initializer() {
    let decls = parse("int arr[] = {1, 2, 3};");
    assert_eq!(decls.len(), 1);

    if let ExternalDecl::Declaration(d) = &decls[0] {
        assert!(d.declarators[0].init.is_some());
    } else {
        panic!("Expected declaration");
    }
}

#[test]
fn test_struct_initializer() {
    let decls = parse("struct Point p = {1, 2};");
    assert_eq!(decls.len(), 1);
}

#[test]
fn test_if_statement() {
    let decls = parse("void foo(void) { if (1) { } }");
    assert_eq!(decls.len(), 1);
}

#[test]
fn test_if_else_statement() {
    let decls = parse("void foo(void) { if (1) { } else { } }");
    assert_eq!(decls.len(), 1);
}

#[test]
fn test_while_statement() {
    let decls = parse("void foo(void) { while (1) { } }");
    assert_eq!(decls.len(), 1);
}

#[test]
fn test_for_statement() {
    let decls = parse("void foo(void) { for (int i = 0; i < 10; i++) { } }");
    assert_eq!(decls.len(), 1);
}

#[test]
fn test_do_while_statement() {
    let decls = parse("void foo(void) { do { } while (1); }");
    assert_eq!(decls.len(), 1);
}

#[test]
fn test_switch_statement() {
    let decls = parse("void foo(int x) { switch (x) { case 1: break; default: break; } }");
    assert_eq!(decls.len(), 1);
}

#[test]
fn test_return_statement() {
    let decls = parse("int foo(void) { return 42; }");
    assert_eq!(decls.len(), 1);
}

#[test]
fn test_break_continue() {
    let decls = parse("void foo(void) { while(1) { break; continue; } }");
    assert_eq!(decls.len(), 1);
}

#[test]
fn test_goto_label() {
    let decls = parse("void foo(void) { start: goto start; }");
    assert_eq!(decls.len(), 1);
}

#[test]
fn test_attribute_on_function() {
    let decls = parse("void foo(void) __attribute__((unused));");
    assert_eq!(decls.len(), 1);
}

#[test]
fn test_attribute_on_parameter() {
    let decls = parse("void foo(int x __attribute__((unused)));");
    assert_eq!(decls.len(), 1);
}

#[test]
fn test_attribute_on_variable() {
    let decls = parse("int x __attribute__((unused));");
    assert_eq!(decls.len(), 1);
}

#[test]
fn test_attribute_on_struct() {
    let decls = parse("struct __attribute__((packed)) S { int x; };");
    assert_eq!(decls.len(), 1);
}

#[test]
fn test_typeof() {
    let decls = parse("__typeof__(1 + 2) x;");
    assert_eq!(decls.len(), 1);
}

#[test]
fn test_statement_expression() {
    let decls = parse("int x = ({ int y = 1; y + 1; });");
    assert_eq!(decls.len(), 1);
}

#[test]
fn test_bool_type() {
    let decls = parse("_Bool b;");
    assert_eq!(decls.len(), 1);
    assert!(has_type_spec(&decls[0], |t| matches!(t, TypeSpec::Bool)));
}

#[test]
fn test_complex_type() {
    let decls = parse("_Complex double z;");
    assert_eq!(decls.len(), 1);
    assert!(has_type_spec(&decls[0], |t| matches!(t, TypeSpec::Complex)));
}

#[test]
fn test_int128() {
    let decls = parse("__int128 big;");
    assert_eq!(decls.len(), 1);
    assert!(has_type_spec(&decls[0], |t| matches!(t, TypeSpec::Int128)));
}

#[test]
fn test_extension_keyword() {
    let decls = parse("__extension__ int x;");
    assert_eq!(decls.len(), 1);
}

#[test]
fn test_asm_in_function() {
    let decls = parse("void foo(void) { __asm__(\"nop\"); }");
    assert_eq!(decls.len(), 1);
}

#[test]
fn test_local_variable_with_attribute() {
    let decls = parse("void foo(void) { int x __attribute__((unused)) = 1; }");
    assert_eq!(decls.len(), 1);
}

// ==================== Macro Marker Handling Tests ====================

/// Helper to parse with macro markers enabled
fn parse_with_markers(source: &str) -> Vec<ExternalDecl> {
    let mut file = NamedTempFile::new().unwrap();
    file.write_all(source.as_bytes()).unwrap();
    file.flush().unwrap();

    let config = PPConfig {
        include_paths: vec![],
        predefined: vec![],
        debug_pp: false,
        target_dir: None,
        emit_markers: true,  // Enable marker emission
        ..Default::default()
    };

    let mut pp = Preprocessor::new(config);
    pp.add_source_file(file.path()).unwrap();

    let mut parser = Parser::new(&mut pp).unwrap();
    parser.set_handle_macro_markers(true).unwrap();  // Enable marker handling
    let tu = parser.parse().unwrap();
    tu.decls
}

#[test]
fn test_parser_with_markers_simple() {
    // Test that parser works normally with markers enabled but no macros
    let decls = parse_with_markers("int x;");
    assert_eq!(decls.len(), 1);
}

#[test]
fn test_parser_with_markers_object_macro() {
    // Test that parser handles object macro expansion markers
    let decls = parse_with_markers("#define VALUE 42\nint x = VALUE;");
    assert_eq!(decls.len(), 1);
}

#[test]
fn test_parser_with_markers_function_macro() {
    // Test that parser handles function macro expansion markers
    let decls = parse_with_markers("#define ADD(a, b) ((a) + (b))\nint x = ADD(1, 2);");
    assert_eq!(decls.len(), 1);
}

#[test]
fn test_parser_with_markers_nested_macro() {
    // Test that parser handles nested macro expansion markers
    let decls = parse_with_markers(
        "#define A B\n#define B 123\nint x = A;"
    );
    assert_eq!(decls.len(), 1);
}

#[test]
fn test_parser_with_markers_multiple_decls() {
    // Test that parser handles multiple declarations with macros
    let decls = parse_with_markers(
        "#define TYPE int\nTYPE a;\nTYPE b;\nTYPE c;"
    );
    assert_eq!(decls.len(), 3);
}

#[test]
fn test_parser_with_markers_function_body() {
    // Test that parser handles macros in function bodies
    let decls = parse_with_markers(
        "#define RETURN_ZERO return 0\nvoid foo(void) { RETURN_ZERO; }"
    );
    assert_eq!(decls.len(), 1);
}

#[test]
fn test_parser_marker_state_not_leaked() {
    // Test that macro context is properly cleaned up between parses
    // (markers should always balance: MacroBegin followed by MacroEnd)
    let decls = parse_with_markers("#define M(x) (x)\nint a = M(1);\nint b = M(2);\nint c = M(3);");
    assert_eq!(decls.len(), 3);
}

/// 先頭宣言の初期化子式を取り出すヘルパ (_Generic テスト用)
fn init_expr(decls: &[ExternalDecl]) -> &Expr {
    match &decls[0] {
        ExternalDecl::Declaration(d) => match &d.declarators[0].init {
            Some(Initializer::Expr(e)) => e,
            other => panic!("unexpected initializer: {:?}", other),
        },
        other => panic!("unexpected decl: {:?}", other),
    }
}

#[test]
fn test_generic_selection_picks_default_branch() {
    // glibc __glibc_const_generic (C23 const 修飾ジェネリック) と同型
    let decls = parse(
        "int x = _Generic(0 ? (const char *)0 : (void *)1, \
         const void *: 1, default: 2);",
    );
    assert_eq!(decls.len(), 1);
    assert!(matches!(&init_expr(&decls).kind, ExprKind::IntLit(2)));
}

#[test]
fn test_generic_selection_no_default_picks_first() {
    let decls = parse("int x = _Generic(1, int: 10, long: 20);");
    assert_eq!(decls.len(), 1);
    assert!(matches!(&init_expr(&decls).kind, ExprKind::IntLit(10)));
}

#[test]
fn test_generic_selection_default_in_first_position() {
    let decls = parse("int x = _Generic(1, default: 30, int: 40);");
    assert_eq!(decls.len(), 1);
    assert!(matches!(&init_expr(&decls).kind, ExprKind::IntLit(30)));
}

#[test]
fn test_generic_selection_in_function_body() {
    // glibc の memchr マクロ展開が関数本体中に現れるケース
    let decls = parse(
        "void *f(const void *p, int c, unsigned long n) {\
           return _Generic(0 ? (p) : (void *) 1, \
                           const void *: (const void *) (p), \
                           default: (void *) (p)); }",
    );
    assert_eq!(decls.len(), 1);
    assert!(matches!(decls[0], ExternalDecl::FunctionDef(_)));
}

/// パースがエラーになることを確認するヘルパ (隣接文字列連結の負のテスト用)
fn parse_fails(source: &str) -> bool {
    let mut file = NamedTempFile::new().unwrap();
    file.write_all(source.as_bytes()).unwrap();
    file.flush().unwrap();

    let config = PPConfig {
        include_paths: vec![],
        predefined: vec![],
        debug_pp: false,
        target_dir: None,
        ..Default::default()
    };

    let mut pp = Preprocessor::new(config);
    pp.add_source_file(file.path()).unwrap();

    let mut parser = Parser::new(&mut pp).unwrap();
    parser.parse().is_err()
}

#[test]
fn test_adjacent_string_literal_concat() {
    // 純リテラル同士の隣接連結 (TinyCC parse_mult_str と同方針、従来挙動の不変確認)
    let decls = parse(r#"char x[] = "abc" "def";"#);
    match &init_expr(&decls).kind {
        ExprKind::StringLit(bytes) => assert_eq!(bytes, b"abcdef"),
        other => panic!("expected StringLit, got {:?}", other),
    }
}

#[test]
fn test_empty_string_concat_with_ident_reduces_to_ident() {
    // STR_WITH_LEN / ASSERT_IS_LITERAL の `("" s "")` イディオム
    // (literal-only assert)。空リテラルとの連結は意味的に s そのもの
    let decls = parse(r#"const char *x = ("" s "");"#);
    match &init_expr(&decls).kind {
        ExprKind::Ident(_) => {}
        other => panic!("expected Ident, got {:?}", other),
    }
}

#[test]
fn test_string_concat_mixing_ident_and_nonempty_literal_is_error() {
    // 非空リテラルと仮引数の連結は未対応 (従来どおりエラー)
    assert!(parse_fails(r#"const char *x = ("abc" s);"#));
    // Ident 2 個以上も同様
    assert!(parse_fails(r#"const char *x = ("" a b "");"#));
}
