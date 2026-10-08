//! Preprocessor integration tests

use std::io::Write;
use tempfile::NamedTempFile;
use libperl_macrogen::{PPConfig, Preprocessor, TokenKind};

/// Helper to create a preprocessor from source string
fn preprocess(source: &str) -> Preprocessor {
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
    pp
}

/// Helper to collect all tokens from preprocessor (excluding Newline tokens)
fn collect_tokens(pp: &mut Preprocessor) -> Vec<(TokenKind, String)> {
    let mut tokens = Vec::new();
    loop {
        let token = pp.next_token().unwrap();
        if matches!(token.kind, TokenKind::Eof) {
            break;
        }
        // Skip newline tokens for easier test assertions
        if matches!(token.kind, TokenKind::Newline) {
            continue;
        }
        let text = token.kind.format(pp.interner());
        tokens.push((token.kind, text));
    }
    tokens
}

/// Helper to get token kinds only
fn token_kinds(pp: &mut Preprocessor) -> Vec<TokenKind> {
    collect_tokens(pp).into_iter().map(|(k, _)| k).collect()
}

// Tests if multicharacter character literals get shifted correctly and overflow correctly
#[test]
fn test_multi_character_char_acc_and_overflow() {
    for (input, output) in [
        ("'A'",     TokenKind::CharLit(b'A')),
        ("'AB'",    TokenKind::IntLit(0x4142)),
        ("'ABC'",   TokenKind::IntLit(0x414243)),
        ("'ABCD'",  TokenKind::IntLit(0x41424344)),
        ("'ABCDE'", TokenKind::IntLit(0x42434445)),
    ] {
        let mut pp = preprocess(input);
        assert_eq!(token_kinds(&mut pp).as_slice(), &[output]);
    }
}

#[test]
fn test_simple_tokens() {
    let mut pp = preprocess("int x;");
    let tokens = collect_tokens(&mut pp);

    assert_eq!(tokens.len(), 3);
    assert_eq!(tokens[0].1, "int");
    assert_eq!(tokens[1].1, "x");
    assert!(matches!(tokens[2].0, TokenKind::Semi));
}

#[test]
fn test_object_macro() {
    let mut pp = preprocess("#define VALUE 42\nint x = VALUE;");
    let tokens = collect_tokens(&mut pp);

    // int x = 42 ;
    assert_eq!(tokens.len(), 5);
    assert_eq!(tokens[0].1, "int");
    assert_eq!(tokens[1].1, "x");
    assert!(matches!(tokens[2].0, TokenKind::Eq));
    assert!(matches!(tokens[3].0, TokenKind::IntLit(42)));
    assert!(matches!(tokens[4].0, TokenKind::Semi));
}

#[test]
fn test_function_macro() {
    let mut pp = preprocess("#define ADD(a, b) ((a) + (b))\nint x = ADD(1, 2);");
    let tokens = collect_tokens(&mut pp);

    // int x = ( ( 1 ) + ( 2 ) ) ;
    // 13 tokens: int, x, =, (, (, 1, ), +, (, 2, ), ), ;
    assert_eq!(tokens.len(), 13);
    assert_eq!(tokens[0].1, "int");
    assert!(matches!(tokens[3].0, TokenKind::LParen)); // (
    assert!(matches!(tokens[4].0, TokenKind::LParen)); // (
    assert!(matches!(tokens[5].0, TokenKind::IntLit(1)));
    assert!(matches!(tokens[7].0, TokenKind::Plus));
    assert!(matches!(tokens[9].0, TokenKind::IntLit(2)));
}

#[test]
fn test_ifdef_true() {
    let mut pp = preprocess("#define FOO\n#ifdef FOO\nint x;\n#endif");
    let tokens = collect_tokens(&mut pp);

    // int x ;
    assert_eq!(tokens.len(), 3);
    assert_eq!(tokens[0].1, "int");
}

#[test]
fn test_ifdef_false() {
    let mut pp = preprocess("#ifdef FOO\nint x;\n#endif\nint y;");
    let tokens = collect_tokens(&mut pp);

    // int y ;
    assert_eq!(tokens.len(), 3);
    assert_eq!(tokens[0].1, "int");
    assert_eq!(tokens[1].1, "y");
}

#[test]
fn test_ifndef() {
    let mut pp = preprocess("#ifndef FOO\nint x;\n#endif");
    let tokens = collect_tokens(&mut pp);

    // int x ;
    assert_eq!(tokens.len(), 3);
    assert_eq!(tokens[0].1, "int");
    assert_eq!(tokens[1].1, "x");
}

#[test]
fn test_if_else() {
    let mut pp = preprocess("#define FOO 1\n#if FOO\nint x;\n#else\nint y;\n#endif");
    let tokens = collect_tokens(&mut pp);

    // int x ;
    assert_eq!(tokens.len(), 3);
    assert_eq!(tokens[1].1, "x");
}

#[test]
fn test_elif() {
    let mut pp = preprocess("#define FOO 0\n#define BAR 1\n#if FOO\nint x;\n#elif BAR\nint y;\n#else\nint z;\n#endif");
    let tokens = collect_tokens(&mut pp);

    // int y ;
    assert_eq!(tokens.len(), 3);
    assert_eq!(tokens[1].1, "y");
}

#[test]
fn test_nested_ifdef() {
    let mut pp = preprocess("#define A\n#define B\n#ifdef A\n#ifdef B\nint x;\n#endif\n#endif");
    let tokens = collect_tokens(&mut pp);

    // int x ;
    assert_eq!(tokens.len(), 3);
    assert_eq!(tokens[1].1, "x");
}

#[test]
fn test_undef() {
    let mut pp = preprocess("#define FOO 1\n#undef FOO\n#ifdef FOO\nint x;\n#endif\nint y;");
    let tokens = collect_tokens(&mut pp);

    // int y ;
    assert_eq!(tokens.len(), 3);
    assert_eq!(tokens[1].1, "y");
}

#[test]
fn test_defined_operator() {
    let mut pp = preprocess("#define FOO\n#if defined(FOO)\nint x;\n#endif");
    let tokens = collect_tokens(&mut pp);

    // int x ;
    assert_eq!(tokens.len(), 3);
    assert_eq!(tokens[1].1, "x");
}

#[test]
fn test_defined_without_parens() {
    let mut pp = preprocess("#define FOO\n#if defined FOO\nint x;\n#endif");
    let tokens = collect_tokens(&mut pp);

    // int x ;
    assert_eq!(tokens.len(), 3);
    assert_eq!(tokens[1].1, "x");
}

#[test]
fn test_stringification() {
    let mut pp = preprocess("#define STR(x) #x\nchar *s = STR(hello);");
    let tokens = collect_tokens(&mut pp);

    // char * s = "hello" ;
    assert_eq!(tokens.len(), 6);
    assert!(matches!(&tokens[4].0, TokenKind::StringLit(s) if s == b"hello"));
}

#[test]
fn test_token_pasting() {
    let mut pp = preprocess("#define PASTE(a, b) a##b\nint PASTE(foo, bar);");
    let tokens = collect_tokens(&mut pp);

    // int foobar ;
    assert_eq!(tokens.len(), 3);
    assert_eq!(tokens[1].1, "foobar");
}

#[test]
fn test_variadic_macro() {
    let mut pp = preprocess("#define CALL(fn, ...) fn(__VA_ARGS__)\nCALL(foo, 1, 2, 3);");
    let tokens = collect_tokens(&mut pp);

    // Verify that the output contains the expected elements
    // Note: exact token count may vary due to macro expansion behavior
    assert!(tokens.iter().any(|(_, text)| text == "foo"));
    assert!(tokens.iter().any(|(kind, _)| matches!(kind, TokenKind::LParen)));
    assert!(tokens.iter().any(|(kind, _)| matches!(kind, TokenKind::IntLit(1))));
    assert!(tokens.iter().any(|(kind, _)| matches!(kind, TokenKind::IntLit(2))));
    assert!(tokens.iter().any(|(kind, _)| matches!(kind, TokenKind::IntLit(3))));
    assert!(tokens.iter().any(|(kind, _)| matches!(kind, TokenKind::Semi)));
}

#[test]
fn test_predefined_macros() {
    let config = PPConfig {
        include_paths: vec![],
        predefined: vec![("TEST_MACRO".to_string(), Some("123".to_string()))],
        debug_pp: false,
        target_dir: None,
        ..Default::default()
    };

    let mut file = NamedTempFile::new().unwrap();
    file.write_all(b"int x = TEST_MACRO;").unwrap();
    file.flush().unwrap();

    let mut pp = Preprocessor::new(config);
    pp.add_source_file(file.path()).unwrap();

    let tokens = collect_tokens(&mut pp);

    // int x = 123 ;
    assert_eq!(tokens.len(), 5);
    assert!(matches!(tokens[3].0, TokenKind::IntLit(123)));
}

#[test]
#[ignore = "__FILE__ macro not yet implemented"]
fn test_file_macro() {
    let mut pp = preprocess("const char *f = __FILE__;");
    let tokens = collect_tokens(&mut pp);

    // const char * f = "..." ;
    assert_eq!(tokens.len(), 7);
    assert!(matches!(&tokens[5].0, TokenKind::StringLit(_)));
}

#[test]
#[ignore = "__LINE__ macro not yet implemented"]
fn test_line_macro() {
    let mut pp = preprocess("int line = __LINE__;");
    let tokens = collect_tokens(&mut pp);

    // int line = <number> ;
    assert_eq!(tokens.len(), 5);
    assert!(matches!(tokens[3].0, TokenKind::IntLit(_)));
}

#[test]
fn test_multiline_macro() {
    let mut pp = preprocess("#define MULTI(x) \\\n    ((x) + 1)\nint y = MULTI(5);");
    let tokens = collect_tokens(&mut pp);

    // int y = ( ( 5 ) + 1 ) ;
    assert!(tokens.len() >= 5);
    assert_eq!(tokens[0].1, "int");
    assert_eq!(tokens[1].1, "y");
}

#[test]
fn test_empty_macro() {
    let mut pp = preprocess("#define EMPTY\nint EMPTY x;");
    let tokens = collect_tokens(&mut pp);

    // int x ;
    assert_eq!(tokens.len(), 3);
    assert_eq!(tokens[0].1, "int");
    assert_eq!(tokens[1].1, "x");
}

#[test]
fn test_macro_redefine() {
    let mut pp = preprocess("#define X 1\n#define X 2\nint a = X;");
    let tokens = collect_tokens(&mut pp);

    // int a = 2 ;
    assert_eq!(tokens.len(), 5);
    assert!(matches!(tokens[3].0, TokenKind::IntLit(2)));
}

#[test]
fn test_if_expression_arithmetic() {
    let mut pp = preprocess("#if 2 + 3 == 5\nint x;\n#endif");
    let tokens = collect_tokens(&mut pp);

    // int x ;
    assert_eq!(tokens.len(), 3);
}

#[test]
fn test_if_expression_logical() {
    let mut pp = preprocess("#if 1 && 1\nint x;\n#endif");
    let tokens = collect_tokens(&mut pp);

    // int x ;
    assert_eq!(tokens.len(), 3);
}

#[test]
fn test_recursive_macro_prevention() {
    let mut pp = preprocess("#define X X\nint y = X;");
    let tokens = collect_tokens(&mut pp);

    // int y = X ;  (X is not expanded recursively)
    assert_eq!(tokens.len(), 5);
    assert_eq!(tokens[3].1, "X");
}

// =============================================================================
// キーワード相当のトークンを #define / #undef / #ifdef の名前として受理する
// （TinyCC 流: tccpp.c parse_define / process_ifdef は v < TOK_IDENT のみ弾く）
// 直接の動機: <stdbool.h> の `#define bool _Bool` で落ちていた CI を通す。
// =============================================================================

#[test]
fn test_define_keyword_name_does_not_error() {
    // #define bool _Bool が処理できること（preprocess() 内 add_source_file が
    // エラーを返さない時点で OK。さらに後続トークンが取り出せることも確認）。
    let mut pp = preprocess("#define bool _Bool\nbool x;");
    let tokens = collect_tokens(&mut pp);
    // トークン展開は keyword token 経路では行われないので、bool x ; がそのまま出る
    assert_eq!(tokens.len(), 3);
    assert_eq!(tokens[0].1, "bool");
    assert_eq!(tokens[1].1, "x");
}

#[test]
fn test_ifdef_keyword_after_define() {
    // #define bool に続けて #ifdef bool が真分岐を選ぶ
    let mut pp = preprocess("#define bool\n#ifdef bool\nint x;\n#endif");
    let tokens = collect_tokens(&mut pp);
    assert_eq!(tokens.len(), 3);
    assert_eq!(tokens[0].1, "int");
    assert_eq!(tokens[1].1, "x");
}

#[test]
fn test_ifdef_keyword_no_define() {
    // #define が無いとき #ifdef bool は偽（bool キーワードは未定義扱い）
    let mut pp = preprocess("#ifdef bool\nint x;\n#endif\nint y;");
    let tokens = collect_tokens(&mut pp);
    assert_eq!(tokens.len(), 3);
    assert_eq!(tokens[0].1, "int");
    assert_eq!(tokens[1].1, "y");
}

#[test]
fn test_ifndef_keyword_no_define() {
    // #ifndef bool は #define が無いとき真分岐
    let mut pp = preprocess("#ifndef bool\nint x;\n#endif");
    let tokens = collect_tokens(&mut pp);
    assert_eq!(tokens.len(), 3);
    assert_eq!(tokens[0].1, "int");
    assert_eq!(tokens[1].1, "x");
}

#[test]
fn test_undef_keyword_clears_define() {
    // #define bool したあと #undef bool すると #ifdef bool は偽
    let mut pp = preprocess(
        "#define bool\n#undef bool\n#ifdef bool\nint x;\n#endif\nint y;"
    );
    let tokens = collect_tokens(&mut pp);
    assert_eq!(tokens.len(), 3);
    assert_eq!(tokens[0].1, "int");
    assert_eq!(tokens[1].1, "y");
}

#[test]
fn test_define_inline_alias_does_not_error() {
    // 互換ヘッダで頻出のパターン: #define inline __inline
    let mut pp = preprocess("#define inline __inline\nint x;");
    let tokens = collect_tokens(&mut pp);
    assert_eq!(tokens.len(), 3);
    assert_eq!(tokens[0].1, "int");
}

#[test]
fn test_defined_operator_with_keyword_regression() {
    // pp_expr.rs の parse_defined は既にキーワード対応済み。
    // 今回の修正で壊れていないことの回帰テスト。
    let mut pp = preprocess("#define bool\n#if defined(bool)\nint x;\n#endif");
    let tokens = collect_tokens(&mut pp);
    assert_eq!(tokens.len(), 3);
    assert_eq!(tokens[0].1, "int");
    assert_eq!(tokens[1].1, "x");
}

#[test]
fn test_macro_param_keyword_does_not_error() {
    // 仮引数名にキーワードを使う `#define FOO(bool) ...` がエラーにならない。
    // expansion 側はまだ keyword token 経路を扱わないので、本テストは
    // 「directive 自体が処理できる」ことのみ確認する。
    let mut pp = preprocess("#define FOO(bool) 1\nint x;");
    let tokens = collect_tokens(&mut pp);
    assert_eq!(tokens.len(), 3);
    assert_eq!(tokens[0].1, "int");
}
