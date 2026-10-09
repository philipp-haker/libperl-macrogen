//! C言語パーサー
//!
//! tinyccのparser部分に相当。再帰下降パーサーで実装。

use std::collections::{HashMap, HashSet};

use crate::ast::*;
use crate::error::{CompileError, ParseError, Result};
use crate::intern::{InternedStr, StringInterner};
use crate::macro_infer::detect_assert_kind;
use crate::preprocessor::Preprocessor;
use crate::lexer::{Lexer, LookupOnly};
use crate::source::{FileId, SourceLocation};
use crate::token::{MacroBeginInfo, MacroInvocationKind, Token, TokenId, TokenKind};
use crate::token_source::{TokenSliceRef, TokenSource};

/// マクロ展開コンテキスト
///
/// パース中のマクロ展開状態を追跡する。
/// MacroBegin マーカーを見つけたらプッシュし、MacroEnd を見つけたらポップする。
#[derive(Debug, Default)]
pub struct MacroContext {
    /// 現在のマクロ展開スタック（外側から内側へ）
    stack: Vec<MacroBeginInfo>,
}

impl MacroContext {
    /// 新しいコンテキストを作成
    pub fn new() -> Self {
        Self { stack: Vec::new() }
    }

    /// マクロ展開を開始
    pub fn push(&mut self, info: MacroBeginInfo) {
        self.stack.push(info);
    }

    /// マクロ展開を終了
    pub fn pop(&mut self) -> Option<MacroBeginInfo> {
        self.stack.pop()
    }

    /// 現在マクロ展開中かどうか
    pub fn is_in_macro(&self) -> bool {
        !self.stack.is_empty()
    }

    /// 現在のマクロ展開情報から MacroExpansionInfo を構築
    pub fn build_macro_info(&self, interner: &StringInterner) -> Option<MacroExpansionInfo> {
        if self.stack.is_empty() {
            return None;
        }

        let mut info = MacroExpansionInfo::new();
        for begin_info in &self.stack {
            let args = match &begin_info.kind {
                crate::token::MacroInvocationKind::Object => None,
                crate::token::MacroInvocationKind::Function { args } => {
                    // トークン列を文字列に変換
                    Some(args.iter().map(|arg_tokens| {
                        arg_tokens.iter()
                            .map(|t| t.kind.format(interner))
                            .collect::<Vec<_>>()
                            .join(" ")
                    }).collect())
                }
            };
            info.push(MacroInvocation {
                name: begin_info.macro_name,
                call_loc: begin_info.call_loc.clone(),
                args,
            });
        }
        Some(info)
    }

    /// 展開スタックの深さ
    pub fn depth(&self) -> usize {
        self.stack.len()
    }
}

/// パーサー
///
/// 汎用のトークンソースからC言語をパースする。
/// `S` は `TokenSource` トレイトを実装する任意の型。
pub struct Parser<'a, S: TokenSource> {
    source: &'a mut S,
    current: Token,
    /// typedef名のセット
    typedefs: HashSet<InternedStr>,
    /// マクロ展開コンテキスト
    macro_ctx: MacroContext,
    /// マクロマーカーを処理するか（emit_markers=true の場合に true にする）
    handle_macro_markers: bool,
    /// do-while 文の末尾セミコロンを省略可能にするフラグ
    allow_missing_semi: bool,
    /// パース中に検出した関数呼び出しの数
    pub function_call_count: usize,
    /// パース中に検出したポインタデリファレンスの数
    pub deref_count: usize,
    /// マクロ仮引数の辞書（マクロ本体パース時のみ使用）
    /// key: 仮引数名, value: パラメータインデックス
    generic_params: HashMap<InternedStr, usize>,
    /// パース中に型として使用が検出された generic param
    detected_type_params: HashSet<InternedStr>,
}

/// Preprocessor 専用の後方互換コンストラクタ
impl<'a> Parser<'a, Preprocessor> {
    /// 新しいパーサーを作成（Preprocessor専用）
    pub fn new(pp: &'a mut Preprocessor) -> Result<Self> {
        Self::from_source(pp)
    }

    /// ストリーミング形式でパース
    ///
    /// 各宣言をパースするたびにコールバックを呼び出す。
    /// パースエラーが発生した場合はコールバックを呼ばずにエラーを返す。
    /// コールバックが `ControlFlow::Break(())` を返した場合はループを終了。
    pub fn parse_each<F>(&mut self, mut callback: F) -> Result<()>
    where
        F: FnMut(&ExternalDecl, &crate::source::SourceLocation, &std::path::Path, &StringInterner) -> std::ops::ControlFlow<()>,
    {
        while !self.is_eof() {
            let loc = self.current.loc.clone();
            let decl = self.parse_external_decl()?;
            let path = self.source.files().get_path(loc.file_id);
            let interner = self.source.interner();
            if callback(&decl, &loc, path, interner).is_break() {
                break;
            }
        }
        Ok(())
    }

    /// ストリーミング形式でパース（Preprocessor アクセス付き）
    ///
    /// `parse_each` と同様だが、コールバックに Preprocessor への可変参照も渡す。
    /// マクロ呼び出しコールバック（MacroCallWatcher など）にアクセスする場合に使用。
    /// パースエラーが発生した場合はコールバックを呼ばずにエラーを返す。
    pub fn parse_each_with_pp<F>(&mut self, mut callback: F) -> Result<()>
    where
        F: FnMut(&ExternalDecl, &crate::source::SourceLocation, &std::path::Path, &mut Preprocessor) -> std::ops::ControlFlow<()>,
    {
        while !self.is_eof() {
            let loc = self.current.loc.clone();
            let decl = self.parse_external_decl()?;
            let path = self.source.files().get_path(loc.file_id).to_path_buf();
            if callback(&decl, &loc, &path, self.source).is_break() {
                break;
            }
        }
        Ok(())
    }
}

/// 汎用のトークンソースに対するパーサー実装
impl<'a, S: TokenSource> Parser<'a, S> {
    /// トークンソースからパーサーを作成
    pub fn from_source(source: &'a mut S) -> Result<Self> {
        // GCC builtin types を事前登録
        let mut typedefs = HashSet::new();
        typedefs.insert(source.interner_mut().intern("__builtin_va_list"));
        // GCC 13 bfloat16 (x86 AVX512-BF16 intrinsics, e.g. avx512bf16vlintrin.h)
        typedefs.insert(source.interner_mut().intern("__bf16"));

        let mut parser = Self {
            source,
            current: Token::default(),
            typedefs,
            macro_ctx: MacroContext::new(),
            handle_macro_markers: false,
            allow_missing_semi: false,
            function_call_count: 0,
            deref_count: 0,
            generic_params: HashMap::new(),
            detected_type_params: HashSet::new(),
        };
        // マーカーをスキップして最初のトークンを取得
        parser.current = parser.inner_next_token()?;

        Ok(parser)
    }

    /// トークンソースからパーサーを作成（既存のtypedef情報を引き継ぐ）
    pub fn from_source_with_typedefs(source: &'a mut S, typedefs: HashSet<InternedStr>) -> Result<Self> {
        let mut parser = Self {
            source,
            current: Token::default(),
            typedefs,
            macro_ctx: MacroContext::new(),
            handle_macro_markers: false,
            allow_missing_semi: false,
            function_call_count: 0,
            deref_count: 0,
            generic_params: HashMap::new(),
            detected_type_params: HashSet::new(),
        };
        // マーカーをスキップして最初のトークンを取得
        parser.current = parser.inner_next_token()?;

        Ok(parser)
    }

    /// マクロマーカー処理を有効にする
    ///
    /// Note: 既に current にマーカートークンがある場合はスキップする
    pub fn set_handle_macro_markers(&mut self, enabled: bool) -> Result<()> {
        self.handle_macro_markers = enabled;

        // 既に current にマーカートークンがある場合はスキップ
        if enabled {
            while matches!(
                self.current.kind,
                TokenKind::MacroBegin(_) | TokenKind::MacroEnd(_)
            ) {
                match &self.current.kind {
                    TokenKind::MacroBegin(info) => {
                        self.macro_ctx.push((**info).clone());
                    }
                    TokenKind::MacroEnd(_) => {
                        self.macro_ctx.pop();
                    }
                    _ => {}
                }
                self.current = self.source.next_token()?;
            }
        }
        Ok(())
    }

    /// StringInterner への参照を取得
    pub fn interner(&self) -> &crate::intern::StringInterner {
        self.source.interner()
    }

    /// typedef名のセットを取得
    pub fn typedefs(&self) -> &HashSet<InternedStr> {
        &self.typedefs
    }

    /// 翻訳単位をパース
    pub fn parse(&mut self) -> Result<TranslationUnit> {
        let mut decls = Vec::new();

        while !self.is_eof() {
            let decl = self.parse_external_decl()?;
            decls.push(decl);
        }

        Ok(TranslationUnit { decls })
    }

    /// 式のみをパース
    ///
    /// マクロ本体など、式だけをパースしたい場合に使用
    pub fn parse_expr_only(&mut self) -> Result<Expr> {
        self.parse_expr()
    }

    /// 文をパース（末尾セミコロン省略可能）
    ///
    /// マクロ body のパースなど、do-while の末尾セミコロンが
    /// 省略されている場合に使用する。
    pub fn parse_stmt_allow_missing_semi(&mut self) -> Result<Stmt> {
        self.allow_missing_semi = true;
        let result = self.parse_stmt();
        self.allow_missing_semi = false;
        result
    }

    /// 外部宣言をパース
    fn parse_external_decl(&mut self) -> Result<ExternalDecl> {
        let comments = self.current.leading_comments.clone();
        let loc = self.current.loc.clone();
        let is_target = self.source.is_file_in_target(loc.file_id);

        // 宣言指定子をパース
        let specs = self.parse_decl_specs()?;

        // ; のみの場合（構造体宣言など）
        if self.check(&TokenKind::Semi) {
            self.advance()?;
            return Ok(ExternalDecl::Declaration(Declaration {
                specs,
                declarators: Vec::new(),
                info: NodeInfo::new(loc),
                comments,
                is_target,
            }));
        }

        // 宣言子をパース
        let declarator = self.parse_declarator()?;

        // __attribute__ をスキップ
        self.try_skip_attribute()?;

        // 関数定義かどうかを判定
        // 関数定義: 宣言子の後に { が来る
        if self.check(&TokenKind::LBrace) {
            // 本体パース前のカウントを記録
            let call_count_before = self.function_call_count;
            let deref_count_before = self.deref_count;

            let body = self.parse_compound_stmt()?;

            // 差分が関数本体のカウント
            let function_call_count = self.function_call_count - call_count_before;
            let deref_count = self.deref_count - deref_count_before;

            return Ok(ExternalDecl::FunctionDef(FunctionDef {
                specs,
                declarator,
                body,
                info: NodeInfo::new(loc),
                comments,
                is_target,
                function_call_count,
                deref_count,
            }));
        }

        // 宣言の続きをパース
        let mut declarators = Vec::new();

        // 最初の宣言子（初期化子あり）
        let init = if self.check(&TokenKind::Eq) {
            self.advance()?;
            Some(self.parse_initializer()?)
        } else {
            None
        };
        declarators.push(InitDeclarator { declarator, init });

        // 追加の宣言子
        while self.check(&TokenKind::Comma) {
            self.advance()?;
            let declarator = self.parse_declarator()?;
            let init = if self.check(&TokenKind::Eq) {
                self.advance()?;
                Some(self.parse_initializer()?)
            } else {
                None
            };
            declarators.push(InitDeclarator { declarator, init });
        }

        // GCC拡張: 宣言の最後の __attribute__((...)) をスキップ
        self.try_skip_attribute()?;

        self.expect(&TokenKind::Semi)?;

        // typedef の場合、名前を登録
        if specs.storage == Some(StorageClass::Typedef) {
            for d in &declarators {
                if let Some(name) = d.declarator.name {
                    self.typedefs.insert(name);
                }
            }
        }

        Ok(ExternalDecl::Declaration(Declaration {
            specs,
            declarators,
            info: NodeInfo::new(loc),
            comments,
            is_target,
        }))
    }

    /// 宣言指定子をパース
    fn parse_decl_specs(&mut self) -> Result<DeclSpecs> {
        let mut specs = DeclSpecs::default();

        loop {
            match &self.current.kind {
                // GCC拡張: __extension__ は無視（TinyCC方式）
                TokenKind::KwExtension => {
                    self.advance()?;
                    continue;
                }
                // C11/GCC: _Thread_local, __thread は無視（TinyCC方式）
                TokenKind::KwThreadLocal | TokenKind::KwThread => {
                    self.advance()?;
                    continue;
                }
                // ストレージクラス
                TokenKind::KwTypedef => {
                    specs.storage = Some(StorageClass::Typedef);
                    self.advance()?;
                }
                TokenKind::KwExtern => {
                    specs.storage = Some(StorageClass::Extern);
                    self.advance()?;
                }
                TokenKind::KwStatic => {
                    specs.storage = Some(StorageClass::Static);
                    self.advance()?;
                }
                TokenKind::KwAuto => {
                    specs.storage = Some(StorageClass::Auto);
                    self.advance()?;
                }
                TokenKind::KwRegister => {
                    specs.storage = Some(StorageClass::Register);
                    self.advance()?;
                }
                // inline
                TokenKind::KwInline | TokenKind::KwInline2 | TokenKind::KwInline3 => {
                    specs.is_inline = true;
                    self.advance()?;
                }
                // 型修飾子
                TokenKind::KwConst | TokenKind::KwConst2 | TokenKind::KwConst3 => {
                    specs.qualifiers.is_const = true;
                    self.advance()?;
                }
                TokenKind::KwVolatile | TokenKind::KwVolatile2 | TokenKind::KwVolatile3 => {
                    specs.qualifiers.is_volatile = true;
                    self.advance()?;
                }
                TokenKind::KwRestrict | TokenKind::KwRestrict2 | TokenKind::KwRestrict3 => {
                    specs.qualifiers.is_restrict = true;
                    self.advance()?;
                }
                TokenKind::KwAtomic => {
                    specs.qualifiers.is_atomic = true;
                    self.advance()?;
                }
                // 型指定子
                TokenKind::KwVoid => {
                    specs.type_specs.push(TypeSpec::Void);
                    self.advance()?;
                }
                TokenKind::KwChar => {
                    specs.type_specs.push(TypeSpec::Char);
                    self.advance()?;
                }
                TokenKind::KwShort => {
                    specs.type_specs.push(TypeSpec::Short);
                    self.advance()?;
                }
                TokenKind::KwInt => {
                    specs.type_specs.push(TypeSpec::Int);
                    self.advance()?;
                }
                TokenKind::KwLong => {
                    specs.type_specs.push(TypeSpec::Long);
                    self.advance()?;
                }
                TokenKind::KwFloat => {
                    specs.type_specs.push(TypeSpec::Float);
                    self.advance()?;
                }
                TokenKind::KwDouble => {
                    specs.type_specs.push(TypeSpec::Double);
                    self.advance()?;
                }
                TokenKind::KwSigned | TokenKind::KwSigned2 => {
                    specs.type_specs.push(TypeSpec::Signed);
                    self.advance()?;
                }
                TokenKind::KwUnsigned => {
                    specs.type_specs.push(TypeSpec::Unsigned);
                    self.advance()?;
                }
                TokenKind::KwBool | TokenKind::KwBool2 => {
                    specs.type_specs.push(TypeSpec::Bool);
                    self.advance()?;
                }
                TokenKind::KwComplex => {
                    specs.type_specs.push(TypeSpec::Complex);
                    self.advance()?;
                }
                // GCC拡張浮動小数点型
                TokenKind::KwFloat16 => {
                    specs.type_specs.push(TypeSpec::Float16);
                    self.advance()?;
                }
                TokenKind::KwFloat32 => {
                    specs.type_specs.push(TypeSpec::Float32);
                    self.advance()?;
                }
                TokenKind::KwFloat64 => {
                    specs.type_specs.push(TypeSpec::Float64);
                    self.advance()?;
                }
                TokenKind::KwFloat128 => {
                    specs.type_specs.push(TypeSpec::Float128);
                    self.advance()?;
                }
                TokenKind::KwFloat32x => {
                    specs.type_specs.push(TypeSpec::Float32x);
                    self.advance()?;
                }
                TokenKind::KwFloat64x => {
                    specs.type_specs.push(TypeSpec::Float64x);
                    self.advance()?;
                }
                // GCC拡張: 128ビット整数
                TokenKind::KwInt128 => {
                    specs.type_specs.push(TypeSpec::Int128);
                    self.advance()?;
                }
                // typeof
                TokenKind::KwTypeof | TokenKind::KwTypeof2 | TokenKind::KwTypeof3 => {
                    self.advance()?;
                    self.expect(&TokenKind::LParen)?;
                    let expr = self.parse_expr()?;
                    self.expect(&TokenKind::RParen)?;
                    specs.type_specs.push(TypeSpec::TypeofExpr(Box::new(expr)));
                }
                // 構造体・共用体・列挙
                TokenKind::KwStruct => {
                    specs.type_specs.push(self.parse_struct_or_union(true)?);
                }
                TokenKind::KwUnion => {
                    specs.type_specs.push(self.parse_struct_or_union(false)?);
                }
                TokenKind::KwEnum => {
                    specs.type_specs.push(self.parse_enum()?);
                }
                // GCC拡張: __attribute__((...)) をスキップ
                TokenKind::KwAttribute | TokenKind::KwAttribute2 => {
                    self.skip_attribute()?;
                }
                // typedef名 or 検出済みの generic 型パラメータ
                TokenKind::Ident(id) if self.typedefs.contains(id) || self.detected_type_params.contains(id) => {
                    let id = *id;
                    specs.type_specs.push(TypeSpec::TypedefName(id));
                    self.advance()?;
                }
                // それ以外はループ終了
                _ => break,
            }
        }

        Ok(specs)
    }

    /// 構造体/共用体をパース
    fn parse_struct_or_union(&mut self, is_struct: bool) -> Result<TypeSpec> {
        let loc = self.current.loc.clone();
        self.advance()?; // struct/union

        // GCC拡張: struct __attribute__((...)) name { ... }
        self.try_skip_attribute()?;

        // 名前（オプション）
        let name = self.current_ident();
        if name.is_some() {
            self.advance()?;
        }

        // メンバーリスト（オプション）
        let members = if self.check(&TokenKind::LBrace) {
            self.advance()?;
            let mut members = Vec::new();
            while !self.check(&TokenKind::RBrace) {
                members.push(self.parse_struct_member()?);
            }
            self.expect(&TokenKind::RBrace)?;
            Some(members)
        } else {
            None
        };

        let spec = StructSpec { name, members, loc };
        if is_struct {
            Ok(TypeSpec::Struct(spec))
        } else {
            Ok(TypeSpec::Union(spec))
        }
    }

    /// 構造体メンバーをパース
    fn parse_struct_member(&mut self) -> Result<StructMember> {
        let specs = self.parse_decl_specs()?;
        let mut declarators = Vec::new();

        loop {
            let declarator = if self.check(&TokenKind::Colon) {
                None
            } else if self.check(&TokenKind::Semi) {
                None
            } else {
                Some(self.parse_declarator()?)
            };

            // GCC拡張: 宣言子の後の __attribute__ をスキップ
            self.try_skip_attribute()?;

            let bitfield = if self.check(&TokenKind::Colon) {
                self.advance()?;
                Some(Box::new(self.parse_conditional_expr()?))
            } else {
                None
            };

            declarators.push(StructDeclarator { declarator, bitfield });

            if !self.check(&TokenKind::Comma) {
                break;
            }
            self.advance()?;
        }

        self.expect(&TokenKind::Semi)?;

        Ok(StructMember { specs, declarators })
    }

    /// 列挙型をパース
    fn parse_enum(&mut self) -> Result<TypeSpec> {
        let loc = self.current.loc.clone();
        self.advance()?; // enum

        // 名前（オプション）
        let name = self.current_ident();
        if name.is_some() {
            self.advance()?;
        }

        // 列挙子リスト（オプション）
        let enumerators = if self.check(&TokenKind::LBrace) {
            self.advance()?;
            let mut enums = Vec::new();
            while !self.check(&TokenKind::RBrace) {
                let eloc = self.current.loc.clone();
                let ename = self.expect_ident()?;
                let value = if self.check(&TokenKind::Eq) {
                    self.advance()?;
                    Some(Box::new(self.parse_conditional_expr()?))
                } else {
                    None
                };
                enums.push(Enumerator {
                    name: ename,
                    value,
                    loc: eloc,
                });
                if !self.check(&TokenKind::Comma) {
                    break;
                }
                self.advance()?;
            }
            self.expect(&TokenKind::RBrace)?;
            Some(enums)
        } else {
            None
        };

        Ok(TypeSpec::Enum(EnumSpec {
            name,
            enumerators,
            loc,
        }))
    }

    /// 宣言子をパース
    fn parse_declarator(&mut self) -> Result<Declarator> {
        let loc = self.current.loc.clone();
        let mut derived = Vec::new();

        // GCC extension: attributes at the start of a declarator,
        // e.g. `int (__cdecl *fp)(void)` with __cdecl = __attribute__((__cdecl__))
        self.try_skip_attribute()?;

        // ポインタ
        while self.check(&TokenKind::Star) {
            self.advance()?;
            let qualifiers = self.parse_type_qualifiers()?;
            derived.push(DerivedDecl::Pointer(qualifiers));
        }

        // 直接宣言子
        let (name, inner_derived) = self.parse_direct_declarator()?;
        derived.extend(inner_derived);

        Ok(Declarator {
            name,
            derived,
            loc,
        })
    }

    /// 直接宣言子をパース
    fn parse_direct_declarator(&mut self) -> Result<(Option<InternedStr>, Vec<DerivedDecl>)> {
        let mut derived = Vec::new();

        // 識別子または ( declarator )
        let name = if self.check(&TokenKind::LParen) {
            self.advance()?;
            let inner = self.parse_declarator()?;
            self.expect(&TokenKind::RParen)?;
            // 内側の派生型を先頭に追加
            derived = inner.derived;
            inner.name
        } else if let Some(id) = self.current_ident() {
            // 識別子（キーワードはTokenKind::Kw*なのでここには来ない）
            self.advance()?;
            Some(id)
        } else {
            None
        };

        // 配列・関数の後置修飾
        loop {
            if self.check(&TokenKind::LBracket) {
                derived.push(self.parse_array_declarator()?);
            } else if self.check(&TokenKind::LParen) {
                derived.push(self.parse_function_declarator()?);
            } else {
                break;
            }
        }

        Ok((name, derived))
    }

    /// 配列宣言子をパース
    fn parse_array_declarator(&mut self) -> Result<DerivedDecl> {
        self.advance()?; // [

        let mut qualifiers = TypeQualifiers::default();
        let mut is_static = false;
        let mut is_vla = false;

        // static と型修飾子
        loop {
            match &self.current.kind {
                TokenKind::KwStatic => {
                    is_static = true;
                    self.advance()?;
                }
                TokenKind::KwConst | TokenKind::KwConst2 | TokenKind::KwConst3 => {
                    qualifiers.is_const = true;
                    self.advance()?;
                }
                TokenKind::KwVolatile | TokenKind::KwVolatile2 | TokenKind::KwVolatile3 => {
                    qualifiers.is_volatile = true;
                    self.advance()?;
                }
                TokenKind::KwRestrict | TokenKind::KwRestrict2 | TokenKind::KwRestrict3 => {
                    qualifiers.is_restrict = true;
                    self.advance()?;
                }
                _ => break,
            }
        }

        // サイズ式
        let size = if self.check(&TokenKind::RBracket) {
            None
        } else if self.check(&TokenKind::Star) {
            is_vla = true;
            self.advance()?;
            None
        } else {
            Some(Box::new(self.parse_assignment_expr()?))
        };

        self.expect(&TokenKind::RBracket)?;

        Ok(DerivedDecl::Array(ArrayDecl {
            size,
            qualifiers,
            is_static,
            is_vla,
        }))
    }

    /// 関数宣言子をパース
    fn parse_function_declarator(&mut self) -> Result<DerivedDecl> {
        self.advance()?; // (

        if self.check(&TokenKind::RParen) {
            self.advance()?;
            return Ok(DerivedDecl::Function(ParamList {
                params: Vec::new(),
                is_variadic: false,
            }));
        }

        let mut params = Vec::new();
        let mut is_variadic = false;

        loop {
            if self.check(&TokenKind::Ellipsis) {
                is_variadic = true;
                self.advance()?;
                break;
            }

            let loc = self.current.loc.clone();
            let specs = self.parse_decl_specs()?;
            let declarator = if self.check(&TokenKind::Comma) || self.check(&TokenKind::RParen) {
                None
            } else {
                Some(self.parse_declarator()?)
            };

            // GCC拡張: パラメータ後の __attribute__((...)) をスキップ
            self.try_skip_attribute()?;

            params.push(ParamDecl {
                specs,
                declarator,
                loc,
            });

            if !self.check(&TokenKind::Comma) {
                break;
            }
            self.advance()?;
        }

        self.expect(&TokenKind::RParen)?;

        Ok(DerivedDecl::Function(ParamList { params, is_variadic }))
    }

    /// 型修飾子をパース
    fn parse_type_qualifiers(&mut self) -> Result<TypeQualifiers> {
        let mut qualifiers = TypeQualifiers::default();

        loop {
            match &self.current.kind {
                TokenKind::KwConst | TokenKind::KwConst2 | TokenKind::KwConst3 => {
                    qualifiers.is_const = true;
                    self.advance()?;
                }
                TokenKind::KwVolatile | TokenKind::KwVolatile2 | TokenKind::KwVolatile3 => {
                    qualifiers.is_volatile = true;
                    self.advance()?;
                }
                TokenKind::KwRestrict | TokenKind::KwRestrict2 | TokenKind::KwRestrict3 => {
                    qualifiers.is_restrict = true;
                    self.advance()?;
                }
                TokenKind::KwAtomic => {
                    qualifiers.is_atomic = true;
                    self.advance()?;
                }
                // GCC extension: attributes in a pointer's qualifier list,
                // e.g. `int * __attribute__((__cdecl__)) f(void)`(mingw-w64 `__cdecl`)
                TokenKind::KwAttribute | TokenKind::KwAttribute2 => {
                    self.skip_attribute()?;
                }
                _ => break,
            }
        }

        Ok(qualifiers)
    }

    /// 初期化子をパース
    fn parse_initializer(&mut self) -> Result<Initializer> {
        if self.check(&TokenKind::LBrace) {
            self.advance()?;
            let mut items = Vec::new();

            while !self.check(&TokenKind::RBrace) {
                let designation = self.parse_designation()?;
                let init = self.parse_initializer()?;
                items.push(InitializerItem { designation, init });

                if !self.check(&TokenKind::Comma) {
                    break;
                }
                self.advance()?;
            }
            self.expect(&TokenKind::RBrace)?;

            Ok(Initializer::List(items))
        } else {
            Ok(Initializer::Expr(Box::new(self.parse_assignment_expr()?)))
        }
    }

    /// 指示子列をパース
    fn parse_designation(&mut self) -> Result<Vec<Designator>> {
        let mut designators = Vec::new();

        loop {
            if self.check(&TokenKind::LBracket) {
                self.advance()?;
                let index = self.parse_conditional_expr()?;
                self.expect(&TokenKind::RBracket)?;
                designators.push(Designator::Index(Box::new(index)));
            } else if self.check(&TokenKind::Dot) {
                self.advance()?;
                let name = self.expect_ident()?;
                designators.push(Designator::Member(name));
            } else {
                break;
            }
        }

        if !designators.is_empty() {
            self.expect(&TokenKind::Eq)?;
        }

        Ok(designators)
    }

    /// 型名をパース
    pub fn parse_type_name(&mut self) -> Result<TypeName> {
        let specs = self.parse_decl_specs()?;
        let declarator = if self.check(&TokenKind::RParen) {
            None
        } else {
            Some(self.parse_abstract_declarator()?)
        };
        Ok(TypeName { specs, declarator })
    }

    /// 抽象宣言子をパース
    fn parse_abstract_declarator(&mut self) -> Result<AbstractDeclarator> {
        let mut derived = Vec::new();

        // ポインタ
        while self.check(&TokenKind::Star) {
            self.advance()?;
            let qualifiers = self.parse_type_qualifiers()?;
            derived.push(DerivedDecl::Pointer(qualifiers));
        }

        // 直接抽象宣言子
        if self.check(&TokenKind::LParen) {
            // ( abstract-declarator ) または ( parameter-list )
            // 先読みで判定が必要だが、簡略化のためここでは単純に処理
            self.advance()?;
            if !self.check(&TokenKind::RParen) && !self.is_type_start() {
                let inner = self.parse_abstract_declarator()?;
                self.expect(&TokenKind::RParen)?;
                derived.extend(inner.derived);
            } else {
                // パラメータリストとして処理
                let params = self.parse_param_list_inner()?;
                derived.push(DerivedDecl::Function(params));
            }
        }

        // 配列・関数の後置修飾
        loop {
            if self.check(&TokenKind::LBracket) {
                derived.push(self.parse_array_declarator()?);
            } else if self.check(&TokenKind::LParen) {
                derived.push(self.parse_function_declarator()?);
            } else {
                break;
            }
        }

        Ok(AbstractDeclarator { derived })
    }

    /// パラメータリストの内部をパース（RParen消費済み前提ではない）
    fn parse_param_list_inner(&mut self) -> Result<ParamList> {
        if self.check(&TokenKind::RParen) {
            self.advance()?;
            return Ok(ParamList {
                params: Vec::new(),
                is_variadic: false,
            });
        }

        let mut params = Vec::new();
        let mut is_variadic = false;

        loop {
            if self.check(&TokenKind::Ellipsis) {
                is_variadic = true;
                self.advance()?;
                break;
            }

            let loc = self.current.loc.clone();
            let specs = self.parse_decl_specs()?;
            let declarator = if self.check(&TokenKind::Comma) || self.check(&TokenKind::RParen) {
                None
            } else if self.check(&TokenKind::Star) || self.check(&TokenKind::LParen) || self.check(&TokenKind::LBracket) {
                Some(self.parse_declarator()?)
            } else if let Some(id) = self.current_ident() {
                // 識別子（キーワードはTokenKind::Kw*なのでここには来ない）
                // typedef名でなければ宣言子
                if !self.typedefs.contains(&id) {
                    Some(self.parse_declarator()?)
                } else {
                    None
                }
            } else {
                None
            };

            // GCC拡張: パラメータ後の __attribute__((...)) をスキップ
            self.try_skip_attribute()?;

            params.push(ParamDecl {
                specs,
                declarator,
                loc,
            });

            if !self.check(&TokenKind::Comma) {
                break;
            }
            self.advance()?;
        }

        self.expect(&TokenKind::RParen)?;

        Ok(ParamList { params, is_variadic })
    }

    // ==================== 文のパース ====================

    /// 複合文をパース
    fn parse_compound_stmt(&mut self) -> Result<CompoundStmt> {
        let loc = self.current.loc.clone();
        self.expect(&TokenKind::LBrace)?;

        let mut items = Vec::new();
        while !self.check(&TokenKind::RBrace) {
            items.push(self.parse_block_item()?);
        }

        self.expect(&TokenKind::RBrace)?;

        Ok(CompoundStmt { items, info: NodeInfo::new(loc) })
    }

    /// ブロック項目をパース
    pub(crate) fn parse_block_item(&mut self) -> Result<BlockItem> {
        if self.is_declaration_start() {
            Ok(BlockItem::Decl(self.parse_declaration()?))
        } else {
            Ok(BlockItem::Stmt(self.parse_stmt()?))
        }
    }

    /// 宣言をパース
    fn parse_declaration(&mut self) -> Result<Declaration> {
        let comments = self.current.leading_comments.clone();
        let loc = self.current.loc.clone();
        let is_target = self.source.is_file_in_target(loc.file_id);
        let specs = self.parse_decl_specs()?;

        if self.check(&TokenKind::Semi) {
            self.advance()?;
            return Ok(Declaration {
                specs,
                declarators: Vec::new(),
                info: NodeInfo::new(loc),
                comments,
                is_target,
            });
        }

        let mut declarators = Vec::new();

        loop {
            let declarator = self.parse_declarator()?;

            // GCC拡張: 宣言子の後の __attribute__((...)) をスキップ
            self.try_skip_attribute()?;

            let init = if self.check(&TokenKind::Eq) {
                self.advance()?;
                Some(self.parse_initializer()?)
            } else {
                None
            };
            declarators.push(InitDeclarator { declarator, init });

            if !self.check(&TokenKind::Comma) {
                break;
            }
            self.advance()?;
        }

        // GCC拡張: 宣言の最後の __attribute__((...)) をスキップ
        self.try_skip_attribute()?;

        self.expect(&TokenKind::Semi)?;

        // typedef の場合、名前を登録
        if specs.storage == Some(StorageClass::Typedef) {
            for d in &declarators {
                if let Some(name) = d.declarator.name {
                    self.typedefs.insert(name);
                }
            }
        }

        Ok(Declaration {
            specs,
            declarators,
            info: NodeInfo::new(loc),
            comments,
            is_target,
        })
    }

    /// 文をパース
    fn parse_stmt(&mut self) -> Result<Stmt> {
        let loc = self.current.loc.clone();

        // キーワードに基づく文のパース
        match &self.current.kind {
            // ラベル文
            TokenKind::KwCase => return self.parse_case_stmt(),
            TokenKind::KwDefault => return self.parse_default_stmt(),
            // 複合文
            TokenKind::LBrace => return Ok(Stmt::Compound(self.parse_compound_stmt()?)),
            // 制御フロー文
            TokenKind::KwIf => return self.parse_if_stmt(),
            TokenKind::KwSwitch => return self.parse_switch_stmt(),
            TokenKind::KwWhile => return self.parse_while_stmt(),
            TokenKind::KwDo => return self.parse_do_while_stmt(),
            TokenKind::KwFor => return self.parse_for_stmt(),
            TokenKind::KwGoto => {
                self.advance()?;
                let name = self.expect_ident()?;
                self.expect(&TokenKind::Semi)?;
                return Ok(Stmt::Goto(name, loc));
            }
            TokenKind::KwContinue => {
                self.advance()?;
                self.expect(&TokenKind::Semi)?;
                return Ok(Stmt::Continue(loc));
            }
            TokenKind::KwBreak => {
                self.advance()?;
                self.expect(&TokenKind::Semi)?;
                return Ok(Stmt::Break(loc));
            }
            TokenKind::KwReturn => {
                self.advance()?;
                let expr = if self.check(&TokenKind::Semi) {
                    None
                } else {
                    Some(Box::new(self.parse_expr()?))
                };
                self.expect(&TokenKind::Semi)?;
                return Ok(Stmt::Return(expr, loc));
            }
            // __asm__ 文
            TokenKind::KwAsm | TokenKind::KwAsm2 | TokenKind::KwAsm3 => {
                return self.parse_asm_stmt();
            }
            _ => {}
        }

        // 式文
        if self.check(&TokenKind::Semi) {
            self.advance()?;
            return Ok(Stmt::Expr(None, loc));
        }

        let expr = self.parse_expr()?;

        // ラベルのチェック（識別子 : の場合）
        if self.check(&TokenKind::Colon) {
            if let ExprKind::Ident(name) = expr.kind {
                self.advance()?;
                let stmt = self.parse_stmt()?;
                return Ok(Stmt::Label {
                    name,
                    stmt: Box::new(stmt),
                    loc,
                });
            }
        }

        self.expect(&TokenKind::Semi)?;
        Ok(Stmt::Expr(Some(Box::new(expr)), loc))
    }

    fn parse_if_stmt(&mut self) -> Result<Stmt> {
        let loc = self.current.loc.clone();
        self.advance()?; // if
        self.expect(&TokenKind::LParen)?;
        let cond = Box::new(self.parse_expr()?);
        self.expect(&TokenKind::RParen)?;
        let then_stmt = Box::new(self.parse_stmt()?);

        let else_stmt = if matches!(self.current.kind, TokenKind::KwElse) {
            self.advance()?;
            Some(Box::new(self.parse_stmt()?))
        } else {
            None
        };

        Ok(Stmt::If {
            cond,
            then_stmt,
            else_stmt,
            loc,
        })
    }

    fn parse_switch_stmt(&mut self) -> Result<Stmt> {
        let loc = self.current.loc.clone();
        self.advance()?; // switch
        self.expect(&TokenKind::LParen)?;
        let expr = Box::new(self.parse_expr()?);
        self.expect(&TokenKind::RParen)?;
        let body = Box::new(self.parse_stmt()?);

        Ok(Stmt::Switch { expr, body, loc })
    }

    fn parse_while_stmt(&mut self) -> Result<Stmt> {
        let loc = self.current.loc.clone();
        self.advance()?; // while
        self.expect(&TokenKind::LParen)?;
        let cond = Box::new(self.parse_expr()?);
        self.expect(&TokenKind::RParen)?;
        let body = Box::new(self.parse_stmt()?);

        Ok(Stmt::While { cond, body, loc })
    }

    fn parse_do_while_stmt(&mut self) -> Result<Stmt> {
        let loc = self.current.loc.clone();
        self.advance()?; // do
        let body = Box::new(self.parse_stmt()?);
        // expect 'while' keyword
        if !matches!(self.current.kind, TokenKind::KwWhile) {
            return Err(CompileError::Parse {
                loc: self.current.loc.clone(),
                kind: ParseError::UnexpectedToken {
                    expected: "while".to_string(),
                    found: self.current.kind.clone(),
                },
            });
        }
        self.advance()?;
        self.expect(&TokenKind::LParen)?;
        let cond = Box::new(self.parse_expr()?);
        self.expect(&TokenKind::RParen)?;

        // allow_missing_semi が true の場合、; は任意
        if self.allow_missing_semi {
            if self.check(&TokenKind::Semi) {
                self.advance()?;
            }
        } else {
            self.expect(&TokenKind::Semi)?;
        }

        Ok(Stmt::DoWhile { body, cond, loc })
    }

    fn parse_for_stmt(&mut self) -> Result<Stmt> {
        let loc = self.current.loc.clone();
        self.advance()?; // for
        self.expect(&TokenKind::LParen)?;

        // 初期化部
        let init = if self.check(&TokenKind::Semi) {
            self.advance()?;
            None
        } else if self.is_declaration_start() {
            let decl = self.parse_declaration()?;
            Some(ForInit::Decl(decl))
        } else {
            let expr = self.parse_expr()?;
            self.expect(&TokenKind::Semi)?;
            Some(ForInit::Expr(Box::new(expr)))
        };

        // 条件部
        let cond = if self.check(&TokenKind::Semi) {
            None
        } else {
            Some(Box::new(self.parse_expr()?))
        };
        self.expect(&TokenKind::Semi)?;

        // 更新部
        let step = if self.check(&TokenKind::RParen) {
            None
        } else {
            Some(Box::new(self.parse_expr()?))
        };
        self.expect(&TokenKind::RParen)?;

        let body = Box::new(self.parse_stmt()?);

        Ok(Stmt::For {
            init,
            cond,
            step,
            body,
            loc,
        })
    }

    fn parse_case_stmt(&mut self) -> Result<Stmt> {
        let loc = self.current.loc.clone();
        self.advance()?; // case
        let expr = Box::new(self.parse_conditional_expr()?);
        self.expect(&TokenKind::Colon)?;
        let stmt = Box::new(self.parse_stmt()?);

        Ok(Stmt::Case { expr, stmt, loc })
    }

    fn parse_default_stmt(&mut self) -> Result<Stmt> {
        let loc = self.current.loc.clone();
        self.advance()?; // default
        self.expect(&TokenKind::Colon)?;
        let stmt = Box::new(self.parse_stmt()?);

        Ok(Stmt::Default { stmt, loc })
    }

    fn parse_asm_stmt(&mut self) -> Result<Stmt> {
        let loc = self.current.loc.clone();
        self.advance()?; // asm / __asm / __asm__

        // volatile / __volatile__ はスキップ
        while matches!(
            self.current.kind,
            TokenKind::KwVolatile | TokenKind::KwVolatile2 | TokenKind::KwVolatile3
        ) {
            self.advance()?;
        }

        // 括弧内をスキップ
        self.expect(&TokenKind::LParen)?;
        self.skip_balanced_parens()?;

        self.expect(&TokenKind::Semi)?;
        Ok(Stmt::Asm { loc })
    }

    // ==================== 式のパース ====================

    /// 式をパース（コンマ式を含む）
    fn parse_expr(&mut self) -> Result<Expr> {
        let lhs = self.parse_assignment_expr()?;

        if self.check(&TokenKind::Comma) {
            let loc = self.current.loc.clone();
            self.advance()?;
            let rhs = self.parse_expr()?;
            return Ok(Expr::new(
                ExprKind::Comma {
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                },
                loc,
            ));
        }

        Ok(lhs)
    }

    /// 代入式をパース
    fn parse_assignment_expr(&mut self) -> Result<Expr> {
        let lhs = self.parse_conditional_expr()?;

        let op = match &self.current.kind {
            TokenKind::Eq => Some(AssignOp::Assign),
            TokenKind::StarEq => Some(AssignOp::MulAssign),
            TokenKind::SlashEq => Some(AssignOp::DivAssign),
            TokenKind::PercentEq => Some(AssignOp::ModAssign),
            TokenKind::PlusEq => Some(AssignOp::AddAssign),
            TokenKind::MinusEq => Some(AssignOp::SubAssign),
            TokenKind::LtLtEq => Some(AssignOp::ShlAssign),
            TokenKind::GtGtEq => Some(AssignOp::ShrAssign),
            TokenKind::AmpEq => Some(AssignOp::AndAssign),
            TokenKind::CaretEq => Some(AssignOp::XorAssign),
            TokenKind::PipeEq => Some(AssignOp::OrAssign),
            _ => None,
        };

        if let Some(op) = op {
            let loc = self.current.loc.clone();
            self.advance()?;
            let rhs = self.parse_assignment_expr()?;
            return Ok(Expr::new(
                ExprKind::Assign {
                    op,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                },
                loc,
            ));
        }

        Ok(lhs)
    }

    /// 条件式をパース
    fn parse_conditional_expr(&mut self) -> Result<Expr> {
        let cond = self.parse_logical_or_expr()?;

        if self.check(&TokenKind::Question) {
            let loc = self.current.loc.clone();
            self.advance()?;
            let then_expr = self.parse_expr()?;
            self.expect(&TokenKind::Colon)?;
            let else_expr = self.parse_conditional_expr()?;
            return Ok(Expr::new(
                ExprKind::Conditional {
                    cond: Box::new(cond),
                    then_expr: Box::new(then_expr),
                    else_expr: Box::new(else_expr),
                },
                loc,
            ));
        }

        Ok(cond)
    }

    /// 論理OR式をパース
    fn parse_logical_or_expr(&mut self) -> Result<Expr> {
        let mut lhs = self.parse_logical_and_expr()?;

        while self.check(&TokenKind::PipePipe) {
            let loc = self.current.loc.clone();
            self.advance()?;
            let rhs = self.parse_logical_and_expr()?;
            lhs = Expr::new(
                ExprKind::Binary {
                    op: BinOp::LogOr,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                },
                loc,
            );
        }

        Ok(lhs)
    }

    /// 論理AND式をパース
    fn parse_logical_and_expr(&mut self) -> Result<Expr> {
        let mut lhs = self.parse_bitwise_or_expr()?;

        while self.check(&TokenKind::AmpAmp) {
            let loc = self.current.loc.clone();
            self.advance()?;
            let rhs = self.parse_bitwise_or_expr()?;
            lhs = Expr::new(
                ExprKind::Binary {
                    op: BinOp::LogAnd,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                },
                loc,
            );
        }

        Ok(lhs)
    }

    /// ビットOR式をパース
    fn parse_bitwise_or_expr(&mut self) -> Result<Expr> {
        let mut lhs = self.parse_bitwise_xor_expr()?;

        while self.check(&TokenKind::Pipe) {
            let loc = self.current.loc.clone();
            self.advance()?;
            let rhs = self.parse_bitwise_xor_expr()?;
            lhs = Expr::new(
                ExprKind::Binary {
                    op: BinOp::BitOr,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                },
                loc,
            );
        }

        Ok(lhs)
    }

    /// ビットXOR式をパース
    fn parse_bitwise_xor_expr(&mut self) -> Result<Expr> {
        let mut lhs = self.parse_bitwise_and_expr()?;

        while self.check(&TokenKind::Caret) {
            let loc = self.current.loc.clone();
            self.advance()?;
            let rhs = self.parse_bitwise_and_expr()?;
            lhs = Expr::new(
                ExprKind::Binary {
                    op: BinOp::BitXor,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                },
                loc,
            );
        }

        Ok(lhs)
    }

    /// ビットAND式をパース
    fn parse_bitwise_and_expr(&mut self) -> Result<Expr> {
        let mut lhs = self.parse_equality_expr()?;

        while self.check(&TokenKind::Amp) {
            let loc = self.current.loc.clone();
            self.advance()?;
            let rhs = self.parse_equality_expr()?;
            lhs = Expr::new(
                ExprKind::Binary {
                    op: BinOp::BitAnd,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                },
                loc,
            );
        }

        Ok(lhs)
    }

    /// 等価式をパース
    fn parse_equality_expr(&mut self) -> Result<Expr> {
        let mut lhs = self.parse_relational_expr()?;

        loop {
            let op = match &self.current.kind {
                TokenKind::EqEq => BinOp::Eq,
                TokenKind::BangEq => BinOp::Ne,
                _ => break,
            };
            let loc = self.current.loc.clone();
            self.advance()?;
            let rhs = self.parse_relational_expr()?;
            lhs = Expr::new(
                ExprKind::Binary {
                    op,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                },
                loc,
            );
        }

        Ok(lhs)
    }

    /// 関係式をパース
    fn parse_relational_expr(&mut self) -> Result<Expr> {
        let mut lhs = self.parse_shift_expr()?;

        loop {
            let op = match &self.current.kind {
                TokenKind::Lt => BinOp::Lt,
                TokenKind::Gt => BinOp::Gt,
                TokenKind::LtEq => BinOp::Le,
                TokenKind::GtEq => BinOp::Ge,
                _ => break,
            };
            let loc = self.current.loc.clone();
            self.advance()?;
            let rhs = self.parse_shift_expr()?;
            lhs = Expr::new(
                ExprKind::Binary {
                    op,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                },
                loc,
            );
        }

        Ok(lhs)
    }

    /// シフト式をパース
    fn parse_shift_expr(&mut self) -> Result<Expr> {
        let mut lhs = self.parse_additive_expr()?;

        loop {
            let op = match &self.current.kind {
                TokenKind::LtLt => BinOp::Shl,
                TokenKind::GtGt => BinOp::Shr,
                _ => break,
            };
            let loc = self.current.loc.clone();
            self.advance()?;
            let rhs = self.parse_additive_expr()?;
            lhs = Expr::new(
                ExprKind::Binary {
                    op,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                },
                loc,
            );
        }

        Ok(lhs)
    }

    /// 加減式をパース
    fn parse_additive_expr(&mut self) -> Result<Expr> {
        let mut lhs = self.parse_multiplicative_expr()?;

        loop {
            let op = match &self.current.kind {
                TokenKind::Plus => BinOp::Add,
                TokenKind::Minus => BinOp::Sub,
                _ => break,
            };
            let loc = self.current.loc.clone();
            self.advance()?;
            let rhs = self.parse_multiplicative_expr()?;
            lhs = Expr::new(
                ExprKind::Binary {
                    op,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                },
                loc,
            );
        }

        Ok(lhs)
    }

    /// 乗除式をパース
    fn parse_multiplicative_expr(&mut self) -> Result<Expr> {
        let mut lhs = self.parse_cast_expr()?;

        loop {
            let op = match &self.current.kind {
                TokenKind::Star => BinOp::Mul,
                TokenKind::Slash => BinOp::Div,
                TokenKind::Percent => BinOp::Mod,
                _ => break,
            };
            let loc = self.current.loc.clone();
            self.advance()?;
            let rhs = self.parse_cast_expr()?;
            lhs = Expr::new(
                ExprKind::Binary {
                    op,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                },
                loc,
            );
        }

        Ok(lhs)
    }

    /// キャスト式をパース
    fn parse_cast_expr(&mut self) -> Result<Expr> {
        // ( type-name ) cast-expression のみをここで処理
        // ( expr ) は parse_primary_expr で処理し、postfix操作を許可する
        if self.check(&TokenKind::LParen) {
            let loc = self.current.loc.clone();
            self.advance()?; // (

            // 1. 確定的な型名（キーワード/typedef/既検出の generic param）
            if self.is_type_name_start() {
                return self.finish_parse_cast_or_compound_lit(loc);
            }

            // 2. 未検出の generic param → 先読みで判定
            if let Some(id) = self.current_ident() {
                if self.generic_params.contains_key(&id)
                    && self.looks_like_generic_cast()
                {
                    // 型パラメータとして記録
                    self.detected_type_params.insert(id);
                    return self.finish_parse_cast_or_compound_lit(loc);
                }
            }

            if self.check(&TokenKind::LBrace) {
                // GCC拡張: ステートメント式 ({ ... })
                let stmt = self.parse_compound_stmt()?;
                self.expect(&TokenKind::RParen)?;
                // ステートメント式の後もpostfixを許可する
                let stmt_expr = Expr::new(ExprKind::StmtExpr(stmt), loc);
                return self.parse_postfix_on(stmt_expr);
            } else {
                // 括弧で囲まれた式 - parse_primary_exprに任せる
                // いったん戻してparse_unary_exprに任せる
                // 注: ここではadvanceを巻き戻す代わりに、式をパースしてからpostfixを処理
                let expr = self.parse_expr()?;
                self.expect(&TokenKind::RParen)?;
                // postfix操作を許可する（->や.など）
                return self.parse_postfix_on(expr);
            }
        }

        self.parse_unary_expr()
    }

    /// キャスト式または複合リテラルのパースを完了する
    ///
    /// `parse_cast_expr` で `(` を消費し、current が型名の先頭にある状態で呼ぶ。
    fn finish_parse_cast_or_compound_lit(&mut self, loc: SourceLocation) -> Result<Expr> {
        let type_name = self.parse_type_name()?;
        self.expect(&TokenKind::RParen)?;

        // 複合リテラルのチェック
        if self.check(&TokenKind::LBrace) {
            self.advance()?;
            let mut items = Vec::new();
            while !self.check(&TokenKind::RBrace) {
                let designation = self.parse_designation()?;
                let init = self.parse_initializer()?;
                items.push(InitializerItem { designation, init });
                if !self.check(&TokenKind::Comma) {
                    break;
                }
                self.advance()?;
            }
            self.expect(&TokenKind::RBrace)?;
            return Ok(Expr::new(
                ExprKind::CompoundLit {
                    type_name: Box::new(type_name),
                    init: items,
                },
                loc,
            ));
        }

        let expr = self.parse_cast_expr()?;
        Ok(Expr::new(
            ExprKind::Cast {
                type_name: Box::new(type_name),
                expr: Box::new(expr),
            },
            loc,
        ))
    }

    /// generic param がキャスト式の型として使われているかを先読みで判定
    ///
    /// 呼び出し時点: current = generic param の Ident
    /// TokenSource::next_token + unget_token で先読み
    fn looks_like_generic_cast(&mut self) -> bool {
        // 先読み1: param の次のトークン
        let next1 = match self.source.next_token() {
            Ok(t) => t,
            Err(_) => return false,
        };

        let result = match &next1.kind {
            // (PARAM *...) — ポインタキャスト
            TokenKind::Star => true,

            // (PARAM) — 値キャスト候補、後続の文脈で判定
            TokenKind::RParen => {
                let next2 = match self.source.next_token() {
                    Ok(t) => t,
                    Err(_) => {
                        self.source.unget_token(next1);
                        return false;
                    }
                };
                // ) の後に式の開始トークンが続くならキャスト
                let is_cast = match &next2.kind {
                    TokenKind::LParen       // (PARAM)(expr)
                    | TokenKind::Ident(_)   // (PARAM)ident
                    | TokenKind::IntLit(_) | TokenKind::UIntLit(_)
                    | TokenKind::FloatLit(_)
                    | TokenKind::StringLit(_) | TokenKind::CharLit(_)
                    | TokenKind::Star       // (PARAM)*ptr (deref)
                    | TokenKind::Amp        // (PARAM)&x
                    | TokenKind::Bang       // (PARAM)!x
                    | TokenKind::Tilde      // (PARAM)~x
                    | TokenKind::KwSizeof   // (PARAM)sizeof(x)
                    => true,

                    // (PARAM)-x: 3トークン先読みで判定
                    // `-` の次が数値リテラルなら cast (e.g. (T)-1)
                    // それ以外なら二項減算 (e.g. (val) - func(...))
                    TokenKind::Minus => {
                        let next3 = match self.source.next_token() {
                            Ok(t) => t,
                            Err(_) => {
                                self.source.unget_token(next2);
                                self.source.unget_token(next1);
                                return false;
                            }
                        };
                        let is_numeric = matches!(&next3.kind,
                            TokenKind::IntLit(_)
                            | TokenKind::UIntLit(_)
                            | TokenKind::FloatLit(_)
                        );
                        self.source.unget_token(next3);
                        is_numeric
                    }

                    _ => false,
                };
                self.source.unget_token(next2);
                is_cast
            }

            // (PARAM + ...) など — 括弧付き式
            _ => false,
        };

        self.source.unget_token(next1);
        result
    }

    /// 既存の式に対してpostfix操作をパース
    fn parse_postfix_on(&mut self, mut expr: Expr) -> Result<Expr> {
        loop {
            let loc = self.current.loc.clone();
            match &self.current.kind {
                TokenKind::LBracket => {
                    self.advance()?;
                    let index = self.parse_expr()?;
                    self.expect(&TokenKind::RBracket)?;
                    expr = Expr::new(
                        ExprKind::Index {
                            expr: Box::new(expr),
                            index: Box::new(index),
                        },
                        loc,
                    );
                }
                TokenKind::LParen => {
                    self.advance()?;
                    let mut args = Vec::new();
                    while !self.check(&TokenKind::RParen) {
                        args.push(self.parse_assignment_expr()?);
                        if !self.check(&TokenKind::Comma) {
                            break;
                        }
                        self.advance()?;
                    }
                    self.expect(&TokenKind::RParen)?;
                    self.function_call_count += 1;
                    expr = Expr::new(
                        ExprKind::Call {
                            func: Box::new(expr),
                            args,
                        },
                        loc,
                    );
                }
                TokenKind::Dot => {
                    self.advance()?;
                    let member = self.expect_ident()?;
                    expr = Expr::new(
                        ExprKind::Member {
                            expr: Box::new(expr),
                            member,
                        },
                        loc,
                    );
                }
                TokenKind::Arrow => {
                    self.advance()?;
                    let member = self.expect_ident()?;
                    self.deref_count += 1;
                    expr = Expr::new(
                        ExprKind::PtrMember {
                            expr: Box::new(expr),
                            member,
                        },
                        loc,
                    );
                }
                TokenKind::PlusPlus => {
                    self.advance()?;
                    expr = Expr::new(ExprKind::PostInc(Box::new(expr)), loc);
                }
                TokenKind::MinusMinus => {
                    self.advance()?;
                    expr = Expr::new(ExprKind::PostDec(Box::new(expr)), loc);
                }
                _ => break,
            }
        }
        Ok(expr)
    }

    /// 単項式をパース
    fn parse_unary_expr(&mut self) -> Result<Expr> {
        let loc = self.current.loc.clone();

        match &self.current.kind {
            TokenKind::PlusPlus => {
                self.advance()?;
                let expr = self.parse_unary_expr()?;
                Ok(Expr::new(ExprKind::PreInc(Box::new(expr)), loc))
            }
            TokenKind::MinusMinus => {
                self.advance()?;
                let expr = self.parse_unary_expr()?;
                Ok(Expr::new(ExprKind::PreDec(Box::new(expr)), loc))
            }
            TokenKind::Amp => {
                self.advance()?;
                let expr = self.parse_cast_expr()?;
                Ok(Expr::new(ExprKind::AddrOf(Box::new(expr)), loc))
            }
            TokenKind::Star => {
                self.advance()?;
                let expr = self.parse_cast_expr()?;
                self.deref_count += 1;
                Ok(Expr::new(ExprKind::Deref(Box::new(expr)), loc))
            }
            TokenKind::Plus => {
                self.advance()?;
                let expr = self.parse_cast_expr()?;
                Ok(Expr::new(ExprKind::UnaryPlus(Box::new(expr)), loc))
            }
            TokenKind::Minus => {
                self.advance()?;
                let expr = self.parse_cast_expr()?;
                Ok(Expr::new(ExprKind::UnaryMinus(Box::new(expr)), loc))
            }
            TokenKind::Tilde => {
                self.advance()?;
                let expr = self.parse_cast_expr()?;
                Ok(Expr::new(ExprKind::BitNot(Box::new(expr)), loc))
            }
            TokenKind::Bang => {
                self.advance()?;
                let expr = self.parse_cast_expr()?;
                Ok(Expr::new(ExprKind::LogNot(Box::new(expr)), loc))
            }
            TokenKind::KwSizeof => {
                self.advance()?;
                if self.check(&TokenKind::LParen) {
                    self.advance()?; // (
                    if self.is_type_name_start() {
                        // sizeof(type)
                        let type_name = self.parse_type_name()?;
                        self.expect(&TokenKind::RParen)?;
                        Ok(Expr::new(ExprKind::SizeofType(Box::new(type_name)), loc))
                    } else {
                        // sizeof(expr) - 括弧付きの式
                        let expr = self.parse_expr()?;
                        self.expect(&TokenKind::RParen)?;
                        Ok(Expr::new(ExprKind::Sizeof(Box::new(expr)), loc))
                    }
                } else {
                    let expr = self.parse_unary_expr()?;
                    Ok(Expr::new(ExprKind::Sizeof(Box::new(expr)), loc))
                }
            }
            TokenKind::KwAlignof | TokenKind::KwAlignof2 | TokenKind::KwAlignof3 => {
                self.advance()?;
                self.expect(&TokenKind::LParen)?;
                let type_name = self.parse_type_name()?;
                self.expect(&TokenKind::RParen)?;
                Ok(Expr::new(ExprKind::Alignof(Box::new(type_name)), loc))
            }
            // GCC extension: ignore __extension__ and continue (TinyCC style)
            TokenKind::KwExtension => {
                self.advance()?;
                self.parse_cast_expr()
            }
            _ => self.parse_postfix_expr(),
        }
    }

    /// 後置式をパース
    fn parse_postfix_expr(&mut self) -> Result<Expr> {
        let mut expr = self.parse_primary_expr()?;

        loop {
            let loc = self.current.loc.clone();
            match &self.current.kind {
                TokenKind::LBracket => {
                    self.advance()?;
                    let index = self.parse_expr()?;
                    self.expect(&TokenKind::RBracket)?;
                    expr = Expr::new(
                        ExprKind::Index {
                            expr: Box::new(expr),
                            index: Box::new(index),
                        },
                        loc,
                    );
                }
                TokenKind::LParen => {
                    // ビルトイン呼び出し判定
                    if let ExprKind::Ident(name) = &expr.kind {
                        if self.is_type_arg_builtin(*name) {
                            let builtin_name = *name;
                            self.advance()?; // (
                            let args = self.parse_builtin_args()?;
                            self.expect(&TokenKind::RParen)?;
                            self.function_call_count += 1;
                            expr = Expr::new(
                                ExprKind::BuiltinCall { name: builtin_name, args },
                                loc,
                            );
                            continue;
                        }
                    }
                    // 通常の関数呼び出し
                    self.advance()?;
                    let mut args = Vec::new();
                    if !self.check(&TokenKind::RParen) {
                        loop {
                            args.push(self.parse_assignment_expr()?);
                            if !self.check(&TokenKind::Comma) {
                                break;
                            }
                            self.advance()?;
                        }
                    }
                    self.expect(&TokenKind::RParen)?;
                    self.function_call_count += 1;
                    expr = Expr::new(
                        ExprKind::Call {
                            func: Box::new(expr),
                            args,
                        },
                        loc,
                    );
                }
                TokenKind::Dot => {
                    self.advance()?;
                    let member = self.expect_ident()?;
                    expr = Expr::new(
                        ExprKind::Member {
                            expr: Box::new(expr),
                            member,
                        },
                        loc,
                    );
                }
                TokenKind::Arrow => {
                    self.advance()?;
                    let member = self.expect_ident()?;
                    self.deref_count += 1;
                    expr = Expr::new(
                        ExprKind::PtrMember {
                            expr: Box::new(expr),
                            member,
                        },
                        loc,
                    );
                }
                TokenKind::PlusPlus => {
                    self.advance()?;
                    expr = Expr::new(ExprKind::PostInc(Box::new(expr)), loc);
                }
                TokenKind::MinusMinus => {
                    self.advance()?;
                    expr = Expr::new(ExprKind::PostDec(Box::new(expr)), loc);
                }
                _ => break,
            }
        }

        Ok(expr)
    }

    /// 一次式をパース
    fn parse_primary_expr(&mut self) -> Result<Expr> {
        let loc = self.current.loc.clone();

        match &self.current.kind {
            TokenKind::Ident(id) => {
                let id = *id;
                self.advance()?;
                Ok(Expr::new(ExprKind::Ident(id), loc))
            }
            TokenKind::IntLit(n) => {
                let n = *n;
                self.advance()?;
                Ok(Expr::new(ExprKind::IntLit(n), loc))
            }
            TokenKind::UIntLit(n) => {
                let n = *n;
                self.advance()?;
                Ok(Expr::new(ExprKind::UIntLit(n), loc))
            }
            TokenKind::FloatLit(f) => {
                let f = *f;
                self.advance()?;
                Ok(Expr::new(ExprKind::FloatLit(f), loc))
            }
            TokenKind::CharLit(c) => {
                let c = *c;
                self.advance()?;
                Ok(Expr::new(ExprKind::CharLit(c), loc))
            }
            TokenKind::StringLit(s) => {
                let mut bytes = s.clone();
                self.advance()?;
                // 連続した文字列リテラルを結合 (C の隣接リテラル連結。
                // TinyCC parse_mult_str と同方針で primary 解析位置で行う)。
                // マクロ本体の解析では `("" s "")` (STR_WITH_LEN /
                // ASSERT_IS_LITERAL の literal-only assert イディオム) の
                // ように仮引数 Ident が挟まるため Ident も受理し、
                // 「Ident 1 個 + 空リテラルのみ」なら意味的に等価なその
                // Ident 式へ還元する。それ以外の Ident 混在 (非空リテラル
                // との連結や Ident 2 個以上) は従来どおりエラー。
                let mut param_ident: Option<InternedStr> = None;
                let mut ident_count = 0usize;
                loop {
                    match &self.current.kind {
                        TokenKind::StringLit(s2) => {
                            bytes.extend_from_slice(s2);
                            self.advance()?;
                        }
                        TokenKind::Ident(id) => {
                            param_ident = Some(*id);
                            ident_count += 1;
                            self.advance()?;
                        }
                        _ => break,
                    }
                }
                match (ident_count, param_ident) {
                    (0, _) => Ok(Expr::new(ExprKind::StringLit(bytes), loc)),
                    (1, Some(id)) if bytes.is_empty() => {
                        Ok(Expr::new(ExprKind::Ident(id), loc))
                    }
                    _ => Err(CompileError::Parse {
                        loc,
                        kind: ParseError::UnexpectedToken {
                            expected: "adjacent string literals (string concatenation \
                                       mixing macro parameters with non-empty literals \
                                       is unsupported)"
                                .to_string(),
                            found: self.current.kind.clone(),
                        },
                    }),
                }
            }
            TokenKind::LParen => {
                self.advance()?;
                // GCC拡張: ステートメント式 ({ ... })
                if self.check(&TokenKind::LBrace) {
                    let stmt = self.parse_compound_stmt()?;
                    self.expect(&TokenKind::RParen)?;
                    Ok(Expr::new(ExprKind::StmtExpr(stmt), loc))
                } else {
                    let expr = self.parse_expr()?;
                    self.expect(&TokenKind::RParen)?;
                    Ok(expr)
                }
            }
            TokenKind::MacroBegin(info) if info.is_wrapped => {
                // wrapped マクロ（assert 等）を処理
                self.parse_wrapped_macro_expr()
            }
            TokenKind::MacroBegin(info) if info.preserve_call => {
                // preserve_call マクロを MacroCall ノードとして処理
                self.parse_macro_call_expr()
            }
            TokenKind::KwGeneric => {
                self.advance()?;
                self.parse_generic_selection(loc)
            }
            _ => Err(CompileError::Parse {
                loc,
                kind: ParseError::UnexpectedToken {
                    expected: "primary expression".to_string(),
                    found: self.current.kind.clone(),
                },
            }),
        }
    }

    /// C11 ジェネリック選択 (_Generic) をパースし、選択枝の式へ脱糖する
    ///
    /// 制御式は未評価オペランド (C11 6.5.1.1) なので AST には残さない。
    /// 本来の選択は制御式の型で決まるが、パース段階では型情報が無いため
    /// default 枝（無ければ先頭の枝）を採用する。実世界での主用途である
    /// glibc の __glibc_const_generic (C23 の const 修飾ジェネリック関数、
    /// 全枝が const キャストの有無を除いて同値) ではこの近似で正しい値になる。
    fn parse_generic_selection(&mut self, loc: SourceLocation) -> Result<Expr> {
        self.expect(&TokenKind::LParen)?;
        // 制御式（未評価なので捨てる）
        let _controlling = self.parse_assignment_expr()?;

        let mut default_expr: Option<Expr> = None;
        let mut first_expr: Option<Expr> = None;

        while self.check(&TokenKind::Comma) {
            self.advance()?;
            let is_default = if self.check(&TokenKind::KwDefault) {
                self.advance()?;
                true
            } else {
                let _type_name = self.parse_type_name()?;
                false
            };
            self.expect(&TokenKind::Colon)?;
            let expr = self.parse_assignment_expr()?;
            if is_default {
                if default_expr.is_none() {
                    default_expr = Some(expr);
                }
            } else if first_expr.is_none() {
                first_expr = Some(expr);
            }
        }
        self.expect(&TokenKind::RParen)?;

        default_expr.or(first_expr).ok_or_else(|| CompileError::Parse {
            loc,
            kind: ParseError::UnexpectedToken {
                expected: "generic association".to_string(),
                found: self.current.kind.clone(),
            },
        })
    }

    /// wrapped マクロ（assert 等）を Assert 式としてパース
    ///
    /// MacroBegin(is_wrapped=true) から args を取得し、condition をパースして
    /// Assert 式を生成する。その後 MacroEnd までスキップ。
    ///
    /// `assert_` や `__ASSERT_` は末尾カンマ形式で、式の前に付ける修飾子として使用される。
    /// 空展開時はカンマがないため、後続の式があれば暗黙的にカンマ式を形成する。
    fn parse_wrapped_macro_expr(&mut self) -> Result<Expr> {
        let loc = self.current.loc.clone();

        // MacroBegin から情報を取得
        let (marker_id, macro_name, args) = match &self.current.kind {
            TokenKind::MacroBegin(info) if info.is_wrapped => {
                let args = match &info.kind {
                    MacroInvocationKind::Function { args } => args.clone(),
                    MacroInvocationKind::Object => {
                        let name = self.source.interner().get(info.macro_name).to_string();
                        return Err(CompileError::Parse {
                            loc,
                            kind: ParseError::AssertNotFunctionMacro { macro_name: name },
                        });
                    }
                };
                (info.marker_id, info.macro_name, args)
            }
            _ => unreachable!("parse_wrapped_macro_expr called without wrapped MacroBegin"),
        };

        // 引数数チェック（assert は 1 引数）
        if args.len() != 1 {
            let name = self.source.interner().get(macro_name).to_string();
            return Err(CompileError::Parse {
                loc,
                kind: ParseError::InvalidAssertArgs {
                    macro_name: name,
                    arg_count: args.len(),
                },
            });
        }

        // args[0] から condition をパース
        let condition = self.parse_expr_from_tokens(&args[0], &loc)?;

        // AssertKind を判定
        let macro_name_str = self.source.interner().get(macro_name);
        let kind = detect_assert_kind(macro_name_str).unwrap_or(AssertKind::Assert);

        // MacroBegin を消費して MacroEnd までスキップ
        // 注意: advance() は inner_next_token() を使うため MacroEnd をスキップしてしまう
        // そのため、ここでは skip_to_macro_end() を使って直接ソースから読み取る
        self.skip_to_macro_end(marker_id)?;

        let assert_expr = Expr::new(ExprKind::Assert {
            kind,
            condition: Box::new(condition),
        }, loc.clone());

        // 末尾カンマ形式（assert_, __ASSERT_）の場合、後続の式と暗黙的にカンマ式を形成
        // 次のトークンが式の開始（別の Assert 含む）であれば結合
        if self.is_expression_start() {
            // 後続の式をパース（再帰的に Assert もパースされる）
            let next_expr = self.parse_assignment_expr()?;
            Ok(Expr::new(
                ExprKind::Comma {
                    lhs: Box::new(assert_expr),
                    rhs: Box::new(next_expr),
                },
                loc,
            ))
        } else {
            Ok(assert_expr)
        }
    }

    /// preserve_call マクロを MacroCall 式としてパース
    ///
    /// MacroBegin(preserve_call=true) から元の引数を取得し、
    /// 展開結果をパースして MacroCall ノードを生成する。
    ///
    /// MacroCall は元のマクロ呼び出し情報と展開結果を両方保持することで：
    /// - 展開結果で型推論を実行
    /// - コード生成時にマクロ呼び出し形式で出力可能
    fn parse_macro_call_expr(&mut self) -> Result<Expr> {
        let loc = self.current.loc.clone();

        // MacroBegin から情報を取得
        let (marker_id, macro_name, raw_args, call_loc) = match &self.current.kind {
            TokenKind::MacroBegin(info) if info.preserve_call => {
                let args = match &info.kind {
                    MacroInvocationKind::Function { args } => args.clone(),
                    MacroInvocationKind::Object => Vec::new(),
                };
                (info.marker_id, info.macro_name, args, info.call_loc.clone())
            }
            _ => unreachable!("parse_macro_call_expr called without preserve_call MacroBegin"),
        };

        // MacroBegin を消費（advance は inner_next_token を使うため MacroEnd もスキップされる可能性がある）
        // そのため、ここでは source から直接読み取って展開トークンを収集
        let expanded_tokens = self.collect_tokens_until_macro_end(marker_id)?;

        // 展開トークンから式をパース
        let expanded = if expanded_tokens.is_empty() {
            // 空展開の場合は 0 を返す（void 式として扱う）
            Expr::new(ExprKind::IntLit(0), loc.clone())
        } else {
            crate::parser::parse_expression_from_tokens_ref(
                expanded_tokens,
                self.source.interner(),
                self.source.files(),
                &self.typedefs,
            )?
        };

        // 元の引数トークン列を式としてパース
        let parsed_args: Result<Vec<Expr>> = raw_args.iter()
            .map(|arg_tokens| {
                if arg_tokens.is_empty() {
                    // 空引数の場合
                    Ok(Expr::new(ExprKind::IntLit(0), loc.clone()))
                } else {
                    crate::parser::parse_expression_from_tokens_ref(
                        arg_tokens.clone(),
                        self.source.interner(),
                        self.source.files(),
                        &self.typedefs,
                    )
                }
            })
            .collect();
        let args = parsed_args?;

        // current を次のトークンで更新
        self.current = self.inner_next_token()?;

        Ok(Expr::new(ExprKind::MacroCall {
            name: macro_name,
            args,
            expanded: Box::new(expanded),
            call_loc,
        }, loc))
    }

    /// 指定した marker_id に対応する MacroEnd までのトークンを収集
    ///
    /// source から直接読み取り、MacroBegin/MacroEnd のネストを追跡する。
    fn collect_tokens_until_macro_end(&mut self, target_marker_id: TokenId) -> Result<Vec<Token>> {
        let mut tokens = Vec::new();
        let mut nested_depth = 0;

        loop {
            let token = self.source.next_token()?;

            match &token.kind {
                TokenKind::MacroBegin(_) => {
                    // ネストされたマクロ展開開始
                    nested_depth += 1;
                    tokens.push(token);
                }
                TokenKind::MacroEnd(info) => {
                    if nested_depth > 0 {
                        // ネストされたマクロ展開終了
                        nested_depth -= 1;
                        tokens.push(token);
                    } else if info.begin_marker_id == target_marker_id {
                        // 目標の MacroEnd を見つけた
                        return Ok(tokens);
                    } else {
                        // 異なる MacroEnd（これは起こらないはず）
                        tokens.push(token);
                    }
                }
                TokenKind::Eof => {
                    return Err(CompileError::Parse {
                        loc: token.loc,
                        kind: ParseError::UnexpectedToken {
                            expected: "MacroEnd".to_string(),
                            found: token.kind,
                        },
                    });
                }
                _ => {
                    tokens.push(token);
                }
            }
        }
    }

    /// 現在のトークンが式の開始かどうかを判定
    fn is_expression_start(&self) -> bool {
        match &self.current.kind {
            // 式の開始になり得るトークン
            TokenKind::Ident(_)
            | TokenKind::IntLit(_)
            | TokenKind::UIntLit(_)
            | TokenKind::FloatLit(_)
            | TokenKind::CharLit(_)
            | TokenKind::WideCharLit(_)
            | TokenKind::StringLit(_)
            | TokenKind::WideStringLit(_)
            | TokenKind::LParen
            | TokenKind::Star      // 間接参照
            | TokenKind::Amp       // アドレス取得
            | TokenKind::Plus      // 単項プラス
            | TokenKind::Minus     // 単項マイナス
            | TokenKind::Bang      // 論理否定
            | TokenKind::Tilde     // ビット反転
            | TokenKind::PlusPlus  // 前置インクリメント
            | TokenKind::MinusMinus // 前置デクリメント
            | TokenKind::KwSizeof
            | TokenKind::KwAlignof
            | TokenKind::KwAlignof2
            | TokenKind::KwAlignof3
            | TokenKind::MacroBegin(_) => true, // 別の Assert も含む
            _ => false,
        }
    }

    /// トークン列から式をパース
    ///
    /// args から取り出したトークン列を一時的なソースとして式をパースする。
    /// 入れ子の wrapped マクロがあればエラー。
    fn parse_expr_from_tokens(&self, tokens: &[Token], loc: &SourceLocation) -> Result<Expr> {
        // 入れ子チェック：tokens 内に wrapped MacroBegin があればエラー
        for token in tokens {
            if let TokenKind::MacroBegin(info) = &token.kind {
                if info.is_wrapped {
                    return Err(CompileError::Parse {
                        loc: loc.clone(),
                        kind: ParseError::NestedAssertNotSupported,
                    });
                }
            }
        }

        // トークン列から式をパース
        crate::parser::parse_expression_from_tokens_ref(
            tokens.to_vec(),
            self.source.interner(),
            self.source.files(),
            &self.typedefs,
        )
    }

    /// 指定した marker_id に対応する MacroEnd までスキップ
    ///
    /// inner_next_token は MacroEnd をスキップするため、
    /// ここでは source から直接読み取る。
    fn skip_to_macro_end(&mut self, target_marker_id: TokenId) -> Result<()> {
        // まず current をチェック（MacroBegin の直後なので通常は MacroEnd ではない）
        loop {
            // source から直接読み取る（inner_next_token は MacroEnd をスキップするため）
            let token = self.source.next_token()?;

            match &token.kind {
                TokenKind::MacroEnd(info) if info.begin_marker_id == target_marker_id => {
                    // 目標の MacroEnd を見つけた
                    // current を次のトークンで更新（inner_next_token を使って正常なフローに戻す）
                    self.current = self.inner_next_token()?;
                    return Ok(());
                }
                TokenKind::Eof => {
                    return Err(CompileError::Parse {
                        loc: token.loc.clone(),
                        kind: ParseError::MacroEndNotFound,
                    });
                }
                _ => {
                    // 他のトークンはスキップ（MacroBegin/MacroEnd 含む）
                    continue;
                }
            }
        }
    }

    // ==================== ユーティリティ ====================

    /// 内部トークン取得メソッド（マーカーを透過的に処理）
    ///
    /// `handle_macro_markers` が true の場合、MacroBegin/MacroEnd マーカーを
    /// 処理してスキップし、通常のトークンのみを返す。
    /// ただし、以下の MacroBegin はスキップせず返す：
    /// - `is_wrapped=true`: assert 等の特殊処理用
    /// - `preserve_call=true`: MacroCall ノード生成用
    /// 空展開でもこれらの MacroBegin を返し、適切な AST ノードを生成する。
    fn inner_next_token(&mut self) -> Result<Token> {
        loop {
            let token = self.source.next_token()?;

            if !self.handle_macro_markers {
                return Ok(token);
            }

            match &token.kind {
                TokenKind::MacroBegin(info) => {
                    if info.is_wrapped {
                        // wrapped マクロ（assert 等）は常に返す
                        // 空展開でも Assert ノードとして AST に残す
                        return Ok(token);
                    }
                    if info.preserve_call {
                        // preserve_call マクロは常に返す
                        // MacroCall ノードとして AST に残す
                        return Ok(token);
                    }
                    // 通常のマクロ展開開始：コンテキストにプッシュ
                    self.macro_ctx.push((**info).clone());
                    continue; // マーカーはスキップ
                }
                TokenKind::MacroEnd(_info) => {
                    // マクロ展開終了：コンテキストからポップ
                    self.macro_ctx.pop();
                    continue; // マーカーはスキップ
                }
                _ => return Ok(token),
            }
        }
    }

    fn advance(&mut self) -> Result<Token> {
        let next = self.inner_next_token()?;
        let old = std::mem::replace(&mut self.current, next);
        Ok(old)
    }

    /// 現在のマクロコンテキストから NodeInfo を作成
    pub fn make_node_info(&self, loc: SourceLocation) -> NodeInfo {
        match self.macro_ctx.build_macro_info(self.source.interner()) {
            Some(macro_info) => NodeInfo::with_macro_info(loc, macro_info),
            None => NodeInfo::new(loc),
        }
    }

    /// 現在マクロ展開中かどうか
    pub fn is_in_macro(&self) -> bool {
        self.macro_ctx.is_in_macro()
    }

    /// マクロ展開の深さ
    pub fn macro_depth(&self) -> usize {
        self.macro_ctx.depth()
    }

    fn expect(&mut self, kind: &TokenKind) -> Result<Token> {
        if self.check(kind) {
            self.advance()
        } else {
            Err(CompileError::Parse {
                loc: self.current.loc.clone(),
                kind: ParseError::UnexpectedToken {
                    expected: format!("{:?}", kind),
                    found: self.current.kind.clone(),
                },
            })
        }
    }

    fn expect_ident(&mut self) -> Result<InternedStr> {
        if let TokenKind::Ident(id) = self.current.kind {
            self.advance()?;
            Ok(id)
        } else {
            Err(CompileError::Parse {
                loc: self.current.loc.clone(),
                kind: ParseError::UnexpectedToken {
                    expected: "identifier".to_string(),
                    found: self.current.kind.clone(),
                },
            })
        }
    }

    fn check(&self, kind: &TokenKind) -> bool {
        std::mem::discriminant(&self.current.kind) == std::mem::discriminant(kind)
    }

    fn is_eof(&self) -> bool {
        matches!(self.current.kind, TokenKind::Eof)
    }

    fn current_ident(&self) -> Option<InternedStr> {
        if let TokenKind::Ident(id) = self.current.kind {
            Some(id)
        } else {
            None
        }
    }

    /// 引数に型名を取りうるビルトイン関数名かどうか
    fn is_type_arg_builtin(&self, name: InternedStr) -> bool {
        let s = self.source.interner().get(name);
        matches!(s, "offsetof" | "__builtin_offsetof"
                  | "__builtin_types_compatible_p"
                  | "__builtin_va_arg"
                  | "STRUCT_OFFSET")
    }

    /// ビルトイン関数の引数をパース（型名と式を自動判定）
    fn parse_builtin_args(&mut self) -> Result<Vec<BuiltinArg>> {
        let mut args = Vec::new();
        if !self.check(&TokenKind::RParen) {
            loop {
                if self.is_type_name_start() {
                    let type_name = self.parse_type_name()?;
                    args.push(BuiltinArg::TypeName(Box::new(type_name)));
                } else {
                    let expr = self.parse_assignment_expr()?;
                    args.push(BuiltinArg::Expr(Box::new(expr)));
                }
                if !self.check(&TokenKind::Comma) {
                    break;
                }
                self.advance()?;
            }
        }
        Ok(args)
    }

    fn is_type_start(&self) -> bool {
        match &self.current.kind {
            // 型指定子キーワード
            TokenKind::KwVoid
            | TokenKind::KwChar
            | TokenKind::KwShort
            | TokenKind::KwInt
            | TokenKind::KwLong
            | TokenKind::KwFloat
            | TokenKind::KwDouble
            | TokenKind::KwSigned
            | TokenKind::KwSigned2
            | TokenKind::KwUnsigned
            | TokenKind::KwBool
            | TokenKind::KwBool2
            | TokenKind::KwComplex
            | TokenKind::KwFloat16
            | TokenKind::KwFloat32
            | TokenKind::KwFloat64
            | TokenKind::KwFloat128
            | TokenKind::KwFloat32x
            | TokenKind::KwFloat64x
            | TokenKind::KwInt128 => true,
            // 型修飾子キーワード
            TokenKind::KwConst
            | TokenKind::KwConst2
            | TokenKind::KwConst3
            | TokenKind::KwVolatile
            | TokenKind::KwVolatile2
            | TokenKind::KwVolatile3
            | TokenKind::KwRestrict
            | TokenKind::KwRestrict2
            | TokenKind::KwRestrict3
            | TokenKind::KwAtomic => true,
            // 構造体・共用体・列挙
            TokenKind::KwStruct | TokenKind::KwUnion | TokenKind::KwEnum => true,
            // typeof
            TokenKind::KwTypeof | TokenKind::KwTypeof2 | TokenKind::KwTypeof3 => true,
            // typedef名 or 検出済みの generic 型パラメータ
            TokenKind::Ident(id) => {
                self.typedefs.contains(id) || self.detected_type_params.contains(id)
            }
            _ => false,
        }
    }

    /// Like `is_type_start`, but also accepts a leading GCC `__attribute__`.
    /// Only for expression contexts after `(` (cast, compound literal, sizeof,
    /// builtin args), where an attribute can only begin a type name,
    /// e.g. `(__attribute__((__vector_size__(16))) int){4,1,2,3}` in xmmintrin.h.
    fn is_type_name_start(&self) -> bool {
        matches!(
            self.current.kind,
            TokenKind::KwAttribute | TokenKind::KwAttribute2
        ) || self.is_type_start()
    }

    fn is_declaration_start(&self) -> bool {
        match &self.current.kind {
            // ストレージクラス
            TokenKind::KwTypedef
            | TokenKind::KwExtern
            | TokenKind::KwStatic
            | TokenKind::KwAuto
            | TokenKind::KwRegister => true,
            // inline
            TokenKind::KwInline | TokenKind::KwInline2 | TokenKind::KwInline3 => true,
            // GCC拡張
            TokenKind::KwExtension => true,
            // thread-local
            TokenKind::KwThreadLocal | TokenKind::KwThread => true,
            _ => self.is_type_start(),
        }
    }

    /// __attribute__ / __asm__ があればスキップ（複数連続も対応）
    fn try_skip_attribute(&mut self) -> Result<()> {
        loop {
            match &self.current.kind {
                TokenKind::KwAttribute | TokenKind::KwAttribute2 => {
                    self.skip_attribute()?;
                }
                TokenKind::KwAsm | TokenKind::KwAsm2 | TokenKind::KwAsm3 => {
                    self.try_skip_asm_label()?;
                }
                _ => break,
            }
        }
        Ok(())
    }

    /// GCC拡張: __attribute__((...)) をスキップ
    fn skip_attribute(&mut self) -> Result<()> {
        self.advance()?; // __attribute__ / __attribute

        // 外側の ( を期待
        if !self.check(&TokenKind::LParen) {
            return Ok(()); // 引数なしの場合
        }
        self.advance()?;

        // 内側の ( を期待
        if !self.check(&TokenKind::LParen) {
            // 単一括弧の場合もある
            self.skip_balanced_parens()?;
            return Ok(());
        }
        self.advance()?;

        // 内側の括弧の中身をスキップ
        self.skip_balanced_parens()?;

        // 外側の ) を期待
        self.expect(&TokenKind::RParen)?;

        Ok(())
    }

    /// __asm__(label) があればスキップ
    fn try_skip_asm_label(&mut self) -> Result<()> {
        if matches!(
            self.current.kind,
            TokenKind::KwAsm | TokenKind::KwAsm2 | TokenKind::KwAsm3
        ) {
            self.advance()?; // asm / __asm / __asm__
            if self.check(&TokenKind::LParen) {
                self.advance()?; // (
                self.skip_balanced_parens()?; // 内容をスキップして ) を消費
            }
        }
        Ok(())
    }

    /// 括弧のバランスを取りながらスキップ
    fn skip_balanced_parens(&mut self) -> Result<()> {
        let mut depth = 1;
        while depth > 0 {
            match &self.current.kind {
                TokenKind::LParen => depth += 1,
                TokenKind::RParen => depth -= 1,
                TokenKind::Eof => {
                    return Err(CompileError::Parse {
                        loc: self.current.loc.clone(),
                        kind: ParseError::UnexpectedToken {
                            expected: ")".to_string(),
                            found: TokenKind::Eof,
                        },
                    });
                }
                _ => {}
            }
            if depth > 0 {
                self.advance()?;
            }
        }
        self.advance()?; // 最後の )
        Ok(())
    }
}

// ==================== ヘルパー関数 ====================

use crate::source::FileRegistry;
use crate::token_source::TokenSlice;

/// トークン列から式をパース
///
/// マクロ本体などのトークン列を式としてパースする際に使用。
///
/// # Arguments
/// * `tokens` - パースするトークン列
/// * `interner` - 文字列インターナー
/// * `files` - ファイルレジストリ
/// * `typedefs` - typedef名のセット（キャスト式の型名判定に使用）
pub fn parse_expression_from_tokens(
    tokens: Vec<Token>,
    interner: StringInterner,
    files: FileRegistry,
    typedefs: HashSet<InternedStr>,
) -> Result<Expr> {
    let mut source = TokenSlice::new(tokens, interner, files);
    let mut parser = Parser::from_source_with_typedefs(&mut source, typedefs)?;
    parser.parse_expr_only()
}

/// トークン列から式をパース（参照ベース版）
///
/// `parse_expression_from_tokens` の参照ベース版。
/// interner, files, typedefs をクローンせずに借用することで、
/// 高頻度の呼び出し時のオーバーヘッドを削減する。
///
/// # Arguments
/// * `tokens` - パースするトークン列
/// * `interner` - 文字列インターナーへの参照
/// * `files` - ファイルレジストリへの参照
/// * `typedefs` - typedef名のセットへの参照
pub fn parse_expression_from_tokens_ref(
    tokens: Vec<Token>,
    interner: &StringInterner,
    files: &FileRegistry,
    typedefs: &HashSet<InternedStr>,
) -> Result<Expr> {
    let mut source = TokenSliceRef::new(tokens, interner, files);
    let mut parser = Parser::from_source_with_typedefs(&mut source, typedefs.clone())?;
    parser.parse_expr_only()
}

/// トークン列を文としてパース（参照ベース版）
///
/// マクロ body のパースに使用。do-while の末尾セミコロンは省略可能。
///
/// # Arguments
/// * `tokens` - パースするトークン列
/// * `interner` - 文字列インターナーへの参照
/// * `files` - ファイルレジストリへの参照
/// * `typedefs` - typedef名のセットへの参照
pub fn parse_statement_from_tokens_ref(
    tokens: Vec<Token>,
    interner: &StringInterner,
    files: &FileRegistry,
    typedefs: &HashSet<InternedStr>,
) -> Result<Stmt> {
    let mut source = TokenSliceRef::new(tokens, interner, files);
    let mut parser = Parser::from_source_with_typedefs(&mut source, typedefs.clone())?;
    parser.parse_stmt_allow_missing_semi()
}

/// パース結果に付随する統計情報
#[derive(Debug, Clone, Default)]
pub struct ParseStats {
    /// 関数呼び出しの数
    pub function_call_count: usize,
    /// ポインタデリファレンスの数
    pub deref_count: usize,
}

impl ParseStats {
    /// unsafe 操作を含むか
    pub fn has_unsafe_ops(&self) -> bool {
        self.function_call_count > 0 || self.deref_count > 0
    }
}

/// トークン列から式をパース（統計情報付き・参照ベース版）
///
/// `parse_expression_from_tokens_ref` と同様だが、
/// パース統計（関数呼び出し数など）も返す。
///
/// # Arguments
/// * `tokens` - パースするトークン列
/// * `interner` - 文字列インターナーへの参照
/// * `files` - ファイルレジストリへの参照
/// * `typedefs` - typedef名のセットへの参照
///
/// # Returns
/// (パースされた式, 統計情報)
pub fn parse_expression_from_tokens_ref_with_stats(
    tokens: Vec<Token>,
    interner: &StringInterner,
    files: &FileRegistry,
    typedefs: &HashSet<InternedStr>,
) -> Result<(Expr, ParseStats)> {
    let mut source = TokenSliceRef::new(tokens, interner, files);
    let mut parser = Parser::from_source_with_typedefs(&mut source, typedefs.clone())?;
    let expr = parser.parse_expr_only()?;
    let stats = ParseStats {
        function_call_count: parser.function_call_count,
        deref_count: parser.deref_count,
    };
    Ok((expr, stats))
}

/// トークン列から式をパース（generic_params 付き）
///
/// マクロ本体のパースに使用。マクロの仮引数名を generic_params として渡し、
/// キャスト式での型パラメータ使用を自動検出する。
///
/// # Returns
/// (パースされた式, 統計情報, 検出された型パラメータ)
pub fn parse_expression_from_tokens_ref_with_generic_params(
    tokens: Vec<Token>,
    interner: &StringInterner,
    files: &FileRegistry,
    typedefs: &HashSet<InternedStr>,
    generic_params: HashMap<InternedStr, usize>,
) -> Result<(Expr, ParseStats, HashSet<InternedStr>)> {
    let mut source = TokenSliceRef::new(tokens, interner, files);
    let mut parser = Parser::from_source_with_typedefs(&mut source, typedefs.clone())?;
    parser.generic_params = generic_params;
    let expr = parser.parse_expr_only()?;
    let stats = ParseStats {
        function_call_count: parser.function_call_count,
        deref_count: parser.deref_count,
    };
    let detected = parser.detected_type_params;
    Ok((expr, stats, detected))
}

/// トークン列を文としてパース（統計情報付き・参照ベース版）
///
/// `parse_statement_from_tokens_ref` と同様だが、
/// パース統計（関数呼び出し数など）も返す。
/// do-while の末尾セミコロンは省略可能。
///
/// # Arguments
/// * `tokens` - パースするトークン列
/// * `interner` - 文字列インターナーへの参照
/// * `files` - ファイルレジストリへの参照
/// * `typedefs` - typedef名のセットへの参照
///
/// # Returns
/// (パースされた文, 統計情報)
pub fn parse_statement_from_tokens_ref_with_stats(
    tokens: Vec<Token>,
    interner: &StringInterner,
    files: &FileRegistry,
    typedefs: &HashSet<InternedStr>,
) -> Result<(Stmt, ParseStats)> {
    let mut source = TokenSliceRef::new(tokens, interner, files);
    let mut parser = Parser::from_source_with_typedefs(&mut source, typedefs.clone())?;
    let stmt = parser.parse_stmt_allow_missing_semi()?;
    let stats = ParseStats {
        function_call_count: parser.function_call_count,
        deref_count: parser.deref_count,
    };
    Ok((stmt, stats))
}

/// トークン列を文としてパース（generic_params 付き）
///
/// マクロ本体のパースに使用。マクロの仮引数名を generic_params として渡し、
/// キャスト式での型パラメータ使用を自動検出する。
///
/// # Returns
/// (パースされた文, 統計情報, 検出された型パラメータ)
pub fn parse_statement_from_tokens_ref_with_generic_params(
    tokens: Vec<Token>,
    interner: &StringInterner,
    files: &FileRegistry,
    typedefs: &HashSet<InternedStr>,
    generic_params: HashMap<InternedStr, usize>,
) -> Result<(Stmt, ParseStats, HashSet<InternedStr>)> {
    let mut source = TokenSliceRef::new(tokens, interner, files);
    let mut parser = Parser::from_source_with_typedefs(&mut source, typedefs.clone())?;
    parser.generic_params = generic_params;
    let stmt = parser.parse_stmt_allow_missing_semi()?;
    let stats = ParseStats {
        function_call_count: parser.function_call_count,
        deref_count: parser.deref_count,
    };
    let detected = parser.detected_type_params;
    Ok((stmt, stats, detected))
}

/// トークン列を複数のブロック項目としてパース
///
/// セミコロン区切りの複数文を含むマクロ本体のパースに使用。
/// `{ }` なしの BlockItem リストをパースし、EOF まで読む。
///
/// # Returns
/// (パースされたブロック項目リスト, 統計情報)
pub fn parse_block_items_from_tokens_ref_with_stats(
    tokens: Vec<Token>,
    interner: &StringInterner,
    files: &FileRegistry,
    typedefs: &HashSet<InternedStr>,
) -> Result<(Vec<BlockItem>, ParseStats)> {
    let mut source = TokenSliceRef::new(tokens, interner, files);
    let mut parser = Parser::from_source_with_typedefs(&mut source, typedefs.clone())?;
    parser.allow_missing_semi = true;
    let mut items = Vec::new();
    while !parser.check(&TokenKind::Eof) {
        items.push(parser.parse_block_item()?);
    }
    let stats = ParseStats {
        function_call_count: parser.function_call_count,
        deref_count: parser.deref_count,
    };
    Ok((items, stats))
}

/// トークン列を複数のブロック項目としてパース（generic_params 付き）
///
/// # Returns
/// (パースされたブロック項目リスト, 統計情報, 検出された型パラメータ)
pub fn parse_block_items_from_tokens_ref_with_generic_params(
    tokens: Vec<Token>,
    interner: &StringInterner,
    files: &FileRegistry,
    typedefs: &HashSet<InternedStr>,
    generic_params: HashMap<InternedStr, usize>,
) -> Result<(Vec<BlockItem>, ParseStats, HashSet<InternedStr>)> {
    let mut source = TokenSliceRef::new(tokens, interner, files);
    let mut parser = Parser::from_source_with_typedefs(&mut source, typedefs.clone())?;
    parser.allow_missing_semi = true;
    parser.generic_params = generic_params;
    let mut items = Vec::new();
    while !parser.check(&TokenKind::Eof) {
        items.push(parser.parse_block_item()?);
    }
    let stats = ParseStats {
        function_call_count: parser.function_call_count,
        deref_count: parser.deref_count,
    };
    let detected = parser.detected_type_params;
    Ok((items, stats, detected))
}

/// 型文字列から TypeName をパース
///
/// apidoc 等の型文字列（例: "SV *", "const char *"）をパースして
/// TypeName AST を返す。ReadOnlyLexer を使用するため、
/// 型文字列内の識別子は既に intern 済みである必要がある。
///
/// # Arguments
/// * `type_str` - パースする型文字列
/// * `interner` - 文字列インターナーへの参照（読み取り専用）
/// * `files` - ファイルレジストリへの参照
/// * `typedefs` - typedef名のセットへの参照
///
/// # Returns
/// パースされた TypeName AST、または識別子が未知の場合はエラー
pub fn parse_type_from_string(
    type_str: &str,
    interner: &StringInterner,
    files: &FileRegistry,
    typedefs: &HashSet<InternedStr>,
) -> Result<TypeName> {
    // 型文字列用のダミー FileId（エラー時の位置情報は限定的）
    let file_id = FileId::default();

    // ReadOnlyLexer でトークン化（新規 intern なし）
    let mut lexer = Lexer::<LookupOnly>::new_readonly(type_str.as_bytes(), file_id, interner);

    let mut tokens = Vec::new();
    loop {
        let token = lexer.next_token()?;
        if matches!(token.kind, TokenKind::Eof) {
            break;
        }
        tokens.push(token);
    }

    // パース
    let mut source = TokenSliceRef::new(tokens, interner, files);
    let mut parser = Parser::from_source_with_typedefs(&mut source, typedefs.clone())?;
    parser.parse_type_name()
}

/// トークン列を struct メンバー宣言の連続としてパースする。
///
/// `_XPVCV_COMMON` のような共通フィールド宣言マクロの本体を解析する用途。
/// 入力末尾に `;` が無い場合は自動補完する（マクロ本体は典型的に最終
/// 宣言の後に `;` を持たない — 展開先で `;` が補われる）。
pub fn parse_struct_members_from_tokens_ref(
    mut tokens: Vec<Token>,
    interner: &StringInterner,
    files: &FileRegistry,
    typedefs: &HashSet<InternedStr>,
) -> Result<Vec<StructMember>> {
    // 末尾に `;` が無ければ補う（最後の意味のあるトークンを見て判定）
    let needs_trailing_semi = tokens
        .iter()
        .rev()
        .find(|t| !matches!(t.kind, TokenKind::Eof))
        .is_some_and(|t| !matches!(t.kind, TokenKind::Semi));
    if needs_trailing_semi {
        // EOF を取り除いて `;` を足し直す
        let eof = tokens.iter().position(|t| matches!(t.kind, TokenKind::Eof))
            .map(|i| tokens.remove(i));
        let semi_loc = tokens.last().map(|t| t.loc.clone()).unwrap_or_default();
        tokens.push(Token::new(TokenKind::Semi, semi_loc));
        if let Some(eof) = eof {
            tokens.push(eof);
        }
    }
    let mut source = TokenSliceRef::new(tokens, interner, files);
    let mut parser = Parser::from_source_with_typedefs(&mut source, typedefs.clone())?;
    let mut members = Vec::new();
    while !parser.check(&TokenKind::Eof) {
        members.push(parser.parse_struct_member()?);
    }
    Ok(members)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preprocessor::PPConfig;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn parse_str(code: &str) -> Result<TranslationUnit> {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(code.as_bytes()).unwrap();

        let mut pp = Preprocessor::new(PPConfig::default());
        pp.add_source_file(file.path())?;

        let mut parser = Parser::new(&mut pp)?;
        parser.parse()
    }

    #[test]
    fn test_simple_function() {
        let tu = parse_str("int main(void) { return 0; }").unwrap();
        assert_eq!(tu.decls.len(), 1);
        assert!(matches!(tu.decls[0], ExternalDecl::FunctionDef(_)));
    }

    #[test]
    fn test_variable_declaration() {
        let tu = parse_str("int x;").unwrap();
        assert_eq!(tu.decls.len(), 1);
        assert!(matches!(tu.decls[0], ExternalDecl::Declaration(_)));
    }

    #[test]
    fn test_struct_declaration() {
        let tu = parse_str("struct Point { int x; int y; };").unwrap();
        assert_eq!(tu.decls.len(), 1);
    }

    #[test]
    fn test_typedef() {
        let tu = parse_str("typedef int INT; INT x;").unwrap();
        assert_eq!(tu.decls.len(), 2);
    }

    #[test]
    fn test_expression() {
        let tu = parse_str("int x = 1 + 2 * 3;").unwrap();
        assert_eq!(tu.decls.len(), 1);
    }
}
