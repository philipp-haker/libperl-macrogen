//! Cプリプロセッサ
//!
//! tinycc の tccpp.c に相当する機能を提供する。
//! next_token() がメインのインターフェースで、マクロ展開済みのトークンを返す。

use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use crate::error::{CompileError, PPError};
use crate::token_source::TokenSource;
use crate::intern::{InternedStr, StringInterner};
use crate::lexer::Lexer;
use crate::macro_def::{MacroDef, MacroKind, MacroTable};
use crate::pp_expr::PPExprEvaluator;
use crate::source::{FileId, FileRegistry, SourceLocation};
use crate::token::{
    Comment, MacroBeginInfo, MacroEndInfo, MacroInvocationKind, Token, TokenId, TokenKind,
};

/// マクロ定義時のコールバックトレイト
///
/// Preprocessor がマクロを定義したときに呼び出される。
/// THX マクロの収集など、マクロ定義時に追加の処理を行いたい場合に使用する。
pub trait MacroDefCallback {
    /// マクロが定義されたときに呼ばれる
    fn on_macro_defined(&mut self, def: &MacroDef);

    /// ダウンキャスト用に Any に変換
    fn into_any(self: Box<Self>) -> Box<dyn Any>;
}

/// 2つのコールバックをペアで保持
pub struct CallbackPair<A, B> {
    pub first: A,
    pub second: B,
}

impl<A, B> CallbackPair<A, B> {
    pub fn new(first: A, second: B) -> Self {
        Self { first, second }
    }
}

impl<A: MacroDefCallback + 'static, B: MacroDefCallback + 'static> MacroDefCallback for CallbackPair<A, B> {
    fn on_macro_defined(&mut self, def: &MacroDef) {
        self.first.on_macro_defined(def);
        self.second.on_macro_defined(def);
    }

    fn into_any(self: Box<Self>) -> Box<dyn Any> {
        self
    }
}

/// マクロ呼び出し時のコールバックトレイト
///
/// Preprocessor で特定のマクロが展開されたときに呼び出される。
/// `set_macro_called_callback` でマクロ名を指定して登録する。
pub trait MacroCalledCallback {
    /// マクロが呼び出され、展開された後に呼ばれる
    /// - args: 引数トークン列（関数形式マクロの場合）
    ///         オブジェクトマクロの場合は None
    /// - interner: トークンを文字列化するために使用
    fn on_macro_called(&mut self, args: Option<&[Vec<Token>]>, interner: &StringInterner);

    /// ダウンキャスト用
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// 特定マクロの呼び出しを監視するシンプルな実装
///
/// フラグベースで呼び出しを検出し、引数も記録する。
pub struct MacroCallWatcher {
    /// 呼び出しフラグ
    called: std::cell::Cell<bool>,
    /// 最後に呼び出された引数（トークン列を文字列化）
    last_args: std::cell::RefCell<Option<Vec<String>>>,
}

impl MacroCallWatcher {
    /// 新しい MacroCallWatcher を作成
    pub fn new() -> Self {
        Self {
            called: std::cell::Cell::new(false),
            last_args: std::cell::RefCell::new(None),
        }
    }

    /// フラグをチェックしてリセット
    pub fn take_called(&self) -> bool {
        self.called.replace(false)
    }

    /// 最後の引数を取得してクリア
    pub fn take_args(&self) -> Option<Vec<String>> {
        self.last_args.borrow_mut().take()
    }

    /// フラグと引数をクリア
    pub fn clear(&self) {
        self.called.set(false);
        *self.last_args.borrow_mut() = None;
    }

    /// 呼び出されたかどうか（リセットなし）
    pub fn was_called(&self) -> bool {
        self.called.get()
    }

    /// 最後の引数を取得（リセットなし）
    pub fn last_args(&self) -> Option<Vec<String>> {
        self.last_args.borrow().clone()
    }

    /// トークン列を文字列に変換
    fn tokens_to_string(tokens: &[Token], interner: &StringInterner) -> String {
        tokens
            .iter()
            .map(|t| t.kind.format(interner))
            .collect::<Vec<_>>()
            .join("")
    }
}

impl Default for MacroCallWatcher {
    fn default() -> Self {
        Self::new()
    }
}

impl MacroCalledCallback for MacroCallWatcher {
    fn on_macro_called(&mut self, args: Option<&[Vec<Token>]>, interner: &StringInterner) {
        self.called.set(true);
        if let Some(args) = args {
            let strs: Vec<String> = args
                .iter()
                .map(|tokens| Self::tokens_to_string(tokens, interner))
                .collect();
            *self.last_args.borrow_mut() = Some(strs);
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// コメント読み込み時のコールバックトレイト
///
/// Preprocessor がコメントを読み込んだときに呼び出される。
/// apidoc の収集など、コメント内容に基づく処理に使用する。
pub trait CommentCallback {
    /// コメントが読み込まれたときに呼ばれる
    ///
    /// - `comment`: コメント内容
    /// - `file_id`: ファイルID
    /// - `is_target`: このファイルが解析対象（samples/wrapper.h からの include）かどうか
    fn on_comment(&mut self, comment: &Comment, file_id: FileId, is_target: bool);

    /// ダウンキャスト用に Any に変換
    fn into_any(self: Box<Self>) -> Box<dyn Any>;
}

/// インクルードパスの種類
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IncludeKind {
    /// <...> システムヘッダ
    System,
    /// "..." ローカルヘッダ
    Local,
}

/// プリプロセッサ設定
#[derive(Debug, Default, Clone)]
pub struct PPConfig {
    /// システムインクルードパス (-I)
    pub include_paths: Vec<PathBuf>,
    /// 事前定義マクロ (-D)
    pub predefined: Vec<(String, Option<String>)>,
    /// プリプロセッサデバッグ出力 (--debug-pp)
    pub debug_pp: bool,
    /// ターゲットディレクトリ（このディレクトリ内で定義されたマクロにis_target=trueを設定）
    pub target_dir: Option<PathBuf>,
    /// マクロ展開マーカーを出力するか（デバッグ/AST用）
    pub emit_markers: bool,
}

/// 条件コンパイル状態
#[derive(Debug, Clone)]
struct CondState {
    /// 現在のブランチが有効か
    active: bool,
    /// いずれかのブランチが有効だったか
    seen_active: bool,
    /// #else を見たか
    seen_else: bool,
    /// ディレクティブの位置
    loc: SourceLocation,
}

/// 展開禁止情報の管理
///
/// トークンごとにマクロ展開禁止リストを管理する。
/// 自己参照マクロの無限再帰を防止するために使用。
#[derive(Debug, Default)]
pub struct NoExpandRegistry {
    map: HashMap<TokenId, HashSet<InternedStr>>,
}

impl NoExpandRegistry {
    /// 新しいレジストリを作成
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    /// トークンに展開禁止マクロを追加
    pub fn add(&mut self, token_id: TokenId, macro_id: InternedStr) {
        self.map.entry(token_id).or_default().insert(macro_id);
    }

    /// トークンに複数の展開禁止マクロを追加
    pub fn extend(&mut self, token_id: TokenId, macros: impl IntoIterator<Item = InternedStr>) {
        self.map.entry(token_id).or_default().extend(macros);
    }

    /// 指定トークンで指定マクロの展開が禁止されているか
    pub fn is_blocked(&self, token_id: TokenId, macro_id: InternedStr) -> bool {
        self.map
            .get(&token_id)
            .map_or(false, |s| s.contains(&macro_id))
    }

    /// あるトークンの展開禁止リストを別のトークンに継承
    pub fn inherit(&mut self, from: TokenId, to: TokenId) {
        if let Some(set) = self.map.get(&from).cloned() {
            self.map.entry(to).or_default().extend(set);
        }
    }

    /// トークンの展開禁止リストを取得（テスト用）
    pub fn get(&self, token_id: TokenId) -> Option<&HashSet<InternedStr>> {
        self.map.get(&token_id)
    }

    /// レジストリが空かどうか
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// 登録されているトークン数
    pub fn len(&self) -> usize {
        self.map.len()
    }
}

/// 入力ソース（ファイルまたはマクロ展開）
struct InputSource {
    /// ソースバイト列
    source: Vec<u8>,
    /// 現在位置
    pos: usize,
    /// 行番号
    line: u32,
    /// 列番号
    column: u32,
    /// ファイルID
    file_id: FileId,
    /// 行頭フラグ（ディレクティブ検出用）
    at_line_start: bool,
    /// トークンバッファ（マクロ展開の場合）
    tokens: Option<Vec<Token>>,
    /// トークンバッファの位置
    token_pos: usize,
}

impl InputSource {
    /// ファイルから作成
    fn from_file(source: Vec<u8>, file_id: FileId) -> Self {
        Self {
            source,
            pos: 0,
            line: 1,
            column: 1,
            file_id,
            at_line_start: true,
            tokens: None,
            token_pos: 0,
        }
    }

    /// トークン列から作成（マクロ展開用、テスト等で有用）
    #[allow(dead_code)]
    fn from_tokens(tokens: Vec<Token>, loc: SourceLocation) -> Self {
        Self {
            source: Vec::new(),
            pos: 0,
            line: loc.line,
            column: loc.column,
            file_id: loc.file_id,
            at_line_start: false,
            tokens: Some(tokens),
            token_pos: 0,
        }
    }

    /// トークンソースかどうか
    fn is_token_source(&self) -> bool {
        self.tokens.is_some()
    }

    /// 次のトークンを取得（トークンソースの場合）
    fn next_buffered_token(&mut self) -> Option<Token> {
        if let Some(ref tokens) = self.tokens {
            if self.token_pos < tokens.len() {
                let token = tokens[self.token_pos].clone();
                self.token_pos += 1;
                return Some(token);
            }
        }
        None
    }

    /// 行頭かどうか
    fn is_at_line_start(&self) -> bool {
        self.at_line_start
    }

    /// 現在位置を取得
    fn current_location(&self) -> SourceLocation {
        SourceLocation::new(self.file_id, self.line, self.column)
    }

    /// 行継続をスキップした実際の位置を取得
    fn skip_line_continuations(&self, start_pos: usize) -> usize {
        let mut pos = start_pos;
        loop {
            // \ の後に改行があれば行継続
            if self.source.get(pos) == Some(&b'\\') {
                let next = self.source.get(pos + 1);
                if next == Some(&b'\n') {
                    pos += 2;
                    continue;
                } else if next == Some(&b'\r') && self.source.get(pos + 2) == Some(&b'\n') {
                    // Windows形式の改行 (\r\n)
                    pos += 3;
                    continue;
                }
            }
            break;
        }
        pos
    }

    /// 現在の文字をピーク（行継続を処理）
    fn peek(&self) -> Option<u8> {
        let pos = self.skip_line_continuations(self.pos);
        self.source.get(pos).copied()
    }

    /// n文字先をピーク（行継続を処理）
    fn peek_n(&self, n: usize) -> Option<u8> {
        let mut pos = self.pos;
        for i in 0..=n {
            pos = self.skip_line_continuations(pos);
            if pos >= self.source.len() {
                return None;
            }
            if i < n {
                pos += 1;
            }
        }
        self.source.get(pos).copied()
    }

    /// 1文字進める（行継続を処理）
    fn advance(&mut self) -> Option<u8> {
        // 行継続をスキップ
        let old_pos = self.pos;
        self.pos = self.skip_line_continuations(self.pos);

        // スキップした行継続の分だけ行番号を更新
        for i in old_pos..self.pos {
            if self.source.get(i) == Some(&b'\n') {
                self.line += 1;
            }
        }

        let c = self.source.get(self.pos).copied()?;
        self.pos += 1;

        if c == b'\n' {
            self.line += 1;
            self.column = 1;
            self.at_line_start = true;
        } else {
            self.column += 1;
            if c != b' ' && c != b'\t' && c != b'\r' {
                self.at_line_start = false;
            }
        }
        Some(c)
    }

    /// 空白をスキップ（改行は含まない）
    fn skip_whitespace(&mut self) {
        while let Some(c) = self.peek() {
            // space, tab, carriage return, form feed (^L), vertical tab
            if c == b' ' || c == b'\t' || c == b'\r' || c == 0x0C || c == 0x0B {
                self.advance();
            } else {
                break;
            }
        }
    }
}

/// プリプロセッサ
pub struct Preprocessor {
    /// ファイルレジストリ
    files: FileRegistry,
    /// 文字列インターナー
    interner: StringInterner,
    /// マクロテーブル
    macros: MacroTable,
    /// 設定
    config: PPConfig,
    /// 入力ソーススタック
    sources: Vec<InputSource>,
    /// 条件コンパイルスタック
    cond_stack: Vec<CondState>,
    /// 先読みトークンバッファ
    lookahead: Vec<Token>,
    /// 収集中のコメント
    pending_comments: Vec<Comment>,
    /// 現在の条件が有効かどうかのキャッシュ
    cond_active: bool,
    /// スペースをトークンとして返すかどうか（TinyCC の PARSE_FLAG_SPACES 相当）
    return_spaces: bool,
    /// コマンドラインマクロを定義中かどうか（is_builtin フラグ用）
    defining_builtin: bool,
    /// トークンごとの展開禁止マクロを管理
    no_expand_registry: NoExpandRegistry,
    /// マクロ定義時のコールバック
    macro_def_callback: Option<Box<dyn MacroDefCallback>>,
    /// マクロ呼び出し時のコールバック（マクロ名ごとに登録）
    macro_called_callbacks: HashMap<InternedStr, Box<dyn MacroCalledCallback>>,
    /// マーカーで囲むマクロの辞書（assert 等の特殊処理用）
    wrapped_macros: HashSet<InternedStr>,
    /// コメント読み込み時のコールバック
    comment_callback: Option<Box<dyn CommentCallback>>,
    /// グローバルな展開抑制マクロ名（bindings.rs の定数など）
    skip_expand_macros: HashSet<InternedStr>,
    /// 明示的に展開する関数マクロ名（preserve_function_macros モードで使用）
    explicit_expand_macros: HashSet<InternedStr>,
}

impl Preprocessor {
    /// 新しいプリプロセッサを作成
    pub fn new(config: PPConfig) -> Self {
        let mut pp = Self {
            files: FileRegistry::new(),
            interner: StringInterner::new(),
            macros: MacroTable::new(),
            config,
            sources: Vec::new(),
            cond_stack: Vec::new(),
            lookahead: Vec::new(),
            pending_comments: Vec::new(),
            cond_active: true,
            return_spaces: false,
            defining_builtin: false,
            no_expand_registry: NoExpandRegistry::new(),
            macro_def_callback: None,
            macro_called_callbacks: HashMap::new(),
            wrapped_macros: HashSet::new(),
            comment_callback: None,
            skip_expand_macros: HashSet::new(),
            explicit_expand_macros: HashSet::new(),
        };

        // 事前定義マクロを登録
        pp.define_predefined_macros();

        pp
    }

    /// マクロ定義コールバックを設定
    pub fn set_macro_def_callback(&mut self, callback: Box<dyn MacroDefCallback>) {
        self.macro_def_callback = Some(callback);
    }

    /// マクロ定義コールバックを取得（所有権を移動）
    pub fn take_macro_def_callback(&mut self) -> Option<Box<dyn MacroDefCallback>> {
        self.macro_def_callback.take()
    }

    /// コメントコールバックを設定
    pub fn set_comment_callback(&mut self, callback: Box<dyn CommentCallback>) {
        self.comment_callback = Some(callback);
    }

    /// コメントコールバックを取得（所有権を移動）
    pub fn take_comment_callback(&mut self) -> Option<Box<dyn CommentCallback>> {
        self.comment_callback.take()
    }

    /// 特定マクロの呼び出しコールバックを設定
    ///
    /// 指定したマクロが展開されたときにコールバックが呼ばれる。
    pub fn set_macro_called_callback(
        &mut self,
        macro_name: InternedStr,
        callback: Box<dyn MacroCalledCallback>,
    ) {
        self.macro_called_callbacks.insert(macro_name, callback);
    }

    /// マクロ呼び出しコールバックを取得（所有権移動）
    pub fn take_macro_called_callback(
        &mut self,
        macro_name: InternedStr,
    ) -> Option<Box<dyn MacroCalledCallback>> {
        self.macro_called_callbacks.remove(&macro_name)
    }

    /// マクロ呼び出しコールバックへの参照を取得
    pub fn get_macro_called_callback(
        &self,
        macro_name: InternedStr,
    ) -> Option<&Box<dyn MacroCalledCallback>> {
        self.macro_called_callbacks.get(&macro_name)
    }

    /// マクロ呼び出しコールバックへの可変参照を取得
    pub fn get_macro_called_callback_mut(
        &mut self,
        macro_name: InternedStr,
    ) -> Option<&mut Box<dyn MacroCalledCallback>> {
        self.macro_called_callbacks.get_mut(&macro_name)
    }

    /// マーカーで囲むマクロを登録（assert 等の特殊処理用）
    ///
    /// 登録されたマクロは展開時に `MacroBegin`/`MacroEnd` マーカーで囲まれ、
    /// `is_wrapped` フラグが true になる。パーサーは args から元の式を復元できる。
    pub fn add_wrapped_macro(&mut self, macro_name: &str) {
        let id = self.interner.intern(macro_name);
        self.wrapped_macros.insert(id);
    }

    /// 展開抑制マクロを追加
    ///
    /// 登録されたマクロは展開されず、識別子としてそのまま出力される。
    /// bindings.rs に存在する定数名を登録することで、コード生成時に
    /// 定数名を保持できる。
    pub fn add_skip_expand_macro(&mut self, name: InternedStr) {
        self.skip_expand_macros.insert(name);
    }

    /// 複数の展開抑制マクロを追加
    pub fn add_skip_expand_macros(&mut self, names: impl IntoIterator<Item = InternedStr>) {
        self.skip_expand_macros.extend(names);
    }

    /// 明示展開マクロを追加（preserve_function_macros モードで展開対象となる）
    pub fn add_explicit_expand_macro(&mut self, name: InternedStr) {
        self.explicit_expand_macros.insert(name);
    }

    /// 複数の明示展開マクロを追加
    pub fn add_explicit_expand_macros(&mut self, names: impl IntoIterator<Item = InternedStr>) {
        self.explicit_expand_macros.extend(names);
    }

    /// 事前定義マクロを登録
    fn define_predefined_macros(&mut self) {
        // TinyCC方式: -Dオプションを#defineディレクティブとして処理
        // これにより関数マクロも正しく処理される
        let mut defines_source = String::new();

        // _Pragma は C99 のオペレータだが、ヘッダー解析時は無視する
        defines_source.push_str("#define _Pragma(x)\n");

        for (name, value) in &self.config.predefined {
            if let Some(val) = value {
                defines_source.push_str(&format!("#define {} {}\n", name, val));
            } else {
                defines_source.push_str(&format!("#define {} 1\n", name));
            }
        }

        if !defines_source.is_empty() {
            // 仮想ファイルとして登録
            let file_id = self.files.register(PathBuf::from("<cmdline>"));
            let input = InputSource::from_file(defines_source.into_bytes(), file_id);
            self.sources.push(input);

            // コマンドラインマクロは builtin としてマーク
            self.defining_builtin = true;

            // ディレクティブを処理
            loop {
                match self.next_raw_token() {
                    Ok(token) => {
                        match token.kind {
                            TokenKind::Eof => break,
                            TokenKind::Hash => {
                                // #define ディレクティブを処理
                                if let Err(_) = self.process_directive(token.loc) {
                                    break;
                                }
                            }
                            TokenKind::Newline => continue,
                            _ => {} // その他のトークンは無視
                        }
                    }
                    Err(_) => break,
                }
            }

            self.defining_builtin = false;

            // 仮想ファイルソースをポップ
            self.sources.pop();
        }
    }

    /// 文字列をトークン列に変換
    fn tokenize_string(&mut self, s: &str) -> Vec<Token> {
        let bytes = s.as_bytes();
        let file_id = FileId::default();
        let mut lexer = Lexer::new(bytes, file_id, &mut self.interner);

        let mut tokens = Vec::new();
        loop {
            match lexer.next_token() {
                Ok(token) => {
                    if matches!(token.kind, TokenKind::Eof) {
                        break;
                    }
                    if !matches!(token.kind, TokenKind::Newline) {
                        tokens.push(token);
                    }
                }
                Err(_) => break,
            }
        }
        tokens
    }

    /// ファイルをソースとして登録する
    ///
    /// 注: この関数はファイルを InputSource として登録するだけで、
    /// 実際のマクロ展開処理は行わない。マクロ展開は `next_token()` や
    /// `collect_tokens()` の呼び出し時に遅延実行される。
    pub fn add_source_file(&mut self, path: &Path) -> Result<(), CompileError> {
        let source = fs::read(path).map_err(|e| {
            CompileError::Preprocess {
                loc: SourceLocation::default(),
                kind: PPError::IoError(path.to_path_buf(), e.to_string()),
            }
        })?;

        let file_id = self.files.register(path.to_path_buf());
        let input = InputSource::from_file(source, file_id);
        self.sources.push(input);

        Ok(())
    }

    /// 現在のソースからレキサー経由でトークンを取得
    fn lex_token_from_source(&mut self) -> Result<Option<Token>, CompileError> {
        // トークンソースかどうかを先にチェック
        {
            let Some(source) = self.sources.last_mut() else {
                return Ok(None);
            };

            if source.is_token_source() {
                return Ok(source.next_buffered_token());
            }

            // return_spaces モードの場合、空白をトークンとして返す
            if self.return_spaces {
                if let Some(c) = source.peek() {
                    // space, tab, form feed, vertical tab
                    if c == b' ' || c == b'\t' || c == 0x0C || c == 0x0B {
                        let loc = source.current_location();
                        source.advance();
                        // 連続する空白は1つのSpaceトークンにまとめる
                        while let Some(c) = source.peek() {
                            if c == b' ' || c == b'\t' || c == 0x0C || c == 0x0B {
                                source.advance();
                            } else {
                                break;
                            }
                        }
                        return Ok(Some(Token::new(TokenKind::Space, loc)));
                    }
                }
            } else {
                source.skip_whitespace();
            }
        }

        // コメントを処理
        let mut leading_comments = Vec::new();
        loop {
            {
                let Some(source) = self.sources.last_mut() else {
                    return Ok(None);
                };
                if !self.return_spaces {
                    source.skip_whitespace();
                }
            }

            let (is_line_comment, is_block_comment) = {
                let Some(source) = self.sources.last() else {
                    return Ok(None);
                };
                (
                    source.peek() == Some(b'/') && source.peek_n(1) == Some(b'/'),
                    source.peek() == Some(b'/') && source.peek_n(1) == Some(b'*'),
                )
            };

            if is_line_comment {
                let comment = self.scan_line_comment();
                leading_comments.push(comment);
            } else if is_block_comment {
                let comment = self.scan_block_comment()?;
                leading_comments.push(comment);
            } else {
                break;
            }
        }

        let loc = {
            let Some(source) = self.sources.last() else {
                return Ok(None);
            };
            source.current_location()
        };

        let kind = self.scan_token_kind()?;

        let mut token = Token::new(kind, loc);
        token.leading_comments = leading_comments;
        Ok(Some(token))
    }

    /// 行コメントをスキャン
    fn scan_line_comment(&mut self) -> Comment {
        let (text, loc, file_id) = {
            let source = self.sources.last_mut().unwrap();
            let loc = source.current_location();
            let file_id = source.file_id;
            source.advance(); // /
            source.advance(); // /

            let start = source.pos;
            while source.peek().is_some_and(|c| c != b'\n') {
                source.advance();
            }
            let text = String::from_utf8_lossy(&source.source[start..source.pos]).to_string();
            (text, loc, file_id)
        };

        let comment = Comment::new(crate::token::CommentKind::Line, text, loc);
        let is_target = self.is_file_in_target(file_id);

        // コールバック呼び出し（is_target なファイルのみ）
        if is_target {
            if let Some(cb) = &mut self.comment_callback {
                cb.on_comment(&comment, file_id, is_target);
            }
        }

        comment
    }

    /// ブロックコメントをスキャン
    fn scan_block_comment(&mut self) -> Result<Comment, CompileError> {
        // まずソースを借用して必要な情報を取得
        let result = {
            let source = self.sources.last_mut().unwrap();
            let loc = source.current_location();
            let file_id = source.file_id;
            source.advance(); // /
            source.advance(); // *

            let start = source.pos;
            loop {
                match (source.peek(), source.peek_n(1)) {
                    (Some(b'*'), Some(b'/')) => {
                        let end = source.pos;
                        source.advance(); // *
                        source.advance(); // /
                        let text = String::from_utf8_lossy(&source.source[start..end]).to_string();
                        break Ok((text, loc, file_id));
                    }
                    (Some(_), _) => {
                        source.advance();
                    }
                    (None, _) => {
                        break Err(CompileError::Lex {
                            loc,
                            kind: crate::error::LexError::UnterminatedComment,
                        });
                    }
                }
            }
        };

        let (text, loc, file_id) = result?;
        let comment = Comment::new(crate::token::CommentKind::Block, text, loc);
        let is_target = self.is_file_in_target(file_id);

        // コールバック呼び出し（is_target なファイルのみ）
        if is_target {
            if let Some(cb) = &mut self.comment_callback {
                cb.on_comment(&comment, file_id, is_target);
            }
        }

        Ok(comment)
    }

    /// トークン種別をスキャン
    fn scan_token_kind(&mut self) -> Result<TokenKind, CompileError> {
        let source = self.sources.last_mut().unwrap();
        let Some(c) = source.peek() else {
            return Ok(TokenKind::Eof);
        };

        match c {
            // Only accept \r right before \n (CRLF headers, e.g. Windows-built perl)
            b'\r' if matches!(source.peek_n(1), Some(b'\n')) => {
                source.advance();
                source.advance();
                Ok(TokenKind::Newline)
            }

            b'\n' => {
                source.advance();
                Ok(TokenKind::Newline)
            }

            // ワイド文字列/文字リテラル
            b'L' if matches!(source.peek_n(1), Some(b'"') | Some(b'\'')) => {
                source.advance(); // L
                if source.peek() == Some(b'"') {
                    self.scan_wide_string()
                } else {
                    self.scan_wide_char()
                }
            }

            // 識別子またはキーワード
            b'a'..=b'z' | b'A'..=b'Z' | b'_' => self.scan_identifier(),

            // 数値リテラル
            b'0'..=b'9' => self.scan_number(),

            // 文字列リテラル
            b'"' => self.scan_string(),

            // 文字リテラル
            b'\'' => self.scan_char(),

            // 演算子・区切り記号
            b'+' => self.scan_operator(b'+', &[(b'+', TokenKind::PlusPlus), (b'=', TokenKind::PlusEq)], TokenKind::Plus),
            b'-' => self.scan_operator(b'-', &[(b'-', TokenKind::MinusMinus), (b'=', TokenKind::MinusEq), (b'>', TokenKind::Arrow)], TokenKind::Minus),
            b'*' => self.scan_operator(b'*', &[(b'=', TokenKind::StarEq)], TokenKind::Star),
            b'/' => self.scan_operator(b'/', &[(b'=', TokenKind::SlashEq)], TokenKind::Slash),
            b'%' => self.scan_operator(b'%', &[(b'=', TokenKind::PercentEq)], TokenKind::Percent),
            b'&' => self.scan_operator(b'&', &[(b'&', TokenKind::AmpAmp), (b'=', TokenKind::AmpEq)], TokenKind::Amp),
            b'|' => self.scan_operator(b'|', &[(b'|', TokenKind::PipePipe), (b'=', TokenKind::PipeEq)], TokenKind::Pipe),
            b'^' => self.scan_operator(b'^', &[(b'=', TokenKind::CaretEq)], TokenKind::Caret),
            b'~' => {
                source.advance();
                Ok(TokenKind::Tilde)
            }
            b'!' => self.scan_operator(b'!', &[(b'=', TokenKind::BangEq)], TokenKind::Bang),
            b'<' => self.scan_lt(),
            b'>' => self.scan_gt(),
            b'=' => self.scan_operator(b'=', &[(b'=', TokenKind::EqEq)], TokenKind::Eq),
            b'?' => {
                source.advance();
                Ok(TokenKind::Question)
            }
            b':' => {
                source.advance();
                Ok(TokenKind::Colon)
            }
            b'.' => self.scan_dot(),
            b',' => {
                source.advance();
                Ok(TokenKind::Comma)
            }
            b';' => {
                source.advance();
                Ok(TokenKind::Semi)
            }
            b'(' => {
                source.advance();
                Ok(TokenKind::LParen)
            }
            b')' => {
                source.advance();
                Ok(TokenKind::RParen)
            }
            b'[' => {
                source.advance();
                Ok(TokenKind::LBracket)
            }
            b']' => {
                source.advance();
                Ok(TokenKind::RBracket)
            }
            b'{' => {
                source.advance();
                Ok(TokenKind::LBrace)
            }
            b'}' => {
                source.advance();
                Ok(TokenKind::RBrace)
            }
            b'#' => {
                source.advance();
                if source.peek() == Some(b'#') {
                    source.advance();
                    Ok(TokenKind::HashHash)
                } else {
                    Ok(TokenKind::Hash)
                }
            }

            // バックスラッシュ（インラインアセンブリマクロなどで使用される）
            b'\\' => {
                source.advance();
                Ok(TokenKind::Backslash)
            }

            _ => {
                let loc = source.current_location();
                source.advance();
                Err(CompileError::Lex {
                    loc,
                    kind: crate::error::LexError::InvalidChar(c as char),
                })
            }
        }
    }

    /// 汎用演算子スキャン
    fn scan_operator(&mut self, _first: u8, continuations: &[(u8, TokenKind)], default: TokenKind) -> Result<TokenKind, CompileError> {
        let source = self.sources.last_mut().unwrap();
        source.advance();
        for (next, kind) in continuations {
            if source.peek() == Some(*next) {
                source.advance();
                return Ok(kind.clone());
            }
        }
        Ok(default)
    }

    /// < 演算子のスキャン
    fn scan_lt(&mut self) -> Result<TokenKind, CompileError> {
        let source = self.sources.last_mut().unwrap();
        source.advance();
        match source.peek() {
            Some(b'<') => {
                source.advance();
                if source.peek() == Some(b'=') {
                    source.advance();
                    Ok(TokenKind::LtLtEq)
                } else {
                    Ok(TokenKind::LtLt)
                }
            }
            Some(b'=') => {
                source.advance();
                Ok(TokenKind::LtEq)
            }
            _ => Ok(TokenKind::Lt),
        }
    }

    /// > 演算子のスキャン
    fn scan_gt(&mut self) -> Result<TokenKind, CompileError> {
        let source = self.sources.last_mut().unwrap();
        source.advance();
        match source.peek() {
            Some(b'>') => {
                source.advance();
                if source.peek() == Some(b'=') {
                    source.advance();
                    Ok(TokenKind::GtGtEq)
                } else {
                    Ok(TokenKind::GtGt)
                }
            }
            Some(b'=') => {
                source.advance();
                Ok(TokenKind::GtEq)
            }
            _ => Ok(TokenKind::Gt),
        }
    }

    /// . 演算子のスキャン
    fn scan_dot(&mut self) -> Result<TokenKind, CompileError> {
        let source = self.sources.last_mut().unwrap();
        source.advance();
        if source.peek() == Some(b'.') && source.peek_n(1) == Some(b'.') {
            source.advance();
            source.advance();
            Ok(TokenKind::Ellipsis)
        } else {
            Ok(TokenKind::Dot)
        }
    }

    /// 識別子またはキーワードをスキャン
    fn scan_identifier(&mut self) -> Result<TokenKind, CompileError> {
        let source = self.sources.last_mut().unwrap();
        let mut chars = Vec::new();
        while let Some(c) = source.peek() {
            if c.is_ascii_alphanumeric() || c == b'_' {
                chars.push(c);
                source.advance();
            } else {
                break;
            }
        }

        let text = std::str::from_utf8(&chars).unwrap();

        // キーワードなら対応するTokenKindを返す
        if let Some(kw) = TokenKind::from_keyword(text) {
            Ok(kw)
        } else {
            let interned = self.interner.intern(text);
            Ok(TokenKind::Ident(interned))
        }
    }

    /// 数値リテラルをスキャン
    fn scan_number(&mut self) -> Result<TokenKind, CompileError> {
        let source = self.sources.last_mut().unwrap();
        let loc = source.current_location();
        let start = source.pos;

        // 16進数、8進数、2進数の判定
        if source.peek() == Some(b'0') {
            source.advance();
            match source.peek() {
                Some(b'x') | Some(b'X') => {
                    source.advance();
                    while source.peek().is_some_and(|c| c.is_ascii_hexdigit()) {
                        source.advance();
                    }
                    // Hex float, cause fml
                    if matches!(source.peek(), Some(b'.') | Some(b'p') | Some(b'P')) {
                        return self.scan_hex_float_from(start, loc)
                    }
                }
                Some(b'b') | Some(b'B') => {
                    source.advance();
                    while matches!(source.peek(), Some(b'0') | Some(b'1')) {
                        source.advance();
                    }
                }
                Some(b'0'..=b'7') => {
                    while source.peek().is_some_and(|c| matches!(c, b'0'..=b'7')) {
                        source.advance();
                    }
                }
                Some(b'.') | Some(b'e') | Some(b'E') => {
                    return self.scan_float_from(start, loc);
                }
                _ => {}
            }
        } else {
            while source.peek().is_some_and(|c| c.is_ascii_digit()) {
                source.advance();
            }
            if matches!(source.peek(), Some(b'.') | Some(b'e') | Some(b'E')) {
                return self.scan_float_from(start, loc);
            }
        }

        self.finish_integer(start, loc)
    }

    /// 浮動小数点数をスキャン
    fn scan_float_from(
        &mut self,
        start: usize,
        loc: SourceLocation,
    ) -> Result<TokenKind, CompileError> {
        let source = self.sources.last_mut().unwrap();

        if source.peek() == Some(b'.') {
            source.advance();
            while source.peek().is_some_and(|c| c.is_ascii_digit()) {
                source.advance();
            }
        }

        if matches!(source.peek(), Some(b'e') | Some(b'E')) {
            source.advance();
            if matches!(source.peek(), Some(b'+') | Some(b'-')) {
                source.advance();
            }
            while source.peek().is_some_and(|c| c.is_ascii_digit()) {
                source.advance();
            }
        }

        // Use new standalone function to parse float suffix
        // As these are of variable length and prefixes of each other, i do it by keeping track of the length
        let suffix_start = source.pos;
        skip_float_suffix(source);

        let text = std::str::from_utf8(&source.source[start..source.pos]).unwrap();
        let number = std::str::from_utf8(&source.source[start..suffix_start]).unwrap();
        let value: f64 = number.parse().map_err(|_| CompileError::Lex {
            loc: loc.clone(),
            kind: crate::error::LexError::InvalidNumber(text.to_string()),
        })?;

        Ok(TokenKind::FloatLit(value))
    }

    /// Scans the rest of a hex float after `0x<hexdigits>`.
    fn scan_hex_float_from(&mut self, start: usize, loc: SourceLocation) -> Result<TokenKind, CompileError> {
        let source = self.sources.last_mut().unwrap();
        let invalid = |source: &InputSource| CompileError::Lex {
            loc: loc.clone(),
            kind: crate::error::LexError::InvalidNumber(
                String::from_utf8_lossy(&source.source[start..source.pos]).into_owned(),
            ),
        };

        if source.peek() == Some(b'.') {
            source.advance();
            while source.peek().as_ref().is_some_and(u8::is_ascii_hexdigit) {
                source.advance();
            }
        }

        // The binary exponent is mandatory for hex floats.
        if !matches!(source.peek(), Some(b'p') | Some(b'P')) {
            return Err(invalid(source));
        }
        source.advance();
        if matches!(source.peek(), Some(b'+') | Some(b'-')) {
            source.advance();
        }
        let exp_start = source.pos;
        while source.peek().as_ref().is_some_and(u8::is_ascii_digit) {
            source.advance();
        }
        if source.pos == exp_start {
            return Err(invalid(source));
        }

        let suffix_start = source.pos;
        skip_float_suffix(source);

        let number = std::str::from_utf8(&source.source[start..suffix_start]).unwrap();
        match parse_hex_float(number) {
            Some(value) => Ok(TokenKind::FloatLit(value)),
            None => Err(invalid(source)),
        }
    }

    /// 整数リテラルの仕上げ
    fn finish_integer(&mut self, start: usize, loc: SourceLocation) -> Result<TokenKind, CompileError> {
        let source = self.sources.last_mut().unwrap();

        // サフィックス
        let mut is_unsigned = false;
        let mut is_long = false;
        let mut is_longlong = false;

        loop {
            match source.peek() {
                Some(b'u') | Some(b'U') => {
                    is_unsigned = true;
                    source.advance();
                }
                Some(b'l') | Some(b'L') => {
                    if is_long {
                        is_longlong = true;
                    }
                    is_long = true;
                    source.advance();
                }
                _ => break,
            }
        }

        let text = std::str::from_utf8(&source.source[start..source.pos]).unwrap();

        // プレフィックスと基数を判定（チェイン trim ではなく排他的に処理）
        let (num_text, radix) = if text.starts_with("0x") || text.starts_with("0X") {
            (&text[2..], 16)
        } else if text.starts_with("0b") || text.starts_with("0B") {
            (&text[2..], 2)
        } else if text.starts_with('0') && text.len() > 1 {
            // 8進数（ただしサフィックスのみの場合は10進数の0として扱う）
            let without_suffix = text.trim_end_matches(|c: char| c == 'u' || c == 'U' || c == 'l' || c == 'L');
            if without_suffix.len() > 1 {
                (without_suffix, 8)
            } else {
                (without_suffix, 10)
            }
        } else {
            (text, 10)
        };

        // サフィックスを除去
        let num_text = num_text.trim_end_matches(|c: char| c == 'u' || c == 'U' || c == 'l' || c == 'L');

        if is_unsigned || is_longlong {
            let value = u64::from_str_radix(num_text, radix).map_err(|_| CompileError::Lex {
                loc: loc.clone(),
                kind: crate::error::LexError::InvalidNumber(text.to_string()),
            })?;
            Ok(TokenKind::UIntLit(value))
        } else {
            // まずi64でパースを試み、失敗したらu64でリトライ
            // (サフィックスなしでも大きな数値に対応)
            match i64::from_str_radix(num_text, radix) {
                Ok(value) => Ok(TokenKind::IntLit(value)),
                Err(_) => {
                    let value = u64::from_str_radix(num_text, radix).map_err(|_| CompileError::Lex {
                        loc: loc.clone(),
                        kind: crate::error::LexError::InvalidNumber(text.to_string()),
                    })?;
                    Ok(TokenKind::UIntLit(value))
                }
            }
        }
    }

    /// 文字列リテラルをスキャン
    fn scan_string(&mut self) -> Result<TokenKind, CompileError> {
        let loc = {
            let source = self.sources.last_mut().unwrap();
            let loc = source.current_location();
            source.advance(); // "
            loc
        };

        let mut bytes = Vec::new();
        loop {
            let c = {
                let source = self.sources.last_mut().unwrap();
                source.peek()
            };

            match c {
                Some(b'"') => {
                    let source = self.sources.last_mut().unwrap();
                    source.advance();
                    return Ok(TokenKind::StringLit(bytes));
                }
                Some(b'\\') => {
                    {
                        let source = self.sources.last_mut().unwrap();
                        source.advance();
                    }
                    let escaped = self.scan_escape_sequence(&loc)?;
                    bytes.push(escaped);
                }
                Some(b'\n') | None => {
                    return Err(CompileError::Lex {
                        loc,
                        kind: crate::error::LexError::UnterminatedString,
                    });
                }
                Some(c) => {
                    let source = self.sources.last_mut().unwrap();
                    source.advance();
                    bytes.push(c);
                }
            }
        }
    }

    /// ワイド文字列をスキャン
    fn scan_wide_string(&mut self) -> Result<TokenKind, CompileError> {
        let loc = {
            let source = self.sources.last_mut().unwrap();
            let loc = source.current_location();
            source.advance(); // "
            loc
        };

        let mut chars = Vec::new();
        loop {
            let c = {
                let source = self.sources.last_mut().unwrap();
                source.peek()
            };

            match c {
                Some(b'"') => {
                    let source = self.sources.last_mut().unwrap();
                    source.advance();
                    return Ok(TokenKind::WideStringLit(chars));
                }
                Some(b'\\') => {
                    {
                        let source = self.sources.last_mut().unwrap();
                        source.advance();
                    }
                    let escaped = self.scan_escape_sequence(&loc)?;
                    chars.push(escaped as u32);
                }
                Some(b'\n') | None => {
                    return Err(CompileError::Lex {
                        loc,
                        kind: crate::error::LexError::UnterminatedString,
                    });
                }
                Some(c) => {
                    let source = self.sources.last_mut().unwrap();
                    source.advance();
                    chars.push(c as u32);
                }
            }
        }
    }

    /// 文字リテラルをスキャン
    fn scan_char(&mut self) -> Result<TokenKind, CompileError> {
        let loc = {
            let source = self.sources.last_mut().unwrap();
            let loc = source.current_location();
            source.advance(); // '
            loc
        };

        let (first_char, second_char) = {
            let source = self.sources.last().unwrap();
            (source.peek(), source.peek_n(1))
        };

        let value = match (first_char, second_char) {
            (Some(b'\''), _) => {
                return Err(CompileError::Lex {
                    loc,
                    kind: crate::error::LexError::EmptyCharLit,
                });
            }
            (Some(b'\\'), _) => {
                {
                    let source = self.sources.last_mut().unwrap();
                    source.advance();
                }
                self.scan_escape_sequence(&loc)?
            }
            (None, _) | (Some(b'\n'), _) | (Some(b'\r'), Some(b'\n')) => {
                return Err(CompileError::Lex {
                    loc,
                    kind: crate::error::LexError::UnterminatedChar,
                });
            },
            (Some(c), _) => {
                let source = self.sources.last_mut().unwrap();
                source.advance();
                c
            }
        };

        // Multi char.
        // Gcc shifts previous value, then sets lower bits to newly discovered value.
        // int x = 'A'  ;   // <- 0x41
        // int x = 'AB' ;   // <- 0x00004142
        // int x = 'ABC';   // <- 0x00414243
        let mut char_count = 1u32;
        let mut multi_value = value as u32;
        loop {
            let source = self.sources.last_mut().unwrap();
            match (source.peek(), source.peek_n(1)) {
                (Some(b'\''), _) => {
                    break;
                },
                (None, _) | (Some(b'\n'), _) | (Some(b'\r'), Some(b'\n')) => {
                    return Err(CompileError::Lex {
                        loc,
                        kind: crate::error::LexError::UnterminatedChar,
                    });
                },
                (Some(c), _) => {
                    source.advance();
                    let c = if c == b'\\' {
                        self.scan_escape_sequence(&loc)?
                    } else {
                        c
                    };
                    multi_value = (multi_value << 8) | (c as u32);
                    char_count += 1;
                }
            };
        }
        self.sources.last_mut().unwrap().advance();

        Ok(if char_count == 1 {
            TokenKind::CharLit(value)
        } else {
            TokenKind::IntLit(multi_value as i32 as i64)
        })
    }

    /// ワイド文字をスキャン
    fn scan_wide_char(&mut self) -> Result<TokenKind, CompileError> {
        let loc = {
            let source = self.sources.last_mut().unwrap();
            let loc = source.current_location();
            source.advance(); // '
            loc
        };

        let first_char = {
            let source = self.sources.last().unwrap();
            source.peek()
        };

        let value = match first_char {
            Some(b'\'') => {
                return Err(CompileError::Lex {
                    loc,
                    kind: crate::error::LexError::EmptyCharLit,
                });
            }
            Some(b'\\') => {
                {
                    let source = self.sources.last_mut().unwrap();
                    source.advance();
                }
                self.scan_escape_sequence(&loc)? as u32
            }
            Some(c) => {
                let source = self.sources.last_mut().unwrap();
                source.advance();
                c as u32
            }
            None => {
                return Err(CompileError::Lex {
                    loc,
                    kind: crate::error::LexError::UnterminatedChar,
                });
            }
        };

        let source = self.sources.last_mut().unwrap();
        if source.peek() != Some(b'\'') {
            return Err(CompileError::Lex {
                loc,
                kind: crate::error::LexError::UnterminatedChar,
            });
        }
        source.advance();

        Ok(TokenKind::WideCharLit(value))
    }

    /// エスケープシーケンスをスキャン
    fn scan_escape_sequence(&mut self, loc: &SourceLocation) -> Result<u8, CompileError> {
        let source = self.sources.last_mut().unwrap();
        match source.peek() {
            Some(b'n') => { source.advance(); Ok(b'\n') }
            Some(b't') => { source.advance(); Ok(b'\t') }
            Some(b'r') => { source.advance(); Ok(b'\r') }
            Some(b'\\') => { source.advance(); Ok(b'\\') }
            Some(b'\'') => { source.advance(); Ok(b'\'') }
            Some(b'"') => { source.advance(); Ok(b'"') }
            Some(b'0') => { source.advance(); Ok(0) }
            Some(b'a') => { source.advance(); Ok(0x07) }
            Some(b'b') => { source.advance(); Ok(0x08) }
            Some(b'f') => { source.advance(); Ok(0x0C) }
            Some(b'v') => { source.advance(); Ok(0x0B) }
            Some(b'x') => {
                source.advance();
                let mut value = 0u8;
                let mut count = 0;
                while let Some(c) = source.peek() {
                    if let Some(digit) = (c as char).to_digit(16) {
                        value = value.wrapping_mul(16).wrapping_add(digit as u8);
                        source.advance();
                        count += 1;
                        if count >= 2 { break; }
                    } else {
                        break;
                    }
                }
                if count == 0 {
                    // GCC互換: \x の後に16進数がない場合は文字 'x' として扱う
                    Ok(b'x')
                } else {
                    Ok(value)
                }
            }
            Some(c @ b'0'..=b'7') => {
                let mut value = (c - b'0') as u8;
                source.advance();
                for _ in 0..2 {
                    if let Some(c @ b'0'..=b'7') = source.peek() {
                        value = value * 8 + (c - b'0');
                        source.advance();
                    } else {
                        break;
                    }
                }
                Ok(value)
            }
            Some(c) => {
                // GCC互換: 未知のエスケープシーケンスは文字そのものとして扱う
                source.advance();
                Ok(c)
            }
            None => Err(CompileError::Lex {
                loc: loc.clone(),
                kind: crate::error::LexError::UnterminatedString,
            }),
        }
    }

    /// 次のトークンを取得（メインインターフェース）
    pub fn next_token(&mut self) -> Result<Token, CompileError> {
        loop {
            // 先読みバッファまたはソースからトークンを取得
            let token = if let Some(token) = self.lookahead.pop() {
                token
            } else {
                match self.lex_token_from_source()? {
                    Some(t) => t,
                    None => {
                        // ソースが空 - ポップして続行
                        if self.sources.len() > 1 {
                            self.sources.pop();
                            continue;
                        }
                        Token::new(TokenKind::Eof, SourceLocation::default())
                    }
                }
            };

            // コメントを収集
            if !token.leading_comments.is_empty() {
                self.pending_comments.extend(token.leading_comments.iter().cloned());
            }

            match &token.kind {
                TokenKind::Eof => {
                    // 現在のソースが終了
                    if self.sources.len() > 1 {
                        self.sources.pop();
                        continue;
                    }

                    // 条件コンパイルスタックのチェック
                    if !self.cond_stack.is_empty() {
                        let state = &self.cond_stack[0];
                        return Err(CompileError::Preprocess {
                            loc: state.loc.clone(),
                            kind: PPError::MissingEndif,
                        });
                    }

                    return Ok(token);
                }

                TokenKind::Newline => {
                    // 改行は通常スキップ
                    continue;
                }

                TokenKind::Hash => {
                    // プリプロセッサディレクティブ
                    let at_line_start = self.sources.last().map(|s| s.is_at_line_start()).unwrap_or(false);
                    if at_line_start || self.sources.last().map(|s| s.is_token_source()).unwrap_or(false) {
                        // ファイルソースで行頭、またはトークンソース（#が先頭にある場合）
                    }
                    self.process_directive(token.loc.clone())?;
                    continue;
                }

                TokenKind::Ident(id) if self.cond_active => {
                    // マクロ展開を試みる
                    let id = *id;
                    if let Some(expanded) = self.try_expand_macro(id, &token)? {
                        // 展開結果を先読みバッファに追加（逆順）
                        for t in expanded.into_iter().rev() {
                            self.lookahead.push(t);
                        }
                        continue;
                    }
                    return Ok(self.attach_comments(token));
                }

                _ if !self.cond_active => {
                    // 条件コンパイルで無効なブランチ
                    continue;
                }

                _ => {
                    return Ok(self.attach_comments(token));
                }
            }
        }
    }

    /// トークンを先読みバッファに戻す
    ///
    /// パーサーが先読みしたトークンを戻す必要がある場合に使用。
    pub fn unget_token(&mut self, token: Token) {
        self.lookahead.push(token);
    }

    /// 生のトークンを取得（マクロ展開なし）
    fn next_raw_token(&mut self) -> Result<Token, CompileError> {
        loop {
            // 先読みバッファから取得
            if let Some(token) = self.lookahead.pop() {
                return Ok(token);
            }

            match self.lex_token_from_source()? {
                Some(token) => {
                    if !token.leading_comments.is_empty() {
                        self.pending_comments.extend(token.leading_comments.iter().cloned());
                    }
                    return Ok(token);
                }
                None => {
                    if self.sources.len() > 1 {
                        self.sources.pop();
                        continue;
                    }
                    return Ok(Token::new(TokenKind::Eof, SourceLocation::default()));
                }
            }
        }
    }

    /// 蓄積したコメントをトークンに付与
    fn attach_comments(&mut self, mut token: Token) -> Token {
        if !self.pending_comments.is_empty() {
            token.leading_comments = std::mem::take(&mut self.pending_comments);
        }
        token
    }

    /// プリプロセッサディレクティブを処理
    fn process_directive(&mut self, loc: SourceLocation) -> Result<(), CompileError> {
        // ディレクティブ名を取得
        let directive_token = self.next_raw_token()?;

        match &directive_token.kind {
            TokenKind::Newline | TokenKind::Eof => {
                // 空のディレクティブ（許可）
                return Ok(());
            }
            TokenKind::Ident(id) => {
                let name = self.interner.get(*id).to_string();
                self.process_directive_by_name(&name, loc)?;
            }
            // プリプロセッサディレクティブはキーワードトークンとして返される可能性がある
            // キーワード名を文字列に変換してディレクティブとして処理
            TokenKind::KwIf => self.process_directive_by_name("if", loc)?,
            TokenKind::KwElse => self.process_directive_by_name("else", loc)?,
            TokenKind::KwFor => self.process_directive_by_name("for", loc)?,  // エラーになる
            TokenKind::IntLit(_) => {
                // #line または # 123 "file" 形式
                self.skip_to_eol()?;
            }
            _ => {
                return Err(CompileError::Preprocess {
                    loc,
                    kind: PPError::InvalidDirective(format!("{:?}", directive_token.kind)),
                });
            }
        }

        Ok(())
    }

    /// ディレクティブ名に基づいて処理
    fn process_directive_by_name(&mut self, name: &str, loc: SourceLocation) -> Result<(), CompileError> {
        match name {
            "define" => {
                if self.cond_active {
                    self.process_define(loc)?;
                } else {
                    self.skip_to_eol()?;
                }
            }
            "undef" => {
                if self.cond_active {
                    self.process_undef()?;
                } else {
                    self.skip_to_eol()?;
                }
            }
            "include" => {
                if self.cond_active {
                    self.process_include(loc, false)?;
                } else {
                    self.skip_to_eol()?;
                }
            }
            "include_next" => {
                if self.cond_active {
                    self.process_include(loc, true)?;
                } else {
                    self.skip_to_eol()?;
                }
            }
            "if" => self.process_if(loc)?,
            "ifdef" => self.process_ifdef(loc, false)?,
            "ifndef" => self.process_ifdef(loc, true)?,
            "elif" => self.process_elif(loc)?,
            "else" => self.process_else(loc)?,
            "endif" => self.process_endif()?,
            "error" => {
                if self.cond_active {
                    self.process_error(loc)?;
                } else {
                    self.skip_to_eol()?;
                }
            }
            "warning" | "pragma" | "line" => {
                self.skip_to_eol()?;
            }
            _ => {
                if self.cond_active {
                    return Err(CompileError::Preprocess {
                        loc,
                        kind: PPError::InvalidDirective(name.to_string()),
                    });
                } else {
                    self.skip_to_eol()?;
                }
            }
        }

        Ok(())
    }

    /// #define を処理
    fn process_define(&mut self, loc: SourceLocation) -> Result<(), CompileError> {
        let name_token = self.next_raw_token()?;
        // TinyCC 流: キーワード相当のトークンも識別子として受理する。
        // 例: <stdbool.h> の `#define bool _Bool` は `bool` が `KwBool2` で
        // 来るが、preprocessor の名前 namespace では Ident と等価に扱う。
        let name = match name_token.kind {
            TokenKind::Ident(id) => id,
            ref kind => {
                if let Some(s) = kind.keyword_str() {
                    self.interner.intern(s)
                } else {
                    return Err(CompileError::Preprocess {
                        loc,
                        kind: PPError::InvalidDirective("expected macro name".to_string()),
                    });
                }
            }
        };

        // TinyCC方式: スペースモードを有効にして次のトークンを取得
        // '(' がマクロ名の直後にある場合のみ関数マクロとして扱う
        self.return_spaces = true;
        let next = self.next_raw_token()?;
        self.return_spaces = false;

        let (kind, body_start) = if matches!(next.kind, TokenKind::LParen) {
            // マクロ名の直後に '(' があるので関数マクロ
            let (params, is_variadic) = self.parse_macro_params()?;
            (MacroKind::Function { params, is_variadic }, None)
        } else if matches!(next.kind, TokenKind::Space) {
            // スペースがあった場合、次のトークンを読んでオブジェクトマクロのボディとする
            let body_first = self.next_raw_token()?;
            (MacroKind::Object, Some(body_first))
        } else {
            // その他（改行など）はそのままオブジェクトマクロ
            (MacroKind::Object, Some(next))
        };

        let mut body = Vec::new();
        let mut need_more = true;
        if let Some(first) = body_start {
            if matches!(first.kind, TokenKind::Newline | TokenKind::Eof) {
                // 値なしマクロ：これ以上読む必要なし
                need_more = false;
            } else {
                body.push(first);
            }
        }

        if need_more {
            loop {
                let token = self.next_raw_token()?;
                match token.kind {
                    TokenKind::Newline | TokenKind::Eof => break,
                    _ => body.push(token),
                }
            }
        }

        let is_target = self.is_current_file_in_target();
        let has_token_pasting = body.iter()
            .any(|t| matches!(t.kind, TokenKind::HashHash));
        let def = MacroDef {
            name,
            kind,
            body,
            def_loc: loc,
            leading_comments: std::mem::take(&mut self.pending_comments),
            is_builtin: self.defining_builtin,
            is_target,
            has_token_pasting,
        };

        // コールバックを呼び出し（define の前に呼ぶことで def への参照を渡せる）
        if let Some(ref mut callback) = self.macro_def_callback {
            callback.on_macro_defined(&def);
        }

        self.macros.define(def, &self.interner);
        Ok(())
    }

    /// 関数マクロのパラメータをパース
    /// GNU拡張: NAME... 形式の可変長引数もサポート
    fn parse_macro_params(&mut self) -> Result<(Vec<InternedStr>, bool), CompileError> {
        let mut params = Vec::new();
        let mut is_variadic = false;

        loop {
            let token = self.next_raw_token()?;
            // process_define と同じ理由で、キーワード相当のトークンも仮引数名として
            // 受理する（例: `#define FOO(bool) ...`）。Ident/Kw* で抽出ロジックを揃える。
            let param_id: Option<InternedStr> = match &token.kind {
                TokenKind::Ident(id) => Some(*id),
                kind => kind.keyword_str().map(|s| self.interner.intern(s)),
            };
            match token.kind {
                TokenKind::RParen => break,
                _ if param_id.is_some() => {
                    params.push(param_id.unwrap());
                    let next = self.next_raw_token()?;
                    match next.kind {
                        TokenKind::Comma => continue,
                        TokenKind::RParen => break,
                        TokenKind::Ellipsis => {
                            // GNU拡張: NAME... 形式
                            // パラメータ名はそのまま保持し、variadic フラグをセット
                            is_variadic = true;
                            let rparen = self.next_raw_token()?;
                            if !matches!(rparen.kind, TokenKind::RParen) {
                                return Err(CompileError::Preprocess {
                                    loc: token.loc,
                                    kind: PPError::InvalidMacroArgs("expected ')' after '...'".to_string()),
                                });
                            }
                            break;
                        }
                        _ => {
                            return Err(CompileError::Preprocess {
                                loc: token.loc,
                                kind: PPError::InvalidMacroArgs("expected ',' or ')'".to_string()),
                            });
                        }
                    }
                }
                TokenKind::Ellipsis => {
                    // 標準 C99: ... のみ（__VA_ARGS__ として扱う）
                    // TinyCC と同様に __VA_ARGS__ をパラメータ名として登録
                    is_variadic = true;
                    let va_args_id = self.interner.intern("__VA_ARGS__");
                    params.push(va_args_id);
                    let next = self.next_raw_token()?;
                    if !matches!(next.kind, TokenKind::RParen) {
                        return Err(CompileError::Preprocess {
                            loc: token.loc,
                            kind: PPError::InvalidMacroArgs("expected ')' after '...'".to_string()),
                        });
                    }
                    break;
                }
                _ => {
                    return Err(CompileError::Preprocess {
                        loc: token.loc,
                        kind: PPError::InvalidMacroArgs("expected parameter name".to_string()),
                    });
                }
            }
        }

        Ok((params, is_variadic))
    }

    /// #undef を処理
    fn process_undef(&mut self) -> Result<(), CompileError> {
        let token = self.next_raw_token()?;
        // process_define と対称: キーワード名で #define されたマクロも undef できる
        let name = match token.kind {
            TokenKind::Ident(id) => Some(id),
            ref kind => kind.keyword_str().map(|s| self.interner.intern(s)),
        };
        if let Some(id) = name {
            self.macros.undefine(id);
        }
        self.skip_to_eol()?;
        Ok(())
    }

    /// #include を処理
    fn process_include(&mut self, loc: SourceLocation, is_include_next: bool) -> Result<(), CompileError> {
        let token = self.next_raw_token()?;

        let (path, kind) = match &token.kind {
            TokenKind::StringLit(bytes) => {
                let path = String::from_utf8_lossy(bytes).to_string();
                (path, IncludeKind::Local)
            }
            TokenKind::Lt => {
                // TinyCC方式: トークナイザを使わず文字レベルで直接読み取る
                // ファイル名に含まれる "64.h" などが FloatLit として誤解析されるのを防ぐ
                let path = self.scan_include_path('>')?;
                (path, IncludeKind::System)
            }
            _ => {
                return Err(CompileError::Preprocess {
                    loc,
                    kind: PPError::InvalidDirective("expected include path".to_string()),
                });
            }
        };

        self.skip_to_eol()?;

        let resolved = self.resolve_include(&path, kind, &loc, is_include_next)?;

        let source = fs::read(&resolved).map_err(|e| {
            CompileError::Preprocess {
                loc: loc.clone(),
                kind: PPError::IoError(resolved.clone(), e.to_string()),
            }
        })?;

        let file_id = self.files.register(resolved);
        let input = InputSource::from_file(source, file_id);
        self.sources.push(input);

        Ok(())
    }

    /// インクルードパスを解決
    fn resolve_include(&self, path: &str, kind: IncludeKind, loc: &SourceLocation, is_include_next: bool) -> Result<PathBuf, CompileError> {
        let path = Path::new(path);

        // #include_next の場合は、現在のファイルのディレクトリ以降から検索開始
        let start_index = if is_include_next {
            self.find_current_include_index()
        } else {
            0
        };

        if kind == IncludeKind::Local && !is_include_next {
            if let Some(source) = self.sources.last() {
                if !source.is_token_source() {
                    let current_path = self.files.get_path(source.file_id);
                    if let Some(parent) = current_path.parent() {
                        let candidate = parent.join(path);
                        if candidate.exists() {
                            return Ok(candidate);
                        }
                    }
                }
            }
        }

        for dir in self.config.include_paths.iter().skip(start_index) {
            let candidate = dir.join(path);
            if candidate.exists() {
                return Ok(candidate);
            }
        }

        Err(CompileError::Preprocess {
            loc: loc.clone(),
            kind: PPError::IncludeNotFound(path.to_path_buf()),
        })
    }

    /// 現在のファイルがどのインクルードパスに属するかを探し、次のインデックスを返す
    fn find_current_include_index(&self) -> usize {
        // 現在のファイルパスを取得
        let current_file_path = if let Some(source) = self.sources.iter().rev().find(|s| !s.is_token_source()) {
            self.files.get_path(source.file_id).to_path_buf()
        } else {
            return 0;
        };

        // どのインクルードパスに含まれているか探す
        for (i, dir) in self.config.include_paths.iter().enumerate() {
            if current_file_path.starts_with(dir) {
                return i + 1; // 次のインデックスから開始
            }
        }

        0
    }

    /// #if を処理
    fn process_if(&mut self, loc: SourceLocation) -> Result<(), CompileError> {
        // 親が無効な場合は文字レベルでスキップ
        if !self.cond_active {
            self.cond_stack.push(CondState {
                active: false,
                seen_active: false,
                seen_else: false,
                loc: loc.clone(),
            });
            self.skip_false_branch(loc)?;
            return Ok(());
        }

        // マクロ展開付きでトークンを収集
        let tokens = self.collect_if_condition()?;

        let mut eval = PPExprEvaluator::new(&tokens, &self.interner, &self.macros, loc.clone());
        let active = eval.evaluate()? != 0;

        self.cond_stack.push(CondState {
            active,
            seen_active: active,
            seen_else: false,
            loc: loc.clone(),
        });

        self.update_cond_active();

        // 条件が偽の場合、TinyCC方式でスキップ
        if !active {
            self.skip_false_branch(loc)?;
        }

        Ok(())
    }

    /// 偽ブランチをスキップし、#else/#elif/#endif を処理
    fn skip_false_branch(&mut self, loc: SourceLocation) -> Result<(), CompileError> {
        loop {
            let directive = self.preprocess_skip()?;
            match directive.as_str() {
                "endif" => {
                    // #endif: スタックからポップして終了
                    self.cond_stack.pop();
                    self.update_cond_active();
                    return Ok(());
                }
                "else" => {
                    // #else: 今までどのブランチも有効でなければこのブランチを有効化
                    if let Some(state) = self.cond_stack.last_mut() {
                        if state.seen_else {
                            return Err(CompileError::Preprocess {
                                loc,
                                kind: PPError::UnmatchedElse,
                            });
                        }
                        state.seen_else = true;
                        if !state.seen_active {
                            state.active = true;
                            state.seen_active = true;
                            self.update_cond_active();
                            return Ok(());
                        }
                        // seen_active が true なら、このelseブランチも偽なので続けてスキップ
                    }
                }
                "elif" => {
                    // #elif: 条件を評価
                    if let Some(state) = self.cond_stack.last() {
                        if state.seen_else {
                            return Err(CompileError::Preprocess {
                                loc,
                                kind: PPError::ElifAfterElse,
                            });
                        }
                        if state.seen_active {
                            // 既に有効なブランチがあったので、この elif もスキップ
                            self.skip_to_eol()?;
                            continue;
                        }
                    }
                    // 条件を評価
                    let tokens = self.collect_if_condition()?;
                    let new_active = {
                        let mut eval = PPExprEvaluator::new(&tokens, &self.interner, &self.macros, loc.clone());
                        eval.evaluate()? != 0
                    };
                    if let Some(state) = self.cond_stack.last_mut() {
                        if new_active {
                            state.active = true;
                            state.seen_active = true;
                            self.update_cond_active();
                            return Ok(());
                        }
                        // 条件が偽なので続けてスキップ
                    }
                }
                _ => unreachable!(),
            }
        }
    }

    /// #ifdef / #ifndef を処理
    fn process_ifdef(&mut self, loc: SourceLocation, negate: bool) -> Result<(), CompileError> {
        // 親が無効な場合は文字レベルでスキップ
        if !self.cond_active {
            self.cond_stack.push(CondState {
                active: false,
                seen_active: false,
                seen_else: false,
                loc: loc.clone(),
            });
            self.skip_false_branch(loc)?;
            return Ok(());
        }

        let token = self.next_raw_token()?;
        // process_define と対称: キーワード名で #define されたマクロも検出する
        let defined = match token.kind {
            TokenKind::Ident(id) => self.macros.is_defined(id),
            ref kind => match kind.keyword_str() {
                Some(s) => {
                    let id = self.interner.intern(s);
                    self.macros.is_defined(id)
                }
                None => false,
            },
        };

        self.skip_to_eol()?;

        let active = if negate { !defined } else { defined };

        self.cond_stack.push(CondState {
            active,
            seen_active: active,
            seen_else: false,
            loc: loc.clone(),
        });

        self.update_cond_active();

        // 条件が偽の場合、TinyCC方式でスキップ
        if !active {
            self.skip_false_branch(loc)?;
        }

        Ok(())
    }

    /// #elif を処理
    /// 注: これは有効なブランチから呼ばれる（そのブランチは終了し、残りをスキップする必要がある）
    fn process_elif(&mut self, loc: SourceLocation) -> Result<(), CompileError> {
        if self.cond_stack.is_empty() {
            return Err(CompileError::Preprocess {
                loc,
                kind: PPError::UnmatchedEndif,
            });
        }

        let seen_else = self.cond_stack.last().unwrap().seen_else;
        if seen_else {
            return Err(CompileError::Preprocess {
                loc,
                kind: PPError::ElifAfterElse,
            });
        }

        // 有効なブランチを見た後なので、#endif までスキップ
        // (seen_active = true を維持したまま)
        self.skip_to_eol()?;
        self.skip_false_branch(loc)?;

        Ok(())
    }

    /// #else を処理
    /// 注: これは有効なブランチから呼ばれる（そのブランチは終了し、#else 以降をスキップする必要がある）
    fn process_else(&mut self, loc: SourceLocation) -> Result<(), CompileError> {
        if self.cond_stack.is_empty() {
            return Err(CompileError::Preprocess {
                loc,
                kind: PPError::UnmatchedElse,
            });
        }

        let seen_else = self.cond_stack.last().unwrap().seen_else;
        if seen_else {
            return Err(CompileError::Preprocess {
                loc,
                kind: PPError::UnmatchedElse,
            });
        }

        // seen_else をマーク
        if let Some(state) = self.cond_stack.last_mut() {
            state.seen_else = true;
        }

        // 有効なブランチを見た後なので、#endif までスキップ
        self.skip_to_eol()?;
        self.skip_false_branch(loc)?;

        Ok(())
    }

    /// #endif を処理
    fn process_endif(&mut self) -> Result<(), CompileError> {
        if self.cond_stack.is_empty() {
            return Err(CompileError::Preprocess {
                loc: SourceLocation::default(),
                kind: PPError::UnmatchedEndif,
            });
        }

        self.cond_stack.pop();
        self.skip_to_eol()?;
        self.update_cond_active();
        Ok(())
    }

    /// #error を処理
    fn process_error(&mut self, loc: SourceLocation) -> Result<(), CompileError> {
        let mut message = String::new();
        loop {
            let token = self.next_raw_token()?;
            match token.kind {
                TokenKind::Newline | TokenKind::Eof => break,
                TokenKind::Ident(id) => {
                    if !message.is_empty() { message.push(' '); }
                    message.push_str(self.interner.get(id));
                }
                TokenKind::StringLit(bytes) => {
                    if !message.is_empty() { message.push(' '); }
                    message.push_str(&String::from_utf8_lossy(&bytes));
                }
                _ => {
                    if !message.is_empty() { message.push(' '); }
                    message.push_str(&format!("{:?}", token.kind));
                }
            }
        }

        Err(CompileError::Preprocess {
            loc,
            kind: PPError::InvalidDirective(format!("#error {}", message)),
        })
    }

    /// 条件アクティブ状態を更新
    fn update_cond_active(&mut self) {
        self.cond_active = self.cond_stack.iter().all(|s| s.active);
    }

    /// #if条件用：マクロ展開付きでトークン収集
    /// TinyCC方式: マクロは展開するが、defined の引数は展開しない
    fn collect_if_condition(&mut self) -> Result<Vec<Token>, CompileError> {
        let mut tokens = Vec::new();
        let defined_id = self.interner.intern("defined");

        loop {
            // 生トークンを読む（先読みバッファから、またはソースから）
            let token = self.next_raw_token()?;

            match &token.kind {
                TokenKind::Newline | TokenKind::Eof => break,
                TokenKind::Ident(id) if *id == defined_id => {
                    // defined演算子の場合、引数は展開しない
                    tokens.push(token);

                    // 次のトークン（パーレンまたは識別子）を収集
                    let next = self.next_raw_token()?;
                    if matches!(next.kind, TokenKind::LParen) {
                        tokens.push(next);
                        // ( 内の識別子を収集（展開しない）
                        let ident = self.next_raw_token()?;
                        tokens.push(ident);
                        let rparen = self.next_raw_token()?;
                        tokens.push(rparen);
                    } else {
                        // defined IDENT 形式（parenthesisなし）
                        tokens.push(next);
                    }
                }
                TokenKind::Ident(id) => {
                    let id = *id;
                    // マクロ展開を試みる
                    if let Some(expanded) = self.try_expand_macro(id, &token)? {
                        // 展開されたトークンを先読みバッファに入れて再処理
                        for t in expanded.into_iter().rev() {
                            self.lookahead.push(t);
                        }
                    } else {
                        // 展開できなかった（未定義の識別子、または展開中のマクロ）
                        tokens.push(token);
                    }
                }
                _ => {
                    tokens.push(token);
                }
            }
        }

        // Debug: print collected tokens
        if self.config.debug_pp {
            eprintln!("DEBUG: collected tokens for #if condition:");
            for t in &tokens {
                eprintln!("  {:?}", t.kind);
            }
        }

        Ok(tokens)
    }

    /// #include <...> のパスを文字レベルで読み取る（TinyCC方式）
    fn scan_include_path(&mut self, terminator: char) -> Result<String, CompileError> {
        let source = self.sources.last_mut().ok_or_else(|| {
            CompileError::Preprocess {
                loc: SourceLocation::default(),
                kind: PPError::InvalidDirective("no source".to_string()),
            }
        })?;

        let loc = source.current_location();
        let mut path = String::new();

        loop {
            match source.peek() {
                Some(c) if c == terminator as u8 => {
                    source.advance();
                    break;
                }
                Some(b'\n') | None => {
                    return Err(CompileError::Preprocess {
                        loc,
                        kind: PPError::InvalidDirective("unterminated include path".to_string()),
                    });
                }
                Some(c) => {
                    source.advance();
                    path.push(c as char);
                }
            }
        }

        Ok(path)
    }

    /// 行末までスキップ
    fn skip_to_eol(&mut self) -> Result<(), CompileError> {
        loop {
            let token = self.next_raw_token()?;
            if matches!(token.kind, TokenKind::Newline | TokenKind::Eof) {
                break;
            }
        }
        Ok(())
    }

    /// 行末までスキップ（ブロックコメントを正しく処理）
    /// preprocess_skip内で使用するための静的メソッド
    fn skip_to_eol_raw(source: &mut InputSource) {
        loop {
            match source.peek() {
                Some(b'\n') | None => break,
                Some(b'/') => {
                    // コメントかどうかチェック
                    if source.peek_n(1) == Some(b'*') {
                        // ブロックコメントをスキップ
                        source.advance(); // '/'
                        source.advance(); // '*'
                        loop {
                            match (source.peek(), source.peek_n(1)) {
                                (Some(b'*'), Some(b'/')) => {
                                    source.advance();
                                    source.advance();
                                    break;
                                }
                                (Some(_), _) => { source.advance(); }
                                (None, _) => break,
                            }
                        }
                    } else if source.peek_n(1) == Some(b'/') {
                        // 行コメント - 行末までスキップ
                        while source.peek().is_some_and(|c| c != b'\n') {
                            source.advance();
                        }
                        break;
                    } else {
                        source.advance();
                    }
                }
                Some(b'\\') => {
                    // 行継続
                    source.advance();
                    if source.peek() == Some(b'\n') {
                        source.advance();
                    } else if source.peek() == Some(b'\r') {
                        source.advance();
                        if source.peek() == Some(b'\n') {
                            source.advance();
                        }
                    }
                }
                Some(_) => { source.advance(); }
            }
        }
    }

    /// TinyCC方式: 条件が偽のブロックをスキップ
    /// トークナイザを使わず文字レベルでスキャンし、#else/#elif/#endif を見つけるまでスキップ
    /// 戻り値: 見つかったディレクティブ名 ("else", "elif", "endif")
    fn preprocess_skip(&mut self) -> Result<String, CompileError> {
        let mut depth = 0i32;  // #if のネスト深度

        loop {
            let source = match self.sources.last_mut() {
                Some(s) => s,
                None => {
                    return Err(CompileError::Preprocess {
                        loc: SourceLocation::default(),
                        kind: PPError::MissingEndif,
                    });
                }
            };

            // 行頭フラグをリセット
            let mut at_line_start = source.is_at_line_start();

            loop {
                let c = match source.peek() {
                    Some(c) => c,
                    None => break, // このソースは終了、外側ループで次のソースへ
                };

                match c {
                    // 空白はスキップ
                    b' ' | b'\t' | b'\r' | 0x0C | 0x0B => {
                        source.advance();
                    }
                    // 改行
                    b'\n' => {
                        source.advance();
                        at_line_start = true;
                    }
                    // 行継続
                    b'\\' => {
                        source.advance();
                        if source.peek() == Some(b'\n') {
                            source.advance();
                        } else if source.peek() == Some(b'\r') {
                            source.advance();
                            if source.peek() == Some(b'\n') {
                                source.advance();
                            }
                        }
                    }
                    // 文字列リテラル（スキップ）
                    b'"' | b'\'' => {
                        let quote = c;
                        source.advance();
                        loop {
                            match source.peek() {
                                Some(c) if c == quote => {
                                    source.advance();
                                    break;
                                }
                                Some(b'\\') => {
                                    source.advance();
                                    source.advance(); // エスケープ文字をスキップ
                                }
                                Some(b'\n') | None => break,
                                Some(_) => {
                                    source.advance();
                                }
                            }
                        }
                        at_line_start = false;
                    }
                    // コメント
                    b'/' => {
                        source.advance();
                        match source.peek() {
                            Some(b'/') => {
                                // 行コメント
                                while source.peek().is_some_and(|c| c != b'\n') {
                                    source.advance();
                                }
                            }
                            Some(b'*') => {
                                // ブロックコメント
                                source.advance();
                                loop {
                                    match (source.peek(), source.peek_n(1)) {
                                        (Some(b'*'), Some(b'/')) => {
                                            source.advance();
                                            source.advance();
                                            break;
                                        }
                                        (Some(_), _) => {
                                            source.advance();
                                        }
                                        (None, _) => break,
                                    }
                                }
                            }
                            _ => {}
                        }
                        at_line_start = false;
                    }
                    // プリプロセッサディレクティブ
                    b'#' if at_line_start => {
                        source.advance();
                        // 空白をスキップ
                        while matches!(source.peek(), Some(b' ') | Some(b'\t')) {
                            source.advance();
                        }
                        // ディレクティブ名を読む
                        let mut directive = String::new();
                        while let Some(c) = source.peek() {
                            if c.is_ascii_alphabetic() || c == b'_' {
                                directive.push(c as char);
                                source.advance();
                            } else {
                                break;
                            }
                        }

                        match directive.as_str() {
                            "if" | "ifdef" | "ifndef" => {
                                depth += 1;
                                // 行末までスキップ
                                while source.peek().is_some_and(|c| c != b'\n') {
                                    source.advance();
                                }
                            }
                            "endif" => {
                                if depth == 0 {
                                    // 行末までスキップしてから戻る（コメントを考慮）
                                    Self::skip_to_eol_raw(source);
                                    return Ok("endif".to_string());
                                }
                                depth -= 1;
                                Self::skip_to_eol_raw(source);
                            }
                            "else" if depth == 0 => {
                                Self::skip_to_eol_raw(source);
                                return Ok("else".to_string());
                            }
                            "elif" if depth == 0 => {
                                // elifの場合は条件式を読む必要があるので、行末までスキップせずに戻る
                                return Ok("elif".to_string());
                            }
                            _ => {
                                // その他のディレクティブは行末までスキップ（コメントを考慮）
                                Self::skip_to_eol_raw(source);
                            }
                        }
                        at_line_start = false;
                    }
                    // その他の文字
                    _ => {
                        source.advance();
                        at_line_start = false;
                    }
                }
            }

            // このソースが終了したら次のソースへ
            if self.sources.len() > 1 {
                self.sources.pop();
            } else {
                return Err(CompileError::Preprocess {
                    loc: SourceLocation::default(),
                    kind: PPError::MissingEndif,
                });
            }
        }
    }

    /// マクロ展開を試みる
    fn try_expand_macro(&mut self, id: InternedStr, token: &Token) -> Result<Option<Vec<Token>>, CompileError> {
        self.try_expand_macro_internal(id, token, false)
    }

    /// マクロ展開を試みる（内部実装）
    fn try_expand_macro_internal(
        &mut self,
        id: InternedStr,
        token: &Token,
        preserve_function_macros: bool,
    ) -> Result<Option<Vec<Token>>, CompileError> {
        // グローバルな展開抑制リストをチェック（bindings.rs の定数など）
        if self.skip_expand_macros.contains(&id) {
            return Ok(None);
        }

        // トークンが展開禁止リストにこのマクロを持っている場合は展開しない
        if self.no_expand_registry.is_blocked(token.id, id) {
            return Ok(None);
        }

        let def = match self.macros.get(id) {
            Some(def) => def.clone(),
            None => return Ok(None),
        };

        // トリガートークンのIDと展開するマクロIDを記録
        let trigger_token_id = token.id;

        // マクロ呼び出し位置を保存
        let call_loc = token.loc.clone();

        match &def.kind {
            MacroKind::Object => {
                let empty = HashMap::new();
                let expanded = self.expand_tokens(&def.body, &empty, &empty)?;
                // 全トークンに展開禁止情報と呼び出し位置を適用
                let marked = self.mark_expanded_with_registry(expanded, trigger_token_id, id, &call_loc);
                // コールバック呼び出し（展開後）
                // 借用の問題を避けるため、一時的にコールバックを取り出す
                if let Some(mut cb) = self.macro_called_callbacks.remove(&id) {
                    cb.on_macro_called(None, &self.interner);
                    self.macro_called_callbacks.insert(id, cb);
                }
                // マーカーで囲む（emit_markers が有効な場合のみ）
                let wrapped = self.wrap_with_markers(
                    marked,
                    id,
                    token,
                    MacroInvocationKind::Object,
                    &call_loc,
                    def.has_token_pasting,
                );
                Ok(Some(wrapped))
            }
            MacroKind::Function { params, is_variadic } => {
                // preserve_function_macros モード: explicit_expand に含まれない関数マクロは保存
                if preserve_function_macros && !self.explicit_expand_macros.contains(&id) {
                    return Ok(None);
                }

                // C標準: 関数形式マクロの識別子と ( の間に空白（改行を含む）があっても良い
                // 改行をスキップして ( を探す
                let mut skipped_newlines = Vec::new();
                let next = loop {
                    let t = self.next_raw_token()?;
                    if matches!(t.kind, TokenKind::Newline) {
                        skipped_newlines.push(t);
                    } else {
                        break t;
                    }
                };
                if !matches!(next.kind, TokenKind::LParen) {
                    // ( がない場合、改行と次のトークンを戻す
                    self.lookahead.push(next);
                    for t in skipped_newlines.into_iter().rev() {
                        self.lookahead.push(t);
                    }
                    return Ok(None);
                }

                let args = self.collect_macro_args(params.len(), *is_variadic)?;

                let mut arg_map = HashMap::new();

                if *is_variadic && !params.is_empty() {
                    // GNU拡張: NAME... → 最後のパラメータが可変長引数名
                    // C99標準: ...    → __VA_ARGS__ がパラメータとして登録済み
                    // どちらの場合も最後のパラメータが可変長引数を受け取る
                    let va_args_id = self.interner.intern("__VA_ARGS__");
                    let last_param = *params.last().unwrap();
                    let is_gnu_style = last_param != va_args_id;

                    // 通常のパラメータをマップ（最後のパラメータは可変長なので除外）
                    let normal_param_count = params.len() - 1;
                    for (i, param) in params.iter().take(normal_param_count).enumerate() {
                        if i < args.len() {
                            arg_map.insert(*param, args[i].clone());
                        } else {
                            arg_map.insert(*param, Vec::new());
                        }
                    }

                    // 可変長引数を構築
                    let mut va = Vec::new();
                    let va_start = normal_param_count;
                    for (i, arg) in args.iter().enumerate().skip(va_start) {
                        if i > va_start {
                            va.push(Token::new(TokenKind::Comma, token.loc.clone()));
                        }
                        va.extend(arg.clone());
                    }

                    if is_gnu_style {
                        // GNU拡張: 名前付きパラメータに格納
                        arg_map.insert(last_param, va.clone());
                        // __VA_ARGS__ もエイリアスとして登録（互換性のため）
                        arg_map.insert(va_args_id, va);
                    } else {
                        // 標準形式: __VA_ARGS__ に格納
                        arg_map.insert(va_args_id, va);
                    }
                } else {
                    // 非可変長マクロ
                    for (i, param) in params.iter().enumerate() {
                        if i < args.len() {
                            arg_map.insert(*param, args[i].clone());
                        } else {
                            arg_map.insert(*param, Vec::new());
                        }
                    }
                }

                // 引数をprescan（# や ## で使われない引数は先に展開される）
                let prescanned_args = self.prescan_args(&arg_map)?;

                let expanded = self.expand_tokens(&def.body, &arg_map, &prescanned_args)?;
                // 全トークンに展開禁止情報と呼び出し位置を適用
                let marked = self.mark_expanded_with_registry(expanded, trigger_token_id, id, &call_loc);
                // コールバック呼び出し（展開後、wrap_with_markers が args を move する前）
                // 借用の問題を避けるため、一時的にコールバックを取り出す
                if let Some(mut cb) = self.macro_called_callbacks.remove(&id) {
                    cb.on_macro_called(Some(&args), &self.interner);
                    self.macro_called_callbacks.insert(id, cb);
                }
                // マーカーで囲む（emit_markers が有効な場合のみ）
                // wrapped マクロ（assert 等）の場合、引数内のマクロも展開する
                let kind = if self.wrapped_macros.contains(&id) {
                    let expanded_args: Result<Vec<_>, _> = args.into_iter()
                        .map(|arg_tokens| {
                            // 関数マクロ保存モードで展開（SvTYPE 等を保存、オブジェクトマクロは展開）
                            let expanded = self.expand_token_list_preserve_fn(&arg_tokens)?;
                            // 展開結果からマーカーを除去（入れ子 assert エラー防止）
                            Ok(expanded.into_iter()
                                .filter(|t| !matches!(t.kind, TokenKind::MacroBegin(_) | TokenKind::MacroEnd(_)))
                                .collect())
                        })
                        .collect();
                    MacroInvocationKind::Function { args: expanded_args? }
                } else {
                    MacroInvocationKind::Function { args }
                };
                let wrapped = self.wrap_with_markers(
                    marked,
                    id,
                    token,
                    kind,
                    &call_loc,
                    def.has_token_pasting,
                );
                Ok(Some(wrapped))
            }
        }
    }

    /// トークン列に展開禁止情報と呼び出し位置を適用（NoExpandRegistry使用版）
    ///
    /// マクロ展開後のトークンには、マクロ呼び出し位置を設定し、
    /// NoExpandRegistryに展開禁止情報を登録する。
    fn mark_expanded_with_registry(
        &mut self,
        tokens: Vec<Token>,
        trigger_token_id: TokenId,
        macro_id: InternedStr,
        call_loc: &SourceLocation,
    ) -> Vec<Token> {
        tokens.into_iter().map(|mut t| {
            // 新しいトークンIDで展開禁止情報を継承
            self.no_expand_registry.inherit(trigger_token_id, t.id);
            // 現在のマクロも展開禁止に追加
            self.no_expand_registry.add(t.id, macro_id);
            // マクロ呼び出し位置を設定
            t.loc = call_loc.clone();
            t
        }).collect()
    }

    /// マクロ展開結果を MacroBegin/MacroEnd マーカーで囲む
    ///
    /// emit_markers が有効な場合、または wrapped_macros に含まれる場合にマーカーを追加する。
    /// マーカーはパーサーがマクロ展開情報をASTに付与するために使用される。
    /// wrapped_macros に含まれるマクロは is_wrapped フラグが true になり、
    /// パーサーで特殊処理（assert の復元など）が可能になる。
    ///
    /// `has_token_pasting`: マクロ本体にトークン連結 (##) を含むか
    fn wrap_with_markers(
        &self,
        tokens: Vec<Token>,
        macro_name: InternedStr,
        trigger_token: &Token,
        kind: MacroInvocationKind,
        call_loc: &SourceLocation,
        has_token_pasting: bool,
    ) -> Vec<Token> {
        let is_wrapped = self.wrapped_macros.contains(&macro_name);

        // emit_markers が off でも、wrapped_macros に含まれていればマーカー出力
        if !self.config.emit_markers && !is_wrapped {
            return tokens;
        }

        let marker_id = TokenId::next();

        // preserve_call を決定:
        // - オブジェクトマクロ → false（引数がないので保存不要）
        // - トークンペースト(##)を含むマクロ → false
        // - explicit_expand_macros に登録されたマクロ → false
        // - その他の関数マクロ → true
        let is_function_macro = matches!(kind, MacroInvocationKind::Function { .. });
        let preserve_call = is_function_macro
            && !has_token_pasting
            && !self.explicit_expand_macros.contains(&macro_name);

        // MacroBegin マーカーを作成
        let begin_info = MacroBeginInfo {
            marker_id,
            trigger_token_id: trigger_token.id,
            macro_name,
            kind,
            call_loc: call_loc.clone(),
            is_wrapped,
            preserve_call,
        };
        let begin_token = Token::new(
            TokenKind::MacroBegin(Box::new(begin_info)),
            call_loc.clone(),
        );

        // MacroEnd マーカーを作成
        let end_info = MacroEndInfo {
            begin_marker_id: marker_id,
        };
        let end_token = Token::new(TokenKind::MacroEnd(end_info), call_loc.clone());

        // [MacroBegin, ...tokens..., MacroEnd] の形式で返す
        let mut result = Vec::with_capacity(tokens.len() + 2);
        result.push(begin_token);
        result.extend(tokens);
        result.push(end_token);
        result
    }

    /// マクロ引数を収集
    fn collect_macro_args(&mut self, param_count: usize, is_variadic: bool) -> Result<Vec<Vec<Token>>, CompileError> {
        let mut args = Vec::new();
        let mut current_arg = Vec::new();
        let mut paren_depth = 0;

        loop {
            let token = self.next_raw_token()?;
            match token.kind {
                TokenKind::LParen => {
                    paren_depth += 1;
                    current_arg.push(token);
                }
                TokenKind::RParen => {
                    if paren_depth == 0 {
                        if !current_arg.is_empty() || !args.is_empty() {
                            args.push(current_arg);
                        }
                        break;
                    }
                    paren_depth -= 1;
                    current_arg.push(token);
                }
                TokenKind::Comma if paren_depth == 0 => {
                    if is_variadic && args.len() >= param_count {
                        current_arg.push(token);
                    } else {
                        args.push(current_arg);
                        current_arg = Vec::new();
                    }
                }
                TokenKind::Eof => {
                    return Err(CompileError::Preprocess {
                        loc: token.loc,
                        kind: PPError::InvalidMacroArgs("unterminated macro arguments".to_string()),
                    });
                }
                TokenKind::Newline => continue,
                _ => current_arg.push(token),
            }
        }

        Ok(args)
    }

    /// マクロ引数をprescan（展開）する
    /// C標準: # や ## で使われない引数は先に展開される
    fn prescan_args(&mut self, args: &HashMap<InternedStr, Vec<Token>>) -> Result<HashMap<InternedStr, Vec<Token>>, CompileError> {
        let mut prescanned = HashMap::new();
        for (param, tokens) in args.iter() {
            // 引数トークンを展開（再帰的マクロ展開）
            let expanded = self.expand_token_list(tokens)?;
            prescanned.insert(*param, expanded);
        }
        Ok(prescanned)
    }

    /// トークンリストを展開（引数prescan用）
    /// マクロ展開を行うが、ソースからは読まない
    fn expand_token_list(&mut self, tokens: &[Token]) -> Result<Vec<Token>, CompileError> {
        self.expand_token_list_internal(tokens, false)
    }

    /// トークンリストを展開（関数マクロ保存モード）
    ///
    /// `explicit_expand_macros` に登録されていない関数マクロは展開されず、
    /// 関数呼び出しとして保存される。オブジェクトマクロは通常通り展開される。
    fn expand_token_list_preserve_fn(&mut self, tokens: &[Token]) -> Result<Vec<Token>, CompileError> {
        self.expand_token_list_internal(tokens, true)
    }

    /// トークンリストを展開（内部実装）
    fn expand_token_list_internal(
        &mut self,
        tokens: &[Token],
        preserve_function_macros: bool,
    ) -> Result<Vec<Token>, CompileError> {
        if tokens.is_empty() {
            return Ok(Vec::new());
        }

        // 既存のlookaheadを保存
        let saved_lookahead = std::mem::take(&mut self.lookahead);

        // トークンを先読みバッファに追加（逆順で追加すると正順で取り出せる）
        // 終端マーカーとしてEofを追加（ソースからの読み込みを防ぐ）
        self.lookahead.push(Token::new(TokenKind::Eof, SourceLocation::default()));
        for token in tokens.iter().rev() {
            self.lookahead.push(token.clone());
        }

        // トークンを1つずつ処理して結果を収集
        let mut result = Vec::new();
        while let Some(token) = self.lookahead.pop() {
            // 終端マーカーに到達したら終了
            if matches!(token.kind, TokenKind::Eof) {
                break;
            }

            // 改行はスキップ
            if matches!(token.kind, TokenKind::Newline) {
                continue;
            }

            // マクロ展開を試みる
            if let TokenKind::Ident(id) = token.kind {
                if let Some(expanded) = self.try_expand_macro_internal(id, &token, preserve_function_macros)? {
                    // 展開結果を逆順でlookaheadに戻す
                    for t in expanded.into_iter().rev() {
                        self.lookahead.push(t);
                    }
                    continue;
                }
            }

            result.push(token);
        }

        // lookaheadを復元
        self.lookahead = saved_lookahead;

        Ok(result)
    }

    /// トークン列を展開
    /// raw_args: # や ## で使用（展開前の引数）
    /// prescanned_args: 通常の置換で使用（展開済みの引数）
    fn expand_tokens(&mut self, tokens: &[Token], raw_args: &HashMap<InternedStr, Vec<Token>>, prescanned_args: &HashMap<InternedStr, Vec<Token>>) -> Result<Vec<Token>, CompileError> {
        let mut result = Vec::new();
        let mut i = 0;

        while i < tokens.len() {
            let token = &tokens[i];

            match &token.kind {
                TokenKind::Hash if i + 1 < tokens.len() => {
                    if let TokenKind::Ident(param_id) = tokens[i + 1].kind {
                        // # はraw引数を使用
                        if let Some(arg_tokens) = raw_args.get(&param_id) {
                            let stringified = self.stringify_tokens(arg_tokens);
                            result.push(Token::new(
                                TokenKind::StringLit(stringified.into_bytes()),
                                token.loc.clone(),
                            ));
                            i += 2;
                            continue;
                        }
                    }
                    return Err(CompileError::Preprocess {
                        loc: token.loc.clone(),
                        kind: PPError::InvalidStringize,
                    });
                }
                TokenKind::HashHash => {
                    if result.is_empty() || i + 1 >= tokens.len() {
                        return Err(CompileError::Preprocess {
                            loc: token.loc.clone(),
                            kind: PPError::InvalidTokenPaste,
                        });
                    }

                    // 左辺のトークンを取得
                    let left = result.pop().unwrap();

                    // 右辺のトークンを取得（## はraw引数を使用）
                    i += 1;
                    let right_token = &tokens[i];
                    let right_tokens = if let TokenKind::Ident(id) = right_token.kind {
                        if let Some(arg_tokens) = raw_args.get(&id) {
                            arg_tokens.clone()
                        } else {
                            vec![right_token.clone()]
                        }
                    } else {
                        vec![right_token.clone()]
                    };

                    // トークン連結を実行
                    let pasted = self.paste_tokens(&left, &right_tokens, &token.loc)?;
                    result.extend(pasted);
                    i += 1;
                    continue;
                }
                TokenKind::Ident(id) => {
                    // 通常の置換はprescanned引数を使用
                    if let Some(arg_tokens) = prescanned_args.get(id) {
                        result.extend(arg_tokens.iter().cloned());
                    } else {
                        result.push(token.clone());
                    }
                }
                _ => result.push(token.clone()),
            }

            i += 1;
        }

        Ok(result)
    }

    /// トークン連結 (##)
    fn paste_tokens(&mut self, left: &Token, right: &[Token], loc: &SourceLocation) -> Result<Vec<Token>, CompileError> {
        // 左辺と右辺の文字列表現を取得
        let left_str = self.token_to_string(left);

        // 右辺が空の場合は左辺のみ返す
        if right.is_empty() {
            return Ok(vec![left.clone()]);
        }

        // 右辺の最初のトークンと連結
        let right_first_str = self.token_to_string(&right[0]);
        let pasted_str = format!("{}{}", left_str, right_first_str);

        // 連結結果を再トークン化
        let pasted_tokens = self.tokenize_string(&pasted_str);

        // 右辺の残りのトークンを追加
        let mut result = pasted_tokens;
        result.extend(right.iter().skip(1).cloned());

        // 位置情報を更新
        for t in &mut result {
            t.loc = loc.clone();
        }

        Ok(result)
    }

    /// トークンを文字列表現に変換
    fn token_to_string(&self, token: &Token) -> String {
        match &token.kind {
            TokenKind::Ident(id) => self.interner.get(*id).to_string(),
            TokenKind::IntLit(n) => n.to_string(),
            TokenKind::UIntLit(n) => n.to_string(),
            TokenKind::FloatLit(f) => f.to_string(),
            TokenKind::StringLit(s) => format!("\"{}\"", String::from_utf8_lossy(s)),
            TokenKind::CharLit(c) => format!("'{}'", *c as char),
            TokenKind::WideCharLit(c) => format!("L'{}'", char::from_u32(*c).unwrap_or('?')),
            TokenKind::Plus => "+".to_string(),
            TokenKind::Minus => "-".to_string(),
            TokenKind::Star => "*".to_string(),
            TokenKind::Slash => "/".to_string(),
            TokenKind::Percent => "%".to_string(),
            TokenKind::Amp => "&".to_string(),
            TokenKind::Pipe => "|".to_string(),
            TokenKind::Caret => "^".to_string(),
            TokenKind::Tilde => "~".to_string(),
            TokenKind::Bang => "!".to_string(),
            TokenKind::Lt => "<".to_string(),
            TokenKind::Gt => ">".to_string(),
            TokenKind::Eq => "=".to_string(),
            TokenKind::Question => "?".to_string(),
            TokenKind::Colon => ":".to_string(),
            TokenKind::Dot => ".".to_string(),
            TokenKind::Comma => ",".to_string(),
            TokenKind::Semi => ";".to_string(),
            TokenKind::LParen => "(".to_string(),
            TokenKind::RParen => ")".to_string(),
            TokenKind::LBracket => "[".to_string(),
            TokenKind::RBracket => "]".to_string(),
            TokenKind::LBrace => "{".to_string(),
            TokenKind::RBrace => "}".to_string(),
            TokenKind::Arrow => "->".to_string(),
            TokenKind::PlusPlus => "++".to_string(),
            TokenKind::MinusMinus => "--".to_string(),
            TokenKind::LtLt => "<<".to_string(),
            TokenKind::GtGt => ">>".to_string(),
            TokenKind::LtEq => "<=".to_string(),
            TokenKind::GtEq => ">=".to_string(),
            TokenKind::EqEq => "==".to_string(),
            TokenKind::BangEq => "!=".to_string(),
            TokenKind::AmpAmp => "&&".to_string(),
            TokenKind::PipePipe => "||".to_string(),
            TokenKind::PlusEq => "+=".to_string(),
            TokenKind::MinusEq => "-=".to_string(),
            TokenKind::StarEq => "*=".to_string(),
            TokenKind::SlashEq => "/=".to_string(),
            TokenKind::PercentEq => "%=".to_string(),
            TokenKind::AmpEq => "&=".to_string(),
            TokenKind::PipeEq => "|=".to_string(),
            TokenKind::CaretEq => "^=".to_string(),
            TokenKind::LtLtEq => "<<=".to_string(),
            TokenKind::GtGtEq => ">>=".to_string(),
            TokenKind::Ellipsis => "...".to_string(),
            TokenKind::Hash => "#".to_string(),
            TokenKind::HashHash => "##".to_string(),
            _ => String::new(),
        }
    }

    /// トークン列を文字列化
    fn stringify_tokens(&self, tokens: &[Token]) -> String {
        let mut result = String::new();
        for (i, token) in tokens.iter().enumerate() {
            if i > 0 { result.push(' '); }
            match &token.kind {
                TokenKind::Ident(id) => result.push_str(self.interner.get(*id)),
                TokenKind::IntLit(n) => result.push_str(&n.to_string()),
                TokenKind::UIntLit(n) => result.push_str(&format!("{}u", n)),
                TokenKind::FloatLit(f) => result.push_str(&f.to_string()),
                TokenKind::StringLit(s) => {
                    result.push('"');
                    result.push_str(&String::from_utf8_lossy(s));
                    result.push('"');
                }
                TokenKind::CharLit(c) => {
                    result.push('\'');
                    result.push(*c as char);
                    result.push('\'');
                }
                _ => result.push_str(&format!("{:?}", token.kind)),
            }
        }
        result
    }

    /// ファイルレジストリへの参照
    pub fn files(&self) -> &FileRegistry {
        &self.files
    }

    /// 文字列インターナーへの参照
    pub fn interner(&self) -> &StringInterner {
        &self.interner
    }

    /// 文字列インターナーへの可変参照
    pub fn interner_mut(&mut self) -> &mut StringInterner {
        &mut self.interner
    }

    /// マクロテーブルへの参照
    pub fn macros(&self) -> &MacroTable {
        &self.macros
    }

    /// マクロ本体を展開する（MacroInferContext 用）
    ///
    /// TokenExpander の代替として、マクロ推論で使用する。
    /// トークンペースト（##）と文字列化（#）を完全にサポートする。
    ///
    /// # Arguments
    /// - `body`: マクロ本体のトークン列
    /// - `params`: パラメータ名のリスト（関数マクロの場合）
    /// - `args`: 各パラメータに対応する引数トークン列
    /// - `in_progress`: 再帰展開防止用のマクロ名セット
    ///
    /// # Returns
    /// (展開結果, 呼び出されたマクロ集合)
    pub fn expand_macro_body_for_inference(
        &mut self,
        body: &[Token],
        params: &[InternedStr],
        args: &[Vec<Token>],
        in_progress: &mut HashSet<InternedStr>,
    ) -> Result<(Vec<Token>, HashSet<InternedStr>), CompileError> {
        let mut called_macros = HashSet::new();

        // パラメータと引数のマッピングを作成
        let mut raw_args = HashMap::new();
        let mut prescanned_args = HashMap::new();

        for (i, &param) in params.iter().enumerate() {
            if let Some(arg_tokens) = args.get(i) {
                // raw_args: ## や # で使用（展開前）
                raw_args.insert(param, arg_tokens.clone());
                // prescanned_args: 通常の置換で使用（展開済み）
                // 引数を展開してから使用
                let (expanded_arg, arg_called) = self.expand_tokens_for_inference(
                    arg_tokens,
                    in_progress,
                )?;
                called_macros.extend(arg_called);
                prescanned_args.insert(param, expanded_arg);
            }
        }

        // パラメータ置換と ##, # を処理
        let substituted = self.expand_tokens(body, &raw_args, &prescanned_args)?;

        // 展開後のトークン列内のマクロを再帰的に展開
        let (result, more_called) = self.expand_tokens_for_inference(&substituted, in_progress)?;
        called_macros.extend(more_called);

        Ok((result, called_macros))
    }

    /// トークン列内のマクロを展開する（MacroInferContext 用）
    ///
    /// 再帰的にマクロを展開し、呼び出されたマクロを記録する。
    fn expand_tokens_for_inference(
        &mut self,
        tokens: &[Token],
        in_progress: &mut HashSet<InternedStr>,
    ) -> Result<(Vec<Token>, HashSet<InternedStr>), CompileError> {
        let mut result = Vec::new();
        let mut called_macros = HashSet::new();
        let mut i = 0;

        while i < tokens.len() {
            let token = &tokens[i];

            if let TokenKind::Ident(id) = token.kind {
                // グローバルな展開抑制リストをチェック
                if self.skip_expand_macros.contains(&id) {
                    result.push(token.clone());
                    i += 1;
                    continue;
                }

                // 再帰展開防止
                if in_progress.contains(&id) {
                    result.push(token.clone());
                    i += 1;
                    continue;
                }

                // マクロ定義を取得
                if let Some(def) = self.macros.get(id).cloned() {
                    match &def.kind {
                        MacroKind::Object => {
                            // オブジェクトマクロは常に展開
                            called_macros.insert(id);
                            in_progress.insert(id);
                            let (expanded, more_called) = self.expand_macro_body_for_inference(
                                &def.body,
                                &[],
                                &[],
                                in_progress,
                            )?;
                            called_macros.extend(more_called);
                            result.extend(expanded);
                            in_progress.remove(&id);
                            i += 1;
                            continue;
                        }
                        MacroKind::Function { params, is_variadic } => {
                            // 関数マクロ: 引数を収集できるかチェック
                            if let Some((args, consumed)) = self.try_collect_args_from_tokens(&tokens[i + 1..], params.len(), *is_variadic) {
                                // 置換列が空の関数マクロ (非 DEBUGGING perl の DEBUG_Xv 等)
                                // は C と同様、呼び出しごと消える — 引数は評価されず
                                // 依存も生まない。preserve してしまうと C では存在しない
                                // 呼び出しが AST に残り、引数内の PTR2UV / PerlIO_printf
                                // などが偽の依存になって cascade を汚染する。
                                // (名前でなくマクロ表の観測形状で判定するので、DEBUGGING
                                //  perl では通常の展開/保存経路に自然に戻る)
                                if def
                                    .body
                                    .iter()
                                    .all(|t| matches!(t.kind, TokenKind::Space | TokenKind::Newline))
                                {
                                    let _ = args; // 未評価で破棄
                                    i += 1 + consumed;
                                    continue;
                                }

                                // 関数マクロの呼び出しを記録
                                called_macros.insert(id);

                                // preserve_function_macros モード: explicit_expand に含まれない場合は保存
                                if !self.explicit_expand_macros.contains(&id) {
                                    // 関数名は保存
                                    result.push(token.clone());

                                    // 引数を展開してから保存
                                    // 開き括弧
                                    result.push(Token::new(TokenKind::LParen, token.loc.clone()));

                                    for (arg_idx, arg_tokens) in args.iter().enumerate() {
                                        if arg_idx > 0 {
                                            result.push(Token::new(TokenKind::Comma, token.loc.clone()));
                                        }
                                        // 引数内のオブジェクトマクロを展開
                                        let (expanded_arg, arg_called) = self.expand_tokens_for_inference(
                                            arg_tokens,
                                            in_progress,
                                        )?;
                                        called_macros.extend(arg_called);
                                        result.extend(expanded_arg);
                                    }

                                    // 閉じ括弧
                                    result.push(Token::new(TokenKind::RParen, token.loc.clone()));

                                    i += 1 + consumed;
                                    continue;
                                }

                                // 展開する
                                in_progress.insert(id);
                                let (expanded, more_called) = self.expand_macro_body_for_inference(
                                    &def.body,
                                    params,
                                    &args,
                                    in_progress,
                                )?;
                                called_macros.extend(more_called);
                                result.extend(expanded);
                                in_progress.remove(&id);
                                i += 1 + consumed;
                                continue;
                            } else {
                                // 引数がない: 関数マクロとして認識されない
                                result.push(token.clone());
                            }
                        }
                    }
                } else {
                    result.push(token.clone());
                }
            } else {
                result.push(token.clone());
            }

            i += 1;
        }

        Ok((result, called_macros))
    }

    /// トークン列から関数マクロの引数を収集する（MacroInferContext 用）
    ///
    /// ファイルからではなく、トークン列から引数を収集する。
    fn try_collect_args_from_tokens(
        &self,
        tokens: &[Token],
        param_count: usize,
        is_variadic: bool,
    ) -> Option<(Vec<Vec<Token>>, usize)> {
        // 空白をスキップして '(' を探す
        let mut start = 0;
        while start < tokens.len() {
            match &tokens[start].kind {
                TokenKind::Space | TokenKind::Newline => start += 1,
                TokenKind::LParen => break,
                _ => return None,
            }
        }

        if start >= tokens.len() || !matches!(tokens[start].kind, TokenKind::LParen) {
            return None;
        }

        // 引数を収集
        let mut args: Vec<Vec<Token>> = Vec::new();
        let mut current_arg = Vec::new();
        let mut paren_depth = 0;
        let mut i = start + 1;

        while i < tokens.len() {
            let token = &tokens[i];
            match &token.kind {
                TokenKind::LParen => {
                    paren_depth += 1;
                    current_arg.push(token.clone());
                }
                TokenKind::RParen => {
                    if paren_depth == 0 {
                        // 引数終了
                        if !current_arg.is_empty() || !args.is_empty() {
                            args.push(current_arg);
                        }
                        // 消費したトークン数を返す（start から i まで、i を含む）
                        return Some((args, i + 1));
                    }
                    paren_depth -= 1;
                    current_arg.push(token.clone());
                }
                TokenKind::Comma if paren_depth == 0 => {
                    // 可変長引数の場合、必要な引数数を超えたらカンマも含める
                    if is_variadic && args.len() >= param_count.saturating_sub(1) {
                        current_arg.push(token.clone());
                    } else {
                        args.push(current_arg);
                        current_arg = Vec::new();
                    }
                }
                TokenKind::Space | TokenKind::Newline => {
                    // 空白は無視（ただし引数内の空白は保持が必要な場合も）
                    if !current_arg.is_empty() {
                        current_arg.push(token.clone());
                    }
                }
                _ => {
                    current_arg.push(token.clone());
                }
            }
            i += 1;
        }

        // ')' が見つからなかった
        None
    }

    /// 現在のファイルがターゲットディレクトリ内かどうかを判定
    fn is_current_file_in_target(&self) -> bool {
        let target_dir = match &self.config.target_dir {
            Some(dir) => dir,
            None => return false,
        };

        let file_id = match self.sources.last() {
            Some(source) => source.file_id,
            None => return false,
        };

        let path = self.files.get_path(file_id);
        path.starts_with(target_dir)
    }

    /// 全トークンを収集
    pub fn collect_tokens(&mut self) -> Result<Vec<Token>, CompileError> {
        let mut tokens = Vec::new();
        loop {
            let token = self.next_token()?;
            if matches!(token.kind, TokenKind::Eof) {
                break;
            }
            tokens.push(token);
        }
        Ok(tokens)
    }
}

/// Utility function that parses suffixes that floats can have:
/// f/F and l/L were already supported.
/// Adding:
/// - f16,  f32,  f64,  f128
/// - f16x, f32x, f64x, f128x
/// - bf16
/// - q/Q
/// - w/W
fn skip_float_suffix(source: &mut InputSource) {
    match source.peek() {
        // These could be the short ones, the f__ or f__x suffixes
        Some(b'f') | Some(b'F') => {
            source.advance(); // f
            // check if digit is next. fx is not allowed. There must be atleast one digit
            if source.peek().as_ref().is_some_and(u8::is_ascii_digit) {
                // Lazy: Skip all the digits instead of checking them.
                // This will also skip the first, so I don't need separate advance
                while source.peek().as_ref().is_some_and(u8::is_ascii_digit) {
                    source.advance();
                }
                // At least one digit was present
                // Optional x suffix
                if matches!(source.peek(), Some(b'x') | Some(b'X')) {
                    source.advance();
                }
            }
        }
        // This must be the bf16
        Some(b'b') | Some(b'B') if matches!(source.peek_n(1), Some(b'f') | Some(b'F')) => {
            source.advance(); // b
            source.advance(); // f
            // lazy: Skip all the digits instead of exactly 2
            while source.peek().as_ref().is_some_and(u8::is_ascii_digit) {
                source.advance();
            }
        }
        // q/Q, w/W, l/L are all single character
        Some(b'q') | Some(b'Q') | Some(b'w') | Some(b'W') | Some(b'l') | Some(b'L') => {
            source.advance();
        }
        // Don't know what this might be
        _ => {}
    }
}

/// Parses the string representation of a hex float to an actual float.
/// Mostly acurate
fn parse_hex_float(s: &str) -> Option<f64> {
    // Don't need the prefix. Also abort if not present.
    let s = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X"))?;
    // Splitting
    let (mantissa, exp) = s.split_once(['p', 'P'])?;
    // 
    let mut e: i64 = exp.parse().ok()?; 
    // If a decimal point is provieded, split there.
    // Otherwise everything is int part.
    let (int_part, frac_part) = mantissa.split_once('.').unwrap_or((mantissa, ""));

    // If there was no (useful) mantissa, something went wrong
    if int_part.is_empty() && frac_part.is_empty() {
        return None;
    }

    let mut m: u64 = 0;
    for c in int_part.chars() {
        let d = c.to_digit(16)? as u64;
        if m >> 60 == 0 {
            m = (m << 4) | d;
        } else {
            e+= 4;
        }
    }

    for c in frac_part.chars() {
        let d = c.to_digit(16)? as u64;
        if m >> 60 == 0 {
            m = (m << 4) | d;
            e -= 4;
        }
    }

    // finally assembling this shit
    let mut x = m as f64;
    
    // p-(sign p/m)-exponent
    let pp1023 = f64::from_bits(0x7FE0_0000_0000_0000); // +2^1023
    let pm1022 = f64::from_bits(0x0010_0000_0000_0000); // -2^1022

    while e > 1023 {
        x *= pp1023;
        e -= 1023;
        if x.is_infinite() { return Some(x); }
    }

    while e < -1022 {
        x *= pm1022;
        e+= 1022;
        if x == 0.0 { return Some(x)}
    }

    Some(
        x * f64::from_bits(((e + 1023) as u64) << 52)
    )
}

/// TokenSource trait の実装
///
/// Parser がプリプロセッサをトークンソースとして使用できるようにする
impl TokenSource for Preprocessor {
    fn next_token(&mut self) -> crate::error::Result<Token> {
        Preprocessor::next_token(self)
    }

    fn unget_token(&mut self, token: Token) {
        Preprocessor::unget_token(self, token)
    }

    fn interner(&self) -> &StringInterner {
        &self.interner
    }

    fn interner_mut(&mut self) -> &mut StringInterner {
        &mut self.interner
    }

    fn files(&self) -> &FileRegistry {
        &self.files
    }

    fn is_file_in_target(&self, file_id: crate::source::FileId) -> bool {
        let target_dir = match &self.config.target_dir {
            Some(dir) => dir,
            None => return false,
        };
        let path = self.files.get_path(file_id);
        path.starts_with(target_dir)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_temp_file(content: &str) -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(content.as_bytes()).unwrap();
        file
    }

    /// 識別子文字列がトークン列に含まれるかチェック
    fn has_ident(pp: &Preprocessor, tokens: &[Token], name: &str) -> bool {
        tokens.iter().any(|t| {
            if let TokenKind::Ident(id) = t.kind {
                pp.interner().get(id) == name
            } else {
                false
            }
        })
    }

    /// キーワードがトークン列に含まれるかチェック
    fn has_keyword(tokens: &[Token], kind: TokenKind) -> bool {
        tokens.iter().any(|t| std::mem::discriminant(&t.kind) == std::mem::discriminant(&kind))
    }

    #[test]
    fn test_parse_hex_float() {
        assert_eq!(parse_hex_float("0x1.8p3"), Some(12.0));
        assert_eq!(parse_hex_float("0x.8p1"), Some(1.0));
        assert_eq!(parse_hex_float("0x1P+4"), Some(16.0));
        assert_eq!(parse_hex_float("0x1p-1"), Some(0.5));
        assert_eq!(parse_hex_float("0x1.fffffffffffffp1023"), Some(f64::MAX));
        assert_eq!(parse_hex_float("0x1p-1022"), Some(f64::MIN_POSITIVE));
        assert_eq!(parse_hex_float("0x1p-1074"), Some(f64::from_bits(1))); // smallest subnormal
        assert_eq!(parse_hex_float("0x1p1024"), Some(f64::INFINITY));
        assert_eq!(parse_hex_float("0x.p1"), None);
    }

    #[test]
    fn test_simple_tokens() {
        let file = create_temp_file("int x;");
        let mut pp = Preprocessor::new(PPConfig::default());
        pp.add_source_file(file.path()).unwrap();

        let tokens = pp.collect_tokens().unwrap();
        // int, x, ; の3トークン
        assert_eq!(tokens.len(), 3);
        // キーワードはキーワードトークンとして返される
        assert!(has_keyword(&tokens, TokenKind::KwInt));
        assert!(has_ident(&pp, &tokens, "x"));
    }

    #[test]
    fn test_object_macro() {
        let file = create_temp_file("#define VALUE 42\nint x = VALUE;");
        let mut pp = Preprocessor::new(PPConfig::default());
        pp.add_source_file(file.path()).unwrap();

        let tokens = pp.collect_tokens().unwrap();
        assert!(tokens.iter().any(|t| matches!(t.kind, TokenKind::IntLit(42))));
    }

    #[test]
    fn test_function_macro() {
        let file = create_temp_file("#define ADD(a, b) a + b\nint x = ADD(1, 2);");
        let mut pp = Preprocessor::new(PPConfig::default());
        pp.add_source_file(file.path()).unwrap();

        let tokens = pp.collect_tokens().unwrap();
        assert!(tokens.iter().any(|t| matches!(t.kind, TokenKind::Plus)));
    }

    #[test]
    fn test_ifdef() {
        let file = create_temp_file("#define FOO\n#ifdef FOO\nint x;\n#endif");
        let mut pp = Preprocessor::new(PPConfig::default());
        pp.add_source_file(file.path()).unwrap();

        let tokens = pp.collect_tokens().unwrap();
        assert!(has_keyword(&tokens, TokenKind::KwInt));
    }

    #[test]
    fn test_ifndef() {
        let file = create_temp_file("#ifndef BAR\nint x;\n#endif");
        let mut pp = Preprocessor::new(PPConfig::default());
        pp.add_source_file(file.path()).unwrap();

        let tokens = pp.collect_tokens().unwrap();
        assert!(has_keyword(&tokens, TokenKind::KwInt));
    }

    #[test]
    fn test_ifdef_else() {
        let file = create_temp_file("#ifdef UNDEFINED\nint x;\n#else\nfloat y;\n#endif");
        let mut pp = Preprocessor::new(PPConfig::default());
        pp.add_source_file(file.path()).unwrap();

        let tokens = pp.collect_tokens().unwrap();
        // UNDEFINED は定義されていないので、int x は出力されない
        assert!(!has_ident(&pp, &tokens, "x"));
        // float y は出力される
        assert!(has_keyword(&tokens, TokenKind::KwFloat));
        assert!(has_ident(&pp, &tokens, "y"));
    }

    #[test]
    fn test_if_expression() {
        let file = create_temp_file("#if 1 + 1 == 2\nint x;\n#endif");
        let mut pp = Preprocessor::new(PPConfig::default());
        pp.add_source_file(file.path()).unwrap();

        let tokens = pp.collect_tokens().unwrap();
        assert!(has_keyword(&tokens, TokenKind::KwInt));
    }

    #[test]
    fn test_predefined_macro() {
        let config = PPConfig {
            predefined: vec![("VERSION".to_string(), Some("100".to_string()))],
            ..Default::default()
        };
        let file = create_temp_file("int v = VERSION;");
        let mut pp = Preprocessor::new(config);
        pp.add_source_file(file.path()).unwrap();

        let tokens = pp.collect_tokens().unwrap();
        assert!(tokens.iter().any(|t| matches!(t.kind, TokenKind::IntLit(100))));
    }

    #[test]
    fn test_undef() {
        let file = create_temp_file("#define FOO 1\n#undef FOO\n#ifdef FOO\nint x;\n#endif");
        let mut pp = Preprocessor::new(PPConfig::default());
        pp.add_source_file(file.path()).unwrap();

        let tokens = pp.collect_tokens().unwrap();
        // FOO は #undef されているので、int x は出力されない
        assert!(!has_ident(&pp, &tokens, "x"));
    }

    #[test]
    fn test_nested_ifdef() {
        let file = create_temp_file(
            "#define A\n#ifdef A\n#ifdef B\nint x;\n#else\nfloat y;\n#endif\n#endif"
        );
        let mut pp = Preprocessor::new(PPConfig::default());
        pp.add_source_file(file.path()).unwrap();

        let tokens = pp.collect_tokens().unwrap();
        // A は定義されているが B は定義されていないので、float y が出力される
        assert!(!has_ident(&pp, &tokens, "x"));
        assert!(has_keyword(&tokens, TokenKind::KwFloat));
        assert!(has_ident(&pp, &tokens, "y"));
    }

    // NoExpandRegistry tests

    #[test]
    fn test_no_expand_registry_new() {
        let registry = NoExpandRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn test_no_expand_registry_add() {
        let mut interner = crate::intern::StringInterner::new();
        let mut registry = NoExpandRegistry::new();

        let token_id = TokenId::next();
        let macro_name = interner.intern("FOO");

        registry.add(token_id, macro_name);

        assert!(registry.is_blocked(token_id, macro_name));
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn test_no_expand_registry_extend() {
        let mut interner = crate::intern::StringInterner::new();
        let mut registry = NoExpandRegistry::new();

        let token_id = TokenId::next();
        let macro1 = interner.intern("FOO");
        let macro2 = interner.intern("BAR");
        let macro3 = interner.intern("BAZ");

        registry.extend(token_id, vec![macro1, macro2, macro3]);

        assert!(registry.is_blocked(token_id, macro1));
        assert!(registry.is_blocked(token_id, macro2));
        assert!(registry.is_blocked(token_id, macro3));
    }

    #[test]
    fn test_no_expand_registry_not_blocked() {
        let mut interner = crate::intern::StringInterner::new();
        let mut registry = NoExpandRegistry::new();

        let token_id = TokenId::next();
        let other_token_id = TokenId::next();
        let macro_name = interner.intern("FOO");
        let other_macro = interner.intern("BAR");

        registry.add(token_id, macro_name);

        // 異なるトークンIDではブロックされない
        assert!(!registry.is_blocked(other_token_id, macro_name));
        // 異なるマクロ名ではブロックされない
        assert!(!registry.is_blocked(token_id, other_macro));
    }

    #[test]
    fn test_no_expand_registry_inherit() {
        let mut interner = crate::intern::StringInterner::new();
        let mut registry = NoExpandRegistry::new();

        let token1 = TokenId::next();
        let token2 = TokenId::next();
        let macro1 = interner.intern("FOO");
        let macro2 = interner.intern("BAR");

        // token1 に FOO と BAR を追加
        registry.add(token1, macro1);
        registry.add(token1, macro2);

        // token1 の禁止リストを token2 に継承
        registry.inherit(token1, token2);

        // token2 も FOO と BAR がブロックされる
        assert!(registry.is_blocked(token2, macro1));
        assert!(registry.is_blocked(token2, macro2));
    }

    #[test]
    fn test_no_expand_registry_inherit_merge() {
        let mut interner = crate::intern::StringInterner::new();
        let mut registry = NoExpandRegistry::new();

        let token1 = TokenId::next();
        let token2 = TokenId::next();
        let macro1 = interner.intern("FOO");
        let macro2 = interner.intern("BAR");
        let macro3 = interner.intern("BAZ");

        // token1 に FOO を追加
        registry.add(token1, macro1);

        // token2 に BAR を追加
        registry.add(token2, macro2);

        // token1 の禁止リストを token2 に継承（マージされる）
        registry.inherit(token1, token2);

        // token2 は FOO と BAR の両方がブロックされる
        assert!(registry.is_blocked(token2, macro1));
        assert!(registry.is_blocked(token2, macro2));

        // token1 は元の FOO のみ（BAR はない）
        assert!(registry.is_blocked(token1, macro1));
        assert!(!registry.is_blocked(token1, macro2));

        // どちらも BAZ はブロックされない
        assert!(!registry.is_blocked(token1, macro3));
        assert!(!registry.is_blocked(token2, macro3));
    }

    // マーカー出力テスト

    #[test]
    fn test_emit_markers_disabled() {
        // emit_markers = false (デフォルト) の場合、マーカーは出力されない
        let file = create_temp_file("#define FOO 42\nint x = FOO;");
        let mut pp = Preprocessor::new(PPConfig::default());
        pp.add_source_file(file.path()).unwrap();

        let tokens = pp.collect_tokens().unwrap();

        // マーカートークンが含まれていないことを確認
        let has_marker = tokens.iter().any(|t| {
            matches!(t.kind, TokenKind::MacroBegin(_) | TokenKind::MacroEnd(_))
        });
        assert!(!has_marker, "Markers should not be emitted when emit_markers is false");
    }

    #[test]
    fn test_emit_markers_object_macro() {
        // emit_markers = true の場合、オブジェクトマクロにマーカーが出力される
        let file = create_temp_file("#define FOO 42\nint x = FOO;");
        let config = PPConfig {
            emit_markers: true,
            ..Default::default()
        };
        let mut pp = Preprocessor::new(config);
        pp.add_source_file(file.path()).unwrap();

        let tokens = pp.collect_tokens().unwrap();

        // MacroBegin と MacroEnd が出力されていることを確認
        let begin_count = tokens.iter().filter(|t| {
            matches!(t.kind, TokenKind::MacroBegin(_))
        }).count();
        let end_count = tokens.iter().filter(|t| {
            matches!(t.kind, TokenKind::MacroEnd(_))
        }).count();

        assert_eq!(begin_count, 1, "Should have exactly one MacroBegin");
        assert_eq!(end_count, 1, "Should have exactly one MacroEnd");

        // マーカーの名前が正しいことを確認
        for t in &tokens {
            if let TokenKind::MacroBegin(info) = &t.kind {
                assert_eq!(pp.interner().get(info.macro_name), "FOO");
                assert!(matches!(info.kind, MacroInvocationKind::Object));
            }
        }
    }

    #[test]
    fn test_emit_markers_function_macro() {
        // 関数マクロにもマーカーが出力される
        let file = create_temp_file("#define ADD(a, b) a + b\nint x = ADD(1, 2);");
        let config = PPConfig {
            emit_markers: true,
            ..Default::default()
        };
        let mut pp = Preprocessor::new(config);
        pp.add_source_file(file.path()).unwrap();

        let tokens = pp.collect_tokens().unwrap();

        // MacroBegin と MacroEnd が出力されていることを確認
        let begin_count = tokens.iter().filter(|t| {
            matches!(t.kind, TokenKind::MacroBegin(_))
        }).count();
        let end_count = tokens.iter().filter(|t| {
            matches!(t.kind, TokenKind::MacroEnd(_))
        }).count();

        assert_eq!(begin_count, 1, "Should have exactly one MacroBegin");
        assert_eq!(end_count, 1, "Should have exactly one MacroEnd");

        // 関数マクロの引数が保持されていることを確認
        for t in &tokens {
            if let TokenKind::MacroBegin(info) = &t.kind {
                assert_eq!(pp.interner().get(info.macro_name), "ADD");
                if let MacroInvocationKind::Function { args } = &info.kind {
                    assert_eq!(args.len(), 2, "ADD macro should have 2 arguments");
                } else {
                    panic!("Expected Function macro kind");
                }
            }
        }
    }

    #[test]
    fn test_emit_markers_begin_end_matching() {
        // MacroBegin と MacroEnd の marker_id が一致することを確認
        let file = create_temp_file("#define FOO 1\nint x = FOO;");
        let config = PPConfig {
            emit_markers: true,
            ..Default::default()
        };
        let mut pp = Preprocessor::new(config);
        pp.add_source_file(file.path()).unwrap();

        let tokens = pp.collect_tokens().unwrap();

        let mut begin_marker_id = None;
        let mut end_marker_id = None;

        for t in &tokens {
            match &t.kind {
                TokenKind::MacroBegin(info) => {
                    begin_marker_id = Some(info.marker_id);
                }
                TokenKind::MacroEnd(info) => {
                    end_marker_id = Some(info.begin_marker_id);
                }
                _ => {}
            }
        }

        assert!(begin_marker_id.is_some(), "Should have MacroBegin");
        assert!(end_marker_id.is_some(), "Should have MacroEnd");
        assert_eq!(
            begin_marker_id.unwrap(),
            end_marker_id.unwrap(),
            "MacroBegin.marker_id should match MacroEnd.begin_marker_id"
        );
    }

    // MacroCallWatcher tests

    #[test]
    fn test_macro_call_watcher_basic() {
        // MacroCallWatcher の基本機能テスト
        let watcher = MacroCallWatcher::new();

        // 初期状態では called は false
        assert!(!watcher.was_called());
        assert!(watcher.last_args().is_none());
    }

    #[test]
    fn test_macro_call_watcher_object_macro() {
        // オブジェクトマクロの呼び出し検出
        let file = create_temp_file("#define TEST_MACRO 42\nint x = TEST_MACRO;");
        let mut pp = Preprocessor::new(PPConfig::default());
        pp.add_source_file(file.path()).unwrap();

        // コールバックを登録
        let macro_name = pp.interner_mut().intern("TEST_MACRO");
        pp.set_macro_called_callback(macro_name, Box::new(MacroCallWatcher::new()));

        // トークンを収集（マクロが展開される）
        let _tokens = pp.collect_tokens().unwrap();

        // コールバックが呼ばれたことを確認
        if let Some(cb) = pp.get_macro_called_callback(macro_name) {
            if let Some(watcher) = cb.as_any().downcast_ref::<MacroCallWatcher>() {
                assert!(watcher.was_called(), "TEST_MACRO should have been called");
                // オブジェクトマクロなので引数は None
                assert!(watcher.last_args().is_none());
            } else {
                panic!("Failed to downcast to MacroCallWatcher");
            }
        } else {
            panic!("Callback not found");
        }
    }

    #[test]
    fn test_macro_call_watcher_function_macro() {
        // 関数マクロの呼び出し検出と引数取得
        let file = create_temp_file("#define ADD(a, b) a + b\nint x = ADD(10, 20);");
        let mut pp = Preprocessor::new(PPConfig::default());
        pp.add_source_file(file.path()).unwrap();

        // コールバックを登録
        let macro_name = pp.interner_mut().intern("ADD");
        pp.set_macro_called_callback(macro_name, Box::new(MacroCallWatcher::new()));

        // トークンを収集（マクロが展開される）
        let _tokens = pp.collect_tokens().unwrap();

        // コールバックが呼ばれたことを確認
        if let Some(cb) = pp.get_macro_called_callback(macro_name) {
            if let Some(watcher) = cb.as_any().downcast_ref::<MacroCallWatcher>() {
                assert!(watcher.was_called(), "ADD should have been called");
                // 関数マクロなので引数がある
                let args = watcher.last_args();
                assert!(args.is_some(), "Function macro should have arguments");
                let args = args.unwrap();
                assert_eq!(args.len(), 2, "ADD has 2 arguments");
                assert_eq!(args[0], "10");
                assert_eq!(args[1], "20");
            } else {
                panic!("Failed to downcast to MacroCallWatcher");
            }
        } else {
            panic!("Callback not found");
        }
    }

    #[test]
    fn test_macro_call_watcher_clear() {
        // clear() メソッドのテスト
        let file = create_temp_file("#define FOO(x) x\nint a = FOO(1);\nint b = FOO(2);");
        let mut pp = Preprocessor::new(PPConfig::default());
        pp.add_source_file(file.path()).unwrap();

        // コールバックを登録
        let macro_name = pp.interner_mut().intern("FOO");
        pp.set_macro_called_callback(macro_name, Box::new(MacroCallWatcher::new()));

        // 最初のトークンを取得（FOO(1) が展開される）
        let mut count = 0;
        while count < 5 {
            // int a = FOO(1) の 5 トークン程度
            if pp.next_token().unwrap().kind == TokenKind::Eof {
                break;
            }
            count += 1;
        }

        // フラグが立っていることを確認
        {
            let cb = pp.get_macro_called_callback(macro_name).unwrap();
            let watcher = cb.as_any().downcast_ref::<MacroCallWatcher>().unwrap();
            assert!(watcher.was_called());
            let args = watcher.last_args().unwrap();
            assert_eq!(args[0], "1");
        }

        // clear() を呼ぶ
        {
            let cb = pp.get_macro_called_callback_mut(macro_name).unwrap();
            let watcher = cb.as_any_mut().downcast_mut::<MacroCallWatcher>().unwrap();
            watcher.clear();
        }

        // フラグがリセットされていることを確認
        {
            let cb = pp.get_macro_called_callback(macro_name).unwrap();
            let watcher = cb.as_any().downcast_ref::<MacroCallWatcher>().unwrap();
            assert!(!watcher.was_called());
            assert!(watcher.last_args().is_none());
        }
    }

    #[test]
    fn test_macro_call_watcher_take_called() {
        // take_called() メソッドのテスト（フラグを取得してリセット）
        let file = create_temp_file("#define BAR 99\nint x = BAR;");
        let mut pp = Preprocessor::new(PPConfig::default());
        pp.add_source_file(file.path()).unwrap();

        let macro_name = pp.interner_mut().intern("BAR");
        pp.set_macro_called_callback(macro_name, Box::new(MacroCallWatcher::new()));

        let _tokens = pp.collect_tokens().unwrap();

        // take_called() は true を返し、フラグをリセットする
        {
            let cb = pp.get_macro_called_callback(macro_name).unwrap();
            let watcher = cb.as_any().downcast_ref::<MacroCallWatcher>().unwrap();
            assert!(watcher.take_called(), "First take_called should return true");
            assert!(!watcher.take_called(), "Second take_called should return false");
        }
    }

    #[test]
    fn test_macro_call_watcher_multiple_macros() {
        // 複数のマクロを監視
        let file = create_temp_file(
            "#define A(x) x\n#define B(x) x\n#define C(x) x\nint a = A(1); int b = B(2);"
        );
        let mut pp = Preprocessor::new(PPConfig::default());
        pp.add_source_file(file.path()).unwrap();

        let macro_a = pp.interner_mut().intern("A");
        let macro_b = pp.interner_mut().intern("B");
        let macro_c = pp.interner_mut().intern("C");

        pp.set_macro_called_callback(macro_a, Box::new(MacroCallWatcher::new()));
        pp.set_macro_called_callback(macro_b, Box::new(MacroCallWatcher::new()));
        pp.set_macro_called_callback(macro_c, Box::new(MacroCallWatcher::new()));

        let _tokens = pp.collect_tokens().unwrap();

        // A と B は呼ばれた、C は呼ばれていない
        {
            let cb = pp.get_macro_called_callback(macro_a).unwrap();
            let watcher = cb.as_any().downcast_ref::<MacroCallWatcher>().unwrap();
            assert!(watcher.was_called(), "A should have been called");
        }
        {
            let cb = pp.get_macro_called_callback(macro_b).unwrap();
            let watcher = cb.as_any().downcast_ref::<MacroCallWatcher>().unwrap();
            assert!(watcher.was_called(), "B should have been called");
        }
        {
            let cb = pp.get_macro_called_callback(macro_c).unwrap();
            let watcher = cb.as_any().downcast_ref::<MacroCallWatcher>().unwrap();
            assert!(!watcher.was_called(), "C should not have been called");
        }
    }

    #[test]
    fn test_macro_call_watcher_sv_head_pattern() {
        // _SV_HEAD パターンのシミュレーション
        // 実際の Perl ヘッダーでは _SV_HEAD(SV) のように使われる
        let file = create_temp_file(
            "#define _SV_HEAD(type) void *sv_any; type *sv_type\n\
             struct sv { _SV_HEAD(SV); };\n\
             struct av { _SV_HEAD(AV); };\n\
             struct other { int x; };"
        );
        let mut pp = Preprocessor::new(PPConfig::default());
        pp.add_source_file(file.path()).unwrap();

        let sv_head = pp.interner_mut().intern("_SV_HEAD");
        pp.set_macro_called_callback(sv_head, Box::new(MacroCallWatcher::new()));

        // トークンを一つずつ読み進める
        // 構造体ごとにフラグをチェック
        let mut sv_family_members = Vec::new();
        let mut current_struct: Option<String> = None;

        loop {
            let token = pp.next_token().unwrap();
            if token.kind == TokenKind::Eof {
                break;
            }

            // struct キーワードを検出
            if token.kind == TokenKind::KwStruct {
                // 新しい構造体の開始時にフラグをクリア
                if let Some(cb) = pp.get_macro_called_callback_mut(sv_head) {
                    let watcher = cb.as_any_mut().downcast_mut::<MacroCallWatcher>().unwrap();
                    watcher.clear();
                }

                let name_token = pp.next_token().unwrap();
                if let TokenKind::Ident(id) = name_token.kind {
                    current_struct = Some(pp.interner().get(id).to_string());
                }
            }

            // 構造体の終わり（セミコロン）を検出
            if token.kind == TokenKind::Semi {
                if let Some(ref struct_name) = current_struct {
                    // _SV_HEAD が呼ばれたかチェック
                    if let Some(cb) = pp.get_macro_called_callback(sv_head) {
                        let watcher = cb.as_any().downcast_ref::<MacroCallWatcher>().unwrap();
                        if watcher.was_called() {
                            sv_family_members.push(struct_name.clone());
                        }
                    }
                }
                current_struct = None;
            }
        }

        // sv と av は _SV_HEAD を使用している、other は使用していない
        assert!(sv_family_members.contains(&"sv".to_string()), "sv should be SV family");
        assert!(sv_family_members.contains(&"av".to_string()), "av should be SV family");
        assert!(!sv_family_members.contains(&"other".to_string()), "other should not be SV family");
    }
}
