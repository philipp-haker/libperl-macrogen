use crate::error::{CompileError, LexError, Result};
use crate::intern::{InternedStr, StringInterner};
use crate::source::{FileId, SourceLocation};
use crate::token::{Comment, CommentKind, Token, TokenKind};

/// 識別子解決トレイト
///
/// Lexer が識別子をトークン化する際に使用する。
/// 通常モード（Interning）では新しい識別子を intern し、
/// 読み取り専用モード（LookupOnly）では既存の識別子のみを lookup する。
pub trait IdentResolver {
    /// 識別子文字列を InternedStr に解決
    ///
    /// 通常モード: intern して常に成功（Some を返す）
    /// 読み取り専用モード: lookup のみ、見つからなければ None
    fn resolve_ident(&mut self, s: &str) -> Option<InternedStr>;
}

/// 通常の intern を行うラッパー
pub struct Interning<'a>(pub &'a mut StringInterner);

impl IdentResolver for Interning<'_> {
    fn resolve_ident(&mut self, s: &str) -> Option<InternedStr> {
        Some(self.0.intern(s)) // 常に成功
    }
}

/// lookup のみ行うラッパー（読み取り専用）
pub struct LookupOnly<'a>(pub &'a StringInterner);

impl IdentResolver for LookupOnly<'_> {
    fn resolve_ident(&mut self, s: &str) -> Option<InternedStr> {
        self.0.lookup(s) // 見つからなければ None
    }
}

/// Lexer（ジェネリック版）
pub struct Lexer<'a, R: IdentResolver> {
    source: &'a [u8],
    pos: usize,
    line: u32,
    column: u32,
    file_id: FileId,
    resolver: R,
    /// スペース/タブをトークンとして返すかどうか（TinyCC の PARSE_FLAG_SPACES 相当）
    return_spaces: bool,
    /// ライフタイムマーカー（'a を使用するため）
    _marker: std::marker::PhantomData<&'a ()>,
}

/// 通常の Lexer（mutable interner を使用）
pub type MutableLexer<'a> = Lexer<'a, Interning<'a>>;

/// 読み取り専用 Lexer（lookup のみ）
pub type ReadOnlyLexer<'a> = Lexer<'a, LookupOnly<'a>>;

impl<'a> Lexer<'a, Interning<'a>> {
    /// 新しいLexerを作成（通常モード）
    pub fn new(source: &'a [u8], file_id: FileId, interner: &'a mut StringInterner) -> Self {
        Self {
            source,
            pos: 0,
            line: 1,
            column: 1,
            file_id,
            resolver: Interning(interner),
            return_spaces: false,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<'a> Lexer<'a, LookupOnly<'a>> {
    /// 新しいLexerを作成（読み取り専用モード）
    ///
    /// このモードでは、既に intern 済みの識別子のみを認識する。
    /// 未知の識別子を検出した場合は LexError::UnknownIdentifier を返す。
    pub fn new_readonly(source: &'a [u8], file_id: FileId, interner: &'a StringInterner) -> Self {
        Self {
            source,
            pos: 0,
            line: 1,
            column: 1,
            file_id,
            resolver: LookupOnly(interner),
            return_spaces: false,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<'a, R: IdentResolver> Lexer<'a, R> {

    /// スペースをトークンとして返すかどうかを設定
    pub fn set_return_spaces(&mut self, enabled: bool) {
        self.return_spaces = enabled;
    }

    /// 現在のスペース返却モードを取得
    pub fn return_spaces(&self) -> bool {
        self.return_spaces
    }

    /// 現在位置を取得
    pub fn current_location(&self) -> SourceLocation {
        SourceLocation::new(self.file_id, self.line, self.column)
    }

    /// ファイルIDを取得
    pub fn file_id(&self) -> FileId {
        self.file_id
    }

    /// 次のトークンを取得
    pub fn next_token(&mut self) -> Result<Token> {
        let mut leading_comments = Vec::new();

        loop {
            // return_spaces モードの場合、空白をトークンとして返す
            if self.return_spaces {
                if let Some(c) = self.peek() {
                    if c == b' ' || c == b'\t' {
                        let loc = self.current_location();
                        self.advance();
                        // 連続する空白は1つのSpaceトークンにまとめる
                        while let Some(c) = self.peek() {
                            if c == b' ' || c == b'\t' {
                                self.advance();
                            } else {
                                break;
                            }
                        }
                        return Ok(Token::with_comments(TokenKind::Space, loc, leading_comments));
                    }
                }
            } else {
                self.skip_whitespace();
            }

            match (self.peek(), self.peek_n(1)) {
                (Some(b'/'), Some(b'/')) => {
                    let comment = self.scan_line_comment();
                    leading_comments.push(comment);
                }
                (Some(b'/'), Some(b'*')) => {
                    let comment = self.scan_block_comment()?;
                    leading_comments.push(comment);
                }
                _ => break,
            }
        }

        let loc = self.current_location();
        let kind = self.scan_token_kind()?;

        Ok(Token::with_comments(kind, loc, leading_comments))
    }

    /// 現在の文字をピーク
    fn peek(&self) -> Option<u8> {
        self.source.get(self.pos).copied()
    }

    /// n文字先をピーク
    fn peek_n(&self, n: usize) -> Option<u8> {
        self.source.get(self.pos + n).copied()
    }

    /// 1文字進める
    fn advance(&mut self) -> Option<u8> {
        let c = self.peek()?;
        self.pos += 1;
        if c == b'\n' {
            self.line += 1;
            self.column = 1;
        } else {
            self.column += 1;
        }
        Some(c)
    }

    /// 空白をスキップ（改行は含まない - プリプロセッサのため）
    fn skip_whitespace(&mut self) {
        while let Some(c) = self.peek() {
            if c == b' ' || c == b'\t' || c == b'\r' {
                self.advance();
            } else {
                break;
            }
        }
    }

    /// 行コメントをスキャン
    fn scan_line_comment(&mut self) -> Comment {
        let loc = self.current_location();
        self.advance(); // /
        self.advance(); // /

        let start = self.pos;
        while self.peek().is_some_and(|c| c != b'\n') {
            self.advance();
        }
        let text = String::from_utf8_lossy(&self.source[start..self.pos]).to_string();

        Comment::new(CommentKind::Line, text, loc)
    }

    /// ブロックコメントをスキャン
    fn scan_block_comment(&mut self) -> Result<Comment> {
        let loc = self.current_location();
        self.advance(); // /
        self.advance(); // *

        let start = self.pos;
        loop {
            match (self.peek(), self.peek_n(1)) {
                (Some(b'*'), Some(b'/')) => {
                    let text = String::from_utf8_lossy(&self.source[start..self.pos]).to_string();
                    self.advance(); // *
                    self.advance(); // /
                    return Ok(Comment::new(CommentKind::Block, text, loc));
                }
                (Some(_), _) => {
                    self.advance();
                }
                (None, _) => {
                    return Err(CompileError::Lex {
                        loc,
                        kind: LexError::UnterminatedComment,
                    });
                }
            }
        }
    }

    /// トークン種別をスキャン
    fn scan_token_kind(&mut self) -> Result<TokenKind> {
        let Some(c) = self.peek() else {
            return Ok(TokenKind::Eof);
        };

        match c {
            // Only accept \r right before \n
            b'\r' if matches!(self.peek_n(1), Some(b'\n')) => {
                // Eat both tokens
                self.advance();
                self.advance();
                Ok(TokenKind::Newline)
            }
            // 改行（プリプロセッサのために独立したトークンとして扱う）
            b'\n' => {
                self.advance();
                Ok(TokenKind::Newline)
            }
            // ワイド文字列/文字リテラル（識別子より先にチェック）
            b'L' if matches!(self.peek_n(1), Some(b'"') | Some(b'\'')) => {
                self.advance(); // L
                if self.peek() == Some(b'"') {
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
            b'+' => self.scan_plus(),
            b'-' => self.scan_minus(),
            b'*' => self.scan_star(),
            b'/' => self.scan_slash(),
            b'%' => self.scan_percent(),
            b'&' => self.scan_amp(),
            b'|' => self.scan_pipe(),
            b'^' => self.scan_caret(),
            b'~' => {
                self.advance();
                Ok(TokenKind::Tilde)
            }
            b'!' => self.scan_bang(),
            b'<' => self.scan_lt(),
            b'>' => self.scan_gt(),
            b'=' => self.scan_eq(),
            b'?' => {
                self.advance();
                Ok(TokenKind::Question)
            }
            b':' => {
                self.advance();
                Ok(TokenKind::Colon)
            }
            b'.' => self.scan_dot(),
            b',' => {
                self.advance();
                Ok(TokenKind::Comma)
            }
            b';' => {
                self.advance();
                Ok(TokenKind::Semi)
            }
            b'(' => {
                self.advance();
                Ok(TokenKind::LParen)
            }
            b')' => {
                self.advance();
                Ok(TokenKind::RParen)
            }
            b'[' => {
                self.advance();
                Ok(TokenKind::LBracket)
            }
            b']' => {
                self.advance();
                Ok(TokenKind::RBracket)
            }
            b'{' => {
                self.advance();
                Ok(TokenKind::LBrace)
            }
            b'}' => {
                self.advance();
                Ok(TokenKind::RBrace)
            }
            b'#' => self.scan_hash(),

            _ => {
                let loc = self.current_location();
                self.advance();
                Err(CompileError::Lex {
                    loc,
                    kind: LexError::InvalidChar(c as char),
                })
            }
        }
    }

    /// 識別子またはキーワードをスキャン
    fn scan_identifier(&mut self) -> Result<TokenKind> {
        let loc = self.current_location();
        let start = self.pos;
        while let Some(c) = self.peek() {
            if c.is_ascii_alphanumeric() || c == b'_' {
                self.advance();
            } else {
                break;
            }
        }

        let text = std::str::from_utf8(&self.source[start..self.pos]).unwrap();

        // キーワードなら対応するTokenKindを返す
        if let Some(kw) = TokenKind::from_keyword(text) {
            Ok(kw)
        } else {
            // resolver を使って識別子を解決
            match self.resolver.resolve_ident(text) {
                Some(interned) => Ok(TokenKind::Ident(interned)),
                None => Err(CompileError::Lex {
                    loc,
                    kind: LexError::UnknownIdentifier(text.to_string()),
                }),
            }
        }
    }

    /// 数値リテラルをスキャン
    fn scan_number(&mut self) -> Result<TokenKind> {
        let loc = self.current_location();
        let start = self.pos;

        // 16進数、8進数、2進数の判定
        if self.peek() == Some(b'0') {
            self.advance();
            match self.peek() {
                Some(b'x') | Some(b'X') => return self.scan_hex_number(start, loc),
                Some(b'b') | Some(b'B') => return self.scan_binary_number(start, loc),
                Some(b'0'..=b'7') => return self.scan_octal_number(start, loc),
                Some(b'.') | Some(b'e') | Some(b'E') => {
                    // 浮動小数点
                    return self.scan_float_number(start, loc);
                }
                _ => {
                    // 単なる 0
                    return self.finish_integer(start, loc);
                }
            }
        }

        // 10進数
        while self.peek().is_some_and(|c| c.is_ascii_digit()) {
            self.advance();
        }

        // 浮動小数点チェック
        if matches!(self.peek(), Some(b'.') | Some(b'e') | Some(b'E')) {
            return self.scan_float_number(start, loc);
        }

        self.finish_integer(start, loc)
    }

    /// 16進数をスキャン
    fn scan_hex_number(&mut self, start: usize, loc: SourceLocation) -> Result<TokenKind> {
        self.advance(); // x or X

        let hex_start = self.pos;
        while self.peek().is_some_and(|c| c.is_ascii_hexdigit()) {
            self.advance();
        }

        if self.pos == hex_start {
            return Err(CompileError::Lex {
                loc,
                kind: LexError::InvalidNumber("0x".to_string()),
            });
        }

        self.finish_integer(start, loc)
    }

    /// 2進数をスキャン
    fn scan_binary_number(&mut self, start: usize, loc: SourceLocation) -> Result<TokenKind> {
        self.advance(); // b or B

        let bin_start = self.pos;
        while matches!(self.peek(), Some(b'0') | Some(b'1')) {
            self.advance();
        }

        if self.pos == bin_start {
            return Err(CompileError::Lex {
                loc,
                kind: LexError::InvalidNumber("0b".to_string()),
            });
        }

        self.finish_integer(start, loc)
    }

    /// 8進数をスキャン
    fn scan_octal_number(&mut self, start: usize, loc: SourceLocation) -> Result<TokenKind> {
        while self.peek().is_some_and(|c| matches!(c, b'0'..=b'7')) {
            self.advance();
        }

        self.finish_integer(start, loc)
    }

    /// 浮動小数点数をスキャン
    fn scan_float_number(&mut self, start: usize, loc: SourceLocation) -> Result<TokenKind> {
        // 小数部
        if self.peek() == Some(b'.') {
            self.advance();
            while self.peek().is_some_and(|c| c.is_ascii_digit()) {
                self.advance();
            }
        }

        // 指数部
        if matches!(self.peek(), Some(b'e') | Some(b'E')) {
            self.advance();
            if matches!(self.peek(), Some(b'+') | Some(b'-')) {
                self.advance();
            }
            while self.peek().is_some_and(|c| c.is_ascii_digit()) {
                self.advance();
            }
        }

        // サフィックス
        let _is_float = matches!(self.peek(), Some(b'f') | Some(b'F'));
        let _is_long = matches!(self.peek(), Some(b'l') | Some(b'L'));
        if _is_float || _is_long {
            self.advance();
        }

        let text = std::str::from_utf8(&self.source[start..self.pos]).unwrap();
        let value: f64 = text
            .trim_end_matches(|c| c == 'f' || c == 'F' || c == 'l' || c == 'L')
            .parse()
            .map_err(|_| CompileError::Lex {
                loc: loc.clone(),
                kind: LexError::InvalidNumber(text.to_string()),
            })?;

        Ok(TokenKind::FloatLit(value))
    }

    /// 整数リテラルの仕上げ（サフィックス処理）
    fn finish_integer(&mut self, start: usize, loc: SourceLocation) -> Result<TokenKind> {
        // サフィックス: u/U, l/L, ll/LL
        let mut is_unsigned = false;
        let mut is_long = false;
        let mut is_longlong = false;

        loop {
            match self.peek() {
                Some(b'u') | Some(b'U') => {
                    is_unsigned = true;
                    self.advance();
                }
                Some(b'l') | Some(b'L') => {
                    if is_long {
                        is_longlong = true;
                    }
                    is_long = true;
                    self.advance();
                }
                _ => break,
            }
        }

        let text = std::str::from_utf8(&self.source[start..self.pos]).unwrap();
        let num_text = text
            .trim_start_matches("0x")
            .trim_start_matches("0X")
            .trim_start_matches("0b")
            .trim_start_matches("0B")
            .trim_end_matches(|c: char| c == 'u' || c == 'U' || c == 'l' || c == 'L');

        let radix = if text.starts_with("0x") || text.starts_with("0X") {
            16
        } else if text.starts_with("0b") || text.starts_with("0B") {
            2
        } else if text.starts_with('0') && text.len() > 1 && !text.contains('.') {
            8
        } else {
            10
        };

        if is_unsigned || is_longlong {
            let value = u64::from_str_radix(num_text, radix).map_err(|_| CompileError::Lex {
                loc: loc.clone(),
                kind: LexError::InvalidNumber(text.to_string()),
            })?;
            Ok(TokenKind::UIntLit(value))
        } else {
            let value = i64::from_str_radix(num_text, radix).map_err(|_| CompileError::Lex {
                loc: loc.clone(),
                kind: LexError::InvalidNumber(text.to_string()),
            })?;
            Ok(TokenKind::IntLit(value))
        }
    }

    /// 文字列リテラルをスキャン
    fn scan_string(&mut self) -> Result<TokenKind> {
        let loc = self.current_location();
        self.advance(); // "

        let mut bytes = Vec::new();
        loop {
            match self.peek() {
                Some(b'"') => {
                    self.advance();
                    return Ok(TokenKind::StringLit(bytes));
                }
                Some(b'\\') => {
                    self.advance();
                    let escaped = self.scan_escape_sequence(&loc)?;
                    bytes.push(escaped);
                }
                Some(b'\n') | None => {
                    return Err(CompileError::Lex {
                        loc,
                        kind: LexError::UnterminatedString,
                    });
                }
                Some(c) => {
                    self.advance();
                    bytes.push(c);
                }
            }
        }
    }

    /// ワイド文字列リテラルをスキャン
    fn scan_wide_string(&mut self) -> Result<TokenKind> {
        let loc = self.current_location();
        self.advance(); // "

        let mut chars = Vec::new();
        loop {
            match self.peek() {
                Some(b'"') => {
                    self.advance();
                    return Ok(TokenKind::WideStringLit(chars));
                }
                Some(b'\\') => {
                    self.advance();
                    let escaped = self.scan_escape_sequence(&loc)?;
                    chars.push(escaped as u32);
                }
                Some(b'\n') | None => {
                    return Err(CompileError::Lex {
                        loc,
                        kind: LexError::UnterminatedString,
                    });
                }
                Some(c) => {
                    self.advance();
                    chars.push(c as u32);
                }
            }
        }
    }

    /// 文字リテラルをスキャン
    fn scan_char(&mut self) -> Result<TokenKind> {
        let loc = self.current_location();
        self.advance(); // '

        let value = match self.peek() {
            Some(b'\'') => {
                return Err(CompileError::Lex {
                    loc,
                    kind: LexError::EmptyCharLit,
                });
            }
            Some(b'\\') => {
                self.advance();
                self.scan_escape_sequence(&loc)?
            }
            Some(c) => {
                self.advance();
                c
            }
            None => {
                return Err(CompileError::Lex {
                    loc,
                    kind: LexError::UnterminatedChar,
                });
            }
        };

        if self.peek() != Some(b'\'') {
            return Err(CompileError::Lex {
                loc,
                kind: LexError::UnterminatedChar,
            });
        }
        self.advance(); // '

        Ok(TokenKind::CharLit(value))
    }

    /// ワイド文字リテラルをスキャン
    fn scan_wide_char(&mut self) -> Result<TokenKind> {
        let loc = self.current_location();
        self.advance(); // '

        let value = match self.peek() {
            Some(b'\'') => {
                return Err(CompileError::Lex {
                    loc,
                    kind: LexError::EmptyCharLit,
                });
            }
            Some(b'\\') => {
                self.advance();
                self.scan_escape_sequence(&loc)? as u32
            }
            Some(c) => {
                self.advance();
                c as u32
            }
            None => {
                return Err(CompileError::Lex {
                    loc,
                    kind: LexError::UnterminatedChar,
                });
            }
        };

        if self.peek() != Some(b'\'') {
            return Err(CompileError::Lex {
                loc,
                kind: LexError::UnterminatedChar,
            });
        }
        self.advance(); // '

        Ok(TokenKind::WideCharLit(value))
    }

    /// エスケープシーケンスをスキャン
    fn scan_escape_sequence(&mut self, loc: &SourceLocation) -> Result<u8> {
        match self.peek() {
            Some(b'n') => {
                self.advance();
                Ok(b'\n')
            }
            Some(b't') => {
                self.advance();
                Ok(b'\t')
            }
            Some(b'r') => {
                self.advance();
                Ok(b'\r')
            }
            Some(b'\\') => {
                self.advance();
                Ok(b'\\')
            }
            Some(b'\'') => {
                self.advance();
                Ok(b'\'')
            }
            Some(b'"') => {
                self.advance();
                Ok(b'"')
            }
            Some(b'0') => {
                self.advance();
                Ok(0)
            }
            Some(b'a') => {
                self.advance();
                Ok(0x07) // bell
            }
            Some(b'b') => {
                self.advance();
                Ok(0x08) // backspace
            }
            Some(b'f') => {
                self.advance();
                Ok(0x0C) // form feed
            }
            Some(b'v') => {
                self.advance();
                Ok(0x0B) // vertical tab
            }
            Some(b'x') => {
                self.advance();
                self.scan_hex_escape(loc)
            }
            Some(c @ b'0'..=b'7') => self.scan_octal_escape(c),
            Some(c) => Err(CompileError::Lex {
                loc: loc.clone(),
                kind: LexError::InvalidEscape(c as char),
            }),
            None => Err(CompileError::Lex {
                loc: loc.clone(),
                kind: LexError::UnterminatedString,
            }),
        }
    }

    /// 16進エスケープをスキャン
    fn scan_hex_escape(&mut self, loc: &SourceLocation) -> Result<u8> {
        let mut value = 0u8;
        let mut count = 0;

        while let Some(c) = self.peek() {
            if let Some(digit) = (c as char).to_digit(16) {
                value = value.wrapping_mul(16).wrapping_add(digit as u8);
                self.advance();
                count += 1;
                if count >= 2 {
                    break;
                }
            } else {
                break;
            }
        }

        if count == 0 {
            return Err(CompileError::Lex {
                loc: loc.clone(),
                kind: LexError::InvalidEscape('x'),
            });
        }

        Ok(value)
    }

    /// 8進エスケープをスキャン
    fn scan_octal_escape(&mut self, first: u8) -> Result<u8> {
        let mut value = (first - b'0') as u8;
        self.advance();

        for _ in 0..2 {
            if let Some(c @ b'0'..=b'7') = self.peek() {
                value = value * 8 + (c - b'0');
                self.advance();
            } else {
                break;
            }
        }

        Ok(value)
    }

    // === 演算子スキャン ===

    fn scan_plus(&mut self) -> Result<TokenKind> {
        self.advance();
        match self.peek() {
            Some(b'+') => {
                self.advance();
                Ok(TokenKind::PlusPlus)
            }
            Some(b'=') => {
                self.advance();
                Ok(TokenKind::PlusEq)
            }
            _ => Ok(TokenKind::Plus),
        }
    }

    fn scan_minus(&mut self) -> Result<TokenKind> {
        self.advance();
        match self.peek() {
            Some(b'-') => {
                self.advance();
                Ok(TokenKind::MinusMinus)
            }
            Some(b'=') => {
                self.advance();
                Ok(TokenKind::MinusEq)
            }
            Some(b'>') => {
                self.advance();
                Ok(TokenKind::Arrow)
            }
            _ => Ok(TokenKind::Minus),
        }
    }

    fn scan_star(&mut self) -> Result<TokenKind> {
        self.advance();
        if self.peek() == Some(b'=') {
            self.advance();
            Ok(TokenKind::StarEq)
        } else {
            Ok(TokenKind::Star)
        }
    }

    fn scan_slash(&mut self) -> Result<TokenKind> {
        self.advance();
        if self.peek() == Some(b'=') {
            self.advance();
            Ok(TokenKind::SlashEq)
        } else {
            Ok(TokenKind::Slash)
        }
    }

    fn scan_percent(&mut self) -> Result<TokenKind> {
        self.advance();
        if self.peek() == Some(b'=') {
            self.advance();
            Ok(TokenKind::PercentEq)
        } else {
            Ok(TokenKind::Percent)
        }
    }

    fn scan_amp(&mut self) -> Result<TokenKind> {
        self.advance();
        match self.peek() {
            Some(b'&') => {
                self.advance();
                Ok(TokenKind::AmpAmp)
            }
            Some(b'=') => {
                self.advance();
                Ok(TokenKind::AmpEq)
            }
            _ => Ok(TokenKind::Amp),
        }
    }

    fn scan_pipe(&mut self) -> Result<TokenKind> {
        self.advance();
        match self.peek() {
            Some(b'|') => {
                self.advance();
                Ok(TokenKind::PipePipe)
            }
            Some(b'=') => {
                self.advance();
                Ok(TokenKind::PipeEq)
            }
            _ => Ok(TokenKind::Pipe),
        }
    }

    fn scan_caret(&mut self) -> Result<TokenKind> {
        self.advance();
        if self.peek() == Some(b'=') {
            self.advance();
            Ok(TokenKind::CaretEq)
        } else {
            Ok(TokenKind::Caret)
        }
    }

    fn scan_bang(&mut self) -> Result<TokenKind> {
        self.advance();
        if self.peek() == Some(b'=') {
            self.advance();
            Ok(TokenKind::BangEq)
        } else {
            Ok(TokenKind::Bang)
        }
    }

    fn scan_lt(&mut self) -> Result<TokenKind> {
        self.advance();
        match self.peek() {
            Some(b'<') => {
                self.advance();
                if self.peek() == Some(b'=') {
                    self.advance();
                    Ok(TokenKind::LtLtEq)
                } else {
                    Ok(TokenKind::LtLt)
                }
            }
            Some(b'=') => {
                self.advance();
                Ok(TokenKind::LtEq)
            }
            _ => Ok(TokenKind::Lt),
        }
    }

    fn scan_gt(&mut self) -> Result<TokenKind> {
        self.advance();
        match self.peek() {
            Some(b'>') => {
                self.advance();
                if self.peek() == Some(b'=') {
                    self.advance();
                    Ok(TokenKind::GtGtEq)
                } else {
                    Ok(TokenKind::GtGt)
                }
            }
            Some(b'=') => {
                self.advance();
                Ok(TokenKind::GtEq)
            }
            _ => Ok(TokenKind::Gt),
        }
    }

    fn scan_eq(&mut self) -> Result<TokenKind> {
        self.advance();
        if self.peek() == Some(b'=') {
            self.advance();
            Ok(TokenKind::EqEq)
        } else {
            Ok(TokenKind::Eq)
        }
    }

    fn scan_dot(&mut self) -> Result<TokenKind> {
        self.advance();
        if self.peek() == Some(b'.') && self.peek_n(1) == Some(b'.') {
            self.advance();
            self.advance();
            Ok(TokenKind::Ellipsis)
        } else {
            Ok(TokenKind::Dot)
        }
    }

    fn scan_hash(&mut self) -> Result<TokenKind> {
        self.advance();
        if self.peek() == Some(b'#') {
            self.advance();
            Ok(TokenKind::HashHash)
        } else {
            Ok(TokenKind::Hash)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lex(source: &str) -> Vec<TokenKind> {
        let mut interner = StringInterner::new();
        let mut lexer = Lexer::new(source.as_bytes(), FileId::default(), &mut interner);
        let mut tokens = Vec::new();
        loop {
            let token = lexer.next_token().unwrap();
            if matches!(token.kind, TokenKind::Eof) {
                break;
            }
            tokens.push(token.kind);
        }
        tokens
    }

    #[test]
    fn test_operators() {
        let tokens = lex("+ - * / % ++ -- += -= -> == != <= >=");
        assert_eq!(
            tokens,
            vec![
                TokenKind::Plus,
                TokenKind::Minus,
                TokenKind::Star,
                TokenKind::Slash,
                TokenKind::Percent,
                TokenKind::PlusPlus,
                TokenKind::MinusMinus,
                TokenKind::PlusEq,
                TokenKind::MinusEq,
                TokenKind::Arrow,
                TokenKind::EqEq,
                TokenKind::BangEq,
                TokenKind::LtEq,
                TokenKind::GtEq,
            ]
        );
    }

    #[test]
    fn test_keywords_and_identifiers() {
        // キーワードはTokenKind::Kw*として返し、識別子はTokenKind::Identとして返す
        let mut interner = StringInterner::new();
        let mut lexer = Lexer::new(
            b"int if else while for return struct foo",
            FileId::default(),
            &mut interner,
        );

        let mut tokens = Vec::new();
        loop {
            let token = lexer.next_token().unwrap();
            if matches!(token.kind, TokenKind::Eof) {
                break;
            }
            tokens.push(token.kind);
        }

        // キーワードはキーワードトークンとして返される
        assert!(matches!(tokens[0], TokenKind::KwInt));
        assert!(matches!(tokens[1], TokenKind::KwIf));
        assert!(matches!(tokens[2], TokenKind::KwElse));
        assert!(matches!(tokens[3], TokenKind::KwWhile));
        assert!(matches!(tokens[4], TokenKind::KwFor));
        assert!(matches!(tokens[5], TokenKind::KwReturn));
        assert!(matches!(tokens[6], TokenKind::KwStruct));
        // 識別子は識別子トークンとして返される
        if let TokenKind::Ident(id) = tokens[7] {
            assert_eq!(interner.get(id), "foo");
        } else {
            panic!("Expected Ident for 'foo'");
        }
    }

    #[test]
    fn test_numbers() {
        let tokens = lex("42 0x1F 0b101 0777 3.14 1e10");
        assert_eq!(
            tokens,
            vec![
                TokenKind::IntLit(42),
                TokenKind::IntLit(0x1F),
                TokenKind::IntLit(0b101),
                TokenKind::IntLit(0o777),
                TokenKind::FloatLit(3.14),
                TokenKind::FloatLit(1e10),
            ]
        );
    }

    #[test]
    fn test_strings() {
        let tokens = lex(r#""hello" "world\n""#);
        assert_eq!(
            tokens,
            vec![
                TokenKind::StringLit(b"hello".to_vec()),
                TokenKind::StringLit(b"world\n".to_vec()),
            ]
        );
    }

    #[test]
    fn test_comments() {
        let mut interner = StringInterner::new();
        let mut lexer = Lexer::new(
            b"// line comment\n42 /* block */ 100",
            FileId::default(),
            &mut interner,
        );

        // コメントの後に改行がある場合、改行トークンにコメントが付く
        let newline = lexer.next_token().unwrap();
        assert_eq!(newline.kind, TokenKind::Newline);
        assert_eq!(newline.leading_comments.len(), 1);
        assert_eq!(newline.leading_comments[0].kind, CommentKind::Line);

        let tok1 = lexer.next_token().unwrap();
        assert_eq!(tok1.kind, TokenKind::IntLit(42));

        let tok2 = lexer.next_token().unwrap();
        assert_eq!(tok2.kind, TokenKind::IntLit(100));
        assert_eq!(tok2.leading_comments.len(), 1);
        assert_eq!(tok2.leading_comments[0].kind, CommentKind::Block);
    }

    #[test]
    fn test_ellipsis() {
        let tokens = lex("...");
        assert_eq!(tokens, vec![TokenKind::Ellipsis]);
    }
}
