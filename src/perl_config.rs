//! Perl Config.pm から設定を取得するモジュール

use std::path::PathBuf;
use std::process::Command;

use crate::preprocessor::PPConfig;

/// Perl Config から取得した設定
#[derive(Debug)]
pub struct PerlConfig {
    /// インクルードパス (incpth + archlib/CORE)
    pub include_paths: Vec<PathBuf>,
    /// プリプロセッサマクロ定義 (cppsymbols)
    pub defines: Vec<(String, Option<String>)>,
    /// 対象 perl の build mode (threaded / non-threaded)
    pub build_mode: PerlBuildMode,
}

/// 対象 perl の build mode
///
/// `Threaded` は `-Dusethreads` でビルドされた perl
/// （`PERL_IMPLICIT_CONTEXT` / `MULTIPLICITY` が定義されている）。
/// 関数は `my_perl: *mut PerlInterpreter` を第一引数に取り、
/// マクロは `aTHX_` で `my_perl` を伝播する。
///
/// `NonThreaded` は `-Uusethreads` の perl。`pTHX_` / `aTHX_` は
/// 空展開され、関数は `my_perl` を取らない。`PL_curcop` 等のグローバル
/// は実 extern 変数として bindings に出る。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerlBuildMode {
    Threaded,
    NonThreaded,
}

impl PerlBuildMode {
    /// `Config{usethreads}` を読んで自動検出する
    ///
    /// 判定順:
    /// 1. `Config{usethreads}` == `"define"` → `Threaded`
    /// 2. それ以外（`"undef"` / 空文字列）→ `NonThreaded`
    pub fn detect_from_perl_config() -> Result<Self, PerlConfigError> {
        let usethreads = get_config_value("usethreads")?;
        if usethreads == "define" {
            Ok(PerlBuildMode::Threaded)
        } else {
            Ok(PerlBuildMode::NonThreaded)
        }
    }

    pub fn is_threaded(self) -> bool {
        matches!(self, Self::Threaded)
    }
}

/// Perl Config 取得エラー
#[derive(Debug)]
pub enum PerlConfigError {
    /// perl コマンド実行失敗
    CommandFailed(String),
    /// Config 値の取得失敗
    ConfigNotFound(String),
    /// パースエラー
    ParseError(String),
}

impl std::fmt::Display for PerlConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PerlConfigError::CommandFailed(msg) => write!(f, "perl command failed: {}", msg),
            PerlConfigError::ConfigNotFound(key) => write!(f, "Config key not found: {}", key),
            PerlConfigError::ParseError(msg) => write!(f, "parse error: {}", msg),
        }
    }
}

impl std::error::Error for PerlConfigError {}

/// Perl Config.pm から指定されたキーの値を取得
fn get_config_value(key: &str) -> Result<String, PerlConfigError> {
    let output = Command::new("perl")
        .args(["-MConfig", "-le", &format!("print $Config{{{}}}", key)])
        .output()
        .map_err(|e| PerlConfigError::CommandFailed(e.to_string()))?;

    if !output.status.success() {
        return Err(PerlConfigError::CommandFailed(
            String::from_utf8_lossy(&output.stderr).to_string(),
        ));
    }

    let value = String::from_utf8_lossy(&output.stdout).trim().to_string();
    Ok(value)
}

/// cppsymbols 文字列をパースして (名前, 値) のペアに変換
///
/// 形式: `NAME=VALUE NAME2=VALUE2 NAME3` (スペース区切り)
/// 値には `\ ` (エスケープされたスペース) を含む場合がある
fn parse_cppsymbols(symbols: &str) -> Vec<(String, Option<String>)> {
    let mut result = Vec::new();
    let mut current = String::new();
    let mut chars = symbols.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '\\' {
            // エスケープシーケンス
            if let Some(&next) = chars.peek() {
                current.push(c);
                current.push(next);
                chars.next();
            }
        } else if c == ' ' || c == '\t' {
            // 区切り文字
            if !current.is_empty() {
                result.push(parse_single_define(&current));
                current.clear();
            }
        } else {
            current.push(c);
        }
    }

    // 最後の要素
    if !current.is_empty() {
        result.push(parse_single_define(&current));
    }

    result
}

/// 単一の定義文字列をパース (NAME または NAME=VALUE)
/// バックスラッシュエスケープ (\ ) をスペースに変換
fn parse_single_define(s: &str) -> (String, Option<String>) {
    if let Some(pos) = s.find('=') {
        let (name, value) = s.split_at(pos);
        // バックスラッシュエスケープを解除 (\ -> スペース)
        let unescaped_value = value[1..].replace("\\ ", " ");
        (name.to_string(), Some(unescaped_value))
    } else {
        (s.to_string(), None)
    }
}

/// incpth 文字列をパースしてパスのベクターに変換
fn parse_incpth(incpth: &str) -> Vec<PathBuf> {
    incpth
        .split_whitespace()
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .collect()
}

/// ExtUtils::Embed の ccopts から -D オプションを抽出
fn get_ccopts_defines() -> Result<Vec<(String, Option<String>)>, PerlConfigError> {
    let output = Command::new("perl")
        .args(["-MExtUtils::Embed", "-e", "print ccopts"])
        .output()
        .map_err(|e| PerlConfigError::CommandFailed(e.to_string()))?;

    if !output.status.success() {
        return Err(PerlConfigError::CommandFailed(
            String::from_utf8_lossy(&output.stderr).to_string(),
        ));
    }

    let ccopts = String::from_utf8_lossy(&output.stdout);
    let mut defines = Vec::new();

    for part in ccopts.split_whitespace() {
        if let Some(def) = part.strip_prefix("-D") {
            defines.push(parse_single_define(def));
        }
    }

    Ok(defines)
}

/// Perl のデフォルトターゲットディレクトリを取得
/// archlib/CORE (例: /usr/lib64/perl5/CORE)
pub fn get_default_target_dir() -> Result<PathBuf, PerlConfigError> {
    let archlib = get_config_value("archlib")?;
    if archlib.is_empty() {
        return Err(PerlConfigError::ConfigNotFound("archlib".to_string()));
    }
    Ok(PathBuf::from(&archlib).join("CORE"))
}

/// Perl のメジャー・マイナーバージョンを取得
///
/// $Config{version} は "5.40.0" のような形式
/// 戻り値: (major, minor) タプル (例: (5, 40))
pub fn get_perl_version() -> Result<(u32, u32), PerlConfigError> {
    let version = get_config_value("version")?;
    if version.is_empty() {
        return Err(PerlConfigError::ConfigNotFound("version".to_string()));
    }

    // "5.40.0" -> ["5", "40", "0"]
    let parts: Vec<&str> = version.split('.').collect();
    if parts.len() < 2 {
        return Err(PerlConfigError::ParseError(format!(
            "invalid version format: {}",
            version
        )));
    }

    let major = parts[0].parse::<u32>().map_err(|_| {
        PerlConfigError::ParseError(format!("invalid major version: {}", parts[0]))
    })?;

    let minor = parts[1].parse::<u32>().map_err(|_| {
        PerlConfigError::ParseError(format!("invalid minor version: {}", parts[1]))
    })?;

    Ok((major, minor))
}

/// Get the compiler's built-in defines (`-dM` dump).
/// 
/// On Windows, `Config{cppsymbols}` is empty (Win32 perl does not run Configure),
/// and `_WIN32` / `__MINGW32__` / `__GNUC__`etc. are only defined as compiler built-ins.
/// System headers (e.g. the  `#error` guard in mingw-w64's `sys/types.h`) require these, so we mirror the built-ins of `CC` (default `cc`).
fn builtin_defines() -> Vec<(String, Option<String>)> {
    const DYNAMIC: [&str; 5] =
        ["__DATE__", "__FILE__", "__TIME__", "__TIMESTAMP__", "__INCLUDE_LEVEL__"];
    let cc = std::env::var("CC").unwrap_or_else(|_| {
        eprintln!("Env CC not defined; using default exe name >cc<");
        "cc".to_string()
    });
    let Ok(output) = Command::new(&cc)
        .args(["-dM", "-E", "-x", "c", "-"])
        .stdin(std::process::Stdio::null())
        .output().inspect_err(|e| {
            eprintln!("Error running >{cc}<.\n{e}");
        })
    else {
        return Vec::new();
    };
    if !output.status.success() {
        eprintln!("Error running >{cc}<. Exited with code {:?}", output.status.code());
        return Vec::new();
    }
    let mut defs = Vec::new();
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        let Some(rest) = line.strip_prefix("#define ") else { continue };
        let (name, value) = match rest.split_once(' ') {
            Some((n, v)) => (n, Some(v)),
            None => (rest, None),
        };
        // Dynamic macros and function-like macros are excluded.
        if DYNAMIC.contains(&name) /* || name.contains('(') */ {
            continue;
        }
        defs.push((name.to_string(), value.map(|s| s.to_string())));
    }
    defs
}

/// Perl Config.pm から設定を取得
pub fn get_perl_config() -> Result<PerlConfig, PerlConfigError> {
    // Helper function that converts function-like macros into just their name
    fn bare(name: &str) -> &str {
        name.split('(').next().unwrap_or(name)
    }

    // インクルードパスを取得
    let incpth = get_config_value("incpth")?;
    let mut include_paths = parse_incpth(&incpth);

    // archlib/CORE を追加 (Perl ヘッダー)
    let archlib = get_config_value("archlib")?;
    if !archlib.is_empty() {
        let core_path = PathBuf::from(&archlib).join("CORE");
        if core_path.exists() {
            include_paths.push(core_path);
        }
    }

    // cppsymbols を取得
    let cppsymbols = get_config_value("cppsymbols")?;
    let mut defines = parse_cppsymbols(&cppsymbols);

    // ccopts から -D オプションを抽出して追加（重複は後で上書きされる）
    if let Ok(ccopts_defines) = get_ccopts_defines() {
        for (name, value) in ccopts_defines {
            // 既存の定義を上書き（ccoptの方が優先）
            if let Some(pos) = defines.iter().position(|(n, _)| n == &name) {
                defines[pos] = (name, value);
            } else {
                defines.push((name, value));
            }
        }
    }

    // Add the compiler's built-in defines (_WIN32 / __MINGW32__/ __GNUC__, etc.).
    // Explicit definitions (cppsymbols / ccopts) take precedence.
    for (name, value) in builtin_defines() {
        let bare_name = bare(name.as_str());
        if !defines.iter().any(|(n, _)| bare(n.as_str()) == bare_name) {
            defines.push((name, value));
        }
    }

    // PERL_CORE を追加 (perl.h内のDFA表などを正しく展開するために必要)
    defines.push(("PERL_CORE".to_string(), None));

    // デバッグ: __x86_64__ が含まれているか確認
    if std::env::var("DEBUG_PERL_CONFIG").is_ok() {
        eprintln!("[perl_config] include_paths: {:?}", include_paths);
        eprintln!("[perl_config] defines count: {}", defines.len());
        for (name, value) in &defines {
            if name.contains("x86") || name.contains("LP64") {
                eprintln!("[perl_config] {} = {:?}", name, value);
            }
        }
    }

    let build_mode = PerlBuildMode::detect_from_perl_config()?;

    Ok(PerlConfig {
        include_paths,
        defines,
        build_mode,
    })
}

/// Perl 環境用の PPConfig を構築
///
/// get_perl_config() と get_default_target_dir() を組み合わせて
/// プリプロセッサ設定を構築する。build.rs から呼び出すことを想定。
pub fn build_pp_config_for_perl() -> Result<PPConfig, PerlConfigError> {
    let perl_cfg = get_perl_config()?;
    let target_dir = get_default_target_dir().ok();
    Ok(PPConfig {
        include_paths: perl_cfg.include_paths,
        predefined: perl_cfg.defines,
        debug_pp: false,
        target_dir,
        emit_markers: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_single_define() {
        assert_eq!(
            parse_single_define("FOO"),
            ("FOO".to_string(), None)
        );
        assert_eq!(
            parse_single_define("FOO=1"),
            ("FOO".to_string(), Some("1".to_string()))
        );
        assert_eq!(
            parse_single_define("__GNUC__=15"),
            ("__GNUC__".to_string(), Some("15".to_string()))
        );
    }

    #[test]
    fn test_parse_cppsymbols_simple() {
        let symbols = "FOO=1 BAR=2 BAZ";
        let result = parse_cppsymbols(symbols);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], ("FOO".to_string(), Some("1".to_string())));
        assert_eq!(result[1], ("BAR".to_string(), Some("2".to_string())));
        assert_eq!(result[2], ("BAZ".to_string(), None));
    }

    #[test]
    fn test_parse_cppsymbols_with_escape() {
        // エスケープされたスペースを含む値 (\ はスペースに変換される)
        let symbols = r#"__VERSION__="15.1.1\ 20250521" FOO=1"#;
        let result = parse_cppsymbols(symbols);
        assert_eq!(result.len(), 2);
        assert_eq!(
            result[0],
            ("__VERSION__".to_string(), Some(r#""15.1.1 20250521""#.to_string()))
        );
        assert_eq!(result[1], ("FOO".to_string(), Some("1".to_string())));
    }

    #[test]
    fn test_parse_incpth() {
        let incpth = "/usr/lib/gcc/x86_64-redhat-linux/15/include /usr/local/include /usr/include";
        let result = parse_incpth(incpth);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], PathBuf::from("/usr/lib/gcc/x86_64-redhat-linux/15/include"));
        assert_eq!(result[1], PathBuf::from("/usr/local/include"));
        assert_eq!(result[2], PathBuf::from("/usr/include"));
    }

    #[test]
    fn test_parse_version() {
        // バージョン文字列のパースをテスト（内部ヘルパー関数で）
        fn parse_version(version: &str) -> Option<(u32, u32)> {
            let parts: Vec<&str> = version.split('.').collect();
            if parts.len() < 2 {
                return None;
            }
            let major = parts[0].parse::<u32>().ok()?;
            let minor = parts[1].parse::<u32>().ok()?;
            Some((major, minor))
        }

        assert_eq!(parse_version("5.40.0"), Some((5, 40)));
        assert_eq!(parse_version("5.38.2"), Some((5, 38)));
        assert_eq!(parse_version("5.10.1"), Some((5, 10)));
        assert_eq!(parse_version("5.8"), Some((5, 8)));
        assert_eq!(parse_version("invalid"), None);
    }
}
