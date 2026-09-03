use std::io::IsTerminal;

use anyhow::Result;
use vision_engine::cli::{ParseOutcome, parse_args, print_usage, validate_config};

const DEFAULT_LOG_FILTER: &str = "info,ort=warn";

fn main() {
    init_tracing();
    if let Err(err) = run() {
        eprintln!("{err:?}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    match parse_args(std::env::args_os().skip(1))? {
        ParseOutcome::Help => {
            print_usage();
            Ok(())
        }
        ParseOutcome::Run(config) => {
            validate_config(&config)?;
            tracing::info!(
                video = %config.video.display(),
                model = %config.model.display(),
                "startup configuration validated"
            );
            vision_engine::pipeline::run(&config)
        }
    }
}

fn init_tracing() {
    // ONNX Runtime bridges its own verbose INFO logging into `tracing` under the
    // `ort` target, which buries application output. Keep it at `warn` unless the
    // user asks for more through `RUST_LOG`.
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(DEFAULT_LOG_FILTER));

    // Colour codes are written between the field name and `=`, which makes
    // redirected logs unparseable by ordinary text tooling. Emit them only when
    // stdout is actually a terminal.
    let use_ansi = std::io::stdout().is_terminal();

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .with_ansi(use_ansi)
        .init();
}
