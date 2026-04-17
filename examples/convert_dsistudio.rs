use std::path::PathBuf;

use odx_rs::{dsistudio, mif, Result};

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let input = PathBuf::from(args.next().expect(
        "usage: convert_dsistudio <input.fib.gz> <output.fz|output.fib.gz> [reference.mif.gz]",
    ));
    let output = PathBuf::from(args.next().expect(
        "usage: convert_dsistudio <input.fib.gz> <output.fz|output.fib.gz> [reference.mif.gz]",
    ));
    let reference = args.next().map(PathBuf::from);

    let affine = if let Some(reference_path) = reference {
        Some(mif::read_mif(&reference_path)?.affine_4x4())
    } else {
        None
    };

    let dataset = dsistudio::load_fibgz(&input, affine)?;

    match output.extension().and_then(|e| e.to_str()) {
        Some("fz") => dsistudio::save_fz(&dataset, &output)?,
        Some("gz") => dsistudio::save_fibgz(&dataset, &output)?,
        other => panic!("unsupported output extension: {other:?}"),
    }

    println!("{}", output.display());
    Ok(())
}
