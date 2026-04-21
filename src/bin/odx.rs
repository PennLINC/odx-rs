use std::io;
use std::path::{Path, PathBuf};

use clap::{Args, CommandFactory, Parser, Subcommand, ValueEnum};
use clap_complete::generate;
use odx_rs::cli_support::{
    detect_target_format, ensure_output_path, load_dataset, load_dataset_with_format,
    render_summary, render_validation, summarize_dataset, validation_report, ConversionSummary,
    DetectedFormat, LoadDatasetOptions,
};
use odx_rs::interop::{
    fit_mrtrix_sh_from_odf, save_dsistudio_from_odx, DenseOdfMode, DsistudioFormat,
    MrtrixToDsistudioOptions, PeakSource, Z0Policy,
};
use odx_rs::mrtrix::{
    self, MrtrixFixelContainer, MrtrixFixelWriteOptions, MrtrixShContainer, MrtrixShWriteOptions,
};
use odx_rs::pam::{self, PamWriteOptions};
use odx_rs::{
    compute_fixel_qc, write_qc_class_dpf, FixelQcOptions, FixelQcReport, OdxDataset, OdxError,
    OdxWritePolicy, ThresholdMode,
};

#[derive(Parser, Debug)]
#[command(name = "odx")]
#[command(about = "ODX conversion, inspection, and validation tools", version)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Print a concise summary of a dataset or supported foreign-format input.
    Info(CommonInputArgs),
    /// Convert between ODX, DSI Studio, and MRtrix representations.
    Convert(ConvertArgs),
    /// Validate internal consistency after normalizing into an ODX dataset.
    Validate(ValidateArgs),
    /// Compute fixel coherence QC metrics and connected/disconnected summaries.
    Qc(QcArgs),
    /// Generate shell completions.
    Completions {
        #[arg(value_enum)]
        shell: clap_complete::Shell,
    },
}

#[derive(Args, Debug)]
struct CommonInputArgs {
    input: PathBuf,
    #[arg(long)]
    sh: Option<PathBuf>,
    #[arg(long = "fixel-dir")]
    fixel_dir: Option<PathBuf>,
    #[arg(long = "mapmri-tensor")]
    mapmri_tensor: Option<PathBuf>,
    #[arg(long = "mapmri-uvec")]
    mapmri_uvec: Option<PathBuf>,
    #[arg(long = "reference-affine")]
    reference_affine: Option<PathBuf>,
    #[arg(long = "input-format", value_enum)]
    input_format: Option<InputFormatOverride>,
    #[arg(long)]
    json: bool,
    #[arg(long)]
    verbose: bool,
}

#[derive(Args, Debug)]
struct ConvertArgs {
    input: PathBuf,
    output: PathBuf,
    #[arg(long)]
    sh: Option<PathBuf>,
    #[arg(long = "fixel-dir")]
    fixel_dir: Option<PathBuf>,
    #[arg(long = "mapmri-tensor")]
    mapmri_tensor: Option<PathBuf>,
    #[arg(long = "mapmri-uvec")]
    mapmri_uvec: Option<PathBuf>,
    #[arg(long = "reference-affine")]
    reference_affine: Option<PathBuf>,
    #[arg(long = "input-format", value_enum)]
    input_format: Option<InputFormatOverride>,
    #[arg(long = "output-format", value_enum)]
    output_format: Option<OutputFormatOverride>,
    #[arg(long)]
    overwrite: bool,
    #[arg(long)]
    quiet: bool,
    #[arg(long)]
    json: bool,
    #[arg(long = "odx-layout", value_enum)]
    odx_layout: Option<OdxLayoutArg>,
    #[arg(long = "quantize-dense")]
    quantize_dense: bool,
    #[arg(long = "quantize-min-len", default_value_t = 4096)]
    quantize_min_len: usize,
    #[arg(long = "out-sh")]
    out_sh: Option<PathBuf>,
    #[arg(long = "mrtrix-fixel-container", value_enum, default_value = "nifti")]
    mrtrix_fixel_container: MrtrixFixelContainerArg,
    #[arg(long = "mrtrix-sh-container", value_enum, default_value = "mif")]
    mrtrix_sh_container: MrtrixShContainerArg,
    #[arg(long = "mrtrix-sh-gzip")]
    mrtrix_sh_gzip: bool,
    #[arg(long = "sh-lmax")]
    sh_lmax: Option<u32>,
    #[arg(long = "dsi-format", value_enum)]
    dsi_format: Option<DsistudioFormatArg>,
    #[arg(long = "dense-odf", value_enum, default_value = "from-sh")]
    dense_odf: DenseOdfModeArg,
    #[arg(long = "peak-source", value_enum, default_value = "fixels")]
    peak_source: PeakSourceArg,
    #[arg(long = "amplitude-key")]
    amplitude_key: Option<String>,
    #[arg(long = "z0", value_enum, default_value = "auto")]
    z0: Z0PolicyArg,
}

#[derive(Args, Debug)]
struct ValidateArgs {
    input: PathBuf,
    #[arg(long)]
    sh: Option<PathBuf>,
    #[arg(long = "fixel-dir")]
    fixel_dir: Option<PathBuf>,
    #[arg(long = "mapmri-tensor")]
    mapmri_tensor: Option<PathBuf>,
    #[arg(long = "mapmri-uvec")]
    mapmri_uvec: Option<PathBuf>,
    #[arg(long = "reference-affine")]
    reference_affine: Option<PathBuf>,
    #[arg(long = "input-format", value_enum)]
    input_format: Option<InputFormatOverride>,
    #[arg(long)]
    json: bool,
    #[arg(long)]
    strict: bool,
}

#[derive(Args, Debug)]
struct QcArgs {
    input: PathBuf,
    #[arg(long)]
    sh: Option<PathBuf>,
    #[arg(long = "fixel-dir")]
    fixel_dir: Option<PathBuf>,
    #[arg(long = "mapmri-tensor")]
    mapmri_tensor: Option<PathBuf>,
    #[arg(long = "mapmri-uvec")]
    mapmri_uvec: Option<PathBuf>,
    #[arg(long = "reference-affine")]
    reference_affine: Option<PathBuf>,
    #[arg(long = "input-format", value_enum)]
    input_format: Option<InputFormatOverride>,
    #[arg(long = "primary-dpf")]
    primary_dpf: Option<String>,
    #[arg(long = "threshold", value_enum, default_value = "otsu")]
    threshold: QcThresholdArg,
    #[arg(long = "threshold-value")]
    threshold_value: Option<f32>,
    #[arg(long = "angle-deg", default_value_t = 15.0)]
    angle_deg: f32,
    #[arg(long = "write-qc-class")]
    write_qc_class: bool,
    #[arg(long = "overwrite-qc-class")]
    overwrite_qc_class: bool,
    #[arg(long)]
    json: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum InputFormatOverride {
    OdxDirectory,
    OdxArchive,
    DsistudioFibgz,
    DsistudioFz,
    DipyPam5,
    TortoiseMapmriNifti,
    MrtrixShImage,
    MrtrixFixelDir,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum OutputFormatOverride {
    OdxDirectory,
    OdxArchive,
    DsistudioFibgz,
    DsistudioFz,
    DipyPam5,
    MrtrixShImage,
    MrtrixFixelDir,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum OdxLayoutArg {
    Directory,
    Archive,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum MrtrixFixelContainerArg {
    Mif,
    Nifti,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum MrtrixShContainerArg {
    Mif,
    Nifti1,
    Nifti2,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum DsistudioFormatArg {
    Fibgz,
    Fz,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum DenseOdfModeArg {
    Off,
    FromSh,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum PeakSourceArg {
    Fixels,
    SampledOdf,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum Z0PolicyArg {
    Auto,
    Never,
    Always,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum QcThresholdArg {
    Otsu,
    Positive,
    All,
    Value,
}

impl From<InputFormatOverride> for DetectedFormat {
    fn from(value: InputFormatOverride) -> Self {
        match value {
            InputFormatOverride::OdxDirectory => DetectedFormat::OdxDirectory,
            InputFormatOverride::OdxArchive => DetectedFormat::OdxArchive,
            InputFormatOverride::DsistudioFibgz => DetectedFormat::DsistudioFibGz,
            InputFormatOverride::DsistudioFz => DetectedFormat::DsistudioFz,
            InputFormatOverride::DipyPam5 => DetectedFormat::DipyPam5,
            InputFormatOverride::TortoiseMapmriNifti => DetectedFormat::TortoiseMapmriNifti,
            InputFormatOverride::MrtrixShImage => DetectedFormat::MrtrixShImage,
            InputFormatOverride::MrtrixFixelDir => DetectedFormat::MrtrixFixelDir,
        }
    }
}

impl From<OutputFormatOverride> for DetectedFormat {
    fn from(value: OutputFormatOverride) -> Self {
        match value {
            OutputFormatOverride::OdxDirectory => DetectedFormat::OdxDirectory,
            OutputFormatOverride::OdxArchive => DetectedFormat::OdxArchive,
            OutputFormatOverride::DsistudioFibgz => DetectedFormat::DsistudioFibGz,
            OutputFormatOverride::DsistudioFz => DetectedFormat::DsistudioFz,
            OutputFormatOverride::DipyPam5 => DetectedFormat::DipyPam5,
            OutputFormatOverride::MrtrixShImage => DetectedFormat::MrtrixShImage,
            OutputFormatOverride::MrtrixFixelDir => DetectedFormat::MrtrixFixelDir,
        }
    }
}

impl From<MrtrixFixelContainerArg> for MrtrixFixelContainer {
    fn from(value: MrtrixFixelContainerArg) -> Self {
        match value {
            MrtrixFixelContainerArg::Mif => MrtrixFixelContainer::Mif,
            MrtrixFixelContainerArg::Nifti => MrtrixFixelContainer::Nifti,
        }
    }
}

impl From<MrtrixShContainerArg> for MrtrixShContainer {
    fn from(value: MrtrixShContainerArg) -> Self {
        match value {
            MrtrixShContainerArg::Mif => MrtrixShContainer::Mif,
            MrtrixShContainerArg::Nifti1 => MrtrixShContainer::Nifti1,
            MrtrixShContainerArg::Nifti2 => MrtrixShContainer::Nifti2,
        }
    }
}

impl From<DsistudioFormatArg> for DsistudioFormat {
    fn from(value: DsistudioFormatArg) -> Self {
        match value {
            DsistudioFormatArg::Fibgz => DsistudioFormat::FibGz,
            DsistudioFormatArg::Fz => DsistudioFormat::Fz,
        }
    }
}

impl From<DenseOdfModeArg> for DenseOdfMode {
    fn from(value: DenseOdfModeArg) -> Self {
        match value {
            DenseOdfModeArg::Off => DenseOdfMode::Off,
            DenseOdfModeArg::FromSh => DenseOdfMode::FromSh,
        }
    }
}

impl From<PeakSourceArg> for PeakSource {
    fn from(value: PeakSourceArg) -> Self {
        match value {
            PeakSourceArg::Fixels => PeakSource::Fixels,
            PeakSourceArg::SampledOdf => PeakSource::SampledOdf,
        }
    }
}

impl From<Z0PolicyArg> for Z0Policy {
    fn from(value: Z0PolicyArg) -> Self {
        match value {
            Z0PolicyArg::Auto => Z0Policy::Auto,
            Z0PolicyArg::Never => Z0Policy::Never,
            Z0PolicyArg::Always => Z0Policy::Always,
        }
    }
}

fn main() {
    let cli = Cli::parse();
    if let Err(err) = run(cli) {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run(cli: Cli) -> odx_rs::Result<()> {
    match cli.command {
        Command::Info(args) => run_info(args),
        Command::Convert(args) => run_convert(args),
        Command::Validate(args) => run_validate(args),
        Command::Qc(args) => run_qc(args),
        Command::Completions { shell } => {
            let mut cmd = Cli::command();
            generate(shell, &mut cmd, "odx", &mut io::stdout());
            Ok(())
        }
    }
}

fn run_info(args: CommonInputArgs) -> odx_rs::Result<()> {
    let (odx, detected) = load_from_args(
        &args.input,
        args.sh.as_deref(),
        args.fixel_dir.as_deref(),
        args.mapmri_tensor.as_deref(),
        args.mapmri_uvec.as_deref(),
        args.reference_affine.as_deref(),
        args.input_format,
    )?;
    let summary = summarize_dataset(&odx, detected);
    if args.json {
        println!("{}", serde_json::to_string_pretty(&summary)?);
    } else {
        print!("{}", render_summary(&summary));
    }

    if args.verbose {
        let report = validation_report(&odx);
        if args.json {
            println!("{}", serde_json::to_string_pretty(&report)?);
        } else {
            print!("{}", render_validation(&report));
        }
    }

    Ok(())
}

fn run_validate(args: ValidateArgs) -> odx_rs::Result<()> {
    let (odx, _detected) = load_from_args(
        &args.input,
        args.sh.as_deref(),
        args.fixel_dir.as_deref(),
        args.mapmri_tensor.as_deref(),
        args.mapmri_uvec.as_deref(),
        args.reference_affine.as_deref(),
        args.input_format,
    )?;
    let report = validation_report(&odx);
    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print!("{}", render_validation(&report));
    }

    if !report.ok {
        return Err(OdxError::Format("validation failed".into()));
    }
    if args.strict && !report.strict_ok {
        return Err(OdxError::Format(
            "validation produced warnings under --strict".into(),
        ));
    }
    Ok(())
}

fn run_qc(args: QcArgs) -> odx_rs::Result<()> {
    let (odx, detected) = load_from_args(
        &args.input,
        args.sh.as_deref(),
        args.fixel_dir.as_deref(),
        args.mapmri_tensor.as_deref(),
        args.mapmri_uvec.as_deref(),
        args.reference_affine.as_deref(),
        args.input_format,
    )?;
    if args.overwrite_qc_class && !args.write_qc_class {
        return Err(OdxError::Argument(
            "--overwrite-qc-class requires --write-qc-class".into(),
        ));
    }
    let threshold = match args.threshold {
        QcThresholdArg::Otsu => {
            if args.threshold_value.is_some() {
                return Err(OdxError::Argument(
                    "--threshold-value is only valid with --threshold value".into(),
                ));
            }
            ThresholdMode::Otsu
        }
        QcThresholdArg::Positive => {
            if args.threshold_value.is_some() {
                return Err(OdxError::Argument(
                    "--threshold-value is only valid with --threshold value".into(),
                ));
            }
            ThresholdMode::Positive
        }
        QcThresholdArg::All => {
            if args.threshold_value.is_some() {
                return Err(OdxError::Argument(
                    "--threshold-value is only valid with --threshold value".into(),
                ));
            }
            ThresholdMode::All
        }
        QcThresholdArg::Value => ThresholdMode::Value(args.threshold_value.ok_or_else(|| {
            OdxError::Argument("--threshold value requires --threshold-value <f32>".into())
        })?),
    };

    let computation = compute_fixel_qc(
        &odx,
        &FixelQcOptions {
            primary_metric: args.primary_dpf,
            threshold,
            angle_degrees: args.angle_deg,
        },
    )?;
    if args.write_qc_class {
        match detected {
            DetectedFormat::OdxDirectory | DetectedFormat::OdxArchive => {
                write_qc_class_dpf(&args.input, &computation.classes, args.overwrite_qc_class)?
            }
            _ => {
                return Err(OdxError::Format(
                    "--write-qc-class requires an ODX directory or .odx archive input".into(),
                ))
            }
        }
    }
    let report = &computation.report;

    if args.json {
        println!("{}", serde_json::to_string_pretty(report)?);
    } else {
        print!("{}", render_fixel_qc(report));
    }
    Ok(())
}

fn run_convert(args: ConvertArgs) -> odx_rs::Result<()> {
    let output_format = resolve_output_format(
        &args.output,
        args.output_format,
        args.odx_layout,
        args.dsi_format,
    )?;

    if output_format == DetectedFormat::MrtrixShImage && args.out_sh.is_some() {
        return Err(OdxError::Argument(
            "--out-sh is only valid when the main output is a MRtrix fixel directory".into(),
        ));
    }

    ensure_output_path(&args.output, args.overwrite)?;
    if let Some(out_sh) = args.out_sh.as_deref() {
        ensure_output_path(out_sh, args.overwrite)?;
    }

    let (odx, input_format) = load_from_args(
        &args.input,
        args.sh.as_deref(),
        args.fixel_dir.as_deref(),
        args.mapmri_tensor.as_deref(),
        args.mapmri_uvec.as_deref(),
        args.reference_affine.as_deref(),
        args.input_format,
    )?;

    let quant_policy = OdxWritePolicy {
        quantize_dense: args.quantize_dense,
        quantize_min_len: args.quantize_min_len,
    };

    match output_format {
        DetectedFormat::OdxDirectory => {
            odx.save_directory_with_policy(&args.output, quant_policy)?;
        }
        DetectedFormat::OdxArchive => {
            odx.save_archive_with_policy(&args.output, quant_policy)?;
        }
        DetectedFormat::DsistudioFibGz | DetectedFormat::DsistudioFz => {
            let options = MrtrixToDsistudioOptions {
                output_format: match output_format {
                    DetectedFormat::DsistudioFibGz => DsistudioFormat::FibGz,
                    DetectedFormat::DsistudioFz => DsistudioFormat::Fz,
                    _ => unreachable!(),
                },
                dense_odf_mode: args.dense_odf.into(),
                peak_source: args.peak_source.into(),
                amplitude_key: args.amplitude_key.clone(),
                write_z0: args.z0.into(),
            };
            save_dsistudio_from_odx(&odx, &args.output, &options)?;
        }
        DetectedFormat::DipyPam5 => {
            pam::save_pam5(&odx, &args.output, &PamWriteOptions::default())?;
        }
        DetectedFormat::TortoiseMapmriNifti => {
            return Err(OdxError::Argument(
                "TORTOISE MAPMRI output is not supported; this format is import-only".into(),
            ));
        }
        DetectedFormat::MrtrixFixelDir => {
            mrtrix::save_mrtrix_fixels(
                &odx,
                &args.output,
                &MrtrixFixelWriteOptions {
                    container: args.mrtrix_fixel_container.into(),
                    include_dpf: true,
                    include_dpv: false,
                },
            )?;
            if let Some(out_sh) = args.out_sh.as_deref() {
                let fitted = if odx.sh::<f32>("coefficients").is_ok() {
                    None
                } else {
                    fit_mrtrix_sh_from_odf(&odx, args.sh_lmax)?
                };
                if odx.sh::<f32>("coefficients").is_err() && fitted.is_none() {
                    return Err(OdxError::Argument(
                        "MRtrix SH output requires existing sh/coefficients or dense ODF data to fit from"
                            .into(),
                    ));
                }
                let sh_dataset = fitted.as_ref().unwrap_or(&odx);
                mrtrix::save_mrtrix_sh(
                    sh_dataset,
                    out_sh,
                    &MrtrixShWriteOptions {
                        array_name: "coefficients".into(),
                        container: args.mrtrix_sh_container.into(),
                        gzip: args.mrtrix_sh_gzip,
                    },
                )?;
            }
        }
        DetectedFormat::MrtrixShImage => {
            let fitted = if odx.sh::<f32>("coefficients").is_ok() {
                None
            } else {
                fit_mrtrix_sh_from_odf(&odx, args.sh_lmax)?
            };
            if odx.sh::<f32>("coefficients").is_err() && fitted.is_none() {
                return Err(OdxError::Argument(
                    "MRtrix SH output requires existing sh/coefficients or dense ODF data to fit from"
                        .into(),
                ));
            }
            let sh_dataset = fitted.as_ref().unwrap_or(&odx);
            mrtrix::save_mrtrix_sh(
                sh_dataset,
                &args.output,
                &MrtrixShWriteOptions {
                    array_name: "coefficients".into(),
                    container: args.mrtrix_sh_container.into(),
                    gzip: args.mrtrix_sh_gzip,
                },
            )?;
        }
    }

    if args.json {
        let summary = ConversionSummary {
            input_format: input_format.as_str().into(),
            output_format: output_format.as_str().into(),
            output_path: args.output.display().to_string(),
            out_sh_path: args.out_sh.as_ref().map(|p| p.display().to_string()),
            nb_voxels: odx.header().nb_voxels,
            nb_peaks: odx.header().nb_peaks,
        };
        println!("{}", serde_json::to_string_pretty(&summary)?);
    } else if !args.quiet {
        println!(
            "converted {} -> {}",
            input_format.as_str(),
            output_format.as_str()
        );
        println!("voxels: {}", odx.header().nb_voxels);
        println!("peaks: {}", odx.header().nb_peaks);
        println!("output: {}", args.output.display());
        if let Some(out_sh) = args.out_sh.as_deref() {
            println!("out_sh: {}", out_sh.display());
        }
    }

    Ok(())
}

fn load_from_args(
    input: &Path,
    sh: Option<&Path>,
    fixel_dir: Option<&Path>,
    mapmri_tensor: Option<&Path>,
    mapmri_uvec: Option<&Path>,
    reference_affine: Option<&Path>,
    input_override: Option<InputFormatOverride>,
) -> odx_rs::Result<(OdxDataset, DetectedFormat)> {
    if let Some(format) = input_override {
        let detected: DetectedFormat = format.into();
        let dataset = load_dataset_with_format(
            input,
            detected,
            LoadDatasetOptions {
                sh_path: sh,
                fixel_dir,
                mapmri_tensor_path: mapmri_tensor,
                mapmri_uvec_path: mapmri_uvec,
                reference_affine,
            },
        )?;
        return Ok((dataset, detected));
    }
    load_dataset(
        input,
        LoadDatasetOptions {
            sh_path: sh,
            fixel_dir,
            mapmri_tensor_path: mapmri_tensor,
            mapmri_uvec_path: mapmri_uvec,
            reference_affine,
        },
    )
}

fn resolve_output_format(
    output: &Path,
    output_format: Option<OutputFormatOverride>,
    odx_layout: Option<OdxLayoutArg>,
    dsi_format: Option<DsistudioFormatArg>,
) -> odx_rs::Result<DetectedFormat> {
    if let Some(format) = output_format {
        return Ok(format.into());
    }
    if let Some(layout) = odx_layout {
        return Ok(match layout {
            OdxLayoutArg::Directory => DetectedFormat::OdxDirectory,
            OdxLayoutArg::Archive => DetectedFormat::OdxArchive,
        });
    }
    if let Some(dsi) = dsi_format {
        return Ok(match dsi {
            DsistudioFormatArg::Fibgz => DetectedFormat::DsistudioFibGz,
            DsistudioFormatArg::Fz => DetectedFormat::DsistudioFz,
        });
    }
    detect_target_format(output)
}

fn render_fixel_qc(report: &FixelQcReport) -> String {
    let mut out = String::new();
    out.push_str(&format!("primary_metric: {}\n", report.primary_metric));
    out.push_str(&format!(
        "threshold_value: {}\n",
        report
            .threshold_value
            .map(|v| format!("{v:.6}"))
            .unwrap_or_else(|| "none".into())
    ));
    out.push_str(&format!("total_fixels: {}\n", report.total_fixels));
    out.push_str(&format!("evaluated_fixels: {}\n", report.evaluated_fixels));
    out.push_str(&format!("excluded_fixels: {}\n", report.excluded_fixels));
    out.push_str(&format!("connected_fixels: {}\n", report.connected_fixels));
    out.push_str(&format!(
        "disconnected_fixels: {}\n",
        report.disconnected_fixels
    ));
    out.push_str(&format!(
        "connected_to_disconnected_ratio: {}\n",
        report
            .connected_to_disconnected_ratio
            .map(|v| format!("{v:.6}"))
            .unwrap_or_else(|| "none".into())
    ));
    out.push_str(&format!(
        "coherence_index: {}\n",
        report
            .coherence_index
            .map(|v| format!("{v:.6}"))
            .unwrap_or_else(|| "none".into())
    ));
    out.push_str(&format!(
        "incoherence_index: {}\n",
        report
            .incoherence_index
            .map(|v| format!("{v:.6}"))
            .unwrap_or_else(|| "none".into())
    ));
    if report.skipped_dpf.is_empty() {
        out.push_str("skipped_dpf: none\n");
    } else {
        out.push_str(&format!("skipped_dpf: {}\n", report.skipped_dpf.join(", ")));
    }
    if !report.per_dpf.is_empty() {
        out.push_str("per_dpf:\n");
        for (name, stats) in &report.per_dpf {
            out.push_str(&format!(
                "  {name}: connected(count={}, mean={}, median={}), disconnected(count={}, mean={}, median={})\n",
                stats.connected.count,
                render_optional_f64(stats.connected.mean),
                render_optional_f32(stats.connected.median),
                stats.disconnected.count,
                render_optional_f64(stats.disconnected.mean),
                render_optional_f32(stats.disconnected.median),
            ));
        }
    }
    out
}

fn render_optional_f64(value: Option<f64>) -> String {
    value
        .map(|v| format!("{v:.6}"))
        .unwrap_or_else(|| "none".into())
}

fn render_optional_f32(value: Option<f32>) -> String {
    value
        .map(|v| format!("{v:.6}"))
        .unwrap_or_else(|| "none".into())
}
