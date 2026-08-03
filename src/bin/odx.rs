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
    combine_odx, compare_odx, compute_fixel_qc, write_qc_class_dpf, CombineInput, CombineOptions,
    CombineOutputs, CombineReport, CompareOptions, CompareReport, FixelQcOptions, FixelQcReport,
    LmaxPolicy, LooMode, MaskCombine, NormalizeFod, OdxDataset, OdxError, OdxWritePolicy,
    PeakFinderConfig, TemplateMethod, ThresholdMode,
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
    /// Pairwise fixel comparison between two ODX files (matching, DPF diffs).
    Compare(CompareArgs),
    /// Combine many template-space ODX into group fixels + per-subject angular
    /// distance to each group fixel (the N-way generalization of `compare`).
    ///
    /// Builds a shared set of group fixels (`--method cluster` pools subject
    /// directions; `--method mean-fod` peak-finds the mean FOD), matches every
    /// subject onto them, and writes a group ODX whose `angle_deg` DPF is an
    /// (n_fixels × n_subjects) matrix — ModelArray's per-scalar `values` shape —
    /// plus a cohort CSV for ModelArrayIO/ModelArray.
    Combine(CombineArgs),
    /// Convert a pyAFQ asymmetric ODF (`*_param-aodf_dwimap.nii.gz`) into ODX.
    /// Stores full-basis descoteaux07 SH and precomputes per-voxel asymmetric peaks.
    ImportAodf(ImportAodfArgs),
    /// Spatially upsample an ODX onto a finer isotropic voxel grid.
    ///
    /// SH coefficients and DPV arrays are trilinearly interpolated (with
    /// boundary renormalization so signal levels are preserved at mask edges).
    /// Fixels are recomputed from the interpolated SH via peak finding.
    /// DPF arrays other than `amplitude` are dropped — they cannot be remapped
    /// to recomputed fixels. Dense ODF data is not supported.
    ///
    /// EXAMPLE — resample a 1.25 mm dataset to 1.0 mm:
    ///   odx upsample subject_1p25mm.odx subject_1mm.odx --voxel-spacing 1.0
    Upsample(UpsampleArgs),
    /// Apply an ANTs/ITK spatial transform to an ODX dataset (grid-resample
    /// SH/DPV, push or pull fixels).
    ///
    /// THE SAME-DIRECTION H5 RULE: this tool resamples a sampled grid, so it
    /// follows the *image-warping* convention — pass the SAME-direction h5
    /// you would give `antsApplyTransforms` for an image. To take an ODX
    /// FROM space A TO space B, pass `from-A_to-B_xfm.h5`. (NOT the
    /// opposite-named convention used by `trxrs`/`giftirs` for points.)
    ///
    /// CARTOON BIDS EXAMPLES — given `sub-01`'s paired h5 files:
    ///
    /// • You have `sub-01_space-ACPC_dwimap.odx` and want ODX in
    ///   MNI152NLin2009cAsym → pass `sub-01_from-ACPC_to-MNI152NLin2009cAsym_xfm.h5`
    ///
    /// • You have `sub-01_space-MNI152NLin2009cAsym_dwimap.odx` and want ODX
    ///   in ACPC → pass `sub-01_from-MNI152NLin2009cAsym_to-ACPC_xfm.h5`
    ///
    /// HEADS-UP: for the SAME subject going to the SAME target, the h5 you
    /// pass to `odx transform` is the OTHER member of the BIDS pair than
    /// the one you'd pass to `trxrs`/`giftirs`:
    ///
    ///   sub-01_space-ACPC_tracts.trx → MNI: trxrs --transform sub-01_from-MNI..._to-ACPC_xfm.h5
    ///   sub-01_space-ACPC_dwimap.odx → MNI: odx   --transform sub-01_from-ACPC_to-MNI..._xfm.h5
    ///
    /// MODES:
    ///
    /// • `--mode mrtrix` (default): pull SH, DPV, AND fixels via the single
    ///   `--transform` h5. Matches `mrtransform` + `fixeltransform`
    ///   semantics. Fixels may be duplicated or dropped at non-uniform
    ///   warp regions (no fixel-correspondence guarantees).
    ///
    /// • `--mode ants`: pull SH and DPV via `--transform` (target→source);
    ///   PUSH fixels via `--transform-inverse` (source→target). Each source
    ///   fixel maps to exactly one target voxel, preserving cardinality.
    ///   Use with an ANTs-style paired h5 set.
    ///
    /// FULL INVOCATION (warp ACPC ODX into MNI, mrtrix mode):
    ///   odx transform sub-01_space-ACPC_dwimap.odx
    ///   sub-01_space-MNI152NLin2009cAsym_dwimap.odx
    ///   --transform sub-01_from-ACPC_to-MNI152NLin2009cAsym_xfm.h5
    ///
    /// FULL INVOCATION (ants mode with paired h5s):
    ///   odx transform sub-01_space-ACPC_dwimap.odx
    ///   sub-01_space-MNI152NLin2009cAsym_dwimap.odx --mode ants
    ///   --transform         sub-01_from-ACPC_to-MNI152NLin2009cAsym_xfm.h5
    ///   --transform-inverse sub-01_from-MNI152NLin2009cAsym_to-ACPC_xfm.h5
    Transform(TransformArgs),
    /// Attach a NIfTI volume to an ODX as a DPV (per-voxel scalar), in
    /// place. The NIfTI grid must match the ODX (dimensions + affine
    /// within 1e-3 mm); voxels outside the ODX mask are silently dropped.
    ///
    /// Examples:
    ///
    ///   odx attach-dpv subject.odx fa.nii.gz --name fa
    ///
    ///   odx attach-dpv subject.odx counts.nii.gz --name counts --dtype u16
    ///
    /// The on-disk DPV dtype defaults to `auto`: narrowest unsigned int
    /// that fits non-negative integer data, else float32. Force a dtype
    /// with `--dtype u8|u16|u32|i16|i32|f32|f64`.
    AttachDpv(AttachDpvArgs),
    /// Generate shell completions.
    Completions {
        #[arg(value_enum)]
        shell: clap_complete::Shell,
    },
}

#[derive(Args, Debug)]
struct AttachDpvArgs {
    /// Existing ODX (directory or `.odx` archive). Edited in place.
    odx: PathBuf,
    /// NIfTI volume to import (`.nii` or `.nii.gz`).
    nifti: PathBuf,
    /// DPV name to register under (e.g. `fa`). Overwrites if it already
    /// exists.
    #[arg(long = "name")]
    name: String,
    /// On-disk DPV datatype. `auto` (default) picks the narrowest
    /// unsigned int that fits non-negative integer data, else `float32`.
    #[arg(long = "dtype", default_value = "auto", value_parser = parse_dpv_dtype_arg)]
    dtype: odx_rs::DpvDtype,
    /// Suppress the per-attach summary line.
    #[arg(long = "quiet")]
    quiet: bool,
}

fn parse_dpv_dtype_arg(s: &str) -> std::result::Result<odx_rs::DpvDtype, String> {
    match s {
        "auto" => Ok(odx_rs::DpvDtype::Auto),
        "u8" | "uint8" => Ok(odx_rs::DpvDtype::UInt8),
        "u16" | "uint16" => Ok(odx_rs::DpvDtype::UInt16),
        "u32" | "uint32" => Ok(odx_rs::DpvDtype::UInt32),
        "i16" | "int16" => Ok(odx_rs::DpvDtype::Int16),
        "i32" | "int32" => Ok(odx_rs::DpvDtype::Int32),
        "f32" | "float32" => Ok(odx_rs::DpvDtype::Float32),
        "f64" | "float64" => Ok(odx_rs::DpvDtype::Float64),
        other => Err(format!(
            "unknown DPV dtype '{other}'; expected one of: \
             auto, u8/uint8, u16/uint16, u32/uint32, i16/int16, i32/int32, \
             f32/float32, f64/float64"
        )),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum TransformModeArg {
    /// Pull SH, DPV, AND fixels via a single forward h5 (chain target→source).
    /// Matches mrtrix3 `mrtransform` + `fixeltransform` conventions: simple,
    /// works with one transform file, fixels may be duplicated or dropped at
    /// non-uniform warp regions (no fixel-correspondence guarantees).
    Mrtrix,
    /// SH and DPV pulled via `--transform`; fixels *pushed* via
    /// `--transform-inverse` (chain source→target). Each source fixel maps to
    /// exactly one target voxel, so cardinality is preserved. Use this when
    /// you have an ANTs-style paired h5 (e.g. `from-ACPC_to-MNI.h5` and
    /// `from-MNI_to-ACPC.h5`).
    Ants,
}

#[derive(Args, Debug)]
struct TransformArgs {
    /// Source ODX (directory or `.odx` archive), e.g.
    /// `sub-01_space-ACPC_dwimap.odx`.
    input: PathBuf,
    /// Output ODX (directory or `.odx` archive), e.g.
    /// `sub-01_space-MNI152NLin2009cAsym_dwimap.odx`.
    output: PathBuf,
    /// ANTs/ITK transform: `Composite.h5` (warp + affines), Insight
    /// Transform File V1.0 (`.txt`, affine-only), or ITK MATLAB (`.mat`,
    /// affine-only). Same-direction file as image warping: to take ODX
    /// from space A onto space B's grid, pass `from-A_to-B_xfm.h5`.
    /// Used to pull SH and DPV onto the target grid in both modes; also
    /// pulls fixels in `--mode mrtrix`.
    #[arg(long = "transform")]
    transform: PathBuf,
    /// Inverse ANTs/ITK transform (the *paired* h5 in BIDS naming).
    /// Required for `--mode ants` — pushes each source fixel to its
    /// corresponding target voxel, preserving fixel cardinality.
    /// Forbidden in `--mode mrtrix`. To take ODX from space A onto
    /// space B with paired h5s, pass `from-B_to-A_xfm.h5` here while
    /// `--transform` gets `from-A_to-B_xfm.h5`.
    #[arg(long = "transform-inverse")]
    transform_inverse: Option<PathBuf>,
    /// Workflow convention. `mrtrix` (default): pull-everything via a single
    /// forward h5, mrtrix3-compatible. `ants`: SH pulled via forward h5,
    /// fixels pushed via `--transform-inverse`.
    #[arg(long = "mode", value_enum, default_value = "mrtrix")]
    mode: TransformModeArg,
    /// Reference NIfTI in target space. Required when the forward h5 has no
    /// displacement field (affine-only).
    #[arg(long = "reference")]
    reference: Option<PathBuf>,
    /// Swap the moving/fixed direction. Only valid with affine-only chains
    /// (warps cannot be numerically inverted in v1).
    #[arg(long)]
    invert: bool,
    /// Opt in to mrtrix-style SH modulation (per-direction
    /// `‖J·d‖/det(J)`, equivalent to `mrtransform -modulate fod`). Off by
    /// default. Fixels are never modulated.
    #[arg(long)]
    modulate: bool,
    /// Number of fibonacci-spiral reference directions for aPSF SH
    /// reorientation. 80 covers lmax 8 reliably; 300 for lmax 12.
    #[arg(long = "apsf-dirs", default_value_t = 80)]
    apsf_dirs: usize,
    #[arg(long = "odx-layout", value_enum, default_value = "directory")]
    odx_layout: OdxLayoutArg,
    #[arg(long)]
    overwrite: bool,
    #[arg(long)]
    json: bool,
}

#[derive(Args, Debug)]
struct UpsampleArgs {
    /// Source ODX (directory or `.odx` archive).
    input: PathBuf,
    /// Output ODX (directory or `.odx` archive).
    output: PathBuf,
    /// Target isotropic voxel spacing in mm.
    #[arg(long = "voxel-spacing")]
    voxel_spacing: f64,
    /// Maximum peaks per voxel.
    #[arg(long = "npeaks", default_value_t = 5)]
    npeaks: usize,
    /// Relative peak threshold (fraction of in-voxel maximum).
    #[arg(long = "peak-threshold", default_value_t = 0.5)]
    peak_threshold: f32,
    /// Minimum angular separation between accepted peaks (degrees).
    #[arg(long = "min-separation-angle", default_value_t = 25.0)]
    min_separation_angle: f32,
    #[arg(long = "odx-layout", value_enum, default_value = "directory")]
    odx_layout: OdxLayoutArg,
    #[arg(long)]
    overwrite: bool,
    #[arg(long)]
    json: bool,
}

#[derive(Args, Debug)]
struct ImportAodfArgs {
    /// Path to the aodf NIfTI (e.g. `..._model-csd_param-aodf_dwimap.nii.gz`).
    input: PathBuf,
    /// Output ODX directory or `.odx` archive.
    output: PathBuf,
    /// Optional sidecar JSON; if omitted we look beside the NIfTI.
    #[arg(long = "sidecar")]
    sidecar: Option<PathBuf>,
    /// Use legacy descoteaux SH (|m| in m<0). Defaults to non-legacy
    /// (matches modern dipy ≥ 1.7 and pyAFQ's `is_legacy=False` default).
    #[arg(long = "legacy-basis")]
    legacy_basis: bool,
    /// Relative peak threshold passed to `peak_directions`-style filtering.
    #[arg(long = "relative-peak-threshold", default_value_t = 0.5)]
    relative_peak_threshold: f32,
    /// Minimum angular separation between accepted peaks (degrees).
    #[arg(long = "min-separation-deg", default_value_t = 25.0)]
    min_separation_deg: f32,
    /// Cap on peaks per voxel.
    #[arg(long = "max-peaks", default_value_t = 5)]
    max_peaks: usize,
    /// Overwrite an existing output path.
    #[arg(long)]
    overwrite: bool,
    #[arg(long = "odx-layout", value_enum, default_value = "directory")]
    odx_layout: OdxLayoutArg,
    #[arg(long)]
    json: bool,
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
    #[arg(long = "quantize-dense")]
    quantize_dense: bool,
    #[arg(long = "quantize-min-len", default_value_t = 4096, hide = true)]
    quantize_min_len: usize,
    #[arg(long = "out-sh")]
    out_sh: Option<PathBuf>,
    #[arg(long = "fixel-container", value_enum, default_value = "nifti")]
    fixel_container: MrtrixFixelContainerArg,
    /// Force NIfTI-2 instead of NIfTI-1 when writing MRtrix SH to a `.nii`/`.nii.gz` output.
    #[arg(long = "nifti2")]
    nifti2: bool,
    #[arg(long = "sh-lmax")]
    sh_lmax: Option<u32>,
    #[arg(long = "dense-odf", value_enum, default_value = "from-sh")]
    dense_odf: DenseOdfModeArg,
    #[arg(long = "peak-source", value_enum, default_value = "fixels", hide = true)]
    peak_source: PeakSourceArg,
    #[arg(long = "amplitude-key", hide = true)]
    amplitude_key: Option<String>,
    #[arg(long = "z0", value_enum, default_value = "auto", hide = true)]
    z0: Z0PolicyArg,
    /// MRtrix-NIfTI only: preserve the input NIfTI's on-disk affine and
    /// (i,j,k) ordering instead of canonicalizing to RAS+. Use when the
    /// resulting ODX must compare with one produced via nibabel-style
    /// ingestion (e.g. cs-odf coeffs.odx).
    #[arg(long = "preserve-affine", hide = true)]
    preserve_affine: bool,
    /// SH-image input only: also compute fixels (peaks) from the SH coefficients
    /// and store them in the output ODX, so the archive carries the SH AND
    /// fixels over the full nonzero-FOD mask (not just the supra-threshold
    /// `fod2fixel` voxels). No-op if the dataset already has fixels.
    #[arg(long = "peaks-from-sh")]
    peaks_from_sh: bool,
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

#[derive(Args, Debug)]
struct CompareArgs {
    /// First ODX file (the geometry of the comparison ODX mirrors this one).
    #[arg(long)]
    a: PathBuf,
    /// Second ODX file.
    #[arg(long)]
    b: PathBuf,
    /// Output directory for per-voxel NIfTIs and the comparison ODX.
    #[arg(long = "out-dir")]
    out_dir: PathBuf,
    /// Optional explicit primary DPF metric (default: amplitude → afd → qa).
    #[arg(long = "primary-dpf")]
    primary_dpf: Option<String>,
    #[arg(long = "threshold", value_enum, default_value = "otsu")]
    threshold: QcThresholdArg,
    #[arg(long = "threshold-value")]
    threshold_value: Option<f32>,
    /// Coherence trajectory/match angle (degrees) passed to the QC pass.
    #[arg(long = "coherence-angle-deg", default_value_t = 15.0)]
    coherence_angle_deg: f32,
    /// Maximum angle (degrees) for fixel mutual matching across A and B.
    #[arg(long = "match-angle-deg", default_value_t = 30.0)]
    match_angle_deg: f32,
    /// Skip writing the comparison.odx archive (NIfTIs only).
    #[arg(long = "no-comparison-odx", default_value_t = false)]
    no_comparison_odx: bool,
    #[arg(long)]
    json: bool,
}

#[derive(Args, Debug)]
struct CombineArgs {
    /// Input ODX files (all must share grid + affine). Repeatable positionally
    /// or via `--input`.
    #[arg(value_name = "ODX")]
    inputs: Vec<PathBuf>,
    /// Additional input ODX file (repeatable alternative to positionals).
    #[arg(long = "input", action = clap::ArgAction::Append)]
    input: Vec<PathBuf>,
    /// How group fixels are built: `cluster` pools subject directions
    /// (amplitude-agnostic); `mean-fod` peak-finds the mean FOD.
    #[arg(long = "method", value_enum, default_value = "cluster")]
    method: TemplateMethodArg,
    /// Adopt this ODX's fixels/geometry as the template, skipping the build.
    #[arg(long = "template")]
    template: Option<PathBuf>,
    /// Template voxel set: voxels covered by ≥1 input (`union`) or all (`intersection`).
    #[arg(long = "mask-combine", value_enum, default_value = "union")]
    mask_combine: MaskCombineArg,
    /// Maximum angle (degrees) for matching a subject fixel to a group fixel.
    #[arg(long = "match-angle-deg", default_value_t = 30.0)]
    match_angle_deg: f32,
    /// `mean-fod` only: per-subject FOD normalization applied before averaging.
    /// `none` is correct for quantitative reconstructions (`consh
    /// --quantitative`, `mtnormalise`d FODs) whose amplitudes already share a
    /// unit; `l0`/`integral` are per-voxel and destroy apparent-fibre-density
    /// contrast, leaving a shape-only template.
    #[arg(long = "normalize-fod", value_enum, default_value = "none")]
    normalize_fod: NormalizeFodArg,
    /// Keep a voxel when at least this FRACTION of inputs cover it. `0` is a
    /// mask union, `1` an intersection; `0.5` is the recommended template
    /// setting. Partially-covered voxels are divided by their own contributor
    /// count, never by N, so the mask edge grows no spurious low-AFD rim.
    /// Generalizes (and overrides) --mask-combine.
    #[arg(long = "min-coverage", conflicts_with = "mask_combine")]
    min_coverage: Option<f32>,
    /// SH order policy when inputs disagree: `min` truncates every input to the
    /// smallest lmax present (default, and the uniform choice), `max` zero-pads,
    /// or give an explicit even integer.
    #[arg(long = "lmax", default_value = "min")]
    lmax: String,
    /// Header/grid/SH-basis reference (default: the first input). Inputs in a
    /// different basis are converted to this one before averaging.
    #[arg(long = "reference")]
    reference: Option<PathBuf>,
    /// Compute the FOD reproducibility block (coverage, l0 spread, angular
    /// correlation) under `--method cluster` or `--template` too. It is always
    /// on for `--method mean-fod`. Needs sh/coefficients on every input.
    #[arg(long = "fod-qc")]
    fod_qc: bool,
    /// Skip the FOD reproducibility block entirely.
    #[arg(long = "no-fod-qc", conflicts_with = "fod_qc")]
    no_fod_qc: bool,
    /// Leave-one-out angular correlation: each subject is scored against a
    /// template with its own contribution removed, so the score is not inflated
    /// by self-similarity. `auto` = on for 3..=20 inputs.
    #[arg(long = "loo", value_enum, default_value = "auto")]
    loo: LooArg,
    /// Lowest SH band included in the angular correlation coefficient. `2`
    /// excludes the isotropic term, which otherwise drives ACC to ~1 even in CSF.
    #[arg(long = "acc-lmin", default_value_t = 2)]
    acc_lmin: usize,
    /// Average this per-voxel scalar (DPV) onto the template (repeatable).
    /// Default: every scalar float DPV present on all inputs.
    #[arg(long = "average-dpv", action = clap::ArgAction::Append)]
    average_dpv: Vec<String>,
    /// Skip DPV averaging.
    #[arg(long = "no-average-dpv", conflicts_with = "average_dpv")]
    no_average_dpv: bool,
    /// Also emit `<name>_sd` beside each averaged DPV.
    #[arg(long = "dpv-sd")]
    dpv_sd: bool,
    /// Also write the JSON report to this path.
    #[arg(long = "out-report")]
    out_report: Option<PathBuf>,
    /// Exit nonzero if any subject is flagged as an outlier. Off by default —
    /// the rule warns loudly and never drops a scan on its own.
    #[arg(long = "fail-on-outlier")]
    fail_on_outlier: bool,
    /// `mean-fod` peak finding: max peaks per voxel.
    #[arg(long = "npeaks", default_value_t = 5)]
    npeaks: usize,
    /// `mean-fod` peak finding: relative peak threshold.
    #[arg(long = "peak-threshold", default_value_t = 0.5)]
    peak_threshold: f32,
    /// `mean-fod` peak finding: minimum peak separation (degrees).
    #[arg(long = "min-separation-angle", default_value_t = 25.0)]
    min_separation_angle: f32,
    /// `cluster` only: drop group fixels supported by fewer than N subjects.
    #[arg(long = "min-subjects", default_value_t = 2)]
    min_subjects: usize,
    /// Restrict carried scalars to these DPF names (default: all shared scalars).
    #[arg(long = "scalar", action = clap::ArgAction::Append)]
    scalar: Vec<String>,
    /// Design table (TSV/CSV) of categorical covariates, joined per input.
    #[arg(long = "design")]
    design: Option<PathBuf>,
    /// Column in the design table identifying each input (default: tries
    /// path/bids_name/source_file/key/id).
    #[arg(long = "design-key-column")]
    design_key_column: Option<String>,
    /// How each input's subject key (DPF column / cohort row) is derived.
    #[arg(long = "input-key", value_enum, default_value = "stem")]
    input_key: InputKeyArg,
    /// Method comparison: require a method-independent external scaffold
    /// (`--template`), erroring if `cluster`/`mean-fod` would build the grid
    /// from the contestants (circular).
    #[arg(long = "require-external-template")]
    require_external_template: bool,
    /// Design column that labels each input's processing method (enables
    /// `n_methods_detecting` and `--reference-method`).
    #[arg(long = "method-column")]
    method_column: Option<String>,
    /// Mark scans whose `--method-column` value equals this as the reference
    /// cohort (defines `scaffold_support`). E.g. `--method-column scheme
    /// --reference-method abcd`.
    #[arg(long = "reference-method")]
    reference_method: Option<String>,
    /// Text file of reference-cohort inputs (one key or path per line) marked
    /// `is_reference`; combined with `--reference-method` if both are given.
    #[arg(long = "reference-cohort")]
    reference_cohort: Option<PathBuf>,
    /// Extra match-angle thresholds (comma-separated degrees) for the
    /// `matched_at_<deg>` detection-sweep planes, e.g. `15,20,25,30,35,45`.
    #[arg(long = "match-angle-sweep")]
    match_angle_sweep: Option<String>,
    /// Group ODX output (multi-column per-subject DPF + summary DPF).
    #[arg(long = "out-odx")]
    out_odx: Option<PathBuf>,
    /// ModelArrayIO/ModelArray cohort CSV (doubles as the phenotype table).
    #[arg(long = "out-cohort")]
    out_cohort: Option<PathBuf>,
    /// Group mask NIfTI output.
    #[arg(long = "out-mask")]
    out_mask: Option<PathBuf>,
    /// Also emit one template-space ODX per subject into this directory.
    #[arg(long = "per-subject-odx")]
    per_subject_odx: Option<PathBuf>,
    /// Optional tidy long table (`.csv`/`.tsv`), one row per (group fixel × subject).
    #[arg(long = "out-table")]
    out_table: Option<PathBuf>,
    /// Optional directory for per-voxel summary NIfTIs.
    #[arg(long = "out-dir")]
    out_dir: Option<PathBuf>,
    #[arg(long)]
    json: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum TemplateMethodArg {
    Cluster,
    MeanFod,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum NormalizeFodArg {
    None,
    L0,
    Integral,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum MaskCombineArg {
    Union,
    Intersection,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum InputKeyArg {
    Stem,
    Path,
}

impl From<TemplateMethodArg> for TemplateMethod {
    fn from(v: TemplateMethodArg) -> Self {
        match v {
            TemplateMethodArg::Cluster => TemplateMethod::Cluster,
            TemplateMethodArg::MeanFod => TemplateMethod::MeanFod,
        }
    }
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum LooArg {
    Auto,
    On,
    Off,
}

impl From<LooArg> for LooMode {
    fn from(v: LooArg) -> Self {
        match v {
            LooArg::Auto => LooMode::Auto,
            LooArg::On => LooMode::On,
            LooArg::Off => LooMode::Off,
        }
    }
}

impl From<NormalizeFodArg> for NormalizeFod {
    fn from(v: NormalizeFodArg) -> Self {
        match v {
            NormalizeFodArg::None => NormalizeFod::None,
            NormalizeFodArg::L0 => NormalizeFod::L0,
            NormalizeFodArg::Integral => NormalizeFod::Integral,
        }
    }
}

impl From<MaskCombineArg> for MaskCombine {
    fn from(v: MaskCombineArg) -> Self {
        match v {
            MaskCombineArg::Union => MaskCombine::Union,
            MaskCombineArg::Intersection => MaskCombine::Intersection,
        }
    }
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
        Command::Compare(args) => run_compare(args),
        Command::Combine(args) => run_combine(args),
        Command::ImportAodf(args) => run_import_aodf(args),
        Command::Upsample(args) => run_upsample(args),
        Command::Transform(args) => run_transform(args),
        Command::AttachDpv(args) => run_attach_dpv(args),
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
        false,
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
        false,
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
        false,
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

fn run_compare(args: CompareArgs) -> odx_rs::Result<()> {
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

    let a = OdxDataset::open(&args.a)?;
    let b = OdxDataset::open(&args.b)?;
    let report = compare_odx(
        &a,
        &b,
        &args.out_dir,
        &CompareOptions {
            primary_metric: args.primary_dpf,
            threshold,
            coherence_angle_deg: args.coherence_angle_deg,
            match_angle_deg: args.match_angle_deg,
            write_comparison_odx: !args.no_comparison_odx,
        },
    )?;

    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print!("{}", render_compare_report(&report));
    }
    Ok(())
}

fn render_compare_report(report: &CompareReport) -> String {
    let mut out = String::new();
    out.push_str(&format!("primary_metric: {}\n", report.primary_metric));
    out.push_str(&format!(
        "coherence_angle_deg: {:.3}\n",
        report.coherence_angle_deg
    ));
    out.push_str(&format!("match_angle_deg: {:.3}\n", report.match_angle_deg));
    out.push_str(&format!(
        "voxels: a={}, b={}, intersection={}\n",
        report.n_voxels_a, report.n_voxels_b, report.n_voxels_intersection
    ));
    out.push_str(&format!(
        "fixels: a={}, b={}\n",
        report.n_fixels_a, report.n_fixels_b
    ));
    out.push_str(&format!(
        "matched: mutual={}, unmatched_a={}, unmatched_b={}\n",
        report.n_mutual_matches, report.n_unmatched_a, report.n_unmatched_b
    ));
    out.push_str(&format!(
        "mean_match_angle_deg: {}\n",
        render_optional_f64(report.mean_match_angle_deg)
    ));
    out.push_str(&format!(
        "coherence_index: a={}, b={}\n",
        render_optional_f64(report.coherence_index_a),
        render_optional_f64(report.coherence_index_b)
    ));
    out.push_str(&format!(
        "shared_dpf_keys: {}\n",
        if report.shared_dpf_keys.is_empty() {
            "none".to_string()
        } else {
            report.shared_dpf_keys.join(", ")
        }
    ));
    out.push_str(&format!("written: {} files\n", report.written_paths.len()));
    out
}

fn run_combine(args: CombineArgs) -> odx_rs::Result<()> {
    let mut paths: Vec<PathBuf> = args.inputs.clone();
    paths.extend(args.input.iter().cloned());
    if paths.is_empty() {
        return Err(OdxError::Argument(
            "combine requires at least one input ODX (positional or --input)".into(),
        ));
    }
    if let Some(t) = args.out_table.as_ref() {
        if t.extension().and_then(|e| e.to_str()) == Some("parquet") {
            return Err(OdxError::Argument(
                "parquet --out-table is not supported yet; use a .csv or .tsv path".into(),
            ));
        }
    }
    if args.require_external_template && args.template.is_none() {
        return Err(OdxError::Argument(
            "--require-external-template needs --template <reference.odx>: for method \
             comparison the scaffold must be method-independent; cluster/mean-fod would \
             build it from the contestants (circular)."
                .into(),
        ));
    }
    if args.out_cohort.is_some() && args.per_subject_odx.is_none() {
        return Err(OdxError::Argument(
            "--out-cohort requires --per-subject-odx <DIR>: cohort rows must point at \
             single-column per-subject ODX files; the group ODX stores per-scan scalars \
             multi-column (carried scalars under a subj_ prefix), which the ModelArrayIO \
             odx loader rejects."
                .into(),
        ));
    }
    if args.reference_method.is_some() && args.method_column.is_none() {
        return Err(OdxError::Argument(
            "--reference-method requires --method-column to identify each scan's method".into(),
        ));
    }
    let match_angle_sweep: Vec<f32> = match args.match_angle_sweep.as_deref() {
        Some(s) => s
            .split(',')
            .map(|x| x.trim())
            .filter(|x| !x.is_empty())
            .map(|x| {
                x.parse::<f32>()
                    .map_err(|e| OdxError::Argument(format!("bad --match-angle-sweep value '{x}': {e}")))
            })
            .collect::<odx_rs::Result<Vec<_>>>()?,
        None => Vec::new(),
    };
    let reference_keys: std::collections::BTreeSet<String> = match args.reference_cohort.as_ref() {
        Some(p) => std::fs::read_to_string(p)
            .map_err(|e| OdxError::Format(format!("read --reference-cohort '{}': {e}", p.display())))?
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty())
            .collect(),
        None => std::collections::BTreeSet::new(),
    };

    let design = match args.design.as_ref() {
        Some(p) => Some(parse_design_table(p, args.design_key_column.as_deref())?),
        None => None,
    };

    let mut inputs = Vec::with_capacity(paths.len());
    for p in &paths {
        let key = derive_input_key(p, args.input_key);
        let categorical = categorical_for(p, &key, design.as_ref())?;
        let method = args.method_column.as_ref().and_then(|col| {
            categorical.iter().find(|(k, _)| k == col).map(|(_, v)| v.clone())
        });
        let path_str = p.to_string_lossy().to_string();
        let is_reference = reference_keys.contains(&key)
            || reference_keys.contains(&path_str)
            || (args.reference_method.is_some()
                && method.as_deref() == args.reference_method.as_deref());
        inputs.push(CombineInput {
            path: p.clone(),
            key,
            categorical,
            is_reference,
            method,
        });
    }
    // Subject keys are the DPF column / cohort-row identity and the per-subject
    // ODX filename, so they must be unique (e.g. method encoded by directory
    // collides under the default --input-key stem; use --input-key path).
    {
        let mut seen: std::collections::HashMap<&str, &Path> = std::collections::HashMap::new();
        for inp in &inputs {
            if let Some(prev) = seen.insert(inp.key.as_str(), inp.path.as_path()) {
                return Err(OdxError::Argument(format!(
                    "duplicate input key '{}' ('{}' and '{}'); keys must be unique — try --input-key path",
                    inp.key,
                    prev.display(),
                    inp.path.display()
                )));
            }
        }
    }

    if let Some(mc) = args.min_coverage {
        if !mc.is_finite() || !(0.0..=1.0).contains(&mc) {
            return Err(OdxError::Argument(format!(
                "--min-coverage must be in [0, 1], got {mc}"
            )));
        }
    }
    if args.acc_lmin % 2 != 0 {
        return Err(OdxError::Argument(format!(
            "--acc-lmin must be even (the symmetric SH bases hold even orders only), got {}",
            args.acc_lmin
        )));
    }

    let opts = CombineOptions {
        method: args.method.into(),
        template_override: args.template.clone(),
        mask_combine: args.mask_combine.into(),
        match_angle_deg: args.match_angle_deg,
        peak_config: PeakFinderConfig {
            npeaks: args.npeaks,
            relative_peak_threshold: args.peak_threshold,
            min_separation_angle_deg: args.min_separation_angle,
        },
        normalize_fod: args.normalize_fod.into(),
        min_coverage: args.min_coverage,
        lmax: LmaxPolicy::parse(&args.lmax)?,
        reference: args.reference.clone(),
        fod_qc: args.fod_qc,
        no_fod_qc: args.no_fod_qc,
        loo: args.loo.into(),
        acc_lmin: args.acc_lmin,
        average_dpv: if args.no_average_dpv {
            Some(Vec::new())
        } else if args.average_dpv.is_empty() {
            None
        } else {
            Some(args.average_dpv.clone())
        },
        dpv_sd: args.dpv_sd,
        min_subjects_per_group_fixel: args.min_subjects,
        matched_scalars: if args.scalar.is_empty() {
            None
        } else {
            Some(args.scalar.clone())
        },
        match_angle_sweep,
    };

    let outputs = CombineOutputs {
        out_odx: args.out_odx.clone(),
        out_cohort: args.out_cohort.clone(),
        out_mask: args.out_mask.clone(),
        per_subject_odx_dir: args.per_subject_odx.clone(),
        out_table: args.out_table.clone(),
        out_dir: args.out_dir.clone(),
    };

    let report = combine_odx(&inputs, &opts, &outputs)?;
    if let Some(p) = args.out_report.as_ref() {
        std::fs::write(p, serde_json::to_string_pretty(&report)?)?;
    }
    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print!("{}", render_combine_report(&report));
    }
    if args.fail_on_outlier && !report.outliers.is_empty() {
        return Err(OdxError::Argument(format!(
            "{} input(s) flagged as outliers: {}",
            report.outliers.len(),
            report.outliers.join(", ")
        )));
    }
    Ok(())
}

fn render_combine_report(r: &CombineReport) -> String {
    let mut out = String::new();
    out.push_str(&format!("method: {}\n", r.method));
    out.push_str(&format!("inputs: {}\n", r.n_inputs));
    out.push_str(&format!("mask_combine: {}\n", r.mask_combine));
    out.push_str(&format!("match_angle_deg: {:.3}\n", r.match_angle_deg));
    if r.normalize_fod != "none" {
        out.push_str(&format!("normalize_fod: {}\n", r.normalize_fod));
    }
    out.push_str(&format!("dims: {:?}\n", r.dims));
    out.push_str(&format!(
        "template: {} voxels, {} fixels\n",
        r.n_template_voxels, r.n_template_fixels
    ));
    out.push_str(&format!(
        "mean_subjects_per_fixel: {:.3}\n",
        r.mean_subjects_per_fixel
    ));
    out.push_str(&format!(
        "mean_angle_deg: {}\n",
        render_optional_f64(r.mean_angle_deg)
    ));
    out.push_str(&format!(
        "reference_scans: {}, mean_unmatched_per_scan: {:.2}\n",
        r.n_reference_scans, r.mean_unmatched_per_scan
    ));
    out.push_str(&format!(
        "matched_scalars: {}\n",
        if r.matched_scalar_keys.is_empty() {
            "none".to_string()
        } else {
            r.matched_scalar_keys.join(", ")
        }
    ));
    out.push_str(&format!(
        "design_columns: {}\n",
        if r.design_columns.is_empty() {
            "none".to_string()
        } else {
            r.design_columns.join(", ")
        }
    ));
    if let (Some(order), Some(basis)) = (r.sh_order, r.sh_basis.as_ref()) {
        out.push_str(&format!(
            "aggregate: lmax {order} {basis} (--lmax {}), min_coverage {:.2}\n",
            r.lmax_policy, r.min_coverage
        ));
    }
    if r.loo != "unavailable" {
        out.push_str(&format!(
            "acc (l>={}) mean: {}   leave-one-out [{}]: {}\n",
            r.acc_lmin,
            render_optional_f64(r.mean_acc),
            r.loo,
            render_optional_f64(r.mean_acc_loo)
        ));
    }
    if !r.averaged_dpv.is_empty() {
        out.push_str(&format!("averaged_dpv: {}\n", r.averaged_dpv.join(", ")));
    }
    if !r.subjects.is_empty() && r.sh_order.is_some() {
        let width = r.subjects.iter().map(|s| s.key.len()).max().unwrap_or(7).max(7);
        out.push_str(&format!(
            "\n{:<width$}  {:>6}  {:>6}  {:>7}  {:>7}  {}\n",
            "subject", "cov", "fixels", "acc", "acc_loo", "flags"
        ));
        for s in &r.subjects {
            out.push_str(&format!(
                "{:<width$}  {:>5.1}%  {:>6}  {:>7}  {:>7}  {}\n",
                s.key,
                s.coverage_frac * 100.0,
                s.n_fixels,
                render_f32(s.mean_acc),
                render_f32(s.mean_acc_loo),
                if s.is_outlier { "OUTLIER" } else { "" }
            ));
        }
    }
    for s in r.subjects.iter().filter(|s| s.is_outlier) {
        out.push_str(&format!("outlier {}: {}\n", s.key, s.outlier_reasons.join("; ")));
    }
    out.push_str(&format!("written: {} files\n", r.written_paths.len()));
    out
}

fn render_f32(v: f32) -> String {
    if v.is_finite() {
        format!("{v:.4}")
    } else {
        "n/a".to_string()
    }
}

/// A parsed design/participants table (TSV/CSV): header columns, the key column
/// used to match inputs, and the data rows.
struct DesignTable {
    columns: Vec<String>,
    key_idx: usize,
    rows: Vec<Vec<String>>,
}

fn parse_design_table(path: &Path, key_col: Option<&str>) -> odx_rs::Result<DesignTable> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| OdxError::Format(format!("read design table '{}': {e}", path.display())))?;
    let sep = if path.extension().and_then(|e| e.to_str()) == Some("tsv") {
        '\t'
    } else {
        ','
    };
    let mut lines = text.lines();
    let header = lines
        .next()
        .ok_or_else(|| OdxError::Argument("design table is empty".into()))?;
    let columns = split_delim(header, sep);
    let key_idx = match key_col {
        Some(name) => columns
            .iter()
            .position(|c| c == name)
            .ok_or_else(|| OdxError::Argument(format!("design key column '{name}' not found")))?,
        None => ["path", "bids_name", "source_file", "key", "id"]
            .iter()
            .find_map(|cand| columns.iter().position(|c| c == cand))
            .ok_or_else(|| {
                OdxError::Argument(
                    "design table has no recognizable key column; pass --design-key-column".into(),
                )
            })?,
    };
    let rows = lines
        .filter(|l| !l.trim().is_empty())
        .map(|l| split_delim(l, sep))
        .collect();
    Ok(DesignTable {
        columns,
        key_idx,
        rows,
    })
}

/// Categorical covariates for one input: joined design row (all non-key
/// columns) if a design table is present, else BIDS entities from the filename.
fn categorical_for(
    path: &Path,
    key: &str,
    design: Option<&DesignTable>,
) -> odx_rs::Result<Vec<(String, String)>> {
    match design {
        Some(table) => {
            let row = design_row_for(table, path, key).ok_or_else(|| {
                OdxError::Argument(format!(
                    "no design row matches input '{}' (key '{}')",
                    path.display(),
                    key
                ))
            })?;
            Ok(table
                .columns
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != table.key_idx)
                .map(|(i, c)| (c.clone(), row.get(i).cloned().unwrap_or_default()))
                .collect())
        }
        None => Ok(bids_entities(path)),
    }
}

fn design_row_for<'a>(
    table: &'a DesignTable,
    path: &Path,
    key: &str,
) -> Option<&'a Vec<String>> {
    let full = path.to_string_lossy();
    let fname = path
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();
    table.rows.iter().find(|row| {
        let cell = row.get(table.key_idx).map(|s| s.as_str()).unwrap_or("");
        cell == full
            || cell == fname
            || cell == key
            || Path::new(cell)
                .file_stem()
                .map(|s| s.to_string_lossy() == key)
                .unwrap_or(false)
    })
}

fn derive_input_key(path: &Path, mode: InputKeyArg) -> String {
    match mode {
        InputKeyArg::Stem => path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| path.to_string_lossy().to_string()),
        InputKeyArg::Path => path.to_string_lossy().to_string(),
    }
}

/// Parse `sub-01_ses-1_acq-abcd_...` style BIDS entities from a filename stem.
fn bids_entities(path: &Path) -> Vec<(String, String)> {
    let stem = path
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();
    stem.split('_')
        .filter_map(|tok| {
            tok.find('-')
                .map(|idx| (tok[..idx].to_string(), tok[idx + 1..].to_string()))
        })
        .collect()
}

/// Split one delimited line, honoring RFC4180-style double-quoted fields.
fn split_delim(line: &str, sep: char) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        if in_quotes {
            if c == '"' {
                if chars.peek() == Some(&'"') {
                    cur.push('"');
                    chars.next();
                } else {
                    in_quotes = false;
                }
            } else {
                cur.push(c);
            }
        } else if c == '"' {
            in_quotes = true;
        } else if c == sep {
            out.push(std::mem::take(&mut cur));
        } else {
            cur.push(c);
        }
    }
    out.push(cur);
    out
}

fn run_convert(args: ConvertArgs) -> odx_rs::Result<()> {
    let output_format = resolve_output_format(&args.output, args.output_format)?;

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
        args.preserve_affine,
    )?;

    // --peaks-from-sh: derive fixels from the SH coefficients (broad FOD mask)
    // so an SH-image conversion produces a self-contained, compare-able archive.
    // No-op if the dataset already carries fixels.
    let odx = if args.peaks_from_sh && odx.nb_peaks() == 0 {
        odx_rs::mrtrix::dataset_with_peaks_from_sh(&odx, odx_rs::PeakFinderConfig::default())?
    } else {
        odx
    };

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
                    container: args.fixel_container.into(),
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
                        container: infer_sh_container(out_sh, args.nifti2),
                        gzip: infer_sh_gzip(out_sh),
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
                    container: infer_sh_container(&args.output, args.nifti2),
                    gzip: infer_sh_gzip(&args.output),
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

fn run_attach_dpv(args: AttachDpvArgs) -> odx_rs::Result<()> {
    use ndarray::Array3;
    use nifti::{IntoNdArray, NiftiObject, NiftiVolume, ReaderOptions};

    if !args.odx.exists() {
        return Err(OdxError::Format(format!(
            "ODX '{}' does not exist",
            args.odx.display()
        )));
    }
    if !args.nifti.exists() {
        return Err(OdxError::Format(format!(
            "NIfTI '{}' does not exist",
            args.nifti.display()
        )));
    }

    let obj = ReaderOptions::new()
        .read_file(&args.nifti)
        .map_err(|e| OdxError::Format(format!("read NIfTI '{}': {e}", args.nifti.display())))?;
    let header = obj.header().clone();
    let volume = obj.into_volume();

    // Read the affine that downstream tools resolve to (sform if active,
    // else qform, else identity); we explicitly want the same priority
    // here as nibabel / FSLeyes use.
    let affine_mat = header.affine::<f64>();
    let mut vol_affine = [[0.0f64; 4]; 4];
    for r in 0..4 {
        for c in 0..4 {
            vol_affine[r][c] = affine_mat[(r, c)];
        }
    }

    let dim = volume.dim();
    if dim.len() < 3 {
        return Err(OdxError::Format(format!(
            "NIfTI '{}' has only {} dimensions; need at least 3",
            args.nifti.display(),
            dim.len()
        )));
    }
    let nx = dim[0] as usize;
    let ny = dim[1] as usize;
    let nz = dim[2] as usize;

    // Take the first 3-D volume (channel/time index 0 if present).
    // Casts everything to f64 internally; nifti-rs handles scl_slope/inter.
    let raw: Vec<f64> = volume
        .into_ndarray::<f64>()
        .map_err(|e| {
            OdxError::Format(format!("decode NIfTI '{}' as f64: {e}", args.nifti.display()))
        })?
        .into_raw_vec_and_offset()
        .0;
    let expected_len = nx * ny * nz;
    if raw.len() < expected_len {
        return Err(OdxError::Format(format!(
            "NIfTI volume '{}' has {} elements but the 3-D grid needs {}",
            args.nifti.display(),
            raw.len(),
            expected_len
        )));
    }
    // Slice off any trailing 4th+ dimensions (use volume[..., 0]). nifti-rs
    // stores in Fortran order; into_ndarray returns it in F-order layout
    // but with C-shape — slice index math respects the shape.
    // Per nifti-rs docs the returned array uses the NIfTI's natural shape
    // and ordering, so directly reshaping the first nx*ny*nz elements
    // works for the 3-D case.
    let mut vol3 = Array3::<f64>::zeros((nx, ny, nz));
    // Fortran storage order (column-major): index linearises as
    //   flat = i + j*nx + k*nx*ny
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let flat = i + j * nx + k * nx * ny;
                vol3[[i, j, k]] = raw[flat];
            }
        }
    }

    let report = odx_rs::attach_dpv_from_volume(
        &args.odx,
        &args.name,
        vol3.view(),
        vol_affine,
        args.dtype,
    )?;

    if !args.quiet {
        println!(
            "attached dpv/{} ({}; {} voxels, {} nonzero{})",
            report.name,
            report.dtype.name(),
            report.nb_voxels,
            report.masked_in_count,
            if report.clamped {
                ", values clamped to dtype range"
            } else {
                ""
            }
        );
    }
    Ok(())
}

fn run_import_aodf(args: ImportAodfArgs) -> odx_rs::Result<()> {
    use odx_rs::formats::pyafq_aodf::{load_pyafq_aodf_with, ImportOptions};

    if args.output.exists() && !args.overwrite {
        return Err(OdxError::Format(format!(
            "output '{}' already exists (pass --overwrite to replace)",
            args.output.display()
        )));
    }

    let options = ImportOptions {
        sidecar_path: args.sidecar,
        legacy_basis: Some(args.legacy_basis),
        relative_peak_threshold: args.relative_peak_threshold,
        min_separation_deg: args.min_separation_deg,
        max_peaks_per_voxel: args.max_peaks,
    };
    let dataset = load_pyafq_aodf_with(&args.input, options)?;

    let policy = OdxWritePolicy {
        quantize_dense: false,
        quantize_min_len: 4096,
    };
    match args.odx_layout {
        OdxLayoutArg::Directory => dataset.save_directory_with_policy(&args.output, policy)?,
        OdxLayoutArg::Archive => dataset.save_archive_with_policy(&args.output, policy)?,
    }

    if args.json {
        let summary = serde_json::json!({
            "output": args.output.display().to_string(),
            "nb_voxels": dataset.header().nb_voxels,
            "nb_peaks": dataset.header().nb_peaks,
            "sh_basis": dataset.header().sh_basis,
            "sh_order": dataset.header().sh_order,
            "sh_full_basis": dataset.header().sh_full_basis,
            "sh_legacy": dataset.header().sh_legacy,
        });
        println!("{}", serde_json::to_string_pretty(&summary)?);
    } else {
        println!(
            "wrote {} ({} voxels, {} peaks; SH basis=descoteaux07 order={} full_basis=true)",
            args.output.display(),
            dataset.header().nb_voxels,
            dataset.header().nb_peaks,
            dataset.header().sh_order.unwrap_or(0)
        );
    }
    Ok(())
}

fn run_transform(args: TransformArgs) -> odx_rs::Result<()> {
    use odx_rs::transform::{apply_transform_h5, TransformMode, TransformOptions};

    ensure_output_path(&args.output, args.overwrite)?;
    let input = OdxDataset::load(&args.input)?;

    let mode = match args.mode {
        TransformModeArg::Mrtrix => TransformMode::Mrtrix,
        TransformModeArg::Ants => TransformMode::Ants,
    };

    let opts = TransformOptions {
        modulate_sh: args.modulate,
        // Fixels are never modulated via the CLI; matches mrtrix3
        // `fixeltransform` (no-modulation) and ANTs-mode push semantics
        // (cardinality already preserved).
        modulate_fixel: false,
        apsf_dirs: args.apsf_dirs,
        ..Default::default()
    };

    let out = apply_transform_h5(
        &input,
        mode,
        &args.transform,
        args.transform_inverse.as_deref(),
        args.reference.as_deref(),
        args.invert,
        &opts,
    )?;

    let policy = OdxWritePolicy::default();
    match args.odx_layout {
        OdxLayoutArg::Directory => out.save_directory_with_policy(&args.output, policy)?,
        OdxLayoutArg::Archive => out.save_archive_with_policy(&args.output, policy)?,
    }

    if args.json {
        let summary = serde_json::json!({
            "output": args.output.display().to_string(),
            "mode": match args.mode {
                TransformModeArg::Mrtrix => "mrtrix",
                TransformModeArg::Ants => "ants",
            },
            "input_nb_voxels": input.header().nb_voxels,
            "input_nb_peaks": input.header().nb_peaks,
            "output_nb_voxels": out.header().nb_voxels,
            "output_nb_peaks": out.header().nb_peaks,
            "modulate_sh": opts.modulate_sh,
            "invert": args.invert,
            "apsf_dirs": opts.apsf_dirs,
        });
        println!("{}", serde_json::to_string_pretty(&summary)?);
    } else {
        println!(
            "wrote {} ({} voxels, {} peaks)",
            args.output.display(),
            out.header().nb_voxels,
            out.header().nb_peaks,
        );
    }
    Ok(())
}

fn run_upsample(args: UpsampleArgs) -> odx_rs::Result<()> {
    use odx_rs::{upsample, UpsampleOptions};
    use odx_rs::PeakFinderConfig;

    ensure_output_path(&args.output, args.overwrite)?;
    let input = OdxDataset::load(&args.input)?;

    let opts = UpsampleOptions {
        peak_config: PeakFinderConfig {
            npeaks: args.npeaks,
            relative_peak_threshold: args.peak_threshold,
            min_separation_angle_deg: args.min_separation_angle,
        },
    };

    let out = upsample(&input, args.voxel_spacing, &opts)?;

    let policy = OdxWritePolicy::default();
    match args.odx_layout {
        OdxLayoutArg::Directory => out.save_directory_with_policy(&args.output, policy)?,
        OdxLayoutArg::Archive => out.save_archive_with_policy(&args.output, policy)?,
    }

    if args.json {
        let summary = serde_json::json!({
            "output": args.output.display().to_string(),
            "voxel_spacing_mm": args.voxel_spacing,
            "input_dims": input.header().dimensions,
            "input_nb_voxels": input.header().nb_voxels,
            "input_nb_peaks": input.header().nb_peaks,
            "output_dims": out.header().dimensions,
            "output_nb_voxels": out.header().nb_voxels,
            "output_nb_peaks": out.header().nb_peaks,
        });
        println!("{}", serde_json::to_string_pretty(&summary)?);
    } else {
        println!(
            "wrote {} ({}→{} voxels, {} peaks; dims {:?}→{:?})",
            args.output.display(),
            input.header().nb_voxels,
            out.header().nb_voxels,
            out.header().nb_peaks,
            input.header().dimensions,
            out.header().dimensions,
        );
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
    preserve_nifti_affine: bool,
) -> odx_rs::Result<(OdxDataset, DetectedFormat)> {
    let opts = LoadDatasetOptions {
        sh_path: sh,
        fixel_dir,
        mapmri_tensor_path: mapmri_tensor,
        mapmri_uvec_path: mapmri_uvec,
        reference_affine,
        preserve_nifti_affine,
    };
    if let Some(format) = input_override {
        let detected: DetectedFormat = format.into();
        let dataset = load_dataset_with_format(input, detected, opts)?;
        return Ok((dataset, detected));
    }
    load_dataset(input, opts)
}

fn resolve_output_format(
    output: &Path,
    output_format: Option<OutputFormatOverride>,
) -> odx_rs::Result<DetectedFormat> {
    if let Some(format) = output_format {
        return Ok(format.into());
    }
    detect_target_format(output)
}

fn infer_sh_container(path: &Path, nifti2: bool) -> MrtrixShContainer {
    let s = path.to_string_lossy().to_lowercase();
    if s.ends_with(".nii") || s.ends_with(".nii.gz") {
        if nifti2 {
            MrtrixShContainer::Nifti2
        } else {
            MrtrixShContainer::Nifti1
        }
    } else {
        MrtrixShContainer::Mif
    }
}

fn infer_sh_gzip(path: &Path) -> bool {
    path.to_string_lossy()
        .to_lowercase()
        .ends_with(".gz")
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
