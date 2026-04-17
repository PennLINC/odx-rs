use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

use odx_rs::{dsistudio, mif, mrtrix, DType, OdxBuilder, OdxError, Result};

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let sh_path = PathBuf::from(args.next().expect(
        "usage: convert_mrtrix_to_fz <sh.mif.gz> <fixels_dir> <sphere_source.fib.gz> <output.fz>",
    ));
    let fixels_dir = PathBuf::from(args.next().expect(
        "usage: convert_mrtrix_to_fz <sh.mif.gz> <fixels_dir> <sphere_source.fib.gz> <output.fz>",
    ));
    let sphere_source = PathBuf::from(args.next().expect(
        "usage: convert_mrtrix_to_fz <sh.mif.gz> <fixels_dir> <sphere_source.fib.gz> <output.fz>",
    ));
    let output = PathBuf::from(args.next().expect(
        "usage: convert_mrtrix_to_fz <sh.mif.gz> <fixels_dir> <sphere_source.fib.gz> <output.fz>",
    ));

    let sh_affine = mif::read_mif(&sh_path)?.affine_4x4();
    let mrtrix_dataset = mrtrix::load_mrtrix_dataset(Some(&sh_path), Some(&fixels_dir))?;
    let sphere_dataset = dsistudio::load_fibgz(&sphere_source, Some(sh_affine))?;

    let sphere_vertices = sphere_dataset
        .sphere_vertices()
        .ok_or_else(|| {
            OdxError::Format("sphere source fib.gz does not contain odf_vertices".into())
        })?
        .to_vec();
    let sphere_faces = sphere_dataset
        .sphere_faces()
        .ok_or_else(|| OdxError::Format("sphere source fib.gz does not contain odf_faces".into()))?
        .to_vec();

    let (odf_amplitudes, n_dirs) =
        sample_sh_on_sphere(&sh_path, mrtrix_dataset.mask(), &sphere_vertices)?;

    let amplitudes = mrtrix_dataset
        .scalar_dpf_f32("afd")
        .or_else(|_| mrtrix_dataset.scalar_dpf_f32("amplitude"))
        .unwrap_or_else(|_| vec![1.0; mrtrix_dataset.nb_peaks()]);
    if amplitudes.len() != mrtrix_dataset.nb_peaks() {
        return Err(OdxError::Format(format!(
            "peak amplitude length {} does not match peak count {}",
            amplitudes.len(),
            mrtrix_dataset.nb_peaks()
        )));
    }

    let mut builder = OdxBuilder::new(
        mrtrix_dataset.header().voxel_to_rasmm,
        mrtrix_dataset.header().dimensions,
        mrtrix_dataset.mask().to_vec(),
    );
    for peaks in mrtrix_dataset.voxel_peaks() {
        builder.push_voxel_peaks(peaks);
    }
    builder.set_sphere(sphere_vertices, sphere_faces);
    builder.set_odf_data(
        "amplitudes",
        bytemuck::cast_slice(&odf_amplitudes).to_vec(),
        n_dirs,
        DType::Float32,
    );
    builder.set_dpf_data(
        "amplitude",
        bytemuck::cast_slice(&amplitudes).to_vec(),
        1,
        DType::Float32,
    );

    let dataset = builder.finalize()?;
    dsistudio::save_fz(&dataset, &output)?;
    println!("{}", output.display());
    Ok(())
}

fn sample_sh_on_sphere(
    sh_path: &Path,
    mask: &[u8],
    sphere_vertices: &[[f32; 3]],
) -> Result<(Vec<f32>, usize)> {
    let tempdir = tempfile::tempdir()?;
    let directions_path = tempdir.path().join("sphere_dirs.txt");
    let amplitudes_path = tempdir.path().join("amplitudes.mif");

    {
        let mut file = std::fs::File::create(&directions_path)?;
        for v in sphere_vertices {
            writeln!(file, "{} {} {}", v[0], v[1], v[2])?;
        }
    }

    let status = Command::new("sh2amp")
        .arg(sh_path)
        .arg(&directions_path)
        .arg(&amplitudes_path)
        .arg("-nonnegative")
        .arg("-datatype")
        .arg("float32")
        .arg("-force")
        .arg("-quiet")
        .status()?;
    if !status.success() {
        return Err(OdxError::Format(format!(
            "sh2amp failed for '{}'",
            sh_path.display()
        )));
    }

    let sampled = mif::read_mif(&amplitudes_path)?;
    let dims = &sampled.header.dimensions;
    if dims.len() < 4 {
        return Err(OdxError::Format(format!(
            "sampled amplitudes must be 4D, found {:?}",
            dims
        )));
    }
    let n_dirs = dims[3];
    if n_dirs != sphere_vertices.len() {
        return Err(OdxError::Format(format!(
            "sampled amplitudes have {n_dirs} directions, expected {}",
            sphere_vertices.len()
        )));
    }

    let full = sampled.logical_f32_vec()?;
    let nvoxels_total = dims[0] * dims[1] * dims[2];
    if full.len() != nvoxels_total * n_dirs {
        return Err(OdxError::Format(format!(
            "sampled amplitude size {} does not match {} voxels x {} directions",
            full.len(),
            nvoxels_total,
            n_dirs
        )));
    }
    if mask.len() != nvoxels_total {
        return Err(OdxError::Format(format!(
            "mask length {} does not match sampled voxel count {}",
            mask.len(),
            nvoxels_total
        )));
    }

    let mut masked = Vec::with_capacity(mask.iter().filter(|&&m| m != 0).count() * n_dirs);
    for (voxel, &m) in mask.iter().enumerate() {
        if m == 0 {
            continue;
        }
        let start = voxel * n_dirs;
        masked.extend_from_slice(&full[start..start + n_dirs]);
    }
    Ok((masked, n_dirs))
}
