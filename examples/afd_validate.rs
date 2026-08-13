//! Dump per-voxel AFD from odx's FMLS so it can be compared against
//! `fod2fixel -afd`.
//!
//! Usage:
//!   afd_validate <input.odx> <out.csv> [max_voxels]
//!
//! Emits one row per (voxel, lobe): `i,j,k,lobe,afd,peak,dx,dy,dz`, with
//! lobes ordered by descending AFD within each voxel. Voxel indices are the
//! ODX's `compact_to_ijk`, so rows can be joined against a `fod2fixel`
//! `index.mif` on `(i,j,k)`.

use odx_rs::fmls::{hemisphere_adjacency, Fmls, FmlsConfig, IntegrationWeights};
use odx_rs::mrtrix_sh::RowSamplePlan;
use odx_rs::odx_file::OdxDataset;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: afd_validate <input.odx> <out.csv> [max_voxels]");
        std::process::exit(2);
    }
    let max_voxels: usize = args
        .get(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(usize::MAX);

    let p = std::path::Path::new(&args[1]);
    let ds = if p.is_dir() {
        OdxDataset::open_directory(p)?
    } else {
        OdxDataset::open(p)?
    };
    let sh_name = ds
        .sh_names()
        .first()
        .map(|s| s.to_string())
        .ok_or("input ODX has no SH array")?;
    let view = ds.sh::<f32>(&sh_name)?;
    let ncoeffs = view.ncols();
    let nvox = view.nrows();
    let ijk = ds.compact_to_ijk();

    // Sphere density. Level 3 = 321 hemisphere directions (mrtrix's
    // tesselation_321); level 4 = 1281, which is what `fod2fixel` uses by
    // default. Overridable so the effect of density can be measured directly.
    let level: usize = std::env::var("AFD_ICO_LEVEL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let sphere = odx_rs::icosphere::icosphere(level);
    let verts = sphere.hemisphere();
    eprintln!(
        "sphere: icosphere level {level} — {} hemisphere directions, {} faces",
        verts.len(),
        sphere.faces.len()
    );
    let plan = RowSamplePlan::for_sh_rows_nonnegative(verts, ncoeffs)?;
    let adj = hemisphere_adjacency(&sphere.vertices, &sphere.faces);
    // Uniform weights: an icosphere is near-equal-area (nearest-neighbour
    // spacing varies < 35%), and any constant-factor error would show up as a
    // uniform ratio against fod2fixel rather than as scatter.
    let weights = IntegrationWeights::uniform(verts.len());
    let cfg = FmlsConfig::default();
    let mut fmls = Fmls::new(verts, &adj, &weights, cfg);

    let mut out = String::from("i,j,k,lobe,afd,peak,dx,dy,dz\n");
    let mut amps = vec![0.0f32; plan.ndir()];
    let n = nvox.min(max_voxels);
    for v in 0..n {
        plan.apply_row_into(view.row(v), &mut amps);
        let lobes = fmls.segment(&amps);
        let (i, j, k) = (ijk[v][0], ijk[v][1], ijk[v][2]);
        for (li, l) in lobes.iter().enumerate() {
            out.push_str(&format!(
                "{i},{j},{k},{li},{:.6},{:.6},{:.4},{:.4},{:.4}\n",
                l.integral, l.peak_value, l.mean_dir[0], l.mean_dir[1], l.mean_dir[2]
            ));
        }
    }
    std::fs::write(&args[2], out)?;
    eprintln!("wrote {} ({} voxels)", args[2], n);
    Ok(())
}
