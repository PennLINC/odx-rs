//! Round-trip identity check: applying the forward chain followed by the
//! reverse chain to the same point should return to (approximately) the
//! original. Operates on the user-provided real ANTs h5 transforms in
//! `transform_test_data/`.

use std::path::Path;

use itk_transforms_rs::read_itk_h5;

fn main() {
    let dir = Path::new("/Users/mcieslak/projects/odx/transform_test_data");
    let fwd = read_itk_h5(&dir.join(
        "sub-0874667_ses-V1_from-ACPC_to-MNI152NLin2009cAsym_mode-image_xfm.h5",
    ))
    .expect("forward h5");
    let rev = read_itk_h5(&dir.join(
        "sub-0874667_ses-V1_from-MNI152NLin2009cAsym_to-ACPC_mode-image_xfm.h5",
    ))
    .expect("reverse h5");

    // For ANTs convention, both chains map fixed→moving in their stored
    // orientation. So for a point in MNI:
    //    chain_fwd(p_mni)        = p_acpc           (forward h5: fixed=MNI, moving=ACPC)
    //    chain_rev(chain_fwd(p)) = (rev: fixed=ACPC) applied to p_acpc → p_mni
    // i.e., chain_rev(chain_fwd(p)) ≈ p for any p in MNI.
    //
    // (The forward h5's chain takes MNI coords to ACPC coords because for
    // resampling ACPC→MNI you walk the MNI grid pulling from ACPC. Same
    // semantics as the rotatez.txt test we did earlier.)

    let test_points = [
        [0.0, 0.0, 0.0],          // MNI origin
        [-30.0, 20.0, 10.0],      // off-origin
        [50.0, -30.0, 20.0],
        [10.0, -50.0, -25.0],
        [-2.0, -45.0, 60.0],      // CC region in MNI
    ];

    println!("Round-trip: chain_rev(chain_fwd(p)) vs p (mm):");
    println!("{:>8} {:>8} {:>8}   →   {:>9} {:>9} {:>9}   delta",
             "px", "py", "pz", "qx", "qy", "qz");

    let mut max_err = 0.0_f64;
    for p in test_points {
        let q = fwd.map_point(p);    // p_mni → p_acpc
        let p2 = rev.map_point(q);   // p_acpc → p_mni
        let err = ((p[0]-p2[0]).powi(2) + (p[1]-p2[1]).powi(2) + (p[2]-p2[2]).powi(2)).sqrt();
        if err > max_err { max_err = err; }
        println!(
            "{:>8.3} {:>8.3} {:>8.3}   →   {:>9.4} {:>9.4} {:>9.4}   {:.4}mm",
            p[0], p[1], p[2], p2[0], p2[1], p2[2], err,
        );
    }
    println!("\nmax round-trip error: {max_err:.4} mm");
    if max_err < 0.5 {
        println!("✓ within typical ANTs warp inversion tolerance (~0.5 mm)");
    } else if max_err < 2.0 {
        println!("? larger than ideal but may be acceptable depending on registration quality");
    } else {
        println!("✗ exceeds 2 mm — suggests a direction-convention bug");
    }
}
