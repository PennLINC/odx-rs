use std::io::Read;
use std::path::Path;

use crate::error::{OdxError, Result};
use crate::formats::mif;

pub fn read_reference_affine(path: &Path) -> Result<[[f64; 4]; 4]> {
    let name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or_default();
    if name.ends_with(".mif") || name.ends_with(".mif.gz") {
        return Ok(mif::read_mif(path)?.affine_4x4());
    }
    if name.ends_with(".nii") || name.ends_with(".nii.gz") {
        return read_nifti_affine(path);
    }
    Err(OdxError::Argument(format!(
        "unsupported reference affine path '{}'; expected .mif/.mif.gz/.nii/.nii.gz",
        path.display()
    )))
}

fn read_nifti_affine(path: &Path) -> Result<[[f64; 4]; 4]> {
    let bytes = read_image_bytes(path)?;
    if bytes.len() < 540 {
        return Err(OdxError::Format(format!(
            "NIfTI file '{}' is too small to contain a valid header",
            path.display()
        )));
    }
    match i32::from_le_bytes(bytes[0..4].try_into().unwrap()) {
        348 => parse_nifti1_affine(&bytes),
        540 => parse_nifti2_affine(&bytes),
        other => Err(OdxError::Format(format!(
            "unsupported NIfTI header size {other} in '{}'",
            path.display()
        ))),
    }
}

fn read_image_bytes(path: &Path) -> Result<Vec<u8>> {
    if path
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext == "gz")
    {
        let file = std::fs::File::open(path)?;
        let mut decoder = flate2::read::MultiGzDecoder::new(file);
        let mut bytes = Vec::new();
        decoder.read_to_end(&mut bytes)?;
        Ok(bytes)
    } else {
        Ok(std::fs::read(path)?)
    }
}

fn parse_nifti1_affine(bytes: &[u8]) -> Result<[[f64; 4]; 4]> {
    let sform_code = read_i16_le(bytes, 254)?;
    if sform_code > 0 {
        return Ok([
            [
                read_f32_le(bytes, 280)? as f64,
                read_f32_le(bytes, 284)? as f64,
                read_f32_le(bytes, 288)? as f64,
                read_f32_le(bytes, 292)? as f64,
            ],
            [
                read_f32_le(bytes, 296)? as f64,
                read_f32_le(bytes, 300)? as f64,
                read_f32_le(bytes, 304)? as f64,
                read_f32_le(bytes, 308)? as f64,
            ],
            [
                read_f32_le(bytes, 312)? as f64,
                read_f32_le(bytes, 316)? as f64,
                read_f32_le(bytes, 320)? as f64,
                read_f32_le(bytes, 324)? as f64,
            ],
            [0.0, 0.0, 0.0, 1.0],
        ]);
    }
    let qform_code = read_i16_le(bytes, 252)?;
    if qform_code > 0 {
        return Ok(qform_to_affine(
            read_f32_le(bytes, 256)? as f64,
            read_f32_le(bytes, 260)? as f64,
            read_f32_le(bytes, 264)? as f64,
            read_f32_le(bytes, 268)? as f64,
            read_f32_le(bytes, 272)? as f64,
            read_f32_le(bytes, 276)? as f64,
            read_f32_le(bytes, 76)? as f64,
            read_f32_le(bytes, 80)? as f64,
            read_f32_le(bytes, 84)? as f64,
            read_f32_le(bytes, 88)? as f64,
        ));
    }
    Ok([
        [read_f32_le(bytes, 80)? as f64, 0.0, 0.0, 0.0],
        [0.0, read_f32_le(bytes, 84)? as f64, 0.0, 0.0],
        [0.0, 0.0, read_f32_le(bytes, 88)? as f64, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
}

fn parse_nifti2_affine(bytes: &[u8]) -> Result<[[f64; 4]; 4]> {
    let sform_code = read_i32_le(bytes, 348)?;
    if sform_code > 0 {
        return Ok([
            [
                read_f64_le(bytes, 400)?,
                read_f64_le(bytes, 408)?,
                read_f64_le(bytes, 416)?,
                read_f64_le(bytes, 424)?,
            ],
            [
                read_f64_le(bytes, 432)?,
                read_f64_le(bytes, 440)?,
                read_f64_le(bytes, 448)?,
                read_f64_le(bytes, 456)?,
            ],
            [
                read_f64_le(bytes, 464)?,
                read_f64_le(bytes, 472)?,
                read_f64_le(bytes, 480)?,
                read_f64_le(bytes, 488)?,
            ],
            [0.0, 0.0, 0.0, 1.0],
        ]);
    }
    let qform_code = read_i32_le(bytes, 344)?;
    if qform_code > 0 {
        return Ok(qform_to_affine(
            read_f64_le(bytes, 352)?,
            read_f64_le(bytes, 360)?,
            read_f64_le(bytes, 368)?,
            read_f64_le(bytes, 376)?,
            read_f64_le(bytes, 384)?,
            read_f64_le(bytes, 392)?,
            read_f64_le(bytes, 104)?,
            read_f64_le(bytes, 112)?,
            read_f64_le(bytes, 120)?,
            read_f64_le(bytes, 128)?,
        ));
    }
    Ok([
        [read_f64_le(bytes, 112)?, 0.0, 0.0, 0.0],
        [0.0, read_f64_le(bytes, 120)?, 0.0, 0.0],
        [0.0, 0.0, read_f64_le(bytes, 128)?, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
}

fn qform_to_affine(
    quatern_b: f64,
    quatern_c: f64,
    quatern_d: f64,
    qoffset_x: f64,
    qoffset_y: f64,
    qoffset_z: f64,
    qfac_raw: f64,
    pixdim_x: f64,
    pixdim_y: f64,
    pixdim_z: f64,
) -> [[f64; 4]; 4] {
    let b = quatern_b;
    let c = quatern_c;
    let d = quatern_d;
    let a_sq = 1.0 - (b * b + c * c + d * d);
    let a = if a_sq < 1.0e-7 { 0.0 } else { a_sq.sqrt() };
    let qfac = if qfac_raw < 0.0 { -1.0 } else { 1.0 };
    let dx = pixdim_x;
    let dy = pixdim_y;
    let dz = qfac * pixdim_z;
    [
        [
            (a * a + b * b - c * c - d * d) * dx,
            (2.0 * b * c - 2.0 * a * d) * dy,
            (2.0 * b * d + 2.0 * a * c) * dz,
            qoffset_x,
        ],
        [
            (2.0 * b * c + 2.0 * a * d) * dx,
            (a * a + c * c - b * b - d * d) * dy,
            (2.0 * c * d - 2.0 * a * b) * dz,
            qoffset_y,
        ],
        [
            (2.0 * b * d - 2.0 * a * c) * dx,
            (2.0 * c * d + 2.0 * a * b) * dy,
            (a * a + d * d - b * b - c * c) * dz,
            qoffset_z,
        ],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn read_i16_le(bytes: &[u8], offset: usize) -> Result<i16> {
    let end = offset + 2;
    bytes
        .get(offset..end)
        .ok_or_else(|| OdxError::Format("unexpected EOF while reading NIfTI header".into()))
        .map(|slice| i16::from_le_bytes(slice.try_into().unwrap()))
}

fn read_i32_le(bytes: &[u8], offset: usize) -> Result<i32> {
    let end = offset + 4;
    bytes
        .get(offset..end)
        .ok_or_else(|| OdxError::Format("unexpected EOF while reading NIfTI header".into()))
        .map(|slice| i32::from_le_bytes(slice.try_into().unwrap()))
}

fn read_f32_le(bytes: &[u8], offset: usize) -> Result<f32> {
    let end = offset + 4;
    bytes
        .get(offset..end)
        .ok_or_else(|| OdxError::Format("unexpected EOF while reading NIfTI header".into()))
        .map(|slice| f32::from_le_bytes(slice.try_into().unwrap()))
}

fn read_f64_le(bytes: &[u8], offset: usize) -> Result<f64> {
    let end = offset + 8;
    bytes
        .get(offset..end)
        .ok_or_else(|| OdxError::Format("unexpected EOF while reading NIfTI header".into()))
        .map(|slice| f64::from_le_bytes(slice.try_into().unwrap()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_nifti1_header(
        qform_code: i16,
        sform_code: i16,
        quatern_b: f32,
        quatern_c: f32,
        quatern_d: f32,
        qoffset: [f32; 3],
        qfac: f32,
        pixdim: [f32; 3],
        srow: Option<[[f32; 4]; 3]>,
    ) -> Vec<u8> {
        let mut bytes = vec![0u8; 348];
        bytes[0..4].copy_from_slice(&348i32.to_le_bytes());
        bytes[76..80].copy_from_slice(&qfac.to_le_bytes());
        bytes[80..84].copy_from_slice(&pixdim[0].to_le_bytes());
        bytes[84..88].copy_from_slice(&pixdim[1].to_le_bytes());
        bytes[88..92].copy_from_slice(&pixdim[2].to_le_bytes());
        bytes[252..254].copy_from_slice(&qform_code.to_le_bytes());
        bytes[254..256].copy_from_slice(&sform_code.to_le_bytes());
        bytes[256..260].copy_from_slice(&quatern_b.to_le_bytes());
        bytes[260..264].copy_from_slice(&quatern_c.to_le_bytes());
        bytes[264..268].copy_from_slice(&quatern_d.to_le_bytes());
        bytes[268..272].copy_from_slice(&qoffset[0].to_le_bytes());
        bytes[272..276].copy_from_slice(&qoffset[1].to_le_bytes());
        bytes[276..280].copy_from_slice(&qoffset[2].to_le_bytes());
        if let Some(s) = srow {
            for (row_idx, row) in s.iter().enumerate() {
                let base = 280 + row_idx * 16;
                for (col_idx, val) in row.iter().enumerate() {
                    let off = base + col_idx * 4;
                    bytes[off..off + 4].copy_from_slice(&val.to_le_bytes());
                }
            }
        }
        bytes
    }

    fn approx_eq(a: [[f64; 4]; 4], b: [[f64; 4]; 4], tol: f64) {
        for r in 0..4 {
            for c in 0..4 {
                assert!(
                    (a[r][c] - b[r][c]).abs() < tol,
                    "mismatch at [{r}][{c}]: {} vs {}",
                    a[r][c],
                    b[r][c]
                );
            }
        }
    }

    #[test]
    fn qform_fallback_recovers_lps_affine_when_sform_absent() {
        // Mirrors the CS bundle: qform_code=1, sform_code=0, 180° rotation
        // around z (LPS axes), pixdim=1.7 isotropic.
        let bytes = synth_nifti1_header(
            1,
            0,
            0.0,
            0.0,
            1.0,
            [95.9, 102.25, -85.75],
            1.0,
            [1.7, 1.7, 1.7],
            None,
        );
        let affine = parse_nifti1_affine(&bytes).unwrap();
        let expected = [
            [-1.7, 0.0, 0.0, 95.9],
            [0.0, -1.7, 0.0, 102.25],
            [0.0, 0.0, 1.7, -85.75],
            [0.0, 0.0, 0.0, 1.0],
        ];
        approx_eq(affine, expected, 1e-5);
    }

    #[test]
    fn sform_takes_priority_over_qform_when_both_set() {
        let srow = [
            [2.0, 0.0, 0.0, 10.0],
            [0.0, 2.0, 0.0, 20.0],
            [0.0, 0.0, 2.0, 30.0],
        ];
        let bytes = synth_nifti1_header(
            1,
            2,
            0.0,
            0.0,
            1.0,
            [95.9, 102.25, -85.75],
            1.0,
            [1.7, 1.7, 1.7],
            Some(srow),
        );
        let affine = parse_nifti1_affine(&bytes).unwrap();
        let expected = [
            [2.0, 0.0, 0.0, 10.0],
            [0.0, 2.0, 0.0, 20.0],
            [0.0, 0.0, 2.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        approx_eq(affine, expected, 1e-5);
    }

    #[test]
    fn qfac_negative_flips_z_column() {
        let bytes = synth_nifti1_header(
            1,
            0,
            0.0,
            0.0,
            0.0,
            [0.0, 0.0, 0.0],
            -1.0,
            [1.0, 1.0, 1.0],
            None,
        );
        let affine = parse_nifti1_affine(&bytes).unwrap();
        assert!((affine[2][2] - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn no_codes_falls_back_to_pixdim_diagonal() {
        let bytes = synth_nifti1_header(
            0,
            0,
            0.0,
            0.0,
            0.0,
            [0.0, 0.0, 0.0],
            1.0,
            [2.5, 2.5, 2.5],
            None,
        );
        let affine = parse_nifti1_affine(&bytes).unwrap();
        let expected = [
            [2.5, 0.0, 0.0, 0.0],
            [0.0, 2.5, 0.0, 0.0],
            [0.0, 0.0, 2.5, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        approx_eq(affine, expected, 1e-6);
    }
}
