use std::collections::HashMap;

use crate::data_array::DataArray;
use crate::dtype::DType;
use crate::error::{OdxError, Result};
use crate::header::{CanonicalDenseRepresentation, Header};
use crate::mmap_backing::{vec_to_bytes, MmapBacking};
use crate::odx_file::{OdxDataset, OdxParts};
use serde_json::Value;

pub struct OdxBuilder {
    affine: [[f64; 4]; 4],
    dimensions: [u64; 3],
    mask: Vec<u8>,
    offsets: Vec<u32>,
    directions: Vec<[f32; 3]>,
    sphere_vertices: Option<Vec<[f32; 3]>>,
    sphere_faces: Option<Vec<[u32; 3]>>,
    odf: HashMap<String, (Vec<u8>, usize, DType)>,
    sh: HashMap<String, (Vec<u8>, usize, DType)>,
    dpv: HashMap<String, (Vec<u8>, usize, DType)>,
    dpf: HashMap<String, (Vec<u8>, usize, DType)>,
    sh_order: Option<u64>,
    sh_basis: Option<String>,
    canonical_dense_representation: Option<CanonicalDenseRepresentation>,
    sphere_id: Option<String>,
    odf_sample_domain: Option<String>,
    extra: HashMap<String, Value>,
}

impl OdxBuilder {
    pub fn new(affine: [[f64; 4]; 4], dimensions: [u64; 3], mask: Vec<u8>) -> Self {
        let expected_len = dimensions[0] as usize * dimensions[1] as usize * dimensions[2] as usize;
        assert_eq!(
            mask.len(),
            expected_len,
            "mask length must equal product of dimensions"
        );
        Self {
            affine,
            dimensions,
            mask,
            offsets: vec![0],
            directions: Vec::new(),
            sphere_vertices: None,
            sphere_faces: None,
            odf: HashMap::new(),
            sh: HashMap::new(),
            dpv: HashMap::new(),
            dpf: HashMap::new(),
            sh_order: None,
            sh_basis: None,
            canonical_dense_representation: None,
            sphere_id: None,
            odf_sample_domain: None,
            extra: HashMap::new(),
        }
    }

    pub fn push_voxel_peaks(&mut self, peaks: &[[f32; 3]]) {
        self.directions.extend_from_slice(peaks);
        self.offsets.push(self.directions.len() as u32);
    }

    pub fn set_sphere(&mut self, vertices: Vec<[f32; 3]>, faces: Vec<[u32; 3]>) {
        self.sphere_vertices = Some(vertices);
        self.sphere_faces = Some(faces);
    }

    pub fn set_sphere_id(&mut self, sphere_id: impl Into<String>) {
        self.sphere_id = Some(sphere_id.into());
    }

    pub fn set_odf_sample_domain(&mut self, domain: impl Into<String>) {
        self.odf_sample_domain = Some(domain.into());
    }

    pub fn set_extra_value(&mut self, key: impl Into<String>, value: Value) {
        self.extra.insert(key.into(), value);
    }

    pub fn set_sh_info(&mut self, order: u64, basis: String) {
        self.sh_order = Some(order);
        self.sh_basis = Some(basis);
        self.canonical_dense_representation = Some(CanonicalDenseRepresentation::Sh);
    }

    pub fn set_canonical_dense_representation(
        &mut self,
        representation: CanonicalDenseRepresentation,
    ) {
        self.canonical_dense_representation = Some(representation);
    }

    pub fn set_odf_data(&mut self, name: &str, data: Vec<u8>, ncols: usize, dtype: DType) {
        self.odf.insert(name.to_string(), (data, ncols, dtype));
        self.canonical_dense_representation
            .get_or_insert(CanonicalDenseRepresentation::Odf);
    }

    pub fn set_sh_data(&mut self, name: &str, data: Vec<u8>, ncols: usize, dtype: DType) {
        self.sh.insert(name.to_string(), (data, ncols, dtype));
        self.canonical_dense_representation
            .get_or_insert(CanonicalDenseRepresentation::Sh);
    }

    pub fn set_dpv_data(&mut self, name: &str, data: Vec<u8>, ncols: usize, dtype: DType) {
        self.dpv.insert(name.to_string(), (data, ncols, dtype));
    }

    pub fn set_dpf_data(&mut self, name: &str, data: Vec<u8>, ncols: usize, dtype: DType) {
        self.dpf.insert(name.to_string(), (data, ncols, dtype));
    }

    pub fn validate(&self) -> Result<()> {
        let nb_voxels = self.offsets.len().saturating_sub(1);
        let nb_peaks = self.directions.len();
        let expected_voxels = self.mask.iter().filter(|&&v| v != 0).count();

        if nb_voxels != expected_voxels {
            return Err(OdxError::Format(format!(
                "offset rows {nb_voxels} do not match mask voxel count {expected_voxels}"
            )));
        }
        if self.offsets.last().copied().unwrap_or(0) as usize != nb_peaks {
            return Err(OdxError::Format(format!(
                "offset sentinel {} does not match peak count {nb_peaks}",
                self.offsets.last().copied().unwrap_or(0)
            )));
        }
        if !self.odf.is_empty() && self.sphere_vertices.is_none() && self.sphere_id.is_none() {
            return Err(OdxError::Format(
                "ODF data requires either explicit sphere vertices or SPHERE_ID".into(),
            ));
        }
        for (name, (data, ncols, dtype)) in &self.odf {
            validate_rows("odf", name, data, *ncols, *dtype, nb_voxels)?;
        }
        for (name, (data, ncols, dtype)) in &self.sh {
            validate_rows("sh", name, data, *ncols, *dtype, nb_voxels)?;
        }
        for (name, (data, ncols, dtype)) in &self.dpv {
            validate_rows("dpv", name, data, *ncols, *dtype, nb_voxels)?;
        }
        for (name, (data, ncols, dtype)) in &self.dpf {
            validate_rows("dpf", name, data, *ncols, *dtype, nb_peaks)?;
        }
        if let Some(order) = self.sh_order {
            let expected = ((order + 1) * (order + 2) / 2) as usize;
            for (name, (_, ncols, _)) in &self.sh {
                if *ncols != expected {
                    return Err(OdxError::Format(format!(
                        "SH array '{name}' has {ncols} columns, expected {expected} for order {order}"
                    )));
                }
            }
        }
        Ok(())
    }

    pub fn finalize(self) -> Result<OdxDataset> {
        self.validate()?;
        let nb_voxels = self.offsets.len() - 1;
        let nb_peaks = self.directions.len();

        let header = Header {
            voxel_to_rasmm: self.affine,
            dimensions: self.dimensions,
            nb_voxels: nb_voxels as u64,
            nb_peaks: nb_peaks as u64,
            nb_sphere_vertices: self.sphere_vertices.as_ref().map(|v| v.len() as u64),
            nb_sphere_faces: self.sphere_faces.as_ref().map(|f| f.len() as u64),
            sh_order: self.sh_order,
            sh_basis: self.sh_basis,
            canonical_dense_representation: self.canonical_dense_representation,
            sphere_id: self.sphere_id,
            odf_sample_domain: self.odf_sample_domain,
            array_quantization: HashMap::new(),
            extra: self.extra,
        };

        let sphere_verts_backing = self
            .sphere_vertices
            .map(|v| MmapBacking::Owned(vec_to_bytes(v)));
        let sphere_faces_backing = self
            .sphere_faces
            .map(|f| MmapBacking::Owned(vec_to_bytes(f)));

        fn build_data_map(
            entries: HashMap<String, (Vec<u8>, usize, DType)>,
        ) -> HashMap<String, DataArray> {
            entries
                .into_iter()
                .map(|(name, (data, ncols, dtype))| {
                    (name, DataArray::owned_bytes(data, ncols, dtype))
                })
                .collect()
        }

        Ok(OdxDataset::from_parts(OdxParts {
            header,
            mask_backing: MmapBacking::Owned(self.mask),
            offsets_backing: MmapBacking::Owned(vec_to_bytes(self.offsets)),
            directions_backing: MmapBacking::Owned(vec_to_bytes(self.directions)),
            sphere_vertices: sphere_verts_backing,
            sphere_faces: sphere_faces_backing,
            odf: build_data_map(self.odf),
            sh: build_data_map(self.sh),
            dpv: build_data_map(self.dpv),
            dpf: build_data_map(self.dpf),
            groups: HashMap::new(),
            dpg: HashMap::new(),
            tempdir: None,
        }))
    }
}

fn validate_rows(
    kind: &str,
    name: &str,
    data: &[u8],
    ncols: usize,
    dtype: DType,
    expected_rows: usize,
) -> Result<()> {
    let row_bytes = ncols
        .checked_mul(dtype.size_of())
        .ok_or_else(|| OdxError::Format(format!("{kind} '{name}' row byte size overflow")))?;
    let actual_rows = if row_bytes == 0 {
        0
    } else {
        data.len() / row_bytes
    };
    if actual_rows != expected_rows {
        return Err(OdxError::Format(format!(
            "{kind} '{name}' has {actual_rows} rows, expected {expected_rows}"
        )));
    }
    Ok(())
}

pub type OdxStream = OdxBuilder;
