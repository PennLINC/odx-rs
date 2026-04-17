use bytemuck::{cast_slice, Pod};
use std::collections::HashMap;
use std::path::Path;

use crate::data_array::{read_scalar_array_as_f32, DataArray, DataArrayInfo, DataPerGroup};
use crate::dtype::DType;
use crate::error::{OdxError, Result};
use crate::header::Header;
use crate::mmap_backing::MmapBacking;
use crate::typed_view::TypedView2D;

pub(crate) struct OdxParts {
    pub header: Header,
    pub mask_backing: MmapBacking,
    pub offsets_backing: MmapBacking,
    pub directions_backing: MmapBacking,
    pub sphere_vertices: Option<MmapBacking>,
    pub sphere_faces: Option<MmapBacking>,
    pub odf: HashMap<String, DataArray>,
    pub sh: HashMap<String, DataArray>,
    pub dpv: HashMap<String, DataArray>,
    pub dpf: HashMap<String, DataArray>,
    pub groups: HashMap<String, DataArray>,
    pub dpg: DataPerGroup,
    pub tempdir: Option<tempfile::TempDir>,
}

#[derive(Debug, Clone, Copy)]
pub struct OdxWritePolicy {
    pub quantize_dense: bool,
    pub quantize_min_len: usize,
}

impl Default for OdxWritePolicy {
    fn default() -> Self {
        Self {
            quantize_dense: false,
            quantize_min_len: 4096,
        }
    }
}

/// Core ODX container with a float32-first public API.
///
/// Runtime dtype metadata is still preserved inside [`DataArray`] for dense
/// scalar arrays, but directions and sphere vertices are always exposed as
/// float32.
pub struct OdxDataset {
    header: Header,
    mask_backing: MmapBacking,
    offsets_backing: MmapBacking,
    directions_backing: MmapBacking,
    sphere_vertices: Option<MmapBacking>,
    sphere_faces: Option<MmapBacking>,
    odf: HashMap<String, DataArray>,
    sh: HashMap<String, DataArray>,
    dpv: HashMap<String, DataArray>,
    dpf: HashMap<String, DataArray>,
    groups: HashMap<String, DataArray>,
    dpg: DataPerGroup,
    _tempdir: Option<tempfile::TempDir>,
}

impl OdxDataset {
    pub(crate) fn from_parts(parts: OdxParts) -> Self {
        Self {
            header: parts.header,
            mask_backing: parts.mask_backing,
            offsets_backing: parts.offsets_backing,
            directions_backing: parts.directions_backing,
            sphere_vertices: parts.sphere_vertices,
            sphere_faces: parts.sphere_faces,
            odf: parts.odf,
            sh: parts.sh,
            dpv: parts.dpv,
            dpf: parts.dpf,
            groups: parts.groups,
            dpg: parts.dpg,
            _tempdir: parts.tempdir,
        }
    }

    pub fn header(&self) -> &Header {
        &self.header
    }

    pub fn mask(&self) -> &[u8] {
        self.mask_backing.as_bytes()
    }

    pub fn nb_voxels(&self) -> usize {
        self.header.nb_voxels as usize
    }

    pub fn offsets(&self) -> &[u32] {
        self.offsets_backing.cast_slice()
    }

    pub fn nb_peaks(&self) -> usize {
        self.header.nb_peaks as usize
    }

    pub fn peaks_per_voxel(&self, i: usize) -> usize {
        let offsets = self.offsets();
        (offsets[i + 1] - offsets[i]) as usize
    }

    pub fn directions(&self) -> &[[f32; 3]] {
        cast_slice(self.directions_backing.as_bytes())
    }

    pub fn directions_bytes(&self) -> &[u8] {
        self.directions_backing.as_bytes()
    }

    pub fn voxel_directions(&self, i: usize) -> &[[f32; 3]] {
        let offsets = self.offsets();
        let start = offsets[i] as usize;
        let end = offsets[i + 1] as usize;
        &self.directions()[start..end]
    }

    pub fn voxel_peaks(&self) -> VoxelPeakIter<'_> {
        VoxelPeakIter {
            directions: self.directions(),
            offsets: self.offsets(),
            index: 0,
        }
    }

    pub fn sphere_vertices(&self) -> Option<&[[f32; 3]]> {
        self.sphere_vertices
            .as_ref()
            .map(|b| cast_slice(b.as_bytes()))
    }

    pub fn sphere_faces(&self) -> Option<&[[u32; 3]]> {
        self.sphere_faces.as_ref().map(|b| cast_slice(b.as_bytes()))
    }

    pub fn sphere_vertices_bytes(&self) -> Option<&[u8]> {
        self.sphere_vertices.as_ref().map(|b| b.as_bytes())
    }

    pub fn sphere_faces_bytes(&self) -> Option<&[u8]> {
        self.sphere_faces.as_ref().map(|b| b.as_bytes())
    }

    pub fn odf<T: Pod>(&self, name: &str) -> Result<TypedView2D<'_, T>> {
        let arr = self
            .odf
            .get(name)
            .ok_or_else(|| OdxError::Argument(format!("no ODF array named '{name}'")))?;
        Ok(arr.typed_view())
    }

    pub fn odf_names(&self) -> Vec<&str> {
        self.odf.keys().map(|s| s.as_str()).collect()
    }

    pub fn sh<T: Pod>(&self, name: &str) -> Result<TypedView2D<'_, T>> {
        let arr = self
            .sh
            .get(name)
            .ok_or_else(|| OdxError::Argument(format!("no SH array named '{name}'")))?;
        Ok(arr.typed_view())
    }

    pub fn sh_names(&self) -> Vec<&str> {
        self.sh.keys().map(|s| s.as_str()).collect()
    }

    pub fn dpv<T: Pod>(&self, name: &str) -> Result<TypedView2D<'_, T>> {
        let arr = self
            .dpv
            .get(name)
            .ok_or_else(|| OdxError::Argument(format!("no DPV named '{name}'")))?;
        Ok(arr.typed_view())
    }

    pub fn dpf<T: Pod>(&self, name: &str) -> Result<TypedView2D<'_, T>> {
        let arr = self
            .dpf
            .get(name)
            .ok_or_else(|| OdxError::Argument(format!("no DPF named '{name}'")))?;
        Ok(arr.typed_view())
    }

    pub fn group(&self, name: &str) -> Result<&[u32]> {
        let arr = self
            .groups
            .get(name)
            .ok_or_else(|| OdxError::Argument(format!("no group named '{name}'")))?;
        Ok(arr.cast_slice())
    }

    pub fn dpg<T: Pod>(&self, group: &str, name: &str) -> Result<TypedView2D<'_, T>> {
        let group_map = self
            .dpg
            .get(group)
            .ok_or_else(|| OdxError::Argument(format!("no DPG group named '{group}'")))?;
        let arr = group_map.get(name).ok_or_else(|| {
            OdxError::Argument(format!("no DPG named '{name}' in group '{group}'"))
        })?;
        Ok(arr.typed_view())
    }

    pub fn dpv_names(&self) -> Vec<&str> {
        self.dpv.keys().map(|s| s.as_str()).collect()
    }

    pub fn dpf_names(&self) -> Vec<&str> {
        self.dpf.keys().map(|s| s.as_str()).collect()
    }

    pub fn group_names(&self) -> Vec<&str> {
        self.groups.keys().map(|s| s.as_str()).collect()
    }

    pub fn scalar_dpv_f32(&self, name: &str) -> Result<Vec<f32>> {
        let arr = self
            .dpv
            .get(name)
            .ok_or_else(|| OdxError::Argument(format!("no DPV named '{name}'")))?;
        read_scalar_array_as_f32(arr, "DPV", name)
    }

    pub fn scalar_dpf_f32(&self, name: &str) -> Result<Vec<f32>> {
        let arr = self
            .dpf
            .get(name)
            .ok_or_else(|| OdxError::Argument(format!("no DPF named '{name}'")))?;
        read_scalar_array_as_f32(arr, "DPF", name)
    }

    pub fn iter_dpv(&self) -> impl Iterator<Item = (&str, DataArrayInfo)> + '_ {
        self.dpv
            .iter()
            .map(|(name, arr)| (name.as_str(), arr.info()))
    }

    pub fn iter_dpf(&self) -> impl Iterator<Item = (&str, DataArrayInfo)> + '_ {
        self.dpf
            .iter()
            .map(|(name, arr)| (name.as_str(), arr.info()))
    }

    pub(crate) fn odf_arrays(&self) -> &HashMap<String, DataArray> {
        &self.odf
    }

    pub(crate) fn sh_arrays(&self) -> &HashMap<String, DataArray> {
        &self.sh
    }

    pub(crate) fn dpv_arrays(&self) -> &HashMap<String, DataArray> {
        &self.dpv
    }

    pub(crate) fn dpf_arrays(&self) -> &HashMap<String, DataArray> {
        &self.dpf
    }

    pub(crate) fn group_arrays(&self) -> &HashMap<String, DataArray> {
        &self.groups
    }

    pub(crate) fn dpg_arrays(&self) -> &DataPerGroup {
        &self.dpg
    }

    pub fn open(path: &Path) -> Result<Self> {
        crate::io::load(path)
    }

    pub fn load(path: &Path) -> Result<Self> {
        Self::open(path)
    }

    pub fn open_directory(path: &Path) -> Result<Self> {
        crate::io::directory::open_directory(path, None)
    }

    pub fn save_directory(&self, path: &Path) -> Result<()> {
        crate::io::directory::save_directory(self, path, OdxWritePolicy::default())
    }

    pub fn save_directory_with_policy(&self, path: &Path, policy: OdxWritePolicy) -> Result<()> {
        crate::io::directory::save_directory(self, path, policy)
    }

    pub fn save_archive(&self, path: &Path) -> Result<()> {
        crate::io::zip::save_archive(self, path, OdxWritePolicy::default())
    }

    pub fn save_archive_with_policy(&self, path: &Path, policy: OdxWritePolicy) -> Result<()> {
        crate::io::zip::save_archive(self, path, policy)
    }

    /// Return the `[i, j, k]` grid position for each nonzero mask voxel,
    /// in the same compact order used by ODF/SH/DPV row indices.
    ///
    /// The mask is stored in C order (i-slowest, k-fastest), so the
    /// iteration order must match: i outermost, k innermost.
    pub fn compact_to_ijk(&self) -> Vec<[u32; 3]> {
        let dims = self.header.dimensions;
        let mask = self.mask();
        let mut out = Vec::with_capacity(self.nb_voxels());
        for i in 0..dims[0] as u32 {
            for j in 0..dims[1] as u32 {
                for k in 0..dims[2] as u32 {
                    let flat = i as u64 * dims[1] * dims[2] + j as u64 * dims[2] + k as u64;
                    if mask[flat as usize] != 0 {
                        out.push([i, j, k]);
                    }
                }
            }
        }
        out
    }

    /// Compute the RAS+ mm center for each nonzero mask voxel, in compact
    /// order.  The affine is applied to `[i, j, k, 1]` (integer indices map
    /// directly to voxel centers under the NIfTI/ODX affine convention).
    pub fn mask_voxel_centers_ras(&self) -> Vec<[f32; 3]> {
        let affine = &self.header.voxel_to_rasmm;
        self.compact_to_ijk()
            .iter()
            .map(|ijk| {
                let v = [ijk[0] as f64, ijk[1] as f64, ijk[2] as f64];
                [
                    (affine[0][0] * v[0] + affine[0][1] * v[1] + affine[0][2] * v[2] + affine[0][3])
                        as f32,
                    (affine[1][0] * v[0] + affine[1][1] * v[1] + affine[1][2] * v[2] + affine[1][3])
                        as f32,
                    (affine[2][0] * v[0] + affine[2][1] * v[1] + affine[2][2] * v[2] + affine[2][3])
                        as f32,
                ]
            })
            .collect()
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if path.is_dir() {
            self.save_directory(path)
        } else if path.extension().and_then(|e| e.to_str()) == Some("odx") {
            self.save_archive(path)
        } else {
            self.save_directory(path)
        }
    }

    pub(crate) fn clone_owned_parts(&self) -> OdxParts {
        OdxParts {
            header: self.header.clone(),
            mask_backing: MmapBacking::Owned(self.mask_backing.as_bytes().to_vec()),
            offsets_backing: MmapBacking::Owned(self.offsets_backing.as_bytes().to_vec()),
            directions_backing: MmapBacking::Owned(self.directions_backing.as_bytes().to_vec()),
            sphere_vertices: self
                .sphere_vertices
                .as_ref()
                .map(|backing| MmapBacking::Owned(backing.as_bytes().to_vec())),
            sphere_faces: self
                .sphere_faces
                .as_ref()
                .map(|backing| MmapBacking::Owned(backing.as_bytes().to_vec())),
            odf: self
                .odf
                .iter()
                .map(|(name, arr)| (name.clone(), arr.clone_owned()))
                .collect(),
            sh: self
                .sh
                .iter()
                .map(|(name, arr)| (name.clone(), arr.clone_owned()))
                .collect(),
            dpv: self
                .dpv
                .iter()
                .map(|(name, arr)| (name.clone(), arr.clone_owned()))
                .collect(),
            dpf: self
                .dpf
                .iter()
                .map(|(name, arr)| (name.clone(), arr.clone_owned()))
                .collect(),
            groups: self
                .groups
                .iter()
                .map(|(name, arr)| (name.clone(), arr.clone_owned()))
                .collect(),
            dpg: self
                .dpg
                .iter()
                .map(|(group, arrays)| {
                    (
                        group.clone(),
                        arrays
                            .iter()
                            .map(|(name, arr)| (name.clone(), arr.clone_owned()))
                            .collect(),
                    )
                })
                .collect(),
            tempdir: None,
        }
    }
}

impl std::fmt::Debug for OdxDataset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OdxDataset")
            .field("direction_dtype", &DType::Float32)
            .field("nb_voxels", &self.nb_voxels())
            .field("nb_peaks", &self.nb_peaks())
            .field("odf", &self.odf_names())
            .field("sh", &self.sh_names())
            .field("dpv", &self.dpv_names())
            .field("dpf", &self.dpf_names())
            .field("groups", &self.group_names())
            .finish()
    }
}

pub struct VoxelPeakIter<'a> {
    directions: &'a [[f32; 3]],
    offsets: &'a [u32],
    index: usize,
}

impl<'a> Iterator for VoxelPeakIter<'a> {
    type Item = &'a [[f32; 3]];

    fn next(&mut self) -> Option<Self::Item> {
        if self.index + 1 >= self.offsets.len() {
            return None;
        }
        let start = self.offsets[self.index] as usize;
        let end = self.offsets[self.index + 1] as usize;
        self.index += 1;
        Some(&self.directions[start..end])
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = if self.offsets.is_empty() {
            0
        } else {
            self.offsets.len() - 1 - self.index
        };
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for VoxelPeakIter<'a> {}

pub type OdxFile = OdxDataset;
