//! Python bindings for `odx-rs`.
//!
//! Exposes the ODX format as a Python package `odx` with a thin pyO3 layer
//! over the Rust `OdxDataset`/`OdxBuilder`. Foreign-format converters
//! (DSI Studio, MRtrix, MIF, pyAFQ, Tortoise) are exposed through dedicated
//! `Odx.save_*` / `from_*` functions; PAM5 goes through the Python-side
//! `odx.adapters.dipy` so the wheel doesn't need HDF5 to function.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use ndarray::Array2;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyArray4, PyArray5, PyArrayMethods,
    PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArray4,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use odx_rs::densify::{
    densify_directions, densify_odf, densify_scalar_dpf, densify_scalar_dpv, densify_sh,
    max_peaks_per_voxel,
};
use odx_rs::dtype::DType;
use odx_rs::header::{CanonicalDenseRepresentation, PamMetadata};
use odx_rs::interop::{convert_sh_basis as core_convert_sh_basis, ShBasisTarget};
use odx_rs::peak_finder::{
    peaks_from_sh_rows_with_basis, PeakFinderConfig as CorePeakFinderConfig,
    SpherePeakFinder as CoreSpherePeakFinder,
};
use odx_rs::sh_basis_evaluator::{
    basis_kind_from_dipy_name, compute_b_matrix as core_compute_b_matrix,
};
use odx_rs::sphere_lookup::{median_nearest_vertex_angle_deg, nearest_vertex_indices};
use odx_rs::{OdxBuilder, OdxDataset};

// ─── Error helpers ───────────────────────────────────────────────────────────

fn map_err<E: std::fmt::Display>(e: E) -> PyErr {
    PyValueError::new_err(format!("{e}"))
}

fn map_io<E: std::fmt::Display>(e: E) -> PyErr {
    PyIOError::new_err(format!("{e}"))
}

// ─── PeakFinderConfig ────────────────────────────────────────────────────────

#[pyclass(name = "PeakFinderConfig", module = "odx._odx")]
#[derive(Clone)]
struct PyPeakFinderConfig {
    inner: CorePeakFinderConfig,
}

#[pymethods]
impl PyPeakFinderConfig {
    #[new]
    #[pyo3(signature = (npeaks=5, relative_peak_threshold=0.5, min_separation_angle_deg=25.0))]
    fn new(
        npeaks: usize,
        relative_peak_threshold: f32,
        min_separation_angle_deg: f32,
    ) -> Self {
        Self {
            inner: CorePeakFinderConfig {
                npeaks,
                relative_peak_threshold,
                min_separation_angle_deg,
            },
        }
    }

    #[getter]
    fn npeaks(&self) -> usize {
        self.inner.npeaks
    }
    #[getter]
    fn relative_peak_threshold(&self) -> f32 {
        self.inner.relative_peak_threshold
    }
    #[getter]
    fn min_separation_angle_deg(&self) -> f32 {
        self.inner.min_separation_angle_deg
    }

    fn __repr__(&self) -> String {
        format!(
            "PeakFinderConfig(npeaks={}, relative_peak_threshold={}, min_separation_angle_deg={})",
            self.inner.npeaks,
            self.inner.relative_peak_threshold,
            self.inner.min_separation_angle_deg
        )
    }
}

// ─── SpherePeakFinder ────────────────────────────────────────────────────────

#[pyclass(name = "SpherePeakFinder", module = "odx._odx")]
struct PySpherePeakFinder {
    inner: CoreSpherePeakFinder,
}

#[pymethods]
impl PySpherePeakFinder {
    #[new]
    #[pyo3(signature = (vertices, faces, config=None))]
    fn new(
        vertices: PyReadonlyArray2<'_, f32>,
        faces: PyReadonlyArray2<'_, u32>,
        config: Option<PyPeakFinderConfig>,
    ) -> PyResult<Self> {
        let v = read_xyz_array(&vertices)?;
        let f = read_face_array(&faces)?;
        let cfg = config.map(|c| c.inner).unwrap_or_default();
        Ok(Self {
            inner: CoreSpherePeakFinder::new(&v, &f, cfg),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (config=None))]
    fn for_dsistudio_odf8(config: Option<PyPeakFinderConfig>) -> Self {
        let cfg = config.map(|c| c.inner).unwrap_or_default();
        Self {
            inner: CoreSpherePeakFinder::for_dsistudio_odf8(cfg),
        }
    }

    /// Find peaks in a single ODF row. Returns `(amps, dirs)` where `amps`
    /// is `(k,) float32` and `dirs` is `(k, 3) float32`.
    fn find_peaks<'py>(
        &self,
        py: Python<'py>,
        odf: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray2<f32>>)> {
        let odf_slice = odf.as_slice().map_err(map_err)?;
        let peaks = self.inner.find_peaks(odf_slice);
        let mut amps = Vec::with_capacity(peaks.len());
        let mut dirs = Vec::with_capacity(peaks.len() * 3);
        for (a, d) in &peaks {
            amps.push(*a);
            dirs.extend_from_slice(d);
        }
        let amps_arr = amps.into_pyarray_bound(py);
        let dirs_arr = Array2::from_shape_vec((peaks.len(), 3), dirs)
            .map_err(map_err)?
            .into_pyarray_bound(py);
        Ok((amps_arr, dirs_arr))
    }

    /// Batch peak finding over `(N, M)` ODF rows. Returns
    /// `(offsets: (N+1,) uint32, directions: (P, 3) float32, amplitudes: (P,) float32)`.
    fn find_peaks_rows<'py>(
        &self,
        py: Python<'py>,
        odf_rows: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<(
        Bound<'py, PyArray1<u32>>,
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray1<f32>>,
    )> {
        let arr = odf_rows.as_array();
        let nrows = arr.shape()[0];
        let flat = arr
            .as_slice()
            .ok_or_else(|| PyValueError::new_err("odf_rows must be C-contiguous"))?;
        let (offsets, dirs, amps) = self.inner.find_peaks_rows(flat, nrows);
        let n_peaks = dirs.len();
        let dirs_flat: Vec<f32> = dirs.into_iter().flatten().collect();
        let dirs_arr = Array2::from_shape_vec((n_peaks, 3), dirs_flat)
            .map_err(map_err)?
            .into_pyarray_bound(py);
        Ok((
            offsets.into_pyarray_bound(py),
            dirs_arr,
            amps.into_pyarray_bound(py),
        ))
    }
}

// ─── PyOdx ───────────────────────────────────────────────────────────────────

#[pyclass(name = "Odx", module = "odx._odx", unsendable)]
struct PyOdx {
    inner: Arc<OdxDataset>,
}

impl PyOdx {
    fn from_dataset(dataset: OdxDataset) -> Self {
        Self {
            inner: Arc::new(dataset),
        }
    }
}

#[pymethods]
impl PyOdx {
    /// 4×4 voxel-to-RAS+mm affine, float64.
    #[getter]
    fn affine<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let a = self.inner.header().voxel_to_rasmm;
        let mut data = Vec::with_capacity(16);
        for row in &a {
            data.extend_from_slice(row);
        }
        ndarray::Array2::from_shape_vec((4, 4), data)
            .unwrap()
            .into_pyarray_bound(py)
    }

    /// Volume dimensions `(X, Y, Z)`.
    #[getter]
    fn dimensions(&self) -> (u64, u64, u64) {
        let d = self.inner.header().dimensions;
        (d[0], d[1], d[2])
    }

    /// Brain mask reshaped to `(X, Y, Z)` uint8. Owned copy.
    #[getter]
    fn mask<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<u8>> {
        let dims = self.inner.header().dimensions;
        let (x, y, z) = (dims[0] as usize, dims[1] as usize, dims[2] as usize);
        let bytes = self.inner.mask().to_vec();
        ndarray::Array3::from_shape_vec((x, y, z), bytes)
            .unwrap()
            .into_pyarray_bound(py)
    }

    /// Per-voxel peak offsets, length `nb_voxels + 1` uint32. Owned copy.
    #[getter]
    fn offsets<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        self.inner.offsets().to_vec().into_pyarray_bound(py)
    }

    /// All peak directions as `(NB_PEAKS, 3) float32`. Owned copy.
    #[getter]
    fn directions<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        let dirs = self.inner.directions();
        let n = dirs.len();
        let mut data = Vec::with_capacity(n * 3);
        for d in dirs {
            data.extend_from_slice(d);
        }
        ndarray::Array2::from_shape_vec((n, 3), data)
            .unwrap()
            .into_pyarray_bound(py)
    }

    #[getter]
    fn nb_voxels(&self) -> usize {
        self.inner.nb_voxels()
    }

    #[getter]
    fn nb_peaks(&self) -> usize {
        self.inner.nb_peaks()
    }

    #[getter]
    fn sh_order(&self) -> Option<u64> {
        self.inner.header().sh_order
    }

    #[getter]
    fn sh_basis(&self) -> Option<String> {
        self.inner.header().sh_basis.clone()
    }

    #[getter]
    fn sh_legacy(&self) -> Option<bool> {
        self.inner.header().sh_legacy
    }

    #[getter]
    fn sh_full_basis(&self) -> Option<bool> {
        self.inner.header().sh_full_basis
    }

    #[getter]
    fn sphere_id(&self) -> Option<String> {
        self.inner.header().sphere_id.clone()
    }

    /// PAM-derived metadata that round-trips through PAM ↔ ODX
    /// (`total_weight`, `ang_thr`, `basis_assumed`). Returns a dict or None.
    #[getter]
    fn pam_metadata<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyDict>> {
        let m = self.inner.header().pam_metadata.as_ref()?;
        let d = PyDict::new_bound(py);
        if let Some(v) = m.total_weight {
            d.set_item("total_weight", v).ok()?;
        }
        if let Some(v) = m.ang_thr {
            d.set_item("ang_thr", v).ok()?;
        }
        if let Some(ref s) = m.basis_assumed {
            d.set_item("basis_assumed", s.as_str()).ok()?;
        }
        Some(d)
    }

    /// Return the canonical dipy basis name (`"tournier07" | "descoteaux07" |
    /// "descoteaux07_legacy"`) or `None` if SH metadata is absent.
    #[getter]
    fn dipy_basis_name(&self) -> Option<String> {
        self.inner.header().dipy_basis_name().map(|s| s.to_string())
    }

    /// `"sh"` or `"odf"` describing the canonical dense representation.
    #[getter]
    fn canonical_dense_representation(&self) -> Option<String> {
        self.inner
            .header()
            .canonical_dense_representation
            .as_ref()
            .map(|c| match c {
                CanonicalDenseRepresentation::Sh => "sh".to_string(),
                CanonicalDenseRepresentation::Odf => "odf".to_string(),
            })
    }

    /// Sphere vertices `(M, 3) float32` if attached, else None.
    #[getter]
    fn sphere_vertices<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f32>>> {
        self.inner.sphere_vertices().map(|v| {
            let n = v.len();
            let mut data = Vec::with_capacity(n * 3);
            for d in v {
                data.extend_from_slice(d);
            }
            ndarray::Array2::from_shape_vec((n, 3), data)
                .unwrap()
                .into_pyarray_bound(py)
        })
    }

    /// Sphere faces `(F, 3) uint32` if attached.
    #[getter]
    fn sphere_faces<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<u32>>> {
        self.inner.sphere_faces().map(|f| {
            let n = f.len();
            let mut data = Vec::with_capacity(n * 3);
            for d in f {
                data.extend_from_slice(d);
            }
            ndarray::Array2::from_shape_vec((n, 3), data)
                .unwrap()
                .into_pyarray_bound(py)
        })
    }

    fn sh_names<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        let names: Vec<String> = self
            .inner
            .sh_names()
            .into_iter()
            .map(|s| s.to_string())
            .collect();
        PyList::new_bound(py, names)
    }

    fn odf_names<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        let names: Vec<String> = self
            .inner
            .odf_names()
            .into_iter()
            .map(|s| s.to_string())
            .collect();
        PyList::new_bound(py, names)
    }

    fn dpv_names<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        let names: Vec<String> = self
            .inner
            .dpv_names()
            .into_iter()
            .map(|s| s.to_string())
            .collect();
        PyList::new_bound(py, names)
    }

    fn dpf_names<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        let names: Vec<String> = self
            .inner
            .dpf_names()
            .into_iter()
            .map(|s| s.to_string())
            .collect();
        PyList::new_bound(py, names)
    }

    /// Get an SH array as a `(NB_VOXELS, K) float32` numpy array. Quantized
    /// uint8 storage is dequantized on the way through.
    #[pyo3(signature = (name="coefficients"))]
    fn sh<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let arr = match self.inner.get_sh(name) {
            Some(a) => a,
            None => {
                return Err(PyValueError::new_err(format!("no SH array '{name}'")));
            }
        };
        let values = arr.to_f32_vec().map_err(map_err)?;
        let ncols = arr.ncols();
        let nrows = values.len() / ncols.max(1);
        Ok(ndarray::Array2::from_shape_vec((nrows, ncols), values)
            .map_err(map_err)?
            .into_pyarray_bound(py))
    }

    #[pyo3(signature = (name="amplitudes"))]
    fn odf<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let arr = match self.inner.get_odf(name) {
            Some(a) => a,
            None => return Err(PyValueError::new_err(format!("no ODF array '{name}'"))),
        };
        let values = arr.to_f32_vec().map_err(map_err)?;
        let ncols = arr.ncols();
        let nrows = values.len() / ncols.max(1);
        Ok(ndarray::Array2::from_shape_vec((nrows, ncols), values)
            .map_err(map_err)?
            .into_pyarray_bound(py))
    }

    fn dpv<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let arr = self
            .inner
            .get_dpv(name)
            .ok_or_else(|| PyValueError::new_err(format!("no DPV '{name}'")))?;
        let values = arr.to_f32_vec().map_err(map_err)?;
        let ncols = arr.ncols();
        let nrows = values.len() / ncols.max(1);
        Ok(ndarray::Array2::from_shape_vec((nrows, ncols), values)
            .map_err(map_err)?
            .into_pyarray_bound(py))
    }

    fn dpf<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let arr = self
            .inner
            .get_dpf(name)
            .ok_or_else(|| PyValueError::new_err(format!("no DPF '{name}'")))?;
        let values = arr.to_f32_vec().map_err(map_err)?;
        let ncols = arr.ncols();
        let nrows = values.len() / ncols.max(1);
        Ok(ndarray::Array2::from_shape_vec((nrows, ncols), values)
            .map_err(map_err)?
            .into_pyarray_bound(py))
    }

    /// Densify peak directions into `(X, Y, Z, N_max, 3) float32`.
    fn densify_directions<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray5<f32>> {
        densify_directions(&self.inner).into_pyarray_bound(py)
    }

    fn densify_dpf<'py>(
        &self,
        py: Python<'py>,
        name: &str,
    ) -> PyResult<Bound<'py, PyArray4<f32>>> {
        Ok(densify_scalar_dpf(&self.inner, name)
            .map_err(map_err)?
            .into_pyarray_bound(py))
    }

    fn densify_dpv<'py>(
        &self,
        py: Python<'py>,
        name: &str,
    ) -> PyResult<Bound<'py, PyArray3<f32>>> {
        Ok(densify_scalar_dpv(&self.inner, name)
            .map_err(map_err)?
            .into_pyarray_bound(py))
    }

    #[pyo3(signature = (name="coefficients"))]
    fn densify_sh<'py>(
        &self,
        py: Python<'py>,
        name: &str,
    ) -> PyResult<Bound<'py, PyArray4<f32>>> {
        Ok(densify_sh(&self.inner, name)
            .map_err(map_err)?
            .into_pyarray_bound(py))
    }

    #[pyo3(signature = (name="amplitudes"))]
    fn densify_odf<'py>(
        &self,
        py: Python<'py>,
        name: &str,
    ) -> PyResult<Bound<'py, PyArray4<f32>>> {
        Ok(densify_odf(&self.inner, name)
            .map_err(map_err)?
            .into_pyarray_bound(py))
    }

    /// Maximum peak count across voxels — the `N_max` dim of densified
    /// peak arrays.
    fn max_peaks_per_voxel(&self) -> usize {
        max_peaks_per_voxel(&self.inner)
    }

    /// Nearest-vertex lookup mapping each peak direction to a sphere index.
    /// `(NB_PEAKS,) int32`. Antipodal symmetry on by default.
    #[pyo3(signature = (sphere_vertices, antipodal=true))]
    fn peak_indices_for<'py>(
        &self,
        py: Python<'py>,
        sphere_vertices: PyReadonlyArray2<'py, f32>,
        antipodal: bool,
    ) -> PyResult<Bound<'py, PyArray1<i32>>> {
        let s = read_xyz_array(&sphere_vertices)?;
        Ok(nearest_vertex_indices(self.inner.directions(), &s, antipodal).into_pyarray_bound(py))
    }

    /// Median angle (degrees) between each peak direction and its nearest
    /// sphere vertex. Used to gauge quantization loss before lossy exports.
    #[pyo3(signature = (sphere_vertices, antipodal=true))]
    fn peak_quantization_error_deg<'py>(
        &self,
        sphere_vertices: PyReadonlyArray2<'py, f32>,
        antipodal: bool,
    ) -> PyResult<f32> {
        let s = read_xyz_array(&sphere_vertices)?;
        Ok(median_nearest_vertex_angle_deg(
            self.inner.directions(),
            &s,
            antipodal,
        ))
    }

    /// Save as a native ODX directory or `.odx` archive (extension-dispatched).
    fn save(&self, path: PathBuf) -> PyResult<()> {
        self.inner.save(&path).map_err(map_io)
    }

    fn to_directory(&self, path: PathBuf) -> PyResult<()> {
        self.inner.save_directory(&path).map_err(map_io)
    }

    fn to_archive(&self, path: PathBuf) -> PyResult<()> {
        self.inner.save_archive(&path).map_err(map_io)
    }

    /// DSI Studio compressed format (`.fz`). Lossy quantization to the DSI
    /// Studio sphere; emits a warning if median quantization error exceeds
    /// 1° unless `lossy_warning=False`.
    #[pyo3(signature = (path, lossy_warning=true))]
    fn save_fz(&self, py: Python<'_>, path: PathBuf, lossy_warning: bool) -> PyResult<()> {
        if lossy_warning {
            maybe_warn_quantization(py, &self.inner, "save_fz")?;
        }
        odx_rs::dsistudio::save_fz(&self.inner, &path).map_err(map_io)
    }

    #[pyo3(signature = (path, lossy_warning=true))]
    fn save_fibgz(&self, py: Python<'_>, path: PathBuf, lossy_warning: bool) -> PyResult<()> {
        if lossy_warning {
            maybe_warn_quantization(py, &self.inner, "save_fibgz")?;
        }
        odx_rs::dsistudio::save_fibgz(&self.inner, &path).map_err(map_io)
    }

    /// MRtrix fixel directory + sibling SH `.mif`. Auto-converts to tournier07
    /// if needed (use `convert_basis="strict"` to refuse non-tournier input).
    #[pyo3(signature = (directory, sh_filename="wmfod.mif".to_string(), convert_basis="auto".to_string()))]
    fn save_mrtrix(
        &self,
        directory: PathBuf,
        sh_filename: String,
        convert_basis: String,
    ) -> PyResult<()> {
        let prepared = ensure_tournier(&self.inner, &convert_basis)?;
        let sh_path = directory.join(&sh_filename);
        odx_rs::mrtrix::save_mrtrix_fixels(
            prepared.as_ref(),
            &directory,
            &odx_rs::mrtrix::MrtrixFixelWriteOptions::default(),
        )
        .map_err(map_io)?;
        odx_rs::mrtrix::save_mrtrix_sh(
            prepared.as_ref(),
            &sh_path,
            &odx_rs::mrtrix::MrtrixShWriteOptions::default(),
        )
        .map_err(map_io)?;
        Ok(())
    }

    /// Single MRtrix `.mif`/`.mif.gz` file. `which="sh"` writes the SH coefficients
    /// (auto-converts basis); `which="odf"` writes ODF amplitudes (no basis conversion).
    #[pyo3(signature = (path, which="sh".to_string(), name="coefficients".to_string(), convert_basis="auto".to_string()))]
    fn save_mif(
        &self,
        path: PathBuf,
        which: String,
        name: String,
        convert_basis: String,
    ) -> PyResult<()> {
        match which.as_str() {
            "sh" => {
                let prepared = ensure_tournier(&self.inner, &convert_basis)?;
                odx_rs::mrtrix::save_mrtrix_sh(
                    prepared.as_ref(),
                    &path,
                    &odx_rs::mrtrix::MrtrixShWriteOptions::default(),
                )
                .map_err(map_io)
            }
            "odf" => {
                let _ = name;
                Err(PyValueError::new_err(
                    "save_mif(which='odf') is not implemented yet; use save_mrtrix",
                ))
            }
            other => Err(PyValueError::new_err(format!(
                "save_mif: which must be 'sh' or 'odf'; got '{other}'"
            ))),
        }
    }

    /// Convert this dataset's SH coefficients to a different basis. Returns
    /// a new `Odx`. Targets: `"tournier07"`, `"descoteaux07"`, `"descoteaux07_legacy"`.
    fn convert_sh_basis_to(&self, target: &str) -> PyResult<PyOdx> {
        let target_kind = parse_basis_target(target)?;
        let new = core_convert_sh_basis(&self.inner, target_kind, None).map_err(map_err)?;
        Ok(PyOdx::from_dataset(new))
    }

    /// Sugar for `convert_sh_basis_to("tournier07")`.
    fn to_tournier(&self) -> PyResult<PyOdx> {
        self.convert_sh_basis_to("tournier07")
    }

    /// Sugar for `convert_sh_basis_to("descoteaux07[_legacy]")`.
    #[pyo3(signature = (legacy=true))]
    fn to_descoteaux(&self, legacy: bool) -> PyResult<PyOdx> {
        if legacy {
            self.convert_sh_basis_to("descoteaux07_legacy")
        } else {
            self.convert_sh_basis_to("descoteaux07")
        }
    }

    /// Run the Rust peak finder on this dataset's SH coefficients and return
    /// a *new* Odx with `directions` and `dpf/amplitude` populated. Niche path
    /// — prefer `OdxBuilder.compute_peaks` or `from_sh_coefficients`.
    #[pyo3(signature = (sphere_vertices=None, sphere_faces=None, config=None))]
    fn with_peaks_from_sh(
        &self,
        sphere_vertices: Option<PyReadonlyArray2<'_, f32>>,
        sphere_faces: Option<PyReadonlyArray2<'_, u32>>,
        config: Option<PyPeakFinderConfig>,
    ) -> PyResult<PyOdx> {
        let header = self.inner.header();
        let sh_order = header.sh_order.ok_or_else(|| {
            PyValueError::new_err("with_peaks_from_sh: sh_order not set on source")
        })? as usize;
        let basis_name = header.dipy_basis_name().ok_or_else(|| {
            PyValueError::new_err("with_peaks_from_sh: SH basis not set on source")
        })?;
        let full_basis = header.sh_full_basis.unwrap_or(false);
        let kind = basis_kind_from_dipy_name(basis_name, sh_order, full_basis).map_err(map_err)?;

        let sh_arr = self
            .inner
            .get_sh("coefficients")
            .ok_or_else(|| PyValueError::new_err("with_peaks_from_sh: no sh/coefficients"))?;
        let sh_rows = sh_arr.to_f32_vec().map_err(map_err)?;
        let nb_voxels = self.inner.nb_voxels();

        let (vertices, faces) = match (sphere_vertices, sphere_faces) {
            (Some(v), Some(f)) => (read_xyz_array(&v)?, read_face_array(&f)?),
            (None, None) => match (self.inner.sphere_vertices(), self.inner.sphere_faces()) {
                (Some(v), Some(f)) => (v.to_vec(), f.to_vec()),
                _ => (
                    odx_rs::formats::dsistudio_odf8::hemisphere_vertices_ras().to_vec(),
                    odx_rs::formats::dsistudio_odf8::faces().to_vec(),
                ),
            },
            _ => {
                return Err(PyValueError::new_err(
                    "with_peaks_from_sh: pass both sphere_vertices and sphere_faces or neither",
                ))
            }
        };

        let cfg = config.map(|c| c.inner).unwrap_or_default();
        let finder = CoreSpherePeakFinder::new(&vertices, &faces, cfg);
        let (offsets, directions, amplitudes) =
            peaks_from_sh_rows_with_basis(&sh_rows, nb_voxels, &finder, kind).map_err(map_err)?;

        let new = self
            .inner
            .with_replaced_peaks(offsets, directions, amplitudes);
        Ok(PyOdx::from_dataset(new))
    }

    fn __repr__(&self) -> String {
        let h = self.inner.header();
        format!(
            "<Odx dimensions=({},{},{}) nb_voxels={} nb_peaks={} sh_basis={:?} sh_order={:?}>",
            h.dimensions[0],
            h.dimensions[1],
            h.dimensions[2],
            self.inner.nb_voxels(),
            self.inner.nb_peaks(),
            h.sh_basis,
            h.sh_order
        )
    }
}

// ─── PyOdxBuilder ────────────────────────────────────────────────────────────

#[pyclass(name = "OdxBuilder", module = "odx._odx", unsendable)]
struct PyOdxBuilder {
    inner: Option<OdxBuilder>,
    /// Track per-voxel push count vs explicit skip_all_peaks state.
    peak_state: PeakState,
}

#[derive(Copy, Clone, PartialEq, Eq)]
enum PeakState {
    /// Builder is fresh; no peak pushes or skip yet.
    Untouched,
    /// User called push_voxel_peaks at least once.
    Pushing,
    /// User called skip_all_peaks; offsets initialized to all zeros.
    Skipped,
    /// User called compute_peaks (overrides any prior pushes).
    Computed,
}

impl PyOdxBuilder {
    fn require(&mut self) -> PyResult<&mut OdxBuilder> {
        self.inner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("OdxBuilder already finalized"))
    }
}

#[pymethods]
impl PyOdxBuilder {
    #[new]
    fn new(
        affine: PyReadonlyArray2<'_, f64>,
        dimensions: (u64, u64, u64),
        mask: PyReadonlyArray1<'_, u8>,
    ) -> PyResult<Self> {
        let aff = read_4x4_affine(&affine)?;
        let mask_vec = mask.as_slice().map_err(map_err)?.to_vec();
        let dims = [dimensions.0, dimensions.1, dimensions.2];
        let expected = dims[0] as usize * dims[1] as usize * dims[2] as usize;
        if mask_vec.len() != expected {
            return Err(PyValueError::new_err(format!(
                "mask length {} does not match dimensions product {}",
                mask_vec.len(),
                expected
            )));
        }
        Ok(Self {
            inner: Some(OdxBuilder::new(aff, dims, mask_vec)),
            peak_state: PeakState::Untouched,
        })
    }

    fn push_voxel_peaks(&mut self, peaks: PyReadonlyArray2<'_, f32>) -> PyResult<()> {
        let v = read_xyz_array(&peaks)?;
        self.require()?.push_voxel_peaks(&v);
        self.peak_state = PeakState::Pushing;
        Ok(())
    }

    fn skip_all_peaks(&mut self) -> PyResult<()> {
        self.require()?.skip_all_peaks();
        self.peak_state = PeakState::Skipped;
        Ok(())
    }

    fn set_sphere(
        &mut self,
        vertices: PyReadonlyArray2<'_, f32>,
        faces: PyReadonlyArray2<'_, u32>,
    ) -> PyResult<()> {
        let v = read_xyz_array(&vertices)?;
        let f = read_face_array(&faces)?;
        self.require()?.set_sphere(v, f);
        Ok(())
    }

    fn set_sh_info(&mut self, order: u64, basis: String) -> PyResult<()> {
        self.require()?.set_sh_info(order, basis);
        Ok(())
    }

    fn set_sh_full_basis(&mut self, full_basis: bool) -> PyResult<()> {
        self.require()?.set_sh_full_basis(full_basis);
        Ok(())
    }

    fn set_sh_legacy(&mut self, legacy: bool) -> PyResult<()> {
        self.require()?.set_sh_legacy(legacy);
        Ok(())
    }

    fn set_sphere_id(&mut self, sphere_id: String) -> PyResult<()> {
        self.require()?.set_sphere_id(sphere_id);
        Ok(())
    }

    /// Stash dipy-PAM-only metadata that round-trips through ODX.
    /// `total_weight` and `ang_thr` come from `PeaksAndMetrics` thresholds;
    /// `basis_assumed` records the dipy basis name as understood at write time.
    #[pyo3(signature = (total_weight=None, ang_thr=None, basis_assumed=None))]
    fn set_pam_metadata(
        &mut self,
        total_weight: Option<f64>,
        ang_thr: Option<f64>,
        basis_assumed: Option<String>,
    ) -> PyResult<()> {
        self.require()?.set_pam_metadata(PamMetadata {
            total_weight,
            ang_thr,
            basis_assumed,
        });
        Ok(())
    }

    /// Set SH coefficients from either a 4D `(X,Y,Z,K)` numpy array (auto
    /// masked-flatten) or a 2D `(NB_VOXELS, K)` masked-flat array.
    /// Stores SH metadata in one call; pair with `compute_peaks` if you want
    /// directions extracted.
    #[pyo3(signature = (sh, basis, sh_order, legacy=true, full_basis=false))]
    fn set_sh_coefficients(
        &mut self,
        py: Python<'_>,
        sh: Bound<'_, PyAny>,
        basis: String,
        sh_order: u64,
        legacy: bool,
        full_basis: bool,
    ) -> PyResult<()> {
        // Accept both 2D (NB_VOXELS, K) and 4D (X,Y,Z,K). Use the builder's
        // mask to flatten 4D inputs.
        let arr_any = sh.downcast::<numpy::PyUntypedArray>().map_err(|_| {
            PyValueError::new_err("set_sh_coefficients: sh must be a numpy array")
        })?;
        let ndim = arr_any.ndim();
        let flat_2d: ndarray::Array2<f32> = match ndim {
            2 => {
                let arr = sh.extract::<PyReadonlyArray2<f32>>()?;
                arr.as_array().to_owned()
            }
            4 => {
                let arr = sh.extract::<PyReadonlyArray4<f32>>()?;
                let view = arr.as_array();
                let shape = view.shape().to_vec();
                let builder = self.require()?;
                let dims = builder_dimensions(builder);
                if shape[0] != dims[0] as usize
                    || shape[1] != dims[1] as usize
                    || shape[2] != dims[2] as usize
                {
                    return Err(PyValueError::new_err(format!(
                        "set_sh_coefficients: 4D sh shape {:?} does not match builder dimensions {:?}",
                        &shape[..3],
                        dims
                    )));
                }
                let ncoeffs = shape[3];
                // Masked flatten: gather (i,j,k) order matching compact_to_ijk.
                let mask = builder_mask(builder);
                let mut flat = Vec::with_capacity(builder_nb_voxels(builder) * ncoeffs);
                for i in 0..dims[0] as usize {
                    for j in 0..dims[1] as usize {
                        for k in 0..dims[2] as usize {
                            let flat_idx =
                                i * (dims[1] as usize) * (dims[2] as usize) + j * (dims[2] as usize) + k;
                            if mask[flat_idx] != 0 {
                                for c in 0..ncoeffs {
                                    flat.push(view[[i, j, k, c]]);
                                }
                            }
                        }
                    }
                }
                ndarray::Array2::from_shape_vec((flat.len() / ncoeffs, ncoeffs), flat)
                    .map_err(map_err)?
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "set_sh_coefficients: sh must be 2D (NB_VOXELS,K) or 4D (X,Y,Z,K); got {other}D"
                )))
            }
        };
        let _ = py;
        let ncols = flat_2d.ncols();
        let bytes: Vec<u8> = bytemuck::cast_slice(
            flat_2d
                .as_slice()
                .ok_or_else(|| PyValueError::new_err("sh must be C-contiguous"))?,
        )
        .to_vec();
        let builder = self.require()?;
        builder.set_sh_data("coefficients", bytes, ncols, DType::Float32);
        builder.set_sh_info(sh_order, basis);
        builder.set_sh_full_basis(full_basis);
        builder.set_sh_legacy(legacy);
        Ok(())
    }

    /// Attach a per-fixel scalar array. `array` must have shape `(NB_PEAKS, ncols)`
    /// and a numeric dtype.
    fn set_dpf(&mut self, name: String, array: Bound<'_, PyAny>) -> PyResult<()> {
        let (bytes, ncols, dtype) = numpy_to_bytes(array)?;
        self.require()?.set_dpf_data(&name, bytes, ncols, dtype);
        Ok(())
    }

    fn set_dpv(&mut self, name: String, array: Bound<'_, PyAny>) -> PyResult<()> {
        let (bytes, ncols, dtype) = numpy_to_bytes(array)?;
        self.require()?.set_dpv_data(&name, bytes, ncols, dtype);
        Ok(())
    }

    fn set_odf(&mut self, name: String, array: Bound<'_, PyAny>) -> PyResult<()> {
        let (bytes, ncols, dtype) = numpy_to_bytes(array)?;
        self.require()?.set_odf_data(&name, bytes, ncols, dtype);
        Ok(())
    }

    /// Run the Rust peak finder on the SH already attached, replacing
    /// directions/offsets and adding `dpf/amplitude` in place.
    #[pyo3(signature = (sphere_vertices=None, sphere_faces=None, config=None))]
    fn compute_peaks(
        &mut self,
        sphere_vertices: Option<PyReadonlyArray2<'_, f32>>,
        sphere_faces: Option<PyReadonlyArray2<'_, u32>>,
        config: Option<PyPeakFinderConfig>,
    ) -> PyResult<()> {
        let sphere = match (sphere_vertices, sphere_faces) {
            (Some(v), Some(f)) => Some((read_xyz_array(&v)?, read_face_array(&f)?)),
            (None, None) => None,
            _ => {
                return Err(PyValueError::new_err(
                    "compute_peaks: pass both sphere_vertices and sphere_faces or neither",
                ))
            }
        };
        let cfg = config.map(|c| c.inner).unwrap_or_default();
        self.require()?.compute_peaks(sphere, cfg).map_err(map_err)?;
        self.peak_state = PeakState::Computed;
        Ok(())
    }

    fn finalize(&mut self) -> PyResult<PyOdx> {
        // If user never set up offsets, default to skip_all_peaks for an SH-only build.
        if self.peak_state == PeakState::Untouched {
            self.require()?.skip_all_peaks();
            self.peak_state = PeakState::Skipped;
        }
        let builder = self
            .inner
            .take()
            .ok_or_else(|| PyValueError::new_err("OdxBuilder already finalized"))?;
        let dataset = builder.finalize().map_err(map_err)?;
        Ok(PyOdx::from_dataset(dataset))
    }
}

// ─── module-level functions ─────────────────────────────────────────────────

/// Load a `.odx` file or directory.
#[pyfunction]
fn load(path: PathBuf) -> PyResult<PyOdx> {
    let dataset = OdxDataset::load(&path).map_err(map_io)?;
    Ok(PyOdx::from_dataset(dataset))
}

/// Save an `Odx` to a path (extension-dispatched).
#[pyfunction]
fn save(path: PathBuf, odx: &PyOdx) -> PyResult<()> {
    odx.inner.save(&path).map_err(map_io)
}

/// Run the Rust peak finder on a flat `(N, K)` SH-coefficient array using
/// the supplied sphere. Returns `(offsets, directions, amplitudes)`.
#[pyfunction]
#[pyo3(signature = (sh_rows, sphere_vertices, sphere_faces, *, basis, sh_order, legacy=false, full_basis=false, config=None))]
fn peaks_from_sh<'py>(
    py: Python<'py>,
    sh_rows: PyReadonlyArray2<'py, f32>,
    sphere_vertices: PyReadonlyArray2<'py, f32>,
    sphere_faces: PyReadonlyArray2<'py, u32>,
    basis: String,
    sh_order: usize,
    legacy: bool,
    full_basis: bool,
    config: Option<PyPeakFinderConfig>,
) -> PyResult<(
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray1<f32>>,
)> {
    let v = read_xyz_array(&sphere_vertices)?;
    let f = read_face_array(&sphere_faces)?;
    let cfg = config.map(|c| c.inner).unwrap_or_default();
    let finder = CoreSpherePeakFinder::new(&v, &f, cfg);

    let dipy_name = if matches!(basis.as_str(), "descoteaux07") && legacy {
        "descoteaux07_legacy".to_string()
    } else {
        basis
    };
    let kind = basis_kind_from_dipy_name(&dipy_name, sh_order, full_basis).map_err(map_err)?;

    let arr = sh_rows.as_array();
    let nrows = arr.shape()[0];
    let flat = arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("sh_rows must be C-contiguous"))?;
    let (offsets, dirs, amps) =
        peaks_from_sh_rows_with_basis(flat, nrows, &finder, kind).map_err(map_err)?;
    let n_peaks = dirs.len();
    let dirs_flat: Vec<f32> = dirs.into_iter().flatten().collect();
    let dirs_arr = Array2::from_shape_vec((n_peaks, 3), dirs_flat)
        .map_err(map_err)?
        .into_pyarray_bound(py);
    Ok((
        offsets.into_pyarray_bound(py),
        dirs_arr,
        amps.into_pyarray_bound(py),
    ))
}

/// One-call constructor: turn dipy-style SH coefficients into a fully-peaked
/// (or SH-only) ODX dataset.
#[pyfunction]
#[pyo3(signature = (
    sh, mask, affine, *, basis, sh_order, legacy=true, full_basis=false,
    sphere_vertices=None, sphere_faces=None, peak_config=None, compute_peaks=true
))]
#[allow(clippy::too_many_arguments)]
fn from_sh_coefficients(
    py: Python<'_>,
    sh: Bound<'_, PyAny>,
    mask: PyReadonlyArray3<'_, u8>,
    affine: PyReadonlyArray2<'_, f64>,
    basis: String,
    sh_order: u64,
    legacy: bool,
    full_basis: bool,
    sphere_vertices: Option<PyReadonlyArray2<'_, f32>>,
    sphere_faces: Option<PyReadonlyArray2<'_, u32>>,
    peak_config: Option<PyPeakFinderConfig>,
    compute_peaks: bool,
) -> PyResult<PyOdx> {
    let mask_view = mask.as_array();
    let dims = mask_view.shape().to_vec();
    if dims.len() != 3 {
        return Err(PyValueError::new_err("mask must be 3D"));
    }
    let mask_flat: Vec<u8> = mask_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("mask must be C-contiguous"))?
        .to_vec();
    let _ = read_4x4_affine(&affine)?; // validate shape; PyOdxBuilder::new re-reads.
    let dimensions = (dims[0] as u64, dims[1] as u64, dims[2] as u64);

    let mask_arr = numpy::PyArray1::from_slice_bound(py, &mask_flat).readonly();
    let py_affine = affine.clone();
    let mut builder = PyOdxBuilder::new(py_affine, dimensions, mask_arr)?;

    builder.set_sh_coefficients(py, sh, basis.clone(), sh_order, legacy, full_basis)?;

    if let (Some(v), Some(f)) = (sphere_vertices.clone(), sphere_faces.clone()) {
        builder.set_sphere(v, f)?;
    }

    if compute_peaks {
        builder.compute_peaks(sphere_vertices, sphere_faces, peak_config)?;
    } else {
        builder.skip_all_peaks()?;
    }

    builder.finalize()
}

/// Convert an Odx's SH coefficients to a different basis, returning a new Odx.
#[pyfunction]
fn convert_sh_basis(odx: &PyOdx, target: &str) -> PyResult<PyOdx> {
    odx.convert_sh_basis_to(target)
}

/// Compute the SH→SF transform matrix for the given sphere and basis.
/// Returns a `(M, K) float32` ndarray where `M = sphere_vertices.shape[0]`
/// and `K = (sh_order+1)(sh_order+2)/2` (or `(sh_order+1)^2` if
/// `full_basis=True`). Equivalent to `dipy.reconst.shm.sh_to_sf_matrix(...)[0]`
/// — handy for populating the PAM5 `B` field without importing dipy.
#[pyfunction]
#[pyo3(signature = (sphere_vertices, sh_order, *, basis="tournier07".to_string(), full_basis=false))]
fn compute_b_matrix<'py>(
    py: Python<'py>,
    sphere_vertices: PyReadonlyArray2<'py, f32>,
    sh_order: usize,
    basis: String,
    full_basis: bool,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let v = read_xyz_array(&sphere_vertices)?;
    let flat = core_compute_b_matrix(&v, sh_order, &basis, full_basis).map_err(map_err)?;
    let ncoeffs = flat.len() / v.len().max(1);
    Ok(ndarray::Array2::from_shape_vec((v.len(), ncoeffs), flat)
        .map_err(map_err)?
        .into_pyarray_bound(py))
}

/// Built-in sphere geometry used as a default for peak finding and FZ export.
/// Returns `(vertices: (M,3) float32, faces: (F,3) uint32)`.
#[pyfunction]
fn dsistudio_odf8_hemisphere<'py>(
    py: Python<'py>,
) -> (Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<u32>>) {
    let v = odx_rs::formats::dsistudio_odf8::hemisphere_vertices_ras();
    let f = odx_rs::formats::dsistudio_odf8::faces();
    let v_data: Vec<f32> = v.iter().flat_map(|d| d.iter().copied()).collect();
    let f_data: Vec<u32> = f.iter().flat_map(|d| d.iter().copied()).collect();
    let v_arr = ndarray::Array2::from_shape_vec((v.len(), 3), v_data)
        .unwrap()
        .into_pyarray_bound(py);
    let f_arr = ndarray::Array2::from_shape_vec((f.len(), 3), f_data)
        .unwrap()
        .into_pyarray_bound(py);
    (v_arr, f_arr)
}

/// DSI Studio full sphere (642 vertices). Used internally by `save_fz` and
/// for measuring quantization error.
#[pyfunction]
fn dsistudio_odf8_full_sphere<'py>(py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
    let v = odx_rs::formats::dsistudio_odf8::full_vertices_ras();
    let v_data: Vec<f32> = v.iter().flat_map(|d| d.iter().copied()).collect();
    ndarray::Array2::from_shape_vec((v.len(), 3), v_data)
        .unwrap()
        .into_pyarray_bound(py)
}

// Foreign-format loaders

#[pyfunction]
fn from_fz(path: PathBuf) -> PyResult<PyOdx> {
    let dataset = odx_rs::dsistudio::load_fz(&path, None).map_err(map_io)?;
    Ok(PyOdx::from_dataset(dataset))
}

#[pyfunction]
#[pyo3(signature = (path, affine=None))]
fn from_fibgz(path: PathBuf, affine: Option<PyReadonlyArray2<'_, f64>>) -> PyResult<PyOdx> {
    let aff = affine.map(|a| read_4x4_affine(&a)).transpose()?;
    let dataset = odx_rs::dsistudio::load_fibgz(&path, aff).map_err(map_io)?;
    Ok(PyOdx::from_dataset(dataset))
}

/// Load an MRtrix dataset from a directory containing a fixel directory and
/// (optionally) a sibling SH `.mif` file. If `sh_filename` is provided the
/// SH is loaded from `directory/sh_filename`; otherwise common defaults are
/// tried (`wmfod.mif`, `WM_FODs.mif`).
#[pyfunction]
#[pyo3(signature = (directory, sh_filename=None))]
fn from_mrtrix(directory: PathBuf, sh_filename: Option<String>) -> PyResult<PyOdx> {
    let candidates: Vec<PathBuf> = match sh_filename {
        Some(name) => vec![directory.join(&name), directory.join(format!("{name}.gz"))],
        None => vec![
            directory.join("wmfod.mif"),
            directory.join("wmfod.mif.gz"),
            directory.join("WM_FODs.mif"),
            directory.join("WM_FODs.mif.gz"),
            directory.join("fods.mif"),
            directory.join("fods.mif.gz"),
        ],
    };
    let sh_path: Option<&Path> = candidates.iter().find(|p| p.exists()).map(|p| p.as_path());
    let dataset = odx_rs::mrtrix::load_mrtrix_dataset(sh_path, Some(&directory)).map_err(map_io)?;
    Ok(PyOdx::from_dataset(dataset))
}

#[pyfunction]
fn from_mapmri(coeff_path: PathBuf, tensor_path: PathBuf, uvec_path: PathBuf) -> PyResult<PyOdx> {
    let dataset = odx_rs::tortoise_mapmri::load_tortoise_mapmri(
        &coeff_path,
        &tensor_path,
        &uvec_path,
    )
    .map_err(map_io)?;
    Ok(PyOdx::from_dataset(dataset))
}

#[pyfunction]
fn from_pyafq_aodf(directory: PathBuf) -> PyResult<PyOdx> {
    let dataset = odx_rs::formats::pyafq_aodf::load_pyafq_aodf(&directory).map_err(map_io)?;
    Ok(PyOdx::from_dataset(dataset))
}

// ─── helpers ─────────────────────────────────────────────────────────────────

/// Convert a 1D or 2D numpy array of any supported numeric dtype into the
/// `(bytes, ncols, DType)` tuple the Rust builder expects. The caller is
/// responsible for ensuring the row count matches what the builder needs.
fn numpy_to_bytes(arr: Bound<'_, PyAny>) -> PyResult<(Vec<u8>, usize, DType)> {
    let untyped = arr
        .downcast::<numpy::PyUntypedArray>()
        .map_err(|_| PyValueError::new_err("expected a numpy ndarray"))?;
    let ndim = untyped.ndim();
    if ndim != 1 && ndim != 2 {
        return Err(PyValueError::new_err(format!(
            "expected 1D or 2D array; got {ndim}D"
        )));
    }
    let dtype_str = untyped
        .dtype()
        .str()
        .map_err(map_err)?
        .to_string();
    let dtype = DType::from_numpy_str(&dtype_str)
        .map_err(|e| PyValueError::new_err(format!("unsupported dtype '{dtype_str}': {e}")))?;

    macro_rules! handle {
        ($ty:ty) => {{
            let bytes = if ndim == 1 {
                let v = arr.extract::<PyReadonlyArray1<$ty>>()?;
                let s = v
                    .as_slice()
                    .map_err(|_| PyValueError::new_err("array must be C-contiguous"))?;
                bytemuck::cast_slice::<$ty, u8>(s).to_vec()
            } else {
                let v = arr.extract::<PyReadonlyArray2<$ty>>()?;
                let arr = v.as_array();
                let s = arr
                    .as_slice()
                    .ok_or_else(|| PyValueError::new_err("array must be C-contiguous"))?;
                bytemuck::cast_slice::<$ty, u8>(s).to_vec()
            };
            let ncols = if ndim == 1 { 1 } else {
                arr.downcast::<numpy::PyUntypedArray>()
                    .unwrap()
                    .shape()[1]
            };
            (bytes, ncols)
        }};
    }

    let (bytes, ncols) = match dtype {
        DType::Float32 => handle!(f32),
        DType::Float64 => handle!(f64),
        DType::UInt8 => handle!(u8),
        DType::UInt16 => handle!(u16),
        DType::UInt32 => handle!(u32),
        DType::UInt64 => handle!(u64),
        DType::Int8 => handle!(i8),
        DType::Int16 => handle!(i16),
        DType::Int32 => handle!(i32),
        DType::Int64 => handle!(i64),
        DType::Float16 => {
            return Err(PyValueError::new_err(
                "float16 numpy arrays not yet supported for builder inputs",
            ))
        }
    };
    Ok((bytes, ncols, dtype))
}

fn read_xyz_array(arr: &PyReadonlyArray2<'_, f32>) -> PyResult<Vec<[f32; 3]>> {
    let view = arr.as_array();
    if view.shape()[1] != 3 {
        return Err(PyValueError::new_err(format!(
            "expected (N, 3) array; got shape {:?}",
            view.shape()
        )));
    }
    let mut out = Vec::with_capacity(view.shape()[0]);
    for row in view.outer_iter() {
        out.push([row[0], row[1], row[2]]);
    }
    Ok(out)
}

fn read_face_array(arr: &PyReadonlyArray2<'_, u32>) -> PyResult<Vec<[u32; 3]>> {
    let view = arr.as_array();
    if view.shape()[1] != 3 {
        return Err(PyValueError::new_err(format!(
            "expected (F, 3) array; got shape {:?}",
            view.shape()
        )));
    }
    let mut out = Vec::with_capacity(view.shape()[0]);
    for row in view.outer_iter() {
        out.push([row[0], row[1], row[2]]);
    }
    Ok(out)
}

fn read_4x4_affine(arr: &PyReadonlyArray2<'_, f64>) -> PyResult<[[f64; 4]; 4]> {
    let view = arr.as_array();
    if view.shape() != [4, 4] {
        return Err(PyValueError::new_err(format!(
            "expected (4, 4) affine; got shape {:?}",
            view.shape()
        )));
    }
    let mut out = [[0.0_f64; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            out[i][j] = view[[i, j]];
        }
    }
    Ok(out)
}

fn parse_basis_target(name: &str) -> PyResult<ShBasisTarget> {
    match name.to_ascii_lowercase().as_str() {
        "tournier07" | "mrtrix" | "mrtrix3" => Ok(ShBasisTarget::Tournier07),
        "descoteaux07" | "dipy" => Ok(ShBasisTarget::Descoteaux07 { legacy: false }),
        "descoteaux07_legacy" => Ok(ShBasisTarget::Descoteaux07 { legacy: true }),
        other => Err(PyValueError::new_err(format!(
            "unknown SH basis target '{other}'"
        ))),
    }
}

/// If `convert_basis` is `"auto"` or `"true"`/`True`, return a converted
/// (or trivially-cloned) dataset whose SH basis is tournier07. If it's
/// `"strict"`/`"false"`/`False`, return the dataset unchanged but error if
/// the basis isn't already tournier07.
fn ensure_tournier(odx: &Arc<OdxDataset>, convert_basis: &str) -> PyResult<Arc<OdxDataset>> {
    let basis = odx.header().dipy_basis_name();
    let mode = convert_basis.to_ascii_lowercase();
    let must_convert = matches!(mode.as_str(), "auto" | "true" | "always");
    let strict = matches!(mode.as_str(), "false" | "strict" | "never");

    if basis == Some("tournier07") {
        return Ok(Arc::clone(odx));
    }
    if strict {
        return Err(PyValueError::new_err(format!(
            "save requires sh_basis='tournier07' (current: {basis:?}); pass convert_basis='auto' to convert"
        )));
    }
    if !must_convert {
        return Err(PyValueError::new_err(format!(
            "unknown convert_basis '{convert_basis}'; expected 'auto'/'true' or 'strict'/'false'"
        )));
    }
    let new = core_convert_sh_basis(odx, ShBasisTarget::Tournier07, None).map_err(map_err)?;
    Ok(Arc::new(new))
}

fn maybe_warn_quantization(py: Python<'_>, odx: &OdxDataset, fn_name: &str) -> PyResult<()> {
    if odx.nb_peaks() == 0 {
        return Ok(());
    }
    let dsi_sphere: Vec<[f32; 3]> =
        odx_rs::formats::dsistudio_odf8::full_vertices_ras().to_vec();
    let median_deg = median_nearest_vertex_angle_deg(odx.directions(), &dsi_sphere, true);
    if median_deg > 1.0 {
        let warnings = py.import_bound("warnings")?;
        let msg = format!(
            "{fn_name} quantizes peaks to the DSI Studio ODF8 sphere; {median_deg:.2}° median angular drift. \
             To avoid: run with_peaks_from_sh(sphere=odx.spheres.dsistudio_odf8()) first, \
             or pass lossy_warning=False to silence."
        );
        warnings.call_method1("warn", (msg,))?;
    }
    Ok(())
}

fn builder_dimensions(b: &OdxBuilder) -> [u64; 3] {
    b.dimensions()
}

fn builder_mask(b: &OdxBuilder) -> &[u8] {
    b.mask()
}

fn builder_nb_voxels(b: &OdxBuilder) -> usize {
    b.mask().iter().filter(|&&v| v != 0).count()
}

// ─── pymodule ────────────────────────────────────────────────────────────────

#[pymodule]
fn _odx(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPeakFinderConfig>()?;
    m.add_class::<PySpherePeakFinder>()?;
    m.add_class::<PyOdx>()?;
    m.add_class::<PyOdxBuilder>()?;
    m.add_function(wrap_pyfunction!(load, m)?)?;
    m.add_function(wrap_pyfunction!(save, m)?)?;
    m.add_function(wrap_pyfunction!(peaks_from_sh, m)?)?;
    m.add_function(wrap_pyfunction!(from_sh_coefficients, m)?)?;
    m.add_function(wrap_pyfunction!(convert_sh_basis, m)?)?;
    m.add_function(wrap_pyfunction!(from_fz, m)?)?;
    m.add_function(wrap_pyfunction!(from_fibgz, m)?)?;
    m.add_function(wrap_pyfunction!(from_mrtrix, m)?)?;
    m.add_function(wrap_pyfunction!(from_mapmri, m)?)?;
    m.add_function(wrap_pyfunction!(from_pyafq_aodf, m)?)?;
    m.add_function(wrap_pyfunction!(dsistudio_odf8_hemisphere, m)?)?;
    m.add_function(wrap_pyfunction!(dsistudio_odf8_full_sphere, m)?)?;
    m.add_function(wrap_pyfunction!(compute_b_matrix, m)?)?;
    Ok(())
}
