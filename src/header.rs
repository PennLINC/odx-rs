use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

use crate::error::Result;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CanonicalDenseRepresentation {
    Sh,
    Odf,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuantizationSpec {
    pub slope: f32,
    pub intercept: f32,
    pub stored_dtype: String,
}

/// PAM5-derived metadata that doesn't have a natural home in the ODX core
/// schema but must round-trip through `from_peaks_and_metrics` →
/// `to_peaks_and_metrics`. These are dipy-specific semantics:
///
/// - `total_weight` / `ang_thr` come from `EuDXDirectionGetter` thresholds.
/// - `basis_assumed` records the dipy basis name as understood at PAM5
///   write time (used when reloading the file under a different default).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PamMetadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_weight: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ang_thr: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub basis_assumed: Option<String>,
}

impl PamMetadata {
    pub fn is_empty(&self) -> bool {
        self.total_weight.is_none() && self.ang_thr.is_none() && self.basis_assumed.is_none()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Header {
    #[serde(rename = "VOXEL_TO_RASMM")]
    pub voxel_to_rasmm: [[f64; 4]; 4],

    #[serde(rename = "DIMENSIONS")]
    pub dimensions: [u64; 3],

    #[serde(rename = "NB_VOXELS")]
    pub nb_voxels: u64,

    #[serde(rename = "NB_PEAKS")]
    pub nb_peaks: u64,

    #[serde(rename = "NB_SPHERE_VERTICES", skip_serializing_if = "Option::is_none")]
    pub nb_sphere_vertices: Option<u64>,

    #[serde(rename = "NB_SPHERE_FACES", skip_serializing_if = "Option::is_none")]
    pub nb_sphere_faces: Option<u64>,

    #[serde(rename = "SH_ORDER", skip_serializing_if = "Option::is_none")]
    pub sh_order: Option<u64>,

    #[serde(rename = "SH_BASIS", skip_serializing_if = "Option::is_none")]
    pub sh_basis: Option<String>,

    #[serde(rename = "SH_FULL_BASIS", skip_serializing_if = "Option::is_none")]
    pub sh_full_basis: Option<bool>,

    #[serde(rename = "SH_LEGACY", skip_serializing_if = "Option::is_none")]
    pub sh_legacy: Option<bool>,

    #[serde(
        rename = "CANONICAL_DENSE_REPRESENTATION",
        skip_serializing_if = "Option::is_none"
    )]
    pub canonical_dense_representation: Option<CanonicalDenseRepresentation>,

    #[serde(rename = "SPHERE_ID", skip_serializing_if = "Option::is_none")]
    pub sphere_id: Option<String>,

    #[serde(rename = "ODF_SAMPLE_DOMAIN", skip_serializing_if = "Option::is_none")]
    pub odf_sample_domain: Option<String>,

    #[serde(
        rename = "ARRAY_QUANTIZATION",
        skip_serializing_if = "HashMap::is_empty",
        default
    )]
    pub array_quantization: HashMap<String, QuantizationSpec>,

    /// PAM5-derived metadata that needs to ride along through
    /// PAM ↔ ODX round-trips. See [`PamMetadata`].
    #[serde(rename = "PAM_METADATA", skip_serializing_if = "Option::is_none", default)]
    pub pam_metadata: Option<PamMetadata>,

    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

impl Header {
    pub fn from_file(path: &Path) -> Result<Self> {
        let data = std::fs::read_to_string(path)?;
        let header: Header = serde_json::from_str(&data)?;
        Ok(header)
    }

    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    pub fn write_to(&self, path: &Path) -> Result<()> {
        let json = self.to_json()?;
        std::fs::write(path, json)?;
        Ok(())
    }

    pub fn identity_affine() -> [[f64; 4]; 4] {
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }

    pub fn mask_volume_size(&self) -> usize {
        self.dimensions[0] as usize * self.dimensions[1] as usize * self.dimensions[2] as usize
    }

    /// Single source of truth mapping `(sh_basis, sh_legacy)` → a canonical
    /// dipy-style basis name. Returns `None` if the SH basis isn't set.
    ///
    /// - `tournier07` → `"tournier07"` (legacy ignored — tournier has no
    ///   legacy variant in dipy).
    /// - `descoteaux07`, legacy=false → `"descoteaux07"`.
    /// - `descoteaux07`, legacy=true → `"descoteaux07_legacy"`.
    pub fn dipy_basis_name(&self) -> Option<&'static str> {
        let basis = self.sh_basis.as_deref()?.to_ascii_lowercase();
        let legacy = self.sh_legacy.unwrap_or(false);
        match basis.as_str() {
            "tournier07" | "mrtrix" | "mrtrix3" => Some("tournier07"),
            "descoteaux07" | "dipy" => {
                if legacy {
                    Some("descoteaux07_legacy")
                } else {
                    Some("descoteaux07")
                }
            }
            _ => None,
        }
    }

    /// Set `sh_basis` and `sh_legacy` from a canonical dipy-style basis name.
    /// Mirrors [`Self::dipy_basis_name`].
    pub fn set_dipy_basis(&mut self, name: &str) -> Result<()> {
        let (canonical, legacy) = crate::sh_basis_evaluator::parse_dipy_basis(name)?;
        self.sh_basis = Some(canonical.to_string());
        self.sh_legacy = Some(legacy);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_serde_round_trip() {
        let header = Header {
            voxel_to_rasmm: Header::identity_affine(),
            dimensions: [145, 174, 145],
            nb_voxels: 72534,
            nb_peaks: 198421,
            nb_sphere_vertices: Some(642),
            nb_sphere_faces: Some(1280),
            sh_order: Some(8),
            sh_basis: Some("descoteaux07".into()),
            sh_full_basis: None,
            sh_legacy: None,
            canonical_dense_representation: Some(CanonicalDenseRepresentation::Sh),
            sphere_id: Some("odf8".into()),
            odf_sample_domain: Some("hemisphere".into()),
            array_quantization: HashMap::new(),
            pam_metadata: None,
            extra: HashMap::new(),
        };

        let json = header.to_json().unwrap();
        let parsed: Header = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.nb_voxels, 72534);
        assert_eq!(parsed.nb_peaks, 198421);
        assert_eq!(parsed.nb_sphere_vertices, Some(642));
        assert_eq!(parsed.sh_order, Some(8));
        assert_eq!(parsed.sh_basis.as_deref(), Some("descoteaux07"));
        assert_eq!(
            parsed.canonical_dense_representation,
            Some(CanonicalDenseRepresentation::Sh)
        );
        assert_eq!(parsed.sphere_id.as_deref(), Some("odf8"));
        assert_eq!(parsed.odf_sample_domain.as_deref(), Some("hemisphere"));
    }

    #[test]
    fn header_minimal() {
        let json = r#"{
            "VOXEL_TO_RASMM": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
            "DIMENSIONS": [100, 100, 100],
            "NB_VOXELS": 42,
            "NB_PEAKS": 100
        }"#;

        let header: Header = serde_json::from_str(json).unwrap();
        assert_eq!(header.nb_voxels, 42);
        assert_eq!(header.nb_sphere_vertices, None);
        assert_eq!(header.sh_order, None);
    }

    #[test]
    fn header_with_extra_fields() {
        let json = r#"{
            "VOXEL_TO_RASMM": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
            "DIMENSIONS": [100, 100, 100],
            "NB_VOXELS": 42,
            "NB_PEAKS": 100,
            "CUSTOM_FIELD": "hello"
        }"#;

        let header: Header = serde_json::from_str(json).unwrap();
        assert_eq!(
            header.extra.get("CUSTOM_FIELD").unwrap(),
            &serde_json::Value::String("hello".into())
        );
    }
}
