use bytemuck::{cast_slice, cast_slice_mut, Pod};
use memmap2::{Mmap, MmapMut};

use crate::error::{OdxError, Result};

pub fn vec_to_bytes<T: Pod>(v: Vec<T>) -> Vec<u8> {
    cast_slice::<T, u8>(&v).to_vec()
}

pub enum MmapBacking {
    ReadOnly(Mmap),
    ReadWrite(MmapMut),
    Owned(Vec<u8>),
}

impl MmapBacking {
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            MmapBacking::ReadOnly(m) => m,
            MmapBacking::ReadWrite(m) => m,
            MmapBacking::Owned(v) => v,
        }
    }

    pub fn as_bytes_mut(&mut self) -> Result<&mut [u8]> {
        match self {
            MmapBacking::ReadOnly(_) => Err(OdxError::Argument(
                "cannot mutably access read-only mmap".into(),
            )),
            MmapBacking::ReadWrite(m) => Ok(m.as_mut()),
            MmapBacking::Owned(v) => Ok(v.as_mut_slice()),
        }
    }

    pub fn len(&self) -> usize {
        self.as_bytes().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn cast_slice<T: Pod>(&self) -> &[T] {
        cast_slice(self.as_bytes())
    }

    pub fn cast_slice_mut<T: Pod>(&mut self) -> Result<&mut [T]> {
        let bytes = self.as_bytes_mut()?;
        Ok(cast_slice_mut(bytes))
    }
}

impl std::fmt::Debug for MmapBacking {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MmapBacking::ReadOnly(m) => write!(f, "ReadOnly({} bytes)", m.len()),
            MmapBacking::ReadWrite(m) => write!(f, "ReadWrite({} bytes)", m.len()),
            MmapBacking::Owned(v) => write!(f, "Owned({} bytes)", v.len()),
        }
    }
}
