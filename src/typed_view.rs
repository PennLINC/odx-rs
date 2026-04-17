use bytemuck::Pod;

#[derive(Debug, Clone, Copy)]
pub struct TypedView2D<'a, T: Pod> {
    data: &'a [T],
    ncols: usize,
}

impl<'a, T: Pod> TypedView2D<'a, T> {
    pub fn new(data: &'a [T], ncols: usize) -> Self {
        assert!(
            ncols > 0 && data.len() % ncols == 0,
            "data length {} is not divisible by ncols {}",
            data.len(),
            ncols,
        );
        Self { data, ncols }
    }

    pub fn nrows(&self) -> usize {
        self.data.len() / self.ncols
    }

    pub fn ncols(&self) -> usize {
        self.ncols
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.nrows(), self.ncols)
    }

    pub fn row(&self, i: usize) -> &'a [T] {
        let start = i * self.ncols;
        &self.data[start..start + self.ncols]
    }

    pub fn as_flat_slice(&self) -> &'a [T] {
        self.data
    }

    pub fn rows(&self) -> impl Iterator<Item = &'a [T]> {
        self.data.chunks_exact(self.ncols)
    }
}
