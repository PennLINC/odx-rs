use std::sync::OnceLock;

static VERTICES_RAS: OnceLock<Vec<[f32; 3]>> = OnceLock::new();
static FACES: OnceLock<Vec<[u32; 3]>> = OnceLock::new();

pub fn full_vertices_ras() -> &'static [[f32; 3]] {
    VERTICES_RAS.get_or_init(parse_vertices).as_slice()
}

pub fn hemisphere_vertices_ras() -> &'static [[f32; 3]] {
    let verts = full_vertices_ras();
    &verts[..verts.len() / 2]
}

pub fn faces() -> &'static [[u32; 3]] {
    FACES.get_or_init(parse_faces).as_slice()
}

pub fn matches_builtin(vertices: &[[f32; 3]], input_faces: &[[u32; 3]]) -> bool {
    const TOL: f32 = 1e-6;
    if vertices.len() != full_vertices_ras().len() || input_faces.len() != faces().len() {
        return false;
    }
    let verts_match = vertices
        .iter()
        .zip(full_vertices_ras().iter())
        .all(|(a, b)| {
            (a[0] - b[0]).abs() <= TOL && (a[1] - b[1]).abs() <= TOL && (a[2] - b[2]).abs() <= TOL
        });
    verts_match && input_faces == faces()
}

fn parse_vertices() -> Vec<[f32; 3]> {
    include_str!("../../assets/dsistudio_odf8_vertices.txt")
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let values = line
                .split_whitespace()
                .map(|token| token.parse::<f32>().expect("bad odf8 vertex"))
                .collect::<Vec<_>>();
            assert_eq!(values.len(), 3, "odf8 vertices must have 3 columns");
            [
                // DSI Studio stores the sphere in LPS. ODX keeps geometry in RAS,
                // so the built-in asset is flipped once here and reused everywhere.
                -values[0], -values[1], values[2],
            ]
        })
        .collect()
}

fn parse_faces() -> Vec<[u32; 3]> {
    include_str!("../../assets/dsistudio_odf8_faces.txt")
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let values = line
                .split_whitespace()
                .map(|token| token.parse::<u32>().expect("bad odf8 face"))
                .collect::<Vec<_>>();
            assert_eq!(values.len(), 3, "odf8 faces must have 3 columns");
            [values[0], values[1], values[2]]
        })
        .collect()
}
