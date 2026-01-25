//! Builder pattern for einsum construction.

use std::collections::HashMap;

use omeco::Label;

use super::Einsum;

/// Builder for constructing einsum specifications.
///
/// # Example
///
/// ```rust,ignore
/// use omeinsum::EinBuilder;
///
/// let ein = EinBuilder::new()
///     .input(&[0, 1])      // A[i,j]
///     .input(&[1, 2])      // B[j,k]
///     .output(&[0, 2])     // C[i,k]
///     .size(0, 10)         // i has size 10
///     .size(1, 20)         // j has size 20
///     .size(2, 30)         // k has size 30
///     .build();
/// ```
pub struct EinBuilder<L: Label = usize> {
    ixs: Vec<Vec<L>>,
    iy: Option<Vec<L>>,
    size_dict: HashMap<L, usize>,
}

impl<L: Label> Default for EinBuilder<L> {
    fn default() -> Self {
        Self::new()
    }
}

impl<L: Label> EinBuilder<L> {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            ixs: Vec::new(),
            iy: None,
            size_dict: HashMap::new(),
        }
    }

    /// Add an input tensor specification.
    pub fn input(mut self, indices: &[L]) -> Self {
        self.ixs.push(indices.to_vec());
        self
    }

    /// Set the output specification.
    pub fn output(mut self, indices: &[L]) -> Self {
        self.iy = Some(indices.to_vec());
        self
    }

    /// Set the size for an index.
    pub fn size(mut self, index: L, size: usize) -> Self {
        self.size_dict.insert(index, size);
        self
    }

    /// Set multiple sizes at once.
    pub fn sizes(mut self, sizes: impl IntoIterator<Item = (L, usize)>) -> Self {
        self.size_dict.extend(sizes);
        self
    }

    /// Build the einsum specification.
    ///
    /// # Panics
    ///
    /// Panics if no output is specified or if sizes are missing.
    pub fn build(self) -> Einsum<L> {
        let iy = self.iy.expect("Output indices not specified");

        // Validate all indices have sizes
        for ix in &self.ixs {
            for i in ix {
                assert!(
                    self.size_dict.contains_key(i),
                    "Size not specified for index {:?}",
                    i
                );
            }
        }
        for i in &iy {
            assert!(
                self.size_dict.contains_key(i),
                "Size not specified for output index {:?}",
                i
            );
        }

        Einsum::new(self.ixs, iy, self.size_dict)
    }
}

/// Convenience macro for creating einsum specifications.
///
/// # Example
///
/// ```rust,ignore
/// // A[i,j] × B[j,k] → C[i,k]
/// let ein = ein!([i, j], [j, k] -> [i, k]; i=10, j=20, k=30);
/// ```
#[macro_export]
macro_rules! ein {
    // Parse: [ix1], [ix2], ... -> [iy]; sizes
    ($([$($ix:expr),*]),+ -> [$($iy:expr),*]; $($label:ident = $size:expr),*) => {{
        let mut builder = $crate::EinBuilder::new();
        $(
            builder = builder.input(&[$($ix),*]);
        )+
        builder = builder.output(&[$($iy),*]);
        $(
            builder = builder.size($label, $size);
        )*
        builder.build()
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder() {
        let ein: Einsum<usize> = EinBuilder::new()
            .input(&[0, 1])
            .input(&[1, 2])
            .output(&[0, 2])
            .size(0, 10)
            .size(1, 20)
            .size(2, 30)
            .build();

        assert_eq!(ein.ixs, vec![vec![0, 1], vec![1, 2]]);
        assert_eq!(ein.iy, vec![0, 2]);
        assert_eq!(ein.size_dict.get(&0), Some(&10));
        assert_eq!(ein.size_dict.get(&1), Some(&20));
        assert_eq!(ein.size_dict.get(&2), Some(&30));
    }

    #[test]
    fn test_builder_with_chars() {
        let ein: Einsum<char> = EinBuilder::new()
            .input(&['i', 'j'])
            .input(&['j', 'k'])
            .output(&['i', 'k'])
            .size('i', 10)
            .size('j', 20)
            .size('k', 30)
            .build();

        assert_eq!(ein.ixs, vec![vec!['i', 'j'], vec!['j', 'k']]);
        assert_eq!(ein.iy, vec!['i', 'k']);
    }
}
