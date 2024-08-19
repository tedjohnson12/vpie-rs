
use std::collections::HashSet;
use nalgebra as na;

mod bic;
mod coeffs;

pub use bic::bic;
pub use bic::log_likelihood;

pub use coeffs::{get_bi, get_mmat, get_weights};


/// Get the $\mathcal{B}$ matrix from the set of spectra given the set $s$
/// 
/// # Arguments
/// 
/// * `f_org` - Original spectra ($m \times n$)
/// * `s` - Set of indices (length $q$)
/// 
/// # Returns
/// 
/// * Basis matrix ($q \times n$)
pub fn get_basis_from_vectors<T>(
    f_org: &na::DMatrix<T>,
    s: &HashSet<usize>
) -> na::DMatrix<T>
where
    T: na::RealField
{
    let n = f_org.ncols();
    let q = s.len();
    let mut bmat = na::DMatrix::<T>::zeros(q, n);
    for (i, _s) in s.into_iter().enumerate() {
        bmat.set_row(i, &f_org.row(*_s))
    }
    bmat
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_basis_from_vectors() {
        let spectra = na::Matrix3x5::<f64>::new(
            1.0, 0.0, 0.0, -1.0, 0.0,
            0.0, 1.0, 0.0, 0.0, 1.0,
            0.0, 0.0, 1.0, 0.0, 0.0,
        );
        let s = HashSet::from([0,1]);

        let bmat_exp = na::Matrix2x5::<f64>::new(
            1.0, 0.0, 0.0, -1.0, 0.0,
            0.0, 1.0, 0.0, 0.0, 1.0,
        );

        let bmat = get_basis_from_vectors::<f64>(&na::convert(spectra), &s);

        assert_eq!(bmat, bmat_exp);

    }
}
