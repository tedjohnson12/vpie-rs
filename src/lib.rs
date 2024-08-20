
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

/// Reconstruct the spectra given a basis and cutoff index
/// 
/// # Arguments
/// 
/// * `f_org` - Original spectra ($m \times n$)
/// * `f_err` - Uncertainty in the data ($m \times n$)
/// * `s` - Set of indices (length $q$)
/// * `cutoff_index` - The number of NIR wavelength points to use in training ($n'$)
/// * `use_mean_error` - Whether to use the mean error in the reconstruction
/// 
/// # Returns
/// 
/// * Reconstructed spectra ($m \times n$)
pub fn get_reconstruction<T>(
    f_org: &na::DMatrix<T>,
    f_err: &na::DMatrix<T>,
    s: &HashSet<usize>,
    cutoff_index: usize,
    use_mean_error: bool
) -> na::DMatrix<T>
where
    T: na::RealField
{
    let _q = s.len();
    let m = f_org.nrows();
    let n = f_org.ncols();
    assert_eq!(f_err.shape(), (m, n), "f_err has shape {:?} but should be {:?}", f_err.shape(), (m, n));

    assert!(cutoff_index < n, "cutoff_index must be less than n, but is {} and n is {}", cutoff_index, n);

    let f_nir = f_org.columns(0, cutoff_index).clone_owned();

    let bmat_nir = get_basis_from_vectors::<T>(&f_nir, s);
    let bmat_full = get_basis_from_vectors::<T>(&f_org, s);
    
    let mut f_rec = na::DMatrix::<T>::zeros(m, n);

    let mut weights: nalgebra::Matrix<T, nalgebra::Dyn, nalgebra::Dyn, nalgebra::VecStorage<T, nalgebra::Dyn, nalgebra::Dyn>>;
    let mut mmat = na::DMatrix::<T>::zeros(cutoff_index, cutoff_index);

    if use_mean_error {
        weights = get_weights(
            &f_err.columns(0, cutoff_index).row_mean().transpose()
        );
        mmat = get_mmat(&weights, &bmat_nir);
    }
    for i in 0..m {
        if !use_mean_error {
            weights = get_weights(&f_err.row(i).columns(0, cutoff_index).transpose());
            mmat = get_mmat(&weights, &bmat_nir);
        }
        let row_to_set = f_nir.row(i) * mmat.clone()*bmat_full.clone();
        f_rec.set_row(i, &row_to_set);
    }
    f_rec
}

pub fn calc_bic<T>(
    f_org: &na::DMatrix<T>,
    f_err: &na::DMatrix<T>,
    s: &HashSet<usize>,
    cutoff_index: usize,
    use_mean_error: bool
)-> T
where
    T: na::RealField
{
    let q = s.len();
    let m = f_org.nrows();
    let _n = cutoff_index;
    let lnl = bic::log_likelihood(
        f_org.columns(0, cutoff_index).clone_owned(),
        get_reconstruction(f_org, f_err, s, cutoff_index, use_mean_error).columns(0, cutoff_index).clone_owned(),
        f_err.columns(0, cutoff_index).clone_owned());
    
    bic::bic(q.try_into().unwrap(), _n.try_into().unwrap(), m.try_into().unwrap(),lnl)
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

    #[test]
    fn test_get_reconstruction() {
        let spectra = na::Matrix5x6::<f64>::new(
            1.0, 0.0, 0.0, -1.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
            1.0, 1.0, 2.0, 1.0, -1.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        );
        let err: na::Matrix5x6<f64> = na::Matrix5x6::zeros().add_scalar(1.0);
        let s = HashSet::from([0,1,2]);
        let cutoff_index = 3;

        let f_rec_exp = na::Matrix5x6::<f64>::new(
            1.0, 0.0, 0.0, -1.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
            1.0, 1.0, 2.0, -1.0, 1.0, 2.0, //(1,1,2)
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //(0,0,0)
        );

        let f_rec = get_reconstruction(
            &na::convert(spectra),
            &na::convert(err),
            &s,
            cutoff_index,
            false);
        assert_eq!(f_rec, f_rec_exp);
    }

    #[test]
    fn test_get_reconstruction_mean_err() {
        let spectra = na::Matrix5x6::<f64>::new(
            1.0, 0.0, 0.0, -1.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
            1.0, 1.0, 2.0, 1.0, -1.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        );

        let err = na::Matrix5x6::<f64>::new(
            1.0, 1.0, -1.0, -1.0, 0.0, 0.0,
            2.0, 1.0, -1.0, 0.0, 1.0, 0.0,
            0.0, 1.0, -1.0, 0.0, 0.0, 1.0,
            -1.0, 1.0, -1.0, 1.0, -1.0, 0.0,
            3.0, 1.0, 9.0, 0.0, 0.0, 1.0,
        ); // First three columns must add to 5 (mean is 1)

        let s = HashSet::from([0,1,2]);
        let cutoff_index = 3;

        let f_rec_exp = na::Matrix5x6::<f64>::new(
            1.0, 0.0, 0.0, -1.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
            1.0, 1.0, 2.0, -1.0, 1.0, 2.0, //(1,1,2)
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //(0,0,0)
        );

        let f_rec = get_reconstruction(
            &na::convert(spectra),
            &na::convert(err),
            &s,
            cutoff_index,
            true);
        assert_eq!(f_rec, f_rec_exp);
        }

        #[test]
        fn test_calc_bic() {
            let spectra = na::Matrix5x6::<f64>::new(
                1.0, 0.0, 0.0, -1.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
                1.0, 1.0, 2.0, 1.0, -1.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            );
            let err: na::Matrix5x6<f64> = na::Matrix5x6::zeros().add_scalar(1.0);
            let cutoff_index = 3;

            assert!(
                calc_bic::<f64>(&na::convert(spectra), &na::convert(err), &HashSet::from([0,1,2]), cutoff_index, false)
                < calc_bic::<f64>(&na::convert(spectra), &na::convert(err), &HashSet::from([0]), cutoff_index, false)
            );
            let cutoff_index = 2;
            assert!(
                calc_bic::<f64>(&na::convert(spectra), &na::convert(err), &HashSet::from([0,1]), cutoff_index, false)
                < calc_bic::<f64>(&na::convert(spectra), &na::convert(err), &HashSet::from([0, 3]), cutoff_index, false),
                "BIC for (0, 1, 2) should be less than BIC for (0, 2, 3), but got {} vs {}",
                calc_bic::<f64>(&na::convert(spectra), &na::convert(err), &HashSet::from([0, 1]), cutoff_index, false),
                calc_bic::<f64>(&na::convert(spectra), &na::convert(err), &HashSet::from([0, 3]), cutoff_index, false)
            );

        }
}
