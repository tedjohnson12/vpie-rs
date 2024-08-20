
use std::collections::HashSet;
use std::result::Result;
use nalgebra as na;

mod bic;
mod coeffs;

pub use bic::bic;
pub use bic::log_likelihood;

pub use coeffs::{get_coeff_mat, get_mmat, get_weights};


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
fn get_reconstruction<T>(
    f_org: &na::DMatrix<T>,
    f_err: &na::DMatrix<T>,
    s: &HashSet<usize>,
    cutoff_index: usize,
    use_mean_error: bool
) -> Result<na::DMatrix<T>, coeffs::MatrixInversionError>
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
        let mmat_result = get_mmat(&weights, &bmat_nir);
        mmat = match mmat_result {
            Ok(_mmat) => _mmat,
            Err(e) =>  return Result::Err(e)
        }
    }
    for i in 0..m {
        if !use_mean_error {
            weights = get_weights(&f_err.row(i).columns(0, cutoff_index).transpose());
            let mmat_result = get_mmat(&weights, &bmat_nir);
            mmat = match mmat_result {
                Ok(_mmat) => _mmat,
                Err(e) =>  return Err(e)
            }
        }
        let row_to_set = f_nir.row(i) * mmat.clone()*bmat_full.clone();
        f_rec.set_row(i, &row_to_set);
    }
    Ok(f_rec)
}

/// Calculate the BIC
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
/// * BIC
pub fn calc_bic<T>(
    f_org: &na::DMatrix<T>,
    f_err: &na::DMatrix<T>,
    s: &HashSet<usize>,
    cutoff_index: usize,
    use_mean_error: bool
)-> Result<T, coeffs::MatrixInversionError>
where
    T: na::RealField
{
    let q = s.len();
    let m = f_org.nrows();
    let _n = cutoff_index;
    let rec_result = get_reconstruction(f_org, f_err, s, cutoff_index, use_mean_error);
    match rec_result {
        Ok(f_rec) => {
            let lnl = bic::log_likelihood(
                f_org.columns(0, cutoff_index).clone_owned(),
                f_rec.columns(0, cutoff_index).clone_owned(),
                f_err.columns(0, cutoff_index).clone_owned());
            
            return Result::Ok(bic::bic(q.try_into().unwrap(), _n.try_into().unwrap(), m.try_into().unwrap(),lnl))
        },
        Err(e) => return Result::Err(e)
    }
    
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
            false).unwrap();
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
            true).unwrap();
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
                calc_bic::<f64>(&na::convert(spectra), &na::convert(err), &HashSet::from([0,1,2]), cutoff_index, false).unwrap()
                < calc_bic::<f64>(&na::convert(spectra), &na::convert(err), &HashSet::from([0]), cutoff_index, false).unwrap()
            );
            let cutoff_index = 2;
            assert!(
                calc_bic::<f64>(&na::convert(spectra), &na::convert(err), &HashSet::from([0,1]), cutoff_index, false).unwrap()
                < calc_bic::<f64>(&na::convert(spectra), &na::convert(err), &HashSet::from([0, 3]), cutoff_index, false).unwrap(),
                "BIC for (0, 1, 2) should be less than BIC for (0, 2, 3), but got {} vs {}",
                calc_bic::<f64>(&na::convert(spectra), &na::convert(err), &HashSet::from([0, 1]), cutoff_index, false).unwrap(),
                calc_bic::<f64>(&na::convert(spectra), &na::convert(err), &HashSet::from([0, 3]), cutoff_index, false).unwrap()
            );

        }

        #[test]
        fn test_get_vpie() {
            let spectra_star = na::Matrix5x4::<f64>::new(
                1.0, 2.0, 1.0, 2.0, // (1, 0)
                -3.0, -1.0, -1.0, 1.0,// (1, -1)
                4.0, 3.0, 2.0, 1.0, // (0, 1)
                2.0, 4.0, 2.0, 4.0,// (2, 0)
                1.0, -0.5, 0.0, -1.5// (-1, 0.5)
            );

            let spectra_planet = na::Matrix5x4::<f64>::new(
                0.0, 0.0, 1.0, 2.0,
                0.0, 0.0, 1.5, 3.0,
                0.0, 0.0, 3.0, 3.5,
                0.0, 0.0, 4.0, 4.0,
                0.0, 0.0, 5.0, 2.0
            );

            let error = na::Matrix5x4::<f64>::zeros().add_scalar(1.0);

            let spectra_total = spectra_planet + spectra_star;
            let s = HashSet::from([0, 2]);
            let cutoff_index = 2;

            let b_of_t = coeffs::get_coeff_mat::<f64>(
                &na::convert(spectra_total.columns(0,cutoff_index).clone_owned()),
                &na::convert(error.columns(0,cutoff_index).clone_owned()),
                &s,
                true
            ).unwrap();

            let basis_total = get_basis_from_vectors::<f64>(&na::convert(spectra_total), &s);
            let basis_planet = get_basis_from_vectors::<f64>(&na::convert(spectra_planet), &s);

            let reconstructed = &b_of_t * basis_total;

            let vpie = spectra_total - reconstructed;
            let vpie_exp = spectra_planet - &b_of_t * basis_planet;
            let res = vpie - vpie_exp;
            assert!(res.abs()< na::Matrix5x4::zeros().add_scalar(1e-10));



        }
}
