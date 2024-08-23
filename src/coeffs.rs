//! # Compute coefficients using Weighted Least Squares
//! 
//! This module is tasked with computing the coefficients $\bm{b}(\bm{t})$ given some set of
//! indices $s'$
//! 
use std::collections::HashSet;
use std::result::Result;
use nalgebra as na;

#[derive(Debug)]
pub struct MatrixInversionError;

impl std::fmt::Display for MatrixInversionError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Matrix inversion failed")
    }
}

impl std::error::Error for MatrixInversionError {}


/// Computes weights based on the uncertainty in the data.
/// 
/// # Arguments
/// 
/// * `f_err` - Uncertainty in the data ($n \times 1$)
/// 
/// # Returns
/// 
/// * Weights ($n \times n$)
pub fn get_weights<T>(
    f_err: &na::DVector<T>
) -> na::DMatrix<T>
where
    T: na::Scalar+na::RealField,
    {
    if f_err.iter().any(|x| x == &T::zero()) {
        panic!("f_err contains zero");
    }
    let diag = na::DVector::<T>::repeat(f_err.len(), T::one()).component_div(f_err);
    na::Matrix::from_diagonal(&diag)
}

/// Computes the matrix $\mathcal{M}$
/// 
/// # Arguments
/// 
/// * `weights` - Weights ($n \times n$). This is $\mathcal{W}(t_i)$
/// * `bmat` - Basis matrix ($q \times n$). This is $\mathbf{B}$
/// 
/// # Returns
/// 
/// * Matrix ($n \times n$)
/// 
/// # Notes
/// 
/// This matrix can be used to compute $\tilde{\bm{f}}$
pub fn get_mmat<T>(
    weights: &na::DMatrix<T>,
    bmat: &na::DMatrix<T>,
) ->  Result<na::DMatrix<T>, MatrixInversionError>
where
    T: na::RealField
    {
    let (_q, n) = bmat.shape();
    assert_eq!(weights.nrows(), n, "weights has shape {:?} but should be {:?}", weights.shape(), (n, n));
    assert_eq!(weights.ncols(), n, "weights has shape {:?} but should be {:?}", weights.shape(), (n, n));

    let inv_res = (bmat*weights*weights*bmat.transpose()).try_inverse();
    
    match inv_res {
        Some(x) => Result::Ok(weights * weights * bmat.transpose() * x),
        None => Result::Err(MatrixInversionError),
    }
}

fn get_coeff_mat_mean_error<T>(
    f_org_nir: &na::DMatrix<T>,
    f_err_nir: &na::DMatrix<T>,
    s: &HashSet<usize>,
) -> Result<na::DMatrix<T>, MatrixInversionError>
where
    T: na::RealField
    {
        let q = s.len();
        let m = f_org_nir.nrows();
        let mut coeff_mat = na::DMatrix::<T>::zeros(m, q);
        let bmat = super::get_basis_from_vectors(f_org_nir, s);
        let weights = get_weights(&f_err_nir.row_mean().transpose());

        let mmat_res = get_mmat(&weights, &bmat);

        match mmat_res {
            Err(e) => return Result::Err(e),
            Ok(mmat) => {
                for i in 0..m {
                    let bi = f_org_nir.row(i) * &mmat;

                    coeff_mat.set_row(i, &bi);
                }
                return Ok(coeff_mat)
            }
        }

    }

fn get_coeff_mat_true_error<T>(
    f_org_nir: &na::DMatrix<T>,
    f_err_nir: &na::DMatrix<T>,
    s: &HashSet<usize>,
) -> Result<na::DMatrix<T>, MatrixInversionError>
where
    T: na::RealField
    {
        let q = s.len();
        let m = f_org_nir.nrows();
        let mut coeff_mat = na::DMatrix::<T>::zeros(m, q);
        let bmat = super::get_basis_from_vectors(f_org_nir, s);
        
        for i in 0..m {
            let weights = get_weights(&f_err_nir.row(i).transpose());
            let mmat_res = get_mmat(&weights, &bmat);
            match mmat_res {
                Err(e) => return Result::Err(e),
                Ok(mmat) => {
                    let bi = f_org_nir.row(i) * &mmat;
                    coeff_mat.set_row(i, &bi);
                }
            }
        }

        return Ok(coeff_mat)
    }


/// Computes the matrix $\bm{b}(\bm{t})$
/// 
/// This matrix can be multiplied by the basis to get the reconstructed spectra
/// 
/// # Arguments
/// 
/// * `f_org_nir` - NIR spectra ($m \times n'$)
/// * `f_err_nir` - Uncertainty in the data ($m \times n'$)
/// * `s` - Set of indices (length $q$)
/// * `use_mean_error` - Whether to use the mean error in the reconstruction (faster)
/// 
/// # Returns
/// 
/// * $\bm{b}(\bm{t})$ ($m \times q$)
pub fn get_coeff_mat<T>(
    f_org_nir: &na::DMatrix<T>,
    f_err_nir: &na::DMatrix<T>,
    s: &HashSet<usize>,
    use_mean_error: bool
) -> Result<na::DMatrix<T>, MatrixInversionError>
where 
    T: na::RealField
    {
        if use_mean_error {
            get_coeff_mat_mean_error(f_org_nir, f_err_nir, s)
        } else {
            get_coeff_mat_true_error(f_org_nir, f_err_nir, s)
        }
    }



#[cfg(test)]
mod tests {
    use core::f64;

    use super::*;

    #[test]
    fn test_get_weights() {
        let s = na::Vector3::new(1.0, 2.0, 4.0);
        let s =na::convert(s);
        

        let w = get_weights(&s);
        let w_exp = na::Matrix3::new(1.0, 0.0, 0.0,
                                    0.0, 0.5, 0.0,
                                    0.0, 0.0, 0.25);
        assert_eq!(w, w_exp);
    }

    #[test]
    fn test_get_mmat() {
        let bmat = na::Matrix3x5::<f64>::new(
            1.0, 0.0, 0.0, -1.0, 0.0,
            0.0, 1.0, 0.0, 0.0, 1.0,
            0.0, 0.0, 1.0, 0.0, 0.0,
        );

        let w = get_weights::<f64>(&na::convert(na::Vector5::new(1.0, 1.0, 1.0, 1.0, 1.0)));
        let m = get_mmat::<f64>(&w, &na::convert(bmat)).unwrap();

        let f = na::Vector5::new(0.0, 0.0, 1.0, 0.0, 0.0);
        assert_eq!(f.transpose(), f.transpose()*m.clone()*bmat.clone(), "mf = f, but is {:?}", m*f);

        let f = na::Vector5::new(1.0, 1.0, 1.0, 1.0, 1.0); // impossible
        assert!(f.transpose()!=f.transpose()*m.clone()*bmat.clone(), "mf = f, but is {:?}", m*f);

        let w = get_weights::<f64>(&na::convert(na::Vector5::new(1.0, 1.0, 1.0, f64::INFINITY, f64::INFINITY)));
        let m = get_mmat::<f64>(&w, &na::convert(bmat)).unwrap();
        let f = na::Vector5::new(1.0, 1.0, 1.0, 1.0, 1.0); // impossible
        let f_exp = na::Vector5::new(1.0, 1.0, 1.0, -1.0, 1.0);
        assert_eq!(f_exp.transpose(), f.clone().transpose()*m.clone()*bmat.clone(), "fm = f, but is {:?}", m*f);

    }

    #[test]
    fn test_get_coeff_mat() {
        let spectra = na::Matrix5x3::<f64>::new(
            1.0, 0.0, 0.0,
            0.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
            1.0, -1.0, -1.0,
            0.0, 0.0, 0.0
        );
        let s = HashSet::from([0, 1]);

        let err = na::Matrix5x3::<f64>::zeros().add_scalar(1.0);

        let coeffs = get_coeff_mat::<f64>(&na::convert(spectra), &na::convert(err), &s, true).unwrap();

        let coeffs_exp = na::Matrix5x2::<f64>::new(
            1.0, 0.0,
            0.0, 1.0,
            1.0, 1.0,
            1.0, -1.0,
            0.0, 0.0,
        );
        assert_eq!(coeffs, coeffs_exp)

    }

}