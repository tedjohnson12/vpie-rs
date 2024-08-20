//! # Compute coefficients using Weighted Least Squares
//! 
//! This module is tasked with computing the coefficients $\bm{b}(\bm{t})$ given some set of
//! indices $s'$
//! 

use nalgebra as na;


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
/// Computes the coefficients $\bm{b}(t_i)$
/// 
/// # Arguments
///
/// * `f_org` - Original data ($n \times 1$). This is $\bm{f}(t_i)$
/// * `weights` - Weights ($n \times n$). This is $\mathcal{W}(t_i)$
/// * `bmat` - Basis matrix ($q \times n$). This is $\mathbf{B}$
pub fn get_bi<T>(
    f_org: &na::DVector<T>,
    weights: &na::DMatrix<T>,
    bmat: &na::DMatrix<T>,
) -> na::DVector<T>
where 
    T: na::RealField
{
    let n:usize = f_org.len();
    let q:usize = bmat.nrows();
    assert_eq!(bmat.ncols(), n, "bmat has shape {:?} but should be {:?}", bmat.shape(), (q, n));
    assert_eq!(weights.nrows(), n, "weights has shape {:?} but should be {:?}", weights.shape(), (n, n));
    assert_eq!(weights.ncols(), n, "weights has shape {:?} but should be {:?}", weights.shape(), (n, n));

    (f_org.transpose() * weights*weights * bmat.transpose() * (bmat*weights*weights*bmat.transpose()).try_inverse().unwrap()).transpose()
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
) -> na::DMatrix<T>
where
    T: na::RealField
    {
    let (_q, n) = bmat.shape();
    assert_eq!(weights.nrows(), n, "weights has shape {:?} but should be {:?}", weights.shape(), (n, n));
    assert_eq!(weights.ncols(), n, "weights has shape {:?} but should be {:?}", weights.shape(), (n, n));

    weights * weights * bmat.transpose() * (bmat*weights*weights*bmat.transpose()).try_inverse().unwrap()
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
    fn test_get_bi() {
        let spectra = na::Matrix3::<f64>::new(1.0,1.0,1.0, // 8
                                        1.0, 0.0, 1.0, // 2
                                        1.0, -1.0, 0.0, // 3
                                        );
        let err = na::Vector3::new(1.0, 1.0, 1.0);
        let weights = get_weights::<f64>(&na::convert(err));


        let f_org = na::Vector3::new(13.0, 5.0, 10.0);

        let b = get_bi::<f64>(&na::convert(f_org), &weights, &na::convert(spectra));
        assert_eq!(b, na::Vector3::new(8.0, 2.0, 3.0), "b should be [8, 2, 3], but is {:?}", b);

    }

    #[test]
    fn test_get_mmat() {
        let bmat = na::Matrix3x5::<f64>::new(
            1.0, 0.0, 0.0, -1.0, 0.0,
            0.0, 1.0, 0.0, 0.0, 1.0,
            0.0, 0.0, 1.0, 0.0, 0.0,
        );

        let w = get_weights::<f64>(&na::convert(na::Vector5::new(1.0, 1.0, 1.0, 1.0, 1.0)));
        let m = get_mmat::<f64>(&w, &na::convert(bmat));

        let f = na::Vector5::new(0.0, 0.0, 1.0, 0.0, 0.0);
        assert_eq!(f.transpose(), f.transpose()*m.clone()*bmat.clone(), "mf = f, but is {:?}", m*f);

        let f = na::Vector5::new(1.0, 1.0, 1.0, 1.0, 1.0); // impossible
        assert!(f.transpose()!=f.transpose()*m.clone()*bmat.clone(), "mf = f, but is {:?}", m*f);

        let w = get_weights::<f64>(&na::convert(na::Vector5::new(1.0, 1.0, 1.0, f64::INFINITY, f64::INFINITY)));
        let m = get_mmat::<f64>(&w, &na::convert(bmat));
        let f = na::Vector5::new(1.0, 1.0, 1.0, 1.0, 1.0); // impossible
        let f_exp = na::Vector5::new(1.0, 1.0, 1.0, -1.0, 1.0);
        let b = get_bi::<f64>(&na::convert(f), &w, &na::convert(bmat));
        let b_exp = na::Vector3::new(1.0, 1.0, 1.0);
        assert_eq!(b, b_exp, "b should be [1, 1, 1], but is {:?}", b);
        assert_eq!(f_exp.transpose(), b.clone().transpose()*bmat.clone(), "bB = f, but is {:?}", b.clone().transpose()*bmat.clone());
        assert_eq!(f_exp.transpose(), f.clone().transpose()*m.clone()*bmat.clone(), "fm = f, but is {:?}", m*f);

    }

}