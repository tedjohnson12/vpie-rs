//! # Bayesian Information Criterion
//!
//! The BIC is a tool used in model selection, weighting goodness of fit with model complexity.
//! 
//! ```math
//! \text{BIC} = q \ln{(mn)} - 2\ln{\hat{L}}
//! ```
//! 
//! Where $q$ is the number of basis vectors, $n$ is the number of spectral bins, $m$ is the number
//! of temporal bins, and $\hat{L}$ is the model likelihood.

use core::f64;

extern crate nalgebra as na;


/// Computes the Bayesian Information Criterion
/// 
/// # Arguments
/// 
/// * `q` - Number of basis vectors
/// * `n` - Number of spectral bins
/// * `m` - Number of temporal bins
/// * `ln_l` - Log likelihood
/// 
/// # Returns
/// 
/// * BIC
/// 
pub fn bic(
    q: u32,
    n: u32,
    m: u32,
    ln_l: f64
) -> f64 {
    q as f64 * (n as f64 * m as f64).ln() - 2.0 * ln_l
}
/// Computes the log likelihood given a set of data.
/// 
/// # Arguments
/// 
/// * `f_org` - Original data
/// * `f_rec` - Reconstructed data
/// * `f_err` - Noise
/// 
/// # Returns
/// 
/// * Log likelihood
///
pub fn log_likelihood<T, R, C, S>(
    f_org: na::Matrix<T, R, C, S>,
    f_rec: na::Matrix<T, R, C, S>,
    f_err: na::Matrix<T, R, C, S>,
)-> T
where
    T: na::Scalar + na::RealField,
    R: na::Dim,
    C: na::Dim,
    S: na::Storage<T, R, C>,
    na::DefaultAllocator: na::allocator::Allocator<R, C>,
{
    // let (m, n) = f_org.shape();
    // assert_eq!(f_rec.shape(), (m, n), "f_rec has shape {:?} but should be {:?}", f_rec.shape(), (m, n));
    // assert_eq!(f_err.shape(), (m, n), "f_err has shape {:?} but should be {:?}", f_err.shape(), (m, n));

    let difference = f_org - f_rec;

    let ln_l = -T::from_f64(0.5).unwrap() * (
        (f_err.component_mul(&f_err) *  T::two_pi()).map(|x| x.ln()).sum()
        + (difference.component_mul(&difference).component_div(&f_err.component_mul(&f_err))).sum()
    );

    ln_l
}


#[cfg(test)]
mod tests{
    use super::*;

    #[test]
    fn test_bic() {
        assert!(bic(1,2,2,0.0) < bic(1,2,2,-0.1)); // Changing the log likelihood
        assert!(bic(1,2,2,0.0) < bic(2,2,2,0.0)); // Changing the number of basis vectors
    }
    
    #[test]
    fn test_log_likelihood() {
        let f_org = na::Matrix2x3::new(1.1, 1.2, 1.3,
                                                                           2.1, 2.2, 2.3);
        let f_rec = na::Matrix2x3::new(1.0, 1.3, 1.2,
                                                                           2.2, 2.3, 2.3);
        let f_err = na::Matrix2x3::new(0.1, 0.1, 0.1,
                                                                           0.1, 0.2, 0.3);
        let f_bad = na::Matrix2x3::new(3.0, 0.1, 1.5,
                                                                           1.2, 3.3, 4.3);
        
        assert!(
            log_likelihood(f_org,f_org, f_err) > log_likelihood(f_org,f_rec, f_err)
        );
        assert!(
            log_likelihood(f_org,f_bad, f_err) < log_likelihood(f_org,f_rec, f_err)
        );
    }
}