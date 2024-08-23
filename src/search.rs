//! # BIC minimization algorithms
//! 
//! There are a couple ways to do this
//! 

use core::f64;
use std::collections::{HashMap, HashSet};
use nalgebra as na;

use super::calc_bic;

pub enum SearchResult<T> {
    Sucess(T, HashSet<usize>),
    Failure,
}


/// Search for the next best basis vector
/// 
/// This starts with $q=1$ and works up.
/// 
/// # Arguments
/// 
/// * `f_org` - Original spectra ($m \times n$)
/// * `f_err` - Uncertainty in the data ($m \times n$)
/// * `cutoff_index` - The number of NIR wavelength points to use in training ($n'$)
/// * `use_mean_error` - Whether to use the mean error in the reconstruction
/// 
/// # Returns
/// 
/// * SearchResult
pub fn agl_next_best<T>(
    f_org: &na::DMatrix<T>,
    f_err: &na::DMatrix<T>,
    cutoff_index: usize,
    use_mean_error: bool
) -> SearchResult<T>
where
    T: na::RealField
{
    let m = f_org.nrows();
    let mut s_best = HashSet::<usize>::new();
    let mut val_best = T::from_f64(f64::INFINITY).unwrap();
    for _ in 1..m-1 {
        let mut candidates = HashMap::<usize, T>::new();
        let s_base = s_best.clone();
        for i in 0..m {
            if s_base.contains(&i){
                continue;
            }
            else{
                let mut s_temp = s_base.clone();
                s_temp.insert(i);
                let val_result = calc_bic(f_org, f_err, &s_temp, cutoff_index, use_mean_error);
                match val_result {
                    Err(_) => continue,
                    Ok(val) => {
                        if val < val_best {
                            candidates.insert(i, val.clone());
                            val_best = val;
                        }
                    }
                }
            }
        }
        if candidates.is_empty() {
            return SearchResult::Sucess(val_best, s_best);
        }
        else {
            let i_best = candidates.iter().find_map(|(key, _val)| if _val==&val_best {Some(key)} else {None}).unwrap();
            s_best.insert(i_best.clone());
        }
    }
    return SearchResult::Failure;
}