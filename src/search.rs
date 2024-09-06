//! # BIC minimization algorithms
//! 
//! There are a couple ways to do this
//! 

use core::f64;
use std::collections::{HashMap, HashSet};
use nalgebra as na;
use ndarray as nd;

use super::{calc_bic,convert_from_ndarray};

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


#[cfg(test)]
mod test {

    use super::*;

    fn approx_planck<D>(wavelen: nd::Array<f64, D>, temp: f64) -> nd::Array<f64, D>
    where
        D: nd::Dimension,
    {
        wavelen.powi(-5) * 1.0 / (
            (1.0/wavelen/temp).exp() - 1.0
        )
    }

    fn meshgrid(a: &nd::Array1<f64>, b: &nd::Array1<f64>) -> (nd::Array2<f64>, nd::Array2<f64>)
    {
        let mut aa = nd::Array2::<f64>::zeros((a.len(), b.len()));
        let mut bb = nd::Array2::<f64>::zeros((a.len(), b.len()));
        for (i, v) in a.iter().enumerate() {
            for (j, w) in b.iter().enumerate() {
                aa[[i, j]] = v.clone();
                bb[[i, j]] = w.clone();
            }
        }
        (aa, bb)
    }

    fn read_data() -> (na::DMatrix<f64>, na::DMatrix<f64>)
    {
        let wl = nd::Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let t = nd::Array1::from_vec(vec![
            -3.0, -2.9, -2.8, -2.7, -2.6, -2.5, -2.4, -2.3, -2.2, -2.1,
            -2.0, -1.9, -1.8, -1.7, -1.6, -1.5, -1.4, -1.3, -1.2, -1.1,
            -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,
            0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
            1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
            2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9,
        ]);
        let spotfrac = t.sin() * 0.1 + 0.3;
        let temp1 = 0.2;
        let temp2 = 0.4;
        let (ww, ss) = meshgrid(&wl, &spotfrac);
        let f_star = approx_planck(ww.clone(), temp1) * ss.clone() + 
            approx_planck(ww.clone(), temp2) * (1.0 - ss.clone());
        let err = (0.000000001*f_star.clone()).sqrt();
        let f_star = convert_from_ndarray::<f64>(f_star);
        let err = convert_from_ndarray::<f64>(err);
        (f_star, err)
    }

    #[test]
    fn test_next_best() {
        let (f_star, err) = read_data();
        let f_org = f_star.clone();
        let f_err = err.clone();
        let cutoff_index = 3;

        let res = agl_next_best(&f_org, &f_err, cutoff_index, false);
        match res {
            SearchResult::Failure => assert!(false, "Failure"),
            SearchResult::Sucess(v,s ) =>{
                print!("v = {}, s = {:?}", v, s);
            }
        }
    }



}