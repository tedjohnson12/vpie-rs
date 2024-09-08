

use std::result;
use std::collections::HashSet;

use pyo3::{pymodule,pyfunction,PyResult, Python, wrap_pyfunction, Py};
use pyo3::types::{PyModule, PySet,};
use pyo3::create_exception;
use pyo3::exceptions::PyRuntimeError;
use numpy as np;
use ndarray as nd;
use nalgebra as na;

mod bic;
mod coeffs;
pub mod search;
pub mod core;


#[pyfunction]
fn search_next_best<'py>(py: Python<'py>,
    f_org: np::PyReadonlyArray2<f64>,
    f_err: np::PyReadonlyArray2<f64>,
    cutoff_index: usize,
    use_mean_error: bool
    ) -> PyResult<&'py PySet> {
        let f_org = f_org.as_array().to_owned();
        let f_org = core::convert_from_ndarray::<f64>(f_org);

        let f_err = f_err.as_array().to_owned();
        let f_err = core::convert_from_ndarray::<f64>(f_err);

        let result = search::agl_next_best(&f_org, &f_err, cutoff_index, use_mean_error);
        match result {
            search::SearchResult::Failure => PyResult::Err(PyRuntimeError::new_err("Failure in search")),
            search::SearchResult::Sucess(_, s) => {
                let set = PySet::new(py, &(s.clone())).unwrap();
                PyResult::Ok(set)
            }
        }

    }

#[pyfunction]
fn get_coeffs<'py>(
    py: Python<'py>,
    f_org_nir: np::PyReadonlyArray2<f64>,
    f_err_nir: np::PyReadonlyArray2<f64>,
    s: Py<PySet>,
    use_mean_error: bool
) -> PyResult<&'py np::PyArray2<f64>> {

    let f_org_nir = f_org_nir.as_array().to_owned();
    let f_org_nir = core::convert_from_ndarray::<f64>(f_org_nir);

    let f_err_nir = f_err_nir.as_array().to_owned();
    let f_err_nir = core::convert_from_ndarray::<f64>(f_err_nir);

    let mut s_rs: HashSet<usize> = HashSet::new();

    let s = s.as_ref(py);
    for x in s.iter() {
        s_rs.insert(x.extract::<usize>().unwrap());
    }

    let res = coeffs::get_coeff_mat::<f64>(&f_org_nir, &f_err_nir, &s_rs, use_mean_error);

    match res {
        Result::Err(e) => PyResult::Err(PyRuntimeError::new_err(format!("Error: {}", e))),
        Result::Ok(x) => {
            let x = core::convert_to_ndarray(x);
            let x = np::PyArray2::from_array(py, &x);
            PyResult::Ok(x)
        }
    }
}
#[pyfunction]
fn get_reconstruction<'py>(
    py: Python<'py>,
    flux: np::PyReadonlyArray2<f64>,
    coeffs: np::PyReadonlyArray2<f64>,
    s: Py<PySet>,
) -> PyResult<&'py np::PyArray2<f64>> {

    let flux = flux.as_array().to_owned();
    let flux = core::convert_from_ndarray::<f64>(flux);

    let coeffs = coeffs.as_array().to_owned();
    let coeffs = core::convert_from_ndarray::<f64>(coeffs);

    let mut s_rs: HashSet<usize> = HashSet::new();

    let s = s.as_ref(py);
    for x in s.iter() {
        s_rs.insert(x.extract::<usize>().unwrap());
    }

    let basis = core::get_basis_from_vectors::<f64>(&flux, &s_rs);

    let reconstruction = coeffs * basis;
    let reconstruction = core::convert_to_ndarray(reconstruction);
    let reconstruction = np::PyArray2::from_array(py, &reconstruction);
    PyResult::Ok(reconstruction)
}


#[pymodule]
fn _vpie_rs<'py>(_py: Python<'py>, m: &PyModule) -> PyResult<()> {

    m.add_function(wrap_pyfunction!(search_next_best, m)?)?;
    m.add_function(wrap_pyfunction!(get_coeffs, m)?)?;
    m.add_function(wrap_pyfunction!(get_reconstruction, m)?)?;
    

    Ok(())
}