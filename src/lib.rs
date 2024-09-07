

use std::result;

use pyo3::{pymodule,pyfunction,PyResult, Python, wrap_pyfunction};
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


#[pymodule]
fn _vpie_rs<'py>(_py: Python<'py>, m: &PyModule) -> PyResult<()> {

    m.add_function(wrap_pyfunction!(search_next_best, m)?)?;
    

    Ok(())
}