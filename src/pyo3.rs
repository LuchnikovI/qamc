use std::collections::HashMap;
use std::mem::size_of;

use crate::ensemble::Ensemble;
use crate::ising_hamiltonian::IsingHamiltonian;
use anyhow::Result;
use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

#[derive(Debug, Clone)]
#[pyclass]
pub struct IsingsEnsemble(Vec<IsingHamiltonian>);

#[pyfunction]
pub fn get_random_isings_ensemble(
    ensemble_size: usize,
    spins_number: usize,
    couplings_density: f64,
    couplings_ensemble: String,
    couplings_ensemble_parameters: HashMap<String, f64>,
    local_fields_ensemble: String,
    local_fields_ensemble_parameters: HashMap<String, f64>,
) -> PyResult<IsingsEnsemble> {
    let couplings_ensemble: Ensemble =
        (couplings_ensemble.as_str(), &couplings_ensemble_parameters)
            .try_into()
            .map_err(|err| PyErr::new::<PyRuntimeError, _>(format!("{err}")))?;
    let local_fields_ensemble: Ensemble = (
        local_fields_ensemble.as_str(),
        &local_fields_ensemble_parameters,
    )
        .try_into()
        .map_err(|err| PyErr::new::<PyRuntimeError, _>(format!("{err}")))?;
    let ensemble = (0..ensemble_size)
        .map(|_| {
            IsingHamiltonian::new_random(
                spins_number,
                couplings_density,
                &couplings_ensemble,
                &local_fields_ensemble,
            )
        })
        .collect::<Result<Vec<_>>>()
        .map_err(|err| PyErr::new::<PyRuntimeError, _>(format!("{err}")))?;
    Ok(IsingsEnsemble(ensemble))
}

#[pymethods]
impl IsingsEnsemble {
    #[new]
    pub fn new(
        couplings: Vec<Vec<(usize, usize, f64)>>,
        local_fields: Vec<Vec<f64>>,
    ) -> PyResult<Self> {
        let ensemble = couplings
            .iter()
            .zip(local_fields)
            .map(|(c, lf)| IsingHamiltonian::new(c, lf))
            .collect::<Result<Vec<_>>>()
            .map_err(|err| PyErr::new::<PyRuntimeError, _>(format!("{err}")))?;
        Ok(IsingsEnsemble(ensemble))
    }

    pub fn get_diagonals<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray2<f64>> {
        let rows_number = self.0.len();
        let columns_number = 1 << self.0[0].get_spins_number();
        let diagonals = unsafe { PyArray2::new(py, [rows_number, columns_number], false) };
        let start: *mut f64 = unsafe { diagonals.as_array_mut() }.as_mut_ptr();
        let strides = diagonals.strides();
        let (s1, s2) = (
            (strides[0] as usize) / size_of::<f64>(),
            (strides[1] as usize) / size_of::<f64>(),
        );
        for (shift, ising) in self.0.iter().enumerate() {
            let start_shifted = unsafe { start.add(shift * s1) };
            for i in 0..columns_number {
                unsafe { *start_shifted.add(i * s2) = ising.get_energy(i) };
            }
        }
        Ok(diagonals)
    }

    pub fn get_normalizing_factors<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
        let ensemble_size = self.0.len();
        let norm_constants = unsafe { PyArray1::new(py, [ensemble_size], false) };
        let start: *mut f64 = unsafe { norm_constants.as_array_mut() }.as_mut_ptr();
        let stride = (norm_constants.strides()[0] as usize) / size_of::<f64>();
        for (i, ham) in self.0.iter().enumerate() {
            unsafe { *start.add(i * stride) = ham.get_normalizing_factors() };
        }
        norm_constants
    }

    pub fn get_spins_number(&self) -> usize {
        self.0[0].get_spins_number()
    }
}

#[pyfunction]
fn get_local_updates_transition_matrix(matrix: &PyArray2<f64>) -> PyResult<()> {
    let shape = matrix.shape();
    if shape.len() != 2 {
        return Err(PyErr::new::<PyRuntimeError, _>(format!(
            "Array must be a matrix, got an arrau of shape {:?}.",
            shape
        )));
    }
    if shape[0] != shape[1] {
        return Err(PyErr::new::<PyRuntimeError, _>(format!(
            "Matrix must be square, got shape {:?}.",
            shape
        )));
    }
    let size = shape[0];
    let mut spins_number = 0;
    {
        let mut size = size;
        while size & 1 == 0 {
            size >>= 1;
            spins_number += 1;
        }
        if size != 1 {
            return Err(PyErr::new::<PyRuntimeError, _>(format!(
                "Number of columns and rows must be a power of two, get shape {:?}.",
                shape
            )));
        }
    }
    if size != 1 << spins_number {
        return Err(PyErr::new::<PyRuntimeError, _>(format!(
            "Number of columns and rows must be a power of two, get shape {:?}.",
            shape
        )));
    }
    let strides = matrix.strides();
    let stride1 = (strides[0] as usize) / size_of::<f64>();
    let stride2 = (strides[1] as usize) / size_of::<f64>();
    let matrix = unsafe { matrix.as_array_mut().as_mut_ptr() };
    let prob = 1f64 / spins_number as f64;
    for i in 0..size {
        for s in 0..spins_number {
            let j = i ^ (1 << s);
            unsafe { *matrix.add(stride1 * j + stride2 * i) = prob }
        }
    }
    Ok(())
}

#[pymodule]
fn hamiltonian_utils(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_local_updates_transition_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(get_random_isings_ensemble, m)?)?;
    m.add_class::<IsingsEnsemble>()?;
    Ok(())
}
