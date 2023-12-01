use std::fmt::Display;

use crate::{coupling::Coupling, ensemble::Ensemble, utils::get_local_energy};
use anyhow::Result;
use rand::{thread_rng, Rng};

#[derive(Debug, Clone, Copy)]
enum IsingErr {
    WrongLocalFieldsNumber(usize, usize),
}

impl Display for IsingErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IsingErr::WrongLocalFieldsNumber(spins_num, fields_num) => {
                write!(
                    f,
                    "Number of spins is {spins_num} but number of local fields is {fields_num}"
                )
            }
        }
    }
}

impl std::error::Error for IsingErr {}

#[derive(Debug, Clone)]
pub(super) struct IsingHamiltonian {
    couplings: Vec<Coupling>,
    local_fields: Vec<f64>,
}

impl IsingHamiltonian {
    pub(super) fn new(
        couplings: impl AsRef<[(usize, usize, f64)]>,
        local_fields: impl AsRef<[f64]>,
    ) -> Result<Self> {
        let spins_number = couplings.as_ref().iter().fold(0usize, |acc, x| {
            let pair_max = x.0.max(x.1);
            pair_max.max(acc)
        }) + 1;
        let local_fields_number = local_fields.as_ref().len();
        if spins_number != local_fields_number {
            return Err(IsingErr::WrongLocalFieldsNumber(spins_number, local_fields_number).into());
        }
        let couplings: Vec<Coupling> = couplings
            .as_ref()
            .iter()
            .map(|triplet| triplet.into())
            .collect();
        let local_fields = local_fields.as_ref().to_owned();
        Ok(IsingHamiltonian {
            couplings,
            local_fields,
        })
    }

    pub(super) fn new_random(
        spins_number: usize,
        couplings_density: f64,
        couplings_ensemble: &Ensemble,
        local_fields_ensemble: &Ensemble,
    ) -> Result<Self> {
        let couplings_number = if spins_number != 0 {
            (spins_number * (spins_number - 1)) / 2
        } else {
            0
        };
        let mut rng = thread_rng();
        let mut couplings = Vec::with_capacity(couplings_number);
        for i in 0..spins_number {
            for j in (i + 1)..spins_number {
                if rng.gen::<f64>() < couplings_density {
                    couplings.push(Coupling::new_random_amplitude(
                        i,
                        j,
                        couplings_ensemble,
                        &mut rng,
                    )?);
                }
            }
        }
        let local_fields = (0..spins_number)
            .map(|_| local_fields_ensemble.sample(&mut rng))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            couplings,
            local_fields,
        })
    }

    #[inline(always)]
    pub(super) fn get_energy(&self, config: usize) -> f64 {
        let mut energy = 0f64;
        for coupling in &self.couplings {
            energy += coupling.get_energy(config);
        }
        for (position, local_field) in self.local_fields.iter().enumerate() {
            energy += get_local_energy(config, position, *local_field);
        }
        energy
    }

    #[inline(always)]
    pub(super) fn get_spins_number(&self) -> usize {
        self.local_fields.len()
    }

    #[inline(always)]
    pub(super) fn get_normalizing_factors(
        &self
    ) -> f64 {
        let mut denom_sq = 0f64;
        for coupling in &self.couplings {
            let (_, _, ampl) = (*coupling).into();
            denom_sq += ampl.powi(2);
        }
        for local_field in &self.local_fields {
            denom_sq += (*local_field).powi(2);
        }
        (self.get_spins_number() as f64).sqrt() / denom_sq.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::IsingHamiltonian;
    use crate::ensemble::Ensemble;

    fn _test_new_random(spins_number: usize) {
        let ising = IsingHamiltonian::new_random(
            spins_number,
            1.,
            &Ensemble::Discrete,
            &Ensemble::Discrete,
        )
        .unwrap();
        let couplings_number = if spins_number != 0 {
            (spins_number * (spins_number - 1)) / 2
        } else {
            0
        };
        let mut unique_couplings: HashSet<(usize, usize)> = HashSet::new();
        assert_eq!(ising.local_fields.len(), spins_number);
        assert_eq!(ising.couplings.len(), couplings_number);
        for coupling in ising.couplings {
            let (node1, node2, ampl) = coupling.into();
            assert!((ampl == 1f64) | (ampl == -1f64));
            assert!(node1 < node2);
            assert!(node2 < spins_number);
            let newly_inserted = unique_couplings.insert((node1, node2));
            if !newly_inserted {
                panic!("Duplicated coupling")
            }
        }
        let ising = IsingHamiltonian::new_random(
            spins_number,
            1.,
            &Ensemble::Uniform { lb: -0.5, ub: 1.5 },
            &Ensemble::Uniform { lb: -0.5, ub: 1.5 },
        )
        .unwrap();
        let couplings_number = if spins_number != 0 {
            (spins_number * (spins_number - 1)) / 2
        } else {
            0
        };
        let mut unique_couplings: HashSet<(usize, usize)> = HashSet::new();
        assert_eq!(ising.local_fields.len(), spins_number);
        assert_eq!(ising.couplings.len(), couplings_number);
        for coupling in ising.couplings {
            let (node1, node2, ampl) = coupling.into();
            assert!((ampl > -0.5f64) | (ampl < 1.5f64));
            assert!(node1 < node2);
            assert!(node2 < spins_number);
            let newly_inserted = unique_couplings.insert((node1, node2));
            if !newly_inserted {
                panic!("Duplicated coupling")
            }
        }
        let ising = IsingHamiltonian::new_random(
            spins_number,
            0.,
            &Ensemble::Discrete,
            &Ensemble::Discrete,
        )
        .unwrap();
        assert_eq!(ising.local_fields.len(), spins_number);
        assert_eq!(ising.couplings.len(), 0);
    }

    #[test]
    fn test_new_random() {
        _test_new_random(0);
        _test_new_random(1);
        _test_new_random(10);
        _test_new_random(111);
    }

    fn _test_get_energy(ising: &IsingHamiltonian, config: usize, correct_energy: f64) {
        let trial_energy = ising.get_energy(config);
        assert_eq!(trial_energy, correct_energy);
    }

    #[test]
    fn test_get_energy() {
        let ising = IsingHamiltonian::new(
            [
                (0, 1, 1.0),
                (0, 2, -1.0),
                (1, 2, -2.0),
                (1, 3, 3.0),
                (2, 3, 1.5),
            ],
            [0., -1.3, 1.3, 0.],
        )
        .unwrap();
        _test_get_energy(&ising, 0b0000, -2.5);
        _test_get_energy(&ising, 0b1111, -2.5);
        _test_get_energy(&ising, 0b1010, -4.1);
    }
}
