use crate::ensemble::Ensemble;
use anyhow::Result;
use rand::Rng;

#[derive(Debug, Clone, Copy)]
pub(super) struct Coupling {
    node1: usize,
    node2: usize,
    ampl: f64,
}

impl From<&(usize, usize, f64)> for Coupling {
    #[inline(always)]
    fn from(triplet: &(usize, usize, f64)) -> Self {
        Coupling {
            node1: triplet.0,
            node2: triplet.1,
            ampl: triplet.2,
        }
    }
}

impl From<Coupling> for (usize, usize, f64) {
    fn from(coupling: Coupling) -> Self {
        (coupling.node1, coupling.node2, coupling.ampl)
    }
}

impl Coupling {
    #[inline(always)]
    pub(super) fn new(node1: usize, node2: usize, ampl: f64) -> Self {
        Coupling { node1, node2, ampl }
    }

    #[inline(always)]
    pub(super) fn new_random_amplitude(
        node1: usize,
        node2: usize,
        ensemble: &Ensemble,
        rng: &mut impl Rng,
    ) -> Result<Self> {
        Ok(Self::new(node1, node2, ensemble.sample(rng)?))
    }

    #[inline(always)]
    pub(super) fn get_energy(self, config: usize) -> f64 {
        let are_opposite = ((config >> self.node1) & 1) ^ ((config >> self.node2) & 1);
        self.ampl * (2f64 * (are_opposite as f64) - 1f64)
    }
}

#[cfg(test)]
mod tests {
    use super::Coupling;

    fn _test_get_energy(config: usize, ampl: f64, node1: usize, node2: usize, correct_energy: f64) {
        let coupling = Coupling::new(node1, node2, ampl);
        let trial_energy = coupling.get_energy(config);
        assert_eq!(trial_energy, correct_energy);
    }

    #[test]
    fn test_get_energy() {
        _test_get_energy(0b1000100, 1.1, 6, 2, -1.1);
        _test_get_energy(0b1111111111, 1.1, 6, 2, -1.1);
        _test_get_energy(0b1010101, 1.1, 6, 2, -1.1);
        _test_get_energy(0b0000000, 1.1, 6, 2, -1.1);
        _test_get_energy(0b0111000, 1.1, 6, 2, -1.1);
        _test_get_energy(0b0111011, 1.1, 6, 2, -1.1);
        _test_get_energy(0b00, 1.1, 1, 0, -1.1);
        _test_get_energy(0b1000000, 1.1, 6, 2, 1.1);
        _test_get_energy(0b1111011, 1.1, 6, 2, 1.1);
        _test_get_energy(0b000100, 1.1, 6, 2, 1.1);
        _test_get_energy(0b111111, 1.1, 6, 2, 1.1);
        _test_get_energy(0b1111011, 1.1, 6, 2, 1.1);
    }
}
