use anyhow::Result;
use rand::Rng;
use rand_distr::{Normal, Uniform};

#[derive(Debug, Clone, Copy)]
pub(super) enum Ensemble {
    Normal { mu: f64, std: f64 },
    Uniform { lb: f64, ub: f64 },
    Discrete,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct Edge {
    node1: usize,
    node2: usize,
    ampl: f64,
}

impl Edge {
    #[inline(always)]
    pub(super) fn new(node1: usize, node2: usize, ampl: f64) -> Self {
        Edge { node1, node2, ampl }
    }

    #[inline(always)]
    pub(super) fn new_random_amplitude(
        node1: usize,
        node2: usize,
        ensemble: &Ensemble,
        rng: &mut impl Rng,
    ) -> Result<Self> {
        let ampl = match ensemble {
            Ensemble::Normal { mu, std } => rng.sample(Normal::new(*mu, *std)?),
            Ensemble::Uniform { lb, ub } => rng.sample(Uniform::new(lb, ub)),
            Ensemble::Discrete => (2 * (rng.gen::<u8>() % 2)) as f64 - 1f64,
        };
        Ok(Self::new(node1, node2, ampl))
    }

    #[allow(dead_code)]
    pub(super) fn get_amplitude(&self) -> f64 {
        self.ampl
    }
}
