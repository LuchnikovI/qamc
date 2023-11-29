use anyhow::{Error, Result};
use rand::Rng;
use rand_distr::{Normal, Uniform};
use std::collections::HashMap;
use std::fmt::Display;

#[derive(Debug, Clone)]
enum EnsembleErr {
    NoMu,
    NoStd,
    NoUB,
    NoLB,
    UnknownEnsemble(String),
}

impl Display for EnsembleErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EnsembleErr::NoMu => {
                write!(f, "Mean value has not been found among ensemble parameters")
            }
            EnsembleErr::NoStd => write!(
                f,
                "Standard deviation has not been found among ensemble parameters"
            ),
            EnsembleErr::NoUB => write!(
                f,
                "Upper bound has not been found among ensemble parameters"
            ),
            EnsembleErr::NoLB => write!(
                f,
                "Lower bound has not been found among ensemble parameters"
            ),
            EnsembleErr::UnknownEnsemble(s) => write!(f, "Unknown ensemble {s}"),
        }
    }
}

impl std::error::Error for EnsembleErr {}

#[derive(Debug, Clone, Copy)]
pub(super) enum Ensemble {
    Normal { mu: f64, std: f64 },
    Uniform { lb: f64, ub: f64 },
    Discrete,
}

impl TryFrom<(&str, &HashMap<String, f64>)> for Ensemble {
    type Error = Error;
    fn try_from(pair: (&str, &HashMap<String, f64>)) -> Result<Self> {
        match pair.0 {
            "normal" => {
                let mu = *pair.1.get("mu").ok_or(EnsembleErr::NoMu)?;
                let std = *pair.1.get("std").ok_or(EnsembleErr::NoStd)?;
                Ok(Ensemble::Normal { mu, std })
            }
            "uniform" => {
                let lb = *pair.1.get("lb").ok_or(EnsembleErr::NoLB)?;
                let ub = *pair.1.get("ub").ok_or(EnsembleErr::NoUB)?;
                Ok(Ensemble::Uniform { lb, ub })
            }
            "discrete" => Ok(Ensemble::Discrete),
            other => Err(EnsembleErr::UnknownEnsemble(other.to_owned()).into()),
        }
    }
}

impl Ensemble {
    #[inline(always)]
    pub(super) fn sample(&self, rng: &mut impl Rng) -> Result<f64> {
        let val = match self {
            Ensemble::Normal { mu, std } => rng.sample(Normal::new(*mu, *std)?),
            Ensemble::Uniform { lb, ub } => rng.sample(Uniform::new(lb, ub)),
            Ensemble::Discrete => (2 * (rng.gen::<u8>() % 2)) as f64 - 1f64,
        };
        Ok(val)
    }
}
