use crate::edge::{Edge, Ensemble};
use anyhow::Result;
use rand::{thread_rng, Rng};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

#[derive(Debug, Clone)]
pub(super) struct GraphSet(Vec<Vec<Edge>>);

impl GraphSet {
    pub(super) fn new_random(
        graphs_number: usize,
        graph_size: usize,
        edges_density: f64,
        ensemble: &Ensemble,
    ) -> Result<Self> {
        let edges_number = if graph_size != 0 {
            (graph_size * (graph_size - 1)) / 2
        } else {
            0
        };
        let graphs = (0..graphs_number)
            .into_par_iter()
            .map(|_| {
                let mut rng = thread_rng();
                let mut graph = Vec::with_capacity(edges_number);
                for i in 0..graph_size {
                    for j in (i + 1)..graph_size {
                        if rng.gen::<f64>() < edges_density {
                            graph.push(Edge::new_random_amplitude(i, j, ensemble, &mut rng)?);
                        }
                    }
                }
                Ok(graph)
            })
            .collect::<Result<Vec<Vec<_>>>>()?;
        Ok(GraphSet(graphs))
    }
}

#[cfg(test)]
mod tests {
    use super::GraphSet;

    fn _test_new_random(graphs_number: usize, graph_size: usize) {
        let max_edges_number = if graph_size != 0 {
            (graph_size * (graph_size - 1)) / 2
        } else {
            0
        };
        let graphs = GraphSet::new_random(
            graphs_number,
            graph_size,
            1.,
            &crate::edge::Ensemble::Discrete,
        )
        .unwrap();
        assert_eq!(graphs.0.len(), graphs_number);
        for graph in graphs.0 {
            assert_eq!(graph.len(), max_edges_number);
            for edge in graph {
                assert!((edge.get_amplitude() == 1f64) || (edge.get_amplitude() == -1f64));
            }
        }
        let graphs = GraphSet::new_random(
            graphs_number,
            graph_size,
            1.,
            &crate::edge::Ensemble::Uniform { lb: -0.5, ub: 1.5 },
        )
        .unwrap();
        assert_eq!(graphs.0.len(), graphs_number);
        for graph in graphs.0 {
            assert_eq!(graph.len(), max_edges_number);
            for edge in graph {
                assert!((edge.get_amplitude() > -0.5) || (edge.get_amplitude() < -1.5));
            }
        }
        let graphs = GraphSet::new_random(
            graphs_number,
            graph_size,
            0.,
            &crate::edge::Ensemble::Discrete,
        )
        .unwrap();
        assert_eq!(graphs.0.len(), graphs_number);
        for graph in graphs.0 {
            assert!(graph.is_empty())
        }
    }

    #[test]
    fn test_new_random() {
        _test_new_random(0, 0);
        _test_new_random(0, 1);
        _test_new_random(1, 0);
        _test_new_random(1, 1);
        _test_new_random(5, 0);
        _test_new_random(0, 5);
        _test_new_random(5, 1);
        _test_new_random(1, 5);
        _test_new_random(5, 11);
        _test_new_random(11, 5);
    }
}
