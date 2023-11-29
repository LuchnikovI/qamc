#[inline(always)]
pub(super) fn get_local_energy(config: usize, position: usize, local_field: f64) -> f64 {
    let local_state = (config >> position) & 1;
    (2f64 * (local_state as f64) - 1f64) * local_field
}
