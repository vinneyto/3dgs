#[path = "../compute.rs"]
mod compute;
#[path = "../examples/mod.rs"]
mod examples;

fn main() {
    pollster::block_on(examples::prefix_sum_gpu::run()).unwrap();
}
