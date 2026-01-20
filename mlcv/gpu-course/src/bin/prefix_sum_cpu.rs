#[path = "../compute.rs"]
mod compute;
#[path = "../examples/mod.rs"]
mod examples;

fn main() {
    examples::prefix_sum_cpu::run();
}
