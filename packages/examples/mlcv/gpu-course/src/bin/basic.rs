#[path = "../compute.rs"]
mod compute;
#[path = "../examples/mod.rs"]
mod examples;

fn main() {
    pollster::block_on(examples::basic::run()).unwrap();
}
