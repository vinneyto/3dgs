mod compute;
mod examples;

fn main() {
    pollster::block_on(examples::basic::run()).unwrap();
}
