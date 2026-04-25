use std::time::Instant;

pub fn run() {
    let n: usize = 100 * 1000 * 1000;
    let mut input = vec![0u32; n];
    let mut total_sum: u64 = 0;
    for i in 0..n {
        let v = (3 * (i as u64 + 5) + 7) % 17;
        input[i] = v as u32;
        total_sum += v;
        assert!(
            total_sum <= u32::MAX as u64,
            "prefix sum overflow at i={}",
            i
        );
    }

    let mut cpu_prefix_sum = vec![0u32; n];
    let start = Instant::now();
    let mut cpu_sum: u64 = 0;
    for i in 0..n {
        cpu_prefix_sum[i] = cpu_sum as u32;
        cpu_sum += input[i] as u64;
    }
    let elapsed = start.elapsed().as_secs_f64();
    println!("cpu prefix sum time (exclusive, seconds): {}", elapsed);
    let cpu_memory_gb = (std::mem::size_of::<u32>() as f64 * 2.0 * n as f64)
        / 1024.0
        / 1024.0
        / 1024.0;
    println!(
        "cpu effective bandwidth estimate: {} GB/s",
        cpu_memory_gb / elapsed
    );
}
