pub fn shift_right_report_u32(a: u32, shift: u32) -> String {
    let shift = shift % 32;
    let r = a >> shift;

    let mut s = String::new();
    s.push_str("Right shift (u32)\n");
    s.push_str("=================\n\n");
    s.push_str(&format_line("a", a));
    s.push_str(&format!("shift = {shift}\n\n"));

    s.push_str("Binary (32-bit)\n");
    s.push_str("--------------\n");
    s.push_str(&format!("a        = {}\n", bin32(a)));
    s.push_str(&format!("a >> {shift:2} = {}\n\n", bin32(r)));

    s.push_str("Result\n");
    s.push_str("------\n");
    s.push_str(&format!("dec: {r}\n"));
    s.push_str(&format!("hex: {}\n", hex(r)));
    s
}

/// Returns true if the k-th bit of `a` is 1.
/// `k=0` is the least significant bit.
pub fn is_bit_set_u32(a: u32, k: u32) -> bool {
    if k >= 32 {
        return false;
    }
    (a & (1u32 << k)) != 0
}

/// Returns `a` with the k-th bit set to 1.
/// `k=0` is the least significant bit.
pub fn set_bit_u32(a: u32, k: u32) -> u32 {
    if k >= 32 {
        return a;
    }
    a | (1u32 << k)
}

/// Hamming distance between two u32 values: number of differing bits.
pub fn hamming_distance_u32(a: u32, b: u32) -> u32 {
    (a ^ b).count_ones()
}

/// Returns all powers of two that sum to `a` (i.e. for each set bit k, includes 2^k).
/// Example: a=13 -> [1,4,8].
pub fn powers_of_two_u32(a: u32) -> Vec<u32> {
    let mut out = Vec::new();
    for k in 0..32 {
        if ((a >> k) & 1) == 1 {
            out.push(1u32 << k);
        }
    }
    out
}

fn format_line(name: &str, v: u32) -> String {
    format!("{name} = {v}  ({})\n", hex(v))
}

fn hex(v: u32) -> String {
    format!("0x{v:08X}")
}

fn bin32(v: u32) -> String {
    // Group by 4 bits: 0000_0000_...
    let mut out = String::with_capacity(32 + 7);
    for i in (0..32).rev() {
        let bit = (v >> i) & 1;
        out.push(if bit == 1 { '1' } else { '0' });
        if i % 4 == 0 && i != 0 {
            out.push('_');
        }
    }
    out
}


