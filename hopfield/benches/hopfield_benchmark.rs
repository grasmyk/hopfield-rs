use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
// Импортируем все функции из вашей библиотеки (замените `your_crate_name` на имя из Cargo.toml)
use hopfield::*;

fn bench_hopfield_multi_n(c: &mut Criterion) {
    let sizes = [512, 1024, 2048, 4096];
    let p = 100;
    let num_seeds = 10;

    let mut group = c.benchmark_group("Hopfield_Comparison");

    for &n in &sizes {
        // Подготовка 50 тестовых наборов для размера N
        let mut f64_test_cases = Vec::with_capacity(num_seeds);
        let mut bit_test_cases = Vec::with_capacity(num_seeds);

        for seed_idx in 0..num_seeds {
            let seed = seed_idx as u64;

            let states_f64 = generate_states(p, n, seed + 1000);
            let weights = weight_matrix_calculate(&states_f64);
            let noisy_f64 = apply_noise(&states_f64[0], 0.05, seed + 2000);

            let packed_patterns: Vec<Vec<u64>> =
                states_f64.iter().map(|pat| pack_bits(pat)).collect();
            let packed_noisy = pack_bits(&noisy_f64);

            f64_test_cases.push((weights, noisy_f64, seed));
            bit_test_cases.push((packed_patterns, packed_noisy, seed));
        }

        // 1. Базовый f64 движок
        group.bench_with_input(BenchmarkId::new("Baseline_f64", n), &n, |b, _| {
            b.iter(|| {
                for (weights, noisy_f64, seed) in &f64_test_cases {
                    let mut state = noisy_f64.clone();
                    neuron_fix(
                        black_box(&mut state),
                        black_box(weights),
                        black_box(*seed),
                        None,
                    );
                }
            })
        });

        // 2. Битовый u64 движок
        group.bench_with_input(BenchmarkId::new("Optimized_u64", n), &n, |b, _| {
            b.iter(|| {
                for (packed_patterns, packed_noisy, seed) in &bit_test_cases {
                    let mut state = packed_noisy.clone();
                    let mut q_overlaps = calculate_initial_overlaps(&state, packed_patterns, n);
                    neuron_fix_bit(
                        black_box(&mut state),
                        black_box(packed_patterns),
                        black_box(&mut q_overlaps),
                        black_box(n),
                        black_box(*seed),
                        None,
                    );
                }
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_hopfield_multi_n);
criterion_main!(benches);
