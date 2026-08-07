use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

fn main() {
    let p = 10;
    let n = 1000;

    let mut seed = rand::thread_rng().gen_range(0..10000);
    let mut generated_states: Vec<Vec<f64>> = generate_states(p, n, seed);
    let weight_matrix = weight_matrix_calculate(&generated_states);
    let mut state = generated_states[0].clone();

    // 2. Искажаем state (инвертируем 1-й элемент)
    state[0] *= -1.0;

    // 3. Вызываем функцию восстановления (seed = 42)

    let e_before = calculate_energy(&state, &weight_matrix);

    println!("Енергия до: {:.4}", e_before);

    neuron_fix(&mut state, &weight_matrix, 42);

    let e_after = calculate_energy(&state, &weight_matrix);

    println!("Енергия после: {:.4}", e_after);

    // 4. Проверяем, совпал ли результат с оригиналом
    if state == generated_states[0] {
        println!("Успех: сеть восстановила образ!");
    } else {
        println!("Ошибка: сеть не смогла восстановить образ.");
    }
}

fn generate_states(p: usize, n: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut states_vector = Vec::with_capacity(p);

    for _ in 0..p {
        let mut states = Vec::with_capacity(n);
        for _ in 0..n {
            let val = if rng.gen_bool(0.5) { 1.0 } else { -1.0 };
            states.push(val);
        }
        states_vector.push(states);
    }

    return states_vector;
}

fn weight_matrix_calculate(states: &[Vec<f64>]) -> Vec<Vec<f64>> {
    // Если нет паттернов, возвращаем пустую матрицу
    if states.is_empty() {
        return vec![];
    }

    let n = states[0].len(); // Длина одного паттерна (количество нейронов N)
    let mut weight_matrix = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in 0..n {
            if i == j {
                weight_matrix[i][j] = 0.0;
            } else {
                let mut sum = 0.0;
                for states in states {
                    sum += states[i] * states[j];
                }
                weight_matrix[i][j] = sum / (n as f64);
            }
        }
    }

    return weight_matrix;
}

fn neuron_fix(states: &mut Vec<f64>, weight_matrix: &[Vec<f64>], seed: u64) {
    let mut n = states.len();
    let mut indices_of_states: Vec<usize> = (0..n).collect();

    let mut rng = StdRng::seed_from_u64(seed);
    indices_of_states.shuffle(&mut rng);

    for &i in &indices_of_states {
        let mut h_i = 0.0;
        for j in 0..n {
            h_i += weight_matrix[i][j] * states[j];
        }

        if h_i > 0.0 {
            states[i] = 1.0
        } else if h_i < 0.0 {
            states[i] = -1.0
        } else {
            states[i] = states[i]
        }
    }
}

fn calculate_energy(states: &Vec<f64>, weight_matrix: &[Vec<f64>]) -> f64 {
    let n = states.len();
    let mut energy_sum = 0.0;

    for i in 0..n {
        for j in 0..n {
            energy_sum += weight_matrix[i][j] * states[i] * states[j];
        }
    }

    return -0.5 * energy_sum;
}

fn generate_n(seed: u64) -> usize {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut n = rng.gen_range(500..=1000);
    n
}

#[test]
fn test_fixed_point() {
    for _ in 0..50 {
        let mut seed = rand::thread_rng().gen_range(0..10000);
        let n = generate_n(seed);
        let p = 10;

        let states = generate_states(p, n, seed);
        let weights = weight_matrix_calculate(&states);

        for (i, states) in states.iter().enumerate() {
            let mut state = states.clone();
            neuron_fix(&mut state, &weights, seed);
            assert_eq!(
                &state, states,
                "Паттерн {} должен оставаться неподвижной точкой",
                i
            );
        }
    }
}

#[test]
fn test_noise_10() {
    for _ in 0..50 {
        let mut seed = rand::thread_rng().gen_range(0..10000);
        let n = generate_n(seed);
        let p = 10;
        let states = generate_states(p, n, seed);
        let weights = weight_matrix_calculate(&states);

        for states in &states {
            let mut noisy_state = states.clone();

            let mut rng = StdRng::seed_from_u64(seed);

            // Портим ровно 10% нейронов (100 из 1000)
            let mut indices: Vec<usize> = (0..n).collect();
            indices.shuffle(&mut rng);
            for &i in indices.iter().take(n / 10) {
                noisy_state[i] *= -1.0;
            }

            // Делаем несколько итераций восстановления

            for iter in 0..5 {
                neuron_fix(&mut noisy_state, &weights, seed + iter);
            }

            // Считаем % совпадения
            let matches = states
                .iter()
                .zip(noisy_state.iter())
                .filter(|(a, b)| a == b)
                .count();

            let accuracy = matches as f64 / n as f64;
            assert!(
                accuracy >= 0.99,
                "Точность восстановления должна быть >= 99%, получили: {:.2}%",
                accuracy * 100.0
            );
        }
    }
}

#[test]
fn low_load() {
    for _ in 0..50 {
        let mut seed = rand::thread_rng().gen_range(0..10000);
        let n = 1000;
        let p = 20;
        let states = generate_states(p, n, seed);
        let weights = weight_matrix_calculate(&states);
        for i in 0..p {
            let mut rng = StdRng::seed_from_u64(seed);

            for states in &states {
                let mut state = states.clone();
                neuron_fix(&mut state, &weights, seed);
                assert_eq!(
                    &state, states,
                    "При alpha <= 0.02 образ должен быть точной неподвижной точкой"
                );
            }
        }
    }
}

#[test]
fn basic_drop_test() {
    for _ in 0..50 {
        let n = 1000;

        // Вспомогательная функция проверки точности при заданном P
        let mut seed = rand::thread_rng().gen_range(0..10000);
        let evaluate_recovery = |p: usize| -> f64 {
            let states = generate_states(p, n, seed);
            let weights = weight_matrix_calculate(&states);

            let target_pattern = &states[0];
            let mut noisy_state = target_pattern.clone();

            // Портим ровно 5% нейронов (50 из 1000)
            let mut rng = StdRng::seed_from_u64(seed);
            let mut indices: Vec<usize> = (0..n).collect();
            indices.shuffle(&mut rng);
            for &i in indices.iter().take(n / 20) {
                noisy_state[i] *= -1.0;
            }

            // Запускаем несколько итераций восстановления
            for iter in 0..5 {
                neuron_fix(&mut noisy_state, &weights, seed + iter);
            }

            // Считаем долю совпавших нейронов
            let matches = target_pattern
                .iter()
                .zip(noisy_state.iter())
                .filter(|(a, b)| a == b)
                .count();

            matches as f64 / n as f64
        };

        let acc_100 = evaluate_recovery(100);
        let acc_200 = evaluate_recovery(200);

        // При P = 100 восстанавливается хорошо (>= 95%)
        assert!(
            acc_100 >= 0.99,
            "При P=100 образ должен восстановиться, но получилось: {:.2}%",
            acc_100 * 100.0
        );

        // При P = 200 происходит обвал емкости (< 90%)
        assert!(
            acc_200 < acc_100,
            "При P=200 сеть не должна превышать результат при acc_100. Результаты: {:.2} - acc_100, {:.2} - acc_200",
            acc_100 * 100.0,
            acc_200 * 100.0
        );
    }
}

#[test]
fn test_energy() {
    for _ in 0..50 {
        let mut seed = rand::thread_rng().gen_range(0..10000);
        let n = 1000;
        let p = 20;
        let states = generate_states(p, n, seed);
        let weights = weight_matrix_calculate(&states);
        for (idx, state) in states.iter().enumerate() {
            let mut noisy_state = state.clone();
            let mut rng = StdRng::seed_from_u64(seed + idx as u64);
            let mut indices: Vec<usize> = (0..n).collect();
            indices.shuffle(&mut rng);
            for &i in indices.iter().take(n / 5) {
                noisy_state[i] *= -1.0;
            }
            let mut e_before = calculate_energy(&noisy_state, &weights);
            for iter in 0..5 {
                neuron_fix(&mut noisy_state, &weights, seed + iter);
                let e_after = calculate_energy(&noisy_state, &weights);
                assert!(
                    e_before >= e_after,
                    "Енергия должна всегда уменьшаться: Енергия до: {:.2}, Енергия после: {:.2}",
                    e_before,
                    e_after
                );
                e_before = e_after;
            }
        }
    }
}
