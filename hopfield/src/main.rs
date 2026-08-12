use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use std::fs::File;
use std::io::Write;

fn main() {
    let n_values = vec![256, 512, 1024, 2048, 4096];
    let num_seeds = 20; 
    let mut main_seed = rand::thread_rng().gen_range(0..10000);

    let mut file = File::create("capacity_results.csv").expect("Не удалось создать CSV файл");
    writeln!(file, "n,alpha,p,seed,overlap,success").unwrap();

    println!("Запуск эксперимента 1");

    for &n in &n_values {
        println!("Расчет для N = {}...", n);

        let mut alpha = 0.10;
        while alpha <= 0.1601 { // 0.1601 из-за погрешности float
            let p = (alpha * n as f64).round() as usize;

            for seed_idx in 0..num_seeds {
                let seed = main_seed + seed_idx;

                //Генерация состояний и расчет весов (используем твои функции)
                let states = generate_states(p, n, seed + 1000);
                let weights = weight_matrix_calculate(&states);

                //Портим 5% нейронов у первого образа
                let target_pattern = states[0].clone();
                let mut noisy_state = target_pattern.clone();

                let mut rng = StdRng::seed_from_u64(seed + 2000);
                let mut indices: Vec<usize> = (0..n).collect();
                indices.shuffle(&mut rng);

                for &i in indices.iter().take(n / 20) {
                    noisy_state[i] *= -1.0;
                }

                //Динамика до остановки (максимум 100 проходов)
                for iter in 0..100 {
                    let prev_state = noisy_state.clone();
                    neuron_fix(&mut noisy_state, &weights, seed + iter as u64 + 3000);
                    
                    // Если ни один нейрон не изменился за проход — останавливаемся
                    if noisy_state == prev_state {
                        break;
                    }
                }

                //Считаем перекрытие m = (1/N) * sum(x_i * y_i)
                let matches_sum: f64 = target_pattern
                    .iter()
                    .zip(noisy_state.iter())
                    .map(|(a, b)| a * b)
                    .sum();

                let overlap = matches_sum / (n as f64);
                let success = if overlap >= 0.95 { 1 } else { 0 };

                //Запись результатов в CSV
                writeln!(
                    file,
                    "{},{:.3},{},{},{:.4},{}",
                    n, alpha, p, seed, overlap, success
                )
                .unwrap();
            }

            alpha += 0.005;
        }
    }

    println!("Готово! Данные сохранены в файл 'capacity_results.csv'");
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

fn neuron_fix(states: &mut Vec<f64>, weight_matrix: &[Vec<f64>], seed: u64) -> usize {
    let n = states.len();
    let mut indices: Vec<usize> = (0..n).collect();

    let mut rng = StdRng::seed_from_u64(seed);
    indices.shuffle(&mut rng);

    let mut changed_count = 0;

    for &i in &indices {
        let mut h_i = 0.0;
        for j in 0..n {
            h_i += weight_matrix[i][j] * states[j];
        }

        let new_val = if h_i > 0.0 {
            1.0
        } else if h_i < 0.0 {
            -1.0
        } else {
            states[i]
        };

        if new_val != states[i] {
            states[i] = new_val;
            changed_count += 1;
        }
    }

    changed_count
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
// В Н1 
fn generate_n(seed: u64) -> usize {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut n = rng.gen_range(500..=1000);
    n
}

fn apply_noise(pattern: &[f64], noise_ratio: f64, seed: u64) -> Vec<f64> {
    let mut noisy_state = pattern.to_vec();
    let n = noisy_state.len();
    let mut rng = StdRng::seed_from_u64(seed);

    let mut indices: Vec<usize> = (0..n).collect();
    indices.shuffle(&mut rng);

    // Считаем количество элементов для инвертирования (например, 0.05 = 5%)
    let flip_count = ((n as f64) * noise_ratio).round() as usize;

    for &idx in indices.iter().take(flip_count) {
        noisy_state[idx] *= -1.0;
    }

    noisy_state
}

#[test]
fn test_fixed_point() {
    let mut file = File::create("test_fixed_point_results.csv").expect("Не удалось создать CSV файл");
    writeln!(file, "n,seed,state_id").unwrap();
    for _ in 0..50 {
        let mut seed = rand::thread_rng().gen_range(0..10000);
        let n = generate_n(seed);
        let p = 10;

        let states = generate_states(p, n, seed + 1000);
        let weights = weight_matrix_calculate(&states);

        for (i, states) in states.iter().enumerate() {
            let mut state = states.clone();
            neuron_fix(&mut state, &weights, seed + 2000);
            writeln!(file, "{},{},{}",n,seed,i).unwrap();
            assert_eq!(
                &state, states,
                "Паттерн {} должен оставаться неподвижной точкой. Непрошедший сид:{}",
                i, seed
            );
        }
    }
}

#[test]
fn test_noise_10() {
    let mut file = File::create("test_noise_10_results.csv").expect("Не удалось создать CSV файл");
    writeln!(file, "seed,state,percent");
    for _ in 0..50 {
        

        let mut seed = rand::thread_rng().gen_range(0..10000);
        let n = generate_n(seed);
        let p = 10;
        let states = generate_states(p, n, seed + 1000);
        let weights = weight_matrix_calculate(&states);
        let mut index = 0;

        for states in &states {
            let mut noisy_state = apply_noise(states, 0.1, seed + 2000);

            // Делаем несколько итераций восстановления

            for iter in 0..5 {
                neuron_fix(&mut noisy_state, &weights, seed + iter + 3000);
            }

            // Считаем % совпадения
            let matches = states
                .iter()
                .zip(noisy_state.iter())
                .filter(|(a, b)| a == b)
                .count();

            let accuracy = matches as f64 / n as f64;
            writeln!(file, "{},{},{},{:.0}", seed, index, n, accuracy*100.00);
            assert!(
                accuracy >= 0.99,
                "Точность восстановления должна быть >= 99%, получили: {:.2}%",
                accuracy * 100.0
            );
            
            index += 1;
        }
    }
}

#[test]
fn low_load_test() {
    let mut file = File::create("low_load_test_results.csv").expect("Не удалось создать CSV файл");
    writeln!(file, "seed");
    for _ in 0..50 {
        let mut seed = rand::thread_rng().gen_range(0..10000);
        let n = 1000;
        let p = 20;
        let states = generate_states(p, n, seed + 1000);
        let weights = weight_matrix_calculate(&states);
        for i in 0..p {
            for states in &states {
                let mut state = states.clone();
                neuron_fix(&mut state, &weights, seed + 2000);
                assert_eq!(
                    &state, states,
                    "При alpha <= 0.02 образ должен быть точной неподвижной точкой"
                );
            }
        }
        writeln!(file, "{}", seed);
    }
}

#[test]
fn basic_drop_test() {
    let mut file = File::create("basic_drop_test_results.csv").expect("Не удалось создать CSV файл");
    writeln!(file, "seed,acc_100,acc_200,");
    for _ in 0..50 {
        let n = 1000;
        // Вспомогательная функция проверки точности при заданном P
        let mut seed = rand::thread_rng().gen_range(0..10000);
        let evaluate_recovery = |p: usize| -> f64 {
            let states = generate_states(p, n, seed + 1000);
            let weights = weight_matrix_calculate(&states);

            let target_pattern = &states[0];
            let mut noisy_state = apply_noise(target_pattern, 0.05, seed + 2000);



            // Запускаем несколько итераций восстановления
            for iter in 0..5 {
                neuron_fix(&mut noisy_state, &weights, seed + iter + 2000);
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
        writeln!(file,"{},{},{}", seed, acc_100, acc_200);
        // При P = 100 восстанавливается хорошо (>= 95%)
        assert!(
            acc_100 >= 0.99,
            "При P=100 образ должен восстановиться, но получилось: {:.2}%",
            acc_100 * 100.0
        );

        // При P = 200 происходит обвал емкости (< 90%)
        assert!(
            acc_200 <= 0.9,
            "При P=200 сеть не должна превышать результат 90%. Результат:{:.2}%",
            acc_200 * 100.0
        );
    }
}

#[test]
fn test_energy() {
    let mut file = File::create("test_energy_results.csv").expect("Не удалось создать CSV файл");
    writeln!(file, "seed");
    for _ in 0..50 {
        let seed: u64 = rand::thread_rng().gen_range(0..10000);
        let n = 1000;
        let p = 20;
        writeln!(file, "{}",seed);
        let states = generate_states(p, n, seed + 1000);
        let weights = weight_matrix_calculate(&states);

        for (idx, state) in states.iter().enumerate() {
            let mut noisy_state = apply_noise(state, 0.05, seed + 2000 + idx as u64);

            for iter in 0..5 {
                // Перенесённая логика neuron_fix с проверкой энергии на каждый переворот бита
                let mut indices: Vec<usize> = (0..n).collect();
                let mut rng = StdRng::seed_from_u64(seed + iter + 3000);
                indices.shuffle(&mut rng);

                let mut changed_count = 0;
                let mut e_before = calculate_energy(&noisy_state, &weights);

                for &i in &indices {
                    let mut h_i = 0.0;
                    for j in 0..n {
                        h_i += weights[i][j] * noisy_state[j];
                    }

                    let new_val = if h_i > 0.0 {
                        1.0
                    } else if h_i < 0.0 {
                        -1.0
                    } else {
                        noisy_state[i]
                    };

                    // Проверяем энергию СРАЗУ же при изменении каждого конкретного бита
                    if new_val != noisy_state[i] {
                        noisy_state[i] = new_val;
                        changed_count += 1;

                        let e_after = calculate_energy(&noisy_state, &weights);
                        assert!(
                            e_before >= e_after,
                            "Ошибка! Энергия выросла при изменении нейрона {}: До = {:.4}, После = {:.4}",
                            i, e_before, e_after
                        );
                        e_before = e_after;
                    }
                }

                // Если за полный проход ни один нейрон не изменился — достигли локального минимума
                if changed_count == 0 {
                    break;
                }
            }
        }
        
    }
}
