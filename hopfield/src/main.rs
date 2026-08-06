use rand::Rng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand::rngs::StdRng;

fn main() {
    let p = 10;
    let n = 1000;

    let mut generated_patterns: Vec<Vec<f64>> = generate_patterns(p, n);

    println!("Сгенерированные паттерны:");
    for (i, pat) in generated_patterns.iter().enumerate() {
        println!("Паттерн {}: {:?}", i + 1, pat);
    }

    let weight_matrix = weight_matrix_calculate(&generated_patterns);
    
    println!("\nМатрица весов ({}x{}):", n, n);
    for row in &weight_matrix {
        println!("{:?}", row);
    }

    let mut state = generated_patterns[0].clone();
    println!("Исходный state:     {:?}", state);

    // 2. Искажаем state (инвертируем 1-й элемент)
    state[0] *= -1.0;
    println!("Испорченный state:  {:?}", state);

    // 3. Вызываем функцию восстановления (seed = 42)
    neuron_fix(&mut state, &weight_matrix, 42);
    println!("Восстановленный state: {:?}", state);

    // 4. Проверяем, совпал ли результат с оригиналом
    if state == generated_patterns[0] {
        println!("Успех: сеть восстановила образ!");
    } else {
        println!("Ошибка: сеть не смогла восстановить образ.");
    }
}

fn generate_patterns(p: usize, n: usize) -> Vec<Vec<f64>> {
    let mut rng = rand::thread_rng();
    let mut patterns = Vec::with_capacity(p);

    for _ in 0..p {
        let mut pattern = Vec::with_capacity(n);
        for _ in 0..n {
            let val = if rng.gen_bool(0.5) { 1.0 } else { -1.0 };
            pattern.push(val);
        }
        patterns.push(pattern);
    }

    return patterns;
}

fn weight_matrix_calculate(patterns: &[Vec<f64>]) -> Vec<Vec<f64>> {
    // Если нет паттернов, возвращаем пустую матрицу
    if patterns.is_empty() {
        return vec![];
    }

    let n = patterns[0].len(); // Длина одного паттерна (количество нейронов N)
    let mut weight_matrix = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in 0..n {
            if i == j {
                weight_matrix[i][j] = 0.0;
            } else {
                let mut sum = 0.0;
                for pattern in patterns {
                    sum += pattern[i] * pattern[j];
                }
                weight_matrix[i][j] = sum / (n as f64);
            }
        }
    }

    return weight_matrix;
}

fn neuron_fix(states:& mut Vec<f64>, weight_matrix:&[Vec<f64>], seed:u64){
    let mut n = states.len();
    let mut indices_of_states:Vec<usize> = (0..n).collect();

    let mut rng = StdRng::seed_from_u64(seed);
    indices_of_states.shuffle(&mut rng); 

    for &i in &indices_of_states {
        let mut h_i = 0.0;
        for j in 0..n {
            h_i += weight_matrix[i][j] * states[j];
        }
        states[i] = if h_i >= 0.0 { 1.0 } else { -1.0 };
    }
}