use rand::Rng;

fn main() {
    let p = 5;
    let n = 10;

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
