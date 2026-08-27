use hopfield::{
    apply_noise, calculate_energy, calculate_overlap, generate_states,
    neuron_fix, weight_matrix_calculate,
};

fn main() {
    println!("==================================================");
    println!("   Демонстрация работы сети Хопфилда (Hopfield)   ");
    println!("==================================================\n");

    let n = 200; // Количество нейронов N
    let p = 5;   // Количество запоминаемых паттернов P
    let seed = 67;

    println!("1. Генерация {} случайных образов (N = {})...", p, n);
    let states = generate_states(p, n, seed);

    println!("2. Вычисление матрицы весов ...");
    let weights = weight_matrix_calculate(&states);

    let target_pattern = &states[0];
    println!("3. Берём образ №1 и накладываем 15% шума...");
    let mut noisy_state = apply_noise(target_pattern, 0.15, seed + 1000);

    let start_overlap = calculate_overlap(&noisy_state, target_pattern);
    let start_energy = calculate_energy(&noisy_state, &weights);

    println!("   - Начальное совпадение с оригиналом: {:.1}%", start_overlap * 100.0);
    println!("   - Начальная энергия сети: {:.4}\n", start_energy);

    println!("4. Запуск динамики восстановления (асинхронный режим)...");
    for iter in 1..=20 {
        let changed = neuron_fix(&mut noisy_state, &weights, seed + iter as u64 + 2000, None);
        let overlap = calculate_overlap(&noisy_state, target_pattern);
        let energy = calculate_energy(&noisy_state, &weights);

        println!(
            "   Итерация {:2}: Изменено нейронов: {:2} | Совпадение: {:5.1}% | Энергия: {:.4}",
            iter, changed, overlap * 100.0, energy
        );

        if changed == 0 {
            println!("\n Сеть достигла неподвижной точки!");
            break;
        }
    }

    let final_overlap = calculate_overlap(&noisy_state, target_pattern);
    println!("\nРезультат:");
    if final_overlap >= 0.99 {
        println!(" Успех! Образ успешно полностью восстановлен.");
    } else {
        println!(" Сеть сошлась к ближайшему локальному минимуму.");
    }
}
