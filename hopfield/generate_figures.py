import os
import json
from pathlib import Path

import matplotlib
# Отключаем GUI-экран полностью
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')


def generate_fig1():
    csv_filename = "capacity_results.csv"

    if not os.path.exists(csv_filename):
        print(f"[ПРОПУСК] Файл '{csv_filename}' не найден.")
        return

    print(f"--> Генерация Рис. 1 из '{csv_filename}'...")
    df = pd.read_csv(csv_filename)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    n_values = sorted(df['n'].unique())
    colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(n_values)))

    for idx, n in enumerate(n_values):
        sub = df[df['n'] == n]

        stats = sub.groupby('alpha')['success'].agg(
            mean='mean',
            std='std',
            count='count'
        ).reset_index()

        stats['sem'] = stats['std'] / np.sqrt(stats['count'])
        stats['sem'] = stats['sem'].fillna(0)

        ax.plot(
            stats['alpha'], 
            stats['mean'], 
            marker='o', 
            linewidth=2, 
            markersize=6,
            color=colors[idx],
            label=f'N = {n}'
        )

        ax.fill_between(
            stats['alpha'],
            np.clip(stats['mean'] - stats['sem'], 0, 1),
            np.clip(stats['mean'] + stats['sem'], 0, 1),
            color=colors[idx],
            alpha=0.18
        )

    ax.axvline(
        x=0.138, 
        color='#d9534f', 
        linestyle='--', 
        linewidth=2, 
        label=r'Теория $\alpha_c \approx 0.138$'
    )

    ax.set_title("Рис. 1 — Кривые ёмкости сети Хопфилда (Фазовый переход)", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Нагрузка α = P / N", fontsize=12)
    ax.set_ylabel("Доля успешных восстановлений (m ≥ 0.95)", fontsize=12)
    ax.set_xlim(0.098, 0.162)
    ax.set_ylim(-0.03, 1.05)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(title="Размер сети N", fontsize=10, title_fontsize=11, loc='lower left', frameon=True)

    output_fig = "fig1_capacity_curves.png"
    plt.savefig(output_fig, dpi=300)
    plt.close(fig)
    print(f"  [OK] Успешно сохранен файл: '{output_fig}'")


def generate_fig2():
    res_file = "mnist_results.csv"
    samp_file = "mnist_samples.csv"

    if not os.path.exists(res_file) or not os.path.exists(samp_file):
        print(f"[ПРОПУСК] Файлы '{res_file}' или '{samp_file}' не найдены.")
        return

    print(f"--> Генерация Рис. 2 из '{res_file}' и '{samp_file}'...")
    results_df = pd.read_csv(res_file)
    samples_df = pd.read_csv(samp_file)

    print("\n" + "=" * 60)
    print(" ТАБЛИЦА СРЕДНЕГО ПОПАРНОГО ПЕРЕКРЫТИЯ C_ab (Утверждение C6)")
    print("=" * 60)
    cab_summary = results_df.groupby(["dataset", "k"])["c_ab"].mean().unstack()
    print(cab_summary.round(4))
    print("-" * 60)
    print("Вывод: У случайных образов C_ab ≈ 0.0 (образы ортогональны).")
    print("       У цифр MNIST C_ab ≈ 0.5...0.7 (сильная корреляция фоновых пикселей).")
    print("=" * 60 + "\n")

    fig = plt.figure(figsize=(14, 6), dpi=300)
    gs = fig.add_gridspec(5, 4, width_ratios=[1, 1, 1, 3.5], wspace=0.1, hspace=0.1)

    stages = ["original", "corrupted", "restored"]
    titles = ["Оригинал", "Зажато", "Восстановлено"]

    for img_idx in range(5):
        for stage_idx, stage in enumerate(stages):
            ax = fig.add_subplot(gs[img_idx, stage_idx])

            row = samples_df[
                (samples_df["image_idx"] == img_idx) & (samples_df["stage"] == stage)
            ]

            if not row.empty:
                pixels_str = row.iloc[0]["pixels"]
                pixels = np.array(pixels_str.split(), dtype=float).reshape(28, 28)
                ax.imshow(pixels, cmap="gray", vmin=-1.0, vmax=1.0)

            ax.set_xticks([])
            ax.set_yticks([])

            if img_idx == 0:
                ax.set_title(titles[stage_idx], fontsize=9, fontweight="bold")

    ax_plot = fig.add_subplot(gs[:, 3])
    stats = results_df.groupby(["dataset", "k"])["overlap"].mean().reset_index()

    mnist_stats = stats[stats["dataset"] == "mnist"]
    random_stats = stats[stats["dataset"] == "random"]

    ax_plot.plot(
        mnist_stats["k"],
        mnist_stats["overlap"],
        "o-",
        label="MNIST (Высокая корреляция $C_{ab}$)",
        color="#d62728",
        linewidth=2.5,
        markersize=7,
    )
    ax_plot.plot(
        random_stats["k"],
        random_stats["overlap"],
        "s--",
        label="Random Control ($C_{ab} \\approx 0$)",
        color="#1f77b4",
        linewidth=2.5,
        markersize=7,
    )

    ax_plot.set_xlabel("Количество запоминаемых образов ($k$)", fontsize=11)
    ax_plot.set_ylabel("Среднее перекрытие $m$ с оригиналом", fontsize=11)
    ax_plot.set_title("Качество восстановления: MNIST vs Control", fontsize=12, fontweight="bold")
    ax_plot.set_ylim(-0.05, 1.05)
    ax_plot.set_xticks([2, 5, 10, 20, 50])
    ax_plot.grid(True, linestyle="--", alpha=0.6)
    ax_plot.legend(fontsize=10, loc="lower left")

    output_fig = "fig2_mnist_vs_control.png"
    plt.savefig(output_fig, dpi=300)
    plt.close(fig)
    print(f"  [OK] Успешно сохранен файл: '{output_fig}'")


def generate_fig3():
    sizes = [512, 1024, 2048, 4096]
    base_dir = Path("target/criterion/Hopfield_Comparison")

    if not base_dir.exists():
        print(f"[ПРОПУСК] Директория '{base_dir}' не найдена. Сначала запустите `cargo bench`.")
        return

    print("--> Генерация Рис. 3 (Графики бенчмарков) из результатов Criterion...")

    n_vals, f64_times, u64_times, speedups = [], [], [], []

    for n in sizes:
        f64_file = base_dir / f"Baseline_f64/{n}/base/estimates.json"
        u64_file = base_dir / f"Optimized_u64/{n}/base/estimates.json"

        if f64_file.exists() and u64_file.exists():
            with open(f64_file, 'r', encoding='utf-8') as f:
                f64_ns = json.load(f)["mean"]["point_estimate"]
            with open(u64_file, 'r', encoding='utf-8') as f:
                u64_ns = json.load(f)["mean"]["point_estimate"]

            f64_ms = f64_ns / 1e6
            u64_ms = u64_ns / 1e6
            speedup = f64_ms / u64_ms if u64_ms > 0 else 0.0

            n_vals.append(str(n))
            f64_times.append(f64_ms)
            u64_times.append(u64_ms)
            speedups.append(speedup)

    if not n_vals:
        print("  [ВНИМАНИЕ] Файлы estimates.json не найдены по указанному пути.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Левый график: Время выполнения (логарифмическая шкала)
    x = np.arange(len(n_vals))
    width = 0.35

    rects1 = ax1.bar(x - width/2, f64_times, width, label='Baseline f64', color='#d9534f', alpha=0.85)
    rects2 = ax1.bar(x + width/2, u64_times, width, label='BitEngine u64', color='#2b8cbe', alpha=0.85)

    ax1.set_yscale('log')
    ax1.set_xlabel('Размер сети (N)', fontsize=11)
    ax1.set_ylabel('Время выполнения (ms, лог. шкала)', fontsize=11)
    ax1.set_title('Время выполнения 50 сидов (f64 vs u64)', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(n_vals)
    ax1.legend(fontsize=10)
    ax1.grid(True, which="both", linestyle=":", alpha=0.6)

    # Значения над столбцами
    for bar in rects1:
        height = bar.get_height()
        ax1.annotate(f'{height:.2f}ms', xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    for bar in rects2:
        height = bar.get_height()
        ax1.annotate(f'{height:.2f}ms', xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    # Правый график: Коэффициент ускорения (Speedup)
    ax2.plot(n_vals, speedups, marker='o', linewidth=2.5, markersize=8, color='#2ca02c')
    for i, txt in enumerate(speedups):
        ax2.annotate(f'{txt:.1f}x', (n_vals[i], speedups[i]), xytext=(0, 8), 
                     textcoords='offset points', ha='center', fontweight='bold', fontsize=10)

    ax2.set_xlabel('Размер сети (N)', fontsize=11)
    ax2.set_ylabel('Ускорение (Baseline / BitEngine)', fontsize=11)
    ax2.set_title('Коэффициент ускорения (Speedup Factor)', fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    output_fig = "fig3_benchmark_results.png"
    plt.savefig(output_fig, dpi=300)
    plt.close(fig)
    print(f"  [OK] Успешно сохранен файл: '{output_fig}'")


if __name__ == "__main__":
    print("=" * 50)
    print("  Генерация графиков проекта Hopfield Networks")
    print("=" * 50)
    generate_fig1()
    generate_fig2()
    generate_fig3()
    print("Готово!")