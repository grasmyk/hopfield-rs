import os
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


if __name__ == "__main__":
    print("=" * 50)
    print("  Генерация графиков проекта Hopfield Networks")
    print("=" * 50)
    generate_fig1()
    generate_fig2()
    print("Готово!")