import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Имя файла с результатами
csv_filename = "capacity_results.csv"

if not os.path.exists(csv_filename):
    print(f"Ошибка: Файл {csv_filename} не найден в текущей директории!")
    exit(1)

print(f"Загрузка данных из {csv_filename}...")
df = pd.read_csv(csv_filename)

# Настройка визуального стиля
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

n_values = sorted(df['n'].unique())
colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(n_values)))

for idx, n in enumerate(n_values):
    sub = df[df['n'] == n]
    
    # Агрегируем по alpha: среднее и стандартная ошибка
    stats = sub.groupby('alpha')['success'].agg(
        mean='mean',
        std='std',
        count='count'
    ).reset_index()
    
    stats['sem'] = stats['std'] / np.sqrt(stats['count'])
    stats['sem'] = stats['sem'].fillna(0)
    
    # Кривая ёмкости
    ax.plot(
        stats['alpha'], 
        stats['mean'], 
        marker='o', 
        linewidth=2, 
        markersize=6,
        color=colors[idx],
        label=f'N = {n}'
    )
    
    # Область разброса / доверительный интервал (±1 SEM)
    ax.fill_between(
        stats['alpha'],
        np.clip(stats['mean'] - stats['sem'], 0, 1),
        np.clip(stats['mean'] + stats['sem'], 0, 1),
        color=colors[idx],
        alpha=0.18
    )

# Теоретический предел alpha ≈ 0.138
ax.axvline(
    x=0.138, 
    color='#d9534f', 
    linestyle='--', 
    linewidth=2, 
    label=r'Теория $\alpha_c \approx 0.138$'
)

# Оформление графика
ax.set_title("Рис. 1 — Кривые ёмкости сети Хопфилда (Фазовый переход)", fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel("Нагрузка α = P / N", fontsize=12)
ax.set_ylabel("Доля успешных восстановлений (m ≥ 0.95)", fontsize=12)

ax.set_xlim(0.098, 0.162)
ax.set_ylim(-0.03, 1.05)

ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(title="Размер сети N", fontsize=10, title_fontsize=11, loc='lower left', frameon=True)

plt.tight_layout()
output_fig = "fig1_capacity_curves.png"
plt.savefig(output_fig, dpi=300)
print(f"График успешно сохранен в '{output_fig}'")
plt.show()