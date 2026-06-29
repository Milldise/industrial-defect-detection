import numpy as np
import matplotlib.pyplot as plt

# Названия осей (критериев из Таблицы 1)
labels = ['Cost efficiency', 'Accuracy', 'Flexibility', 'Real-time capability', 'Data logging']
num_vars = len(labels)

# Количественные баллы, переведенные из качественных оценок
systems = {
    'Manual Inspection': [2, 3, 5, 2, 1],
    'Classical CV': [4, 3, 1, 5, 3],
    'Commercial Systems': [1, 5, 2, 5, 5],
    'Deep Learning (Custom)': [4, 5, 5, 5, 5]
}

# Вычисление углов для лепестковой диаграммы
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # Замыкаем петлю графика

# Создание полярной сетки (без использования plt.figure())
fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

# Корректировка направления осей (чтобы Cost efficiency была сверху)
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# Отрисовка подписей осей
plt.xticks(angles[:-1], labels, color='black', size=11)

# Настройка круговой шкалы от 1 до 5
ax.set_rlabel_position(0)
plt.yticks([1, 2, 3, 4, 5], ["1", "2", "3", "4", "5"], color="grey", size=9)
ax.set_ylim(0, 5)

# Цветовая палитра для полигонов
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# Построение графиков для каждой из систем
for (name, data), color in zip(systems.items(), colors):
    values = data + data[:1]  # Замыкаем петлю данных
    ax.plot(angles, values, label=name, linewidth=2, color=color)
    ax.fill(angles, values, alpha=0.1, color=color)

# Добавление легенды и заголовка
plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), fontsize=10)

# Оптимизация полей и сохранение файла
plt.tight_layout()
plt.savefig('figure_1_3.png', dpi=300, bbox_inches='tight')