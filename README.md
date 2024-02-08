Завдання 1:
У цьому завданні був створений граф, що представляє модель певної реальної мережі, використовуючи бібліотеку NetworkX. Граф був візуалізований за допомогою Matplotlib для кращого розуміння його структури.
| Характеристика | Значення |
|:------------------------|-----------:|
| Кількість вершин | 15 |
| Кількість ребер | 31 |
| Середній ступінь вершин | 4.13333 |

Завдання 2:
Алгоритми DFS і BFS:
Були реалізовані алгоритми DFS і BFS для знаходження шляхів у графі. Результати виконання обох алгоритмів були порівняні, і виявлені різниці в отриманих шляхах.
| DFS Paths | [0, 1, 3, 5, 8, 11, 14, 7, 4, 2, 6, 9, 12, 10, 13] |
|:------------|:-----------------------------------------------------|
| BFS Paths | [0, 1, 3, 5, 8, 11, 14, 11] |

Пояснення різниці в шляхах:
Різниці в шляхах, знайдених алгоритмами DFS і BFS, можуть бути пояснені їхніми принципами роботи. DFS вибирає один шлях та глибоко досліджує його, що може призвести до іншого вигляду шляхів порівняно із шляхами, знайденими BFS, який розглядає всі можливі шляхи на одному рівні глибини перед переходом на наступний рівень.

Завдання 3:
Алгоритм Дейкстри:
Була додана вага до ребер графа, і програмно реалізований алгоритм Дейкстри для знаходження найкоротших шляхів між вершинами графа. Це дозволило знайти оптимальні шляхи з урахуванням ваг ребер.
| Маршрут | Відстань |
|:------------|-----------:|
| Від 0 до 0 | 0 |
| Від 0 до 1 | 10 |
| Від 0 до 2 | 5 |
| Від 0 до 3 | 18 |
| Від 0 до 4 | 11 |
| Від 0 до 5 | 15 |
| Від 0 до 6 | 12 |
| Від 0 до 7 | 15 |
| Від 0 до 8 | 20 |
| Від 0 до 9 | 18 |
| Від 0 до 10 | 23 |
| Від 0 до 11 | 29 |
| Від 0 до 12 | 25 |
| Від 0 до 13 | 29 |
| Від 0 до 14 | 34 |

Загальні висновки:
Завершено завдання з моделювання та аналізу графів, а також реалізації алгоритмів пошуку шляхів та знаходження найкоротших шляхів. Робота включає в себе визначення графічної моделі, аналіз характеристик, порівняння результатів алгоритмів та висновки з отриманих даних.