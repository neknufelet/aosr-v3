"""讓 ``blueprint`` 變成一個正常的套件（不是命名空間套件）。

**為什麼要有這一支。** 產生器（``blueprint/generate_config_cut1_answers.py``）與考卷
（``tests/engine/``）都要讀同一份 case 表 ``blueprint/config_cut1_cases.py``——
產生器要照它去上一代跑值，考卷要照它確認「答案檔裡有哪些 case」。兩邊要 import 同一個
模組名，所以這一層得是一個真的套件。
"""
