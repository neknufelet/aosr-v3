"""新家的設定層（第 2 層）——這裡刻意不做事。

**為什麼這一支是空的（只留文件字串）。** 上一代那個套件的門面檔只做 5 行
re-export，其中 3 行指向會把 JAX（算數用的函式庫）拉進來的模組；那 3 行是上一代第 2 層
**唯一**的 JAX 破口——20 支模組本身一支都沒碰 JAX，但只要有人去 import 那個門面，
它就先把 JAX 拉進來。新家不做門面：**不轉手任何名字**，要拿哪一支就寫全名
（``import aosr.config.spl_output``）。決定與量測見決策紙
``docs/decisions/engine-first-block-config-shape.md``（型別那一題）與票 #127 的留言。

**這一層的家規**（規矩卡 ``layers-import-downward-only``）：第 2 層不准去拿上面的
第 3 層以上（materials／geometry／physics／scoring／…），也不准碰 JAX 的全域設定。
誰要拿這裡的東西都可以——方向是往下的。

設定檔的路徑只有一個住處：:mod:`aosr.config.paths`。
"""
