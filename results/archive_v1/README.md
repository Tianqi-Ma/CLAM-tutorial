# v1 产物归档（2026-10 审计前）

这里是 `analysis_multiomics.py` / `analysis_de.py` / `analysis_morph_predict.py` / `analysis_survival.py` / `analysis_cox.py`
在审计前（v1）的输出，原来分别位于 `results/multiomics/` 和 `results/survival/`。保留它们是为了对照，**不要引用其中的结论**：

- `multiomics/`：表达矩阵可能混入了癌旁正常样本；`morph_embeddings.csv` 来自 5 个参数不同的 CLAM 模型，
  PCA 主成分里混着折号（见 `multiomics/fold_artifact_check.txt`，由 `scripts/audit_v1_embedding_folds.py` 生成）；
  "形态可预测基因 ∩ 差异基因"的重叠是循环论证；DE 的 log2FC 用算术均值算，富集背景不对。
- `survival/`：生存表漏取了 GDC 的 follow_ups，留下的 LUAD 全是死亡病例，KM 和 Cox 结果全部作废；
  日志里的"26 例""97 个患者"是 100 张时代写死的旧数字。

v2 脚本的说明见仓库 README 和 `notebooks/CLAM_workflow.ipynb` §8、§9。
