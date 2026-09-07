# Task 7 验收交接：需补完整独立审查

状态：本地实现和普通调用方修复已保存；Task 7 尚未合入 `test`，不是最终候选。当前审查服务限制使完整核心审查中断，没有完整 verdict；不继续重试，也不以普通测试替代。

## 固定审查对象

- 仓库：`TNHTH/SEA-Nav-Code`，本任务 workspace 的 `work/SEA-Nav-Code`。
- 完整 Task 7 BASE：`562d4ae0b56ca977ea433def6dbd05607284e38b`。
- 固定 HEAD：`774027d1a975e318ad2f577b457951f51c82bfad`。
- 有序提交：`2387cf02d85a1a9a52b0e85da54c7ebad09b0cec` → `c7b9aa371b8ab3800adea378a7024f043fb58580` → 固定 HEAD。
- 可读的干净 detached checkout：`work/SEA-Nav-Code-batch7-review-callers-fix1`。保留原先两个审查快照，不移动它们。
- 主仓 `test` 的已接受生产源码仍截至 Task 6 的 `b5b945513acb4d3e48c401653198442654a387e9`；协调文档的后续提交不是 Task 7 源码整合。

## 已有证据及其限制

- `task-7-report.md`：完整实施交接及 fix1 附录；固定 HEAD 实施者全量 CPU 测试 433 passed / 65.70s，Gate A 五项 CPU/static passed、四项 Gym/Lab blocked。外部 Gate A JSON 保存在 workspace 的 `work/task-7-worker-gate-a-fix1-774027d.json`，未提交或上传。
- `task-7-core-functional-review.md`：控制者的有限普通 runner/persistence 验证，身份和选择见正文；不是完整核心/转换器审查。
- `task-7-callers-review.md` 与 `task-7-callers-rereview-1.md`：普通 caller/README 审查及唯一 terrain P2 的修复复审。固定 HEAD 有界 Spec/Quality PASS，独立 119 passed / 1 deselected；只关闭该调用方缺陷。
- 完整基线 ancestry 成立，没有删除基线文件。`main/stable` 未推进，远端未写入。

## 需要外部提供的内容

请由具备相应授权和能力的人工或外部独立审查渠道，补齐固定完整范围的 checkpoint 核心、runner 持久化整合及 operator-only 转换器审查。以 `task-7-brief.md` 的 binding corrections、`checkpoint-contract-audit.md` 和现有完整设计为合同；不要仅审查最后的单行 play 修复。

返回报告至少注明完整 BASE/HEAD、实际覆盖文件/合同、具体发现与严重度、独立验证命令及结果、未覆盖项和明确 verdict。此前服务中断没有产生完整报告；不能标记为通过。当前不请求执行新增探针、降低安全门、开放运行权限或重试受限服务。

完整审查结果到位后才能裁决 Task 7 验收；若有问题仍由单一实施者串行修复。随后才可继续独立的 CBF hotpath P2、最终冻结 Rungs 0–2 和远端事务。仿真、论文指标、真机与再分发许可仍是独立阻塞；远端精确授权及 stable 保护选择见 `publication-preflight.md`，此次交接不授予远端操作权限。
