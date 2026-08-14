---
name: git-backup
description: HSRL 项目的 git 备份/回退协议：修改代码前先确认备份策略并创建快照提交，改动完成后立即提交新版本，以及回退命令速查。任何涉及代码修改、手动备份或回退的任务都应加载本 skill。
whenToUse: Use when about to modify code, create a backup snapshot, or roll back changes in this project.
---

# git-backup：HSRL 备份与回退协议

## 触发时机

- 任何将要修改代码/配置的任务开始前：先确认备份策略，再动手改。
- 用户输入 `/git-backup`：立即按当前工作区状态生成一次快照。

## 会话开始时询问（每会话一次）

未确认备份策略时，用 `ask_user_question` 询问：
1. 每次修改前自动 git 快照（推荐）
2. 只在我明确要求时备份
3. 本次会话不备份

## 修改前快照（策略 1）

工作区不干净时先执行，干净则跳过：

```bash
git add -A && git commit -m "snapshot: <接下来要做的改动>"
```

## 改动完成即提交

```bash
git add -A && git commit -m "<feat|fix|refactor|docs|chore>: <改动简述>"
```

## 回退速查

| 需求 | 命令 |
|---|---|
| 单文件回退 | `git restore -- <file>` |
| 撤销某次提交 | `git revert <commit>` |
| 查看版本历史 | `git log --oneline` |
| 查看某次改动 | `git diff <commit>~1 <commit>` |

## 红线

除非用户明确要求，禁止 `git reset --hard`、强制推送、删除 `.git`、改写历史。
回退前若工作区有未提交改动，先做一次快照。
