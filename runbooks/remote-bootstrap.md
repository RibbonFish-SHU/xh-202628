# Remote Execution Mirror Bootstrap

状态：**READY - 用户已于 2026-08-18 明确批准，尚未执行创建**

- SSH 目标：`lynsdu2@10.0.33.75`
- 唯一允许创建的目录：`/home/user/lynsdu2/xh-202628-agent`
- 用途：接收本地已提交 commit 的只读源码快照并运行 NVIDIA 测试

## 绝对门禁

用户已经针对上述精确路径回复“1、允许”。该许可已写入机器状态；不扩大到其他路径。

获得批准后，按顺序：

1. 在 `state/PROJECT_STATE.md` 记录用户许可原文、Asia/Shanghai 时间和批准的绝对路径。
2. 在 `state/remote-execution.json` 将 `directory_creation.authorized` 改为 `true`，填写同一原文和时间。
3. 同时取得 GPU 使用规则；未取得时保持 `execution.enabled=false`。
4. 提交上述状态变更，并确认本地工作树干净。
5. 执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\initialize-remote-mirror.ps1 -Apply
```

初始化脚本会再次读取机器门禁并拒绝未授权状态。它只创建：

```text
/home/user/lynsdu2/xh-202628-agent/
  .xh-202628-execution-mirror
  incoming/
  locks/
  runs/
```

脚本拒绝复用已经存在的目标目录，不执行 Git 初始化，不配置 GitHub，不安装软件，也不访问其他项目。

## 创建后核对

只核对目标目录自身的绝对路径、owner、权限、marker 和三个空子目录。随后更新 `state/PROJECT_STATE.md` 与 JSON 状态并提交。

若初始化中途失败，保留现场并向用户报告；不得自动递归删除或尝试清理。任何恢复操作都必须先解析并再次核对精确路径。

## 禁止事项

- 不读取或复用其他项目的源码、虚拟环境、凭据、缓存或构建产物。
- 不在 home 目录本身执行 `git init` 或创建散落文件。
- 不在服务器保存 GitHub 凭据或连接 GitHub/XPU-OJ。
- 不删除、移动、chmod/chown 或修改其他路径。
- 未经另行批准，不删除本项目远端 `runs/` 中的历史结果。
