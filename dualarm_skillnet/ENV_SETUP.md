# 本地小模型环境配置记录

## 当前结论

2026-11-15 已完成环境配置，Ollama 安装、官方 Qwen 7B 模型拉取和两项冒烟测试全部通过。配置过程遵循更新后的 `HANDOFF_v1.2_LLM.md` 第 2.2 节：从既有前缀单连接续传，验证完整长度及官方 SHA256，静默安装到默认 C 盘位置，模型文件放在 D 盘。本轮没有更改代理、TLS 验证、驱动或防火墙，没有安装 Python 第三方包。

## 硬件与磁盘

| 项目 | 实测结果 |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| 显存 | 8188 MiB |
| NVIDIA 驱动 | 572.83 |
| 系统物理内存 | 33,601,957,888 字节，31.294 GiB |
| 本次恢复前 C / D 剩余 | 19.392 / 44.777 GiB |
| 安装前 C 盘剩余 | 19.339 GiB，满足至少 8 GB |
| 安装后 C 盘剩余 | 15.206 GiB |
| 下载与冒烟后 C / D 剩余 | 15.003 / 38.785 GiB |
| 实际运行方式 | GPU；`/api/ps` 的 size_vram 与模型运行 size 均为 4,994,472,344 字节 |
| 实际推理上下文 | 8192 token |

下载、安装期间持续检查 C >= 3 GiB、D >= 15 GiB。冒烟后的显卡剩余显存约 1979 MiB，说明模型已实际加载；并非仅根据显卡标称推断运行方式。

## Ollama 安装

- 版本：`0.40.16`，CLI 与本机 `/api/version` 均已验证。
- 安装位置：默认当前用户目录 `C:\Users\t1507\AppData\Local\Programs\Ollama`。
- 来源：官方 winget `Ollama.Ollama` 元数据对应的 Ollama 官方发布地址：`https://github.com/ollama/ollama/releases/download/v0.40.16/OllamaSetup.exe`。
- 完整长度：**1,857,663,944 字节**，与本次续传响应 `Content-Range` 中的服务器总长度一致。
- 预期与实际 SHA256 均为：`84db4f756ddc9e091142a85cdc0f7ae5db68bc9b3e075f114388b326f8ed912d`。
- 从 `out/ollama_installer/prefix.bin` 的 **109,068,288 字节**末尾续传；本次使用单连接，无分段或并行请求，首次尝试即完成，没有消耗重试额度。
- 完整校验后重命名为 `out/ollama_installer/OllamaSetup-0.40.16.exe`，以 `/CURRENTUSER /VERYSILENT /NORESTART` 静默安装，退出码 **0**。没有覆盖默认安装位置，未出现管理员或确认窗口。
- 设置并保留用户级 `OLLAMA_MODELS=D:\ollama\models`，安装后停止自动启动的应用及服务，再从项目目录以该环境重启 `ollama serve`。服务仅监听 `127.0.0.1:11434`。

## 模型来源、摘要与空间

| 项目 | 实际值 |
|---|---|
| 来源 | Ollama 官方模型库，未使用备用来源 |
| 模型名称 | `qwen2.5:7b-instruct` |
| 精确模型 digest | `845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e` |
| 格式 / 量化 | GGUF / Q4_K_M |
| 规格 | Qwen 官方命名 7B，API 参数量标记为 7.6B |
| `/api/tags` 模型大小 | 4,683,087,332 字节 |
| 模型存储目录 | `D:\ollama\models` |
| 目录总占用 | 4,683,094,203 字节，9 个文件（含 manifest 等元数据） |
| 默认 C 盘模型目录 | 本次检查不存在 `C:\Users\t1507\.ollama\models` |
| 下载模型数 | 1，未使用最多 2 个模型的剩余额度 |

选择依据：[Ollama 官方量化标签页面](https://ollama.com/library/qwen2.5:7b-instruct-q4_K_M)。核实的新系列通用文本标签为 8B、9B 或更大规格，因此本次按确切的 7B 限制选择 Qwen2.5 指令版，没有用其他规模替代。

已核实但未使用的备用来源：[魔搭 Qwen 官方 GGUF 页面](https://modelscope.cn/models/Qwen/Qwen2.5-7B-Instruct-GGUF)，含 Q4_K_M 两个分片，合计 4,683,073,632 字节。只有官方拉取失败或速度异常时才允许按交接单使用 `ollama pull modelscope.cn/Qwen/Qwen2.5-7B-Instruct-GGUF:Q4_K_M`；本次未执行该命令，未验证该 registry 拉取结果。

## 冒烟测试

执行：

```powershell
python -X utf8 tools/check_model.py
```

退出码 **0**，两项全部通过。脚本仅使用标准库 `urllib`，只请求本机 `http://127.0.0.1:11434/api/generate`，非流式；关闭请求代理并拒绝重定向。

| 测试 | 结果 | 请求耗时 | 生成速度 |
|---|---|---|---|
| 一句中文回答“一加一等于几” | 通过，返回非空中文 | 3.297 秒（含首次加载） | 56.13 token/s |
| `s1_reach_handover` 的 `with_capabilities` 项目提示词 | 通过，JSON schema 约束输出可由 `parse_model_output` 解析，步骤非空 | 1.437 秒 | 54.08 token/s |

参数：temperature=0、seed=1101、num_ctx=8192；中文问答最多 64 个输出 token，项目 JSON 最多 2048。速度由 API 的 eval_count / eval_duration 计算，和包含加载／提示词处理的请求墙钟时间含义不同。

原始提示词、模型输出、API 响应及耗时保存在 `out/model_smoke.json`。完整环境快照（含 `/api/version`、`/api/tags`、`/api/ps`）在 `out/v1_2_environment_verified.json`。

## 遇到的问题和处理

此前两次安装包下载未成功；第三次单连接取得约 104 MiB 后切换并行，遇 TLS 失败而停止。本次按用户更新后的规则保持单连接，直接从保留前缀续传成功，没有重新下载或合并旧并行分段。安装包完整性验证通过后才执行安装。

官方模型下载初段约 40–46 MB/s，末段降至约 1–2 MB/s，但一直有进展，最终 `ollama pull` 退出码为 0。模型和服务运行均未触发硬件、权限或磁盘停止条件。此前未成功记录已保存在 `out/v1_2_environment_history.md`，以免与当前成功状态混淆。

## 如何在另一台电脑上复现

所有项目命令在 `D:\APEX_MR_Plus\dualarm_skillnet` 执行。

1. 用 `nvidia-smi`、`Get-CimInstance Win32_ComputerSystem` 和 `Get-PSDrive C,D` 检查硬件与空间。安装前 C 盘至少 8 GB，过程中 C 至少 3 GB，D 至少 15 GB。不要自动修改驱动或网络设置。
2. 设置模型目录并让后续 Ollama 进程继承：

   ```powershell
   New-Item -ItemType Directory -Force 'D:\ollama\models'
   [Environment]::SetEnvironmentVariable('OLLAMA_MODELS','D:\ollama\models','User')
   $env:OLLAMA_MODELS = 'D:\ollama\models'
   ```

3. 优先通过官方 winget 安装：

   ```powershell
   winget show --id Ollama.Ollama --exact --source winget --scope user
   winget install --id Ollama.Ollama --exact --source winget --scope user --silent --accept-source-agreements --accept-package-agreements --disable-interactivity
   ```

   若已有官方部分安装包，按交接单第 2.2 节单连接断点续传；临时错误最多重试 5 次，间隔为 10、30、60、120、300 秒。先核对服务器总长度和官方 SHA256，任一不符删除文件并停止。使用核对后的包安装时保持默认位置；遇 UAC 或确认窗口停止。
4. 重启 Ollama，确保服务继承上述模型目录，再验证版本与本机服务。在新终端使用 `ollama --version`；已有终端尚未继承安装器 PATH 时可以调用默认安装目录中的完整 `ollama.exe` 路径，无需手工修改 PATH。
5. 仅拉取指定 7B 模型：

   ```powershell
   ollama pull qwen2.5:7b-instruct
   ollama list
   ollama ps
   ```

   记录完整 digest，确认模型文件在 D 盘。只有官方库失败或异常才使用上文魔搭 Qwen 官方备用模型，全部来源都失败时停止。
6. 运行 `python -X utf8 tools/check_model.py`，确认两项通过，再通过 `/api/ps` 记录实际 GPU／CPU 分配，更新本文件。环境成功后才开始规划链路接入与对比实验。

## 配置证据

- `out/v1_2_official_metadata.txt`：官方安装包元数据。
- `out/v1_2_single_resume.json`、`out/v1_2_single_resume_console.txt`：单线续传与完整性验证。
- `out/v1_2_ollama_install.log`：安装退出前的成功日志。
- `out/v1_2_pull_official.txt`：官方模型拉取日志。
- `out/model_smoke.json`、`out/v1_2_model_smoke_console.txt`：两项真实冒烟。
- `out/v1_2_environment_verified.json`：版本、模型、运行方式和磁盘快照。

清理记录（v1.2.1）：2026-11-15T07:42:54-05:00（America/New_York），删除项目 out/ollama_installer/ 中 6 个安装包及分段文件，共 1,871,395,941 字节（约 1.743 GiB）；保留日志、download.cfg、parts.json 和 resume_single.py，未改动已安装的 Ollama 或 D 盘模型。清理后 D 盘剩余约 40.522 GiB。
