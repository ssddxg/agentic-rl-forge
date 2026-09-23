<h1 align="center">AgenticRLForge</h1>

<p align="center">
  <img src="assets/hero.svg" alt="AgenticRLForge" width="100%" />
</p>

<p align="center">
  <strong>面向搜索与工具智能体的本地强化学习工作台。</strong>
</p>

<p align="center">
  <a href="https://github.com/ssddxg/agentic-rl-forge/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/ssddxg/agentic-rl-forge/actions/workflows/ci.yml/badge.svg" /></a>
  <a href="https://www.python.org/"><img alt="Python 3.10–3.12" src="https://img.shields.io/badge/Python-3.10%E2%80%933.12-3776AB?logo=python&amp;logoColor=white" /></a>
  <a href="LICENSE"><img alt="Apache-2.0 license" src="https://img.shields.io/badge/License-Apache--2.0-6B7280" /></a>
  <a href="https://github.com/ssddxg/agentic-rl-forge"><img alt="本地优先" src="https://img.shields.io/badge/%E6%9C%AC%E5%9C%B0%E4%BC%98%E5%85%88-%E6%97%A0%E4%BD%BF%E7%94%A8%E8%B7%9F%E8%B8%AA-16A34A" /></a>
</p>

<p align="center">
  <a href="README.md">English</a> · <strong>简体中文</strong> ·
  <a href="docs/studio.zh-CN.md">完整使用指南</a> ·
  <a href="https://github.com/ssddxg/agentic-rl-forge/issues">问题反馈</a>
</p>

AgenticRLForge 的核心不是普通聊天网页，也不是单纯的 RAG 知识库。它是一套可以在本地
运行、检查和导出数据的 **Agent 强化学习（Agent RL）工作台**：让智能体调用搜索工具，
记录完整轨迹，计算奖励，进行分组采样和评测，并生成可交给 GRPO/verl 的训练批次。

项目同时保留了一个本地文档知识库，方便整理检索语料和项目资料；它是辅助功能，不是产品
的主要定位。普通电脑可以完成演示、数据验证和离线管线，真正修改大模型权重仍需要 GPU、
PyTorch、模型服务和 verl。

<p align="center">
  <img src="assets/studio-overview.png" alt="AgenticRLForge Studio 总览、Agent RL 数据闭环与运行环境状态" width="100%" />
</p>

<p align="center"><em>本地 Studio：真实轨迹、离线 RL 数据流、环境状态和运行记录集中展示。</em></p>

## 最主要的功能

### 1. 在网页里看懂一次 Agent 强化学习流程

Studio 会真实执行一条“问题 → 搜索动作 → 环境反馈 → 最终答案 → 奖励”的轨迹，并展示：

- 智能体每一步做了什么；
- 工具返回了什么内容；
- 哪些 Token 属于模型生成，哪些来自环境；
- 最终答案是否正确，以及各项奖励如何组成；
- 轨迹、数据集和策略版本的来源信息。

这比只展示一段聊天记录更接近真实的 Agent RL 数据生产过程。

### 2. 一键运行完整的离线 RL 数据管线

可以使用内置样例，也可以上传自己的问答数据和检索语料。系统会在后台完成：

```text
问答任务 + 检索语料
        ↓
同一问题的分组 Rollout
        ↓
轨迹持久化、奖励与评测
        ↓
筛选有学习信号的数据
        ↓
导出并校验训练批次
```

输出包含 SQLite 状态、可恢复分片、评测报告、筛选后的轨迹和训练批次。中断后的运行可恢复，
不会把任务永久留在“处理中”。

### 3. 为 Search-R1、GRPO 和 verl 准备可靠数据

项目提供 Search-R1 风格的搜索/推理协议、分组优势计算、策略版本保护、生成 Token 掩码、
奖励验证和 verl 兼容导出。研究代码还包括 PRM 引导的 MCTS、拒绝采样、Nash-MD、
Hindsight 轨迹重标注、实验矩阵、对比评测、候选晋级和可复现归档。

### 4. 辅助的本地知识库

支持 PDF、DOCX、HTML、Markdown、RST 和纯文本。可以完全离线搜索，也可以连接兼容
OpenAI 接口的本地或在线模型生成带引用的答案。数据默认保存在本机，应用不会向外发送
使用遥测；需要运维监控时仍可主动读取本地 Prometheus 指标。

## 适合谁使用

- **普通用户和初学者**：双击启动，在网页里直观看懂智能体如何调用工具、获得反馈和奖励。
- **数据与算法工程师**：验证问答/检索数据、奖励函数、分组 Rollout 和训练批次是否连通。
- **强化学习研究者**：复用 Search-R1、GRPO、PRM/MCTS、Hindsight、实验与评测组件。
- **本地知识管理用户**：把知识库作为辅助工具，离线搜索自己的文档。

如果只想找一个通用聊天机器人，本项目并不是最简单的选择；如果希望看到并控制 Agent RL
从环境交互到训练数据导出的完整链路，它会更有价值。

## Windows 一键启动

要求：已安装 Python 3.10、3.11 或 3.12。安装 Python 时建议勾选“Add Python to PATH”。

1. 在 GitHub 页面点击 **Code → Download ZIP**，下载后完整解压。
2. 双击根目录中的 `Start AgenticRLForge.cmd`。
3. 首次运行会自动创建独立环境、安装依赖、执行健康检查并打开浏览器。
4. 浏览器访问地址是 <http://127.0.0.1:7860>。

以后继续双击同一个文件即可。使用期间保留启动窗口；退出时在窗口中按 `Ctrl+C`。

> 不要在 ZIP 压缩包预览窗口中直接运行，也不要直接双击网页文件。Studio 需要先启动本地服务。

## Linux 或 macOS

在项目目录中运行：

```bash
bash scripts/start-studio.sh
```

## Docker Compose

已经安装 Docker Desktop 或 Docker Engine 时：

```bash
docker compose up --build
```

然后打开 <http://127.0.0.1:7860>。数据保存在 `studio-data` 数据卷中，普通的容器重建
不会删除资料。

## 第一次打开后怎么用

1. 在“总览”查看运行环境。核心环境通过即可使用本地功能；没有 verl 不影响入门演示。
2. 点击“运行轨迹演示”，观察一次完整的搜索 Agent 循环和奖励结果。
3. 使用内置数据运行“离线 RL 数据管线”，查看分组轨迹、评测和训练批次。
4. 再上传自己的两个 UTF-8 JSONL 文件进行验证。

问答文件每行一条记录：

```json
{"id":"q1","question":"法国首都是什么？","answer":"巴黎"}
```

检索语料每行一条记录：

```json
{"id":"doc1","contents":"巴黎是法国的首都。"}
```

每个问题必须能从语料中检索到包含答案的证据。否则管线会明确报错，避免生成看似成功但
实际上没有学习价值的数据。

更详细的页面说明、模型连接、数据目录、备份和排错方法见
[《Studio 中文使用指南》](docs/studio.zh-CN.md)。

## 命令行用法

安装源码版本：

```bash
python -m venv .venv
.venv/bin/python -m pip install -e ".[studio]"
.venv/bin/arf doctor --profile core --strict
.venv/bin/arf studio
```

Windows 请把上面三处 `.venv/bin/` 改为 `.venv\Scripts\`。

构建本地文档语料并搜索：

```bash
arf corpus-build ./my-documents ./data/my-corpus.jsonl
arf corpus-check ./data/my-corpus.jsonl
arf search "项目如何部署？" --corpus ./data/my-corpus.jsonl --top-k 5
```

运行离线强化学习数据管线：

```bash
arf offline-pipeline examples/data/qa.jsonl examples/data/corpus.jsonl artifacts/offline-run
```

启动 Search-R1 兼容检索服务：

```bash
arf serve-retriever examples/data/corpus.jsonl --host 127.0.0.1 --port 8000
```

高级实验、GPU 训练、分布式 Rollout、对象存储和候选晋级命令请查阅
[英文 README](README.md) 与 [`docs/`](docs/) 目录。

## 能力边界

| 场景 | 普通电脑直接可用 | 额外要求 |
| --- | --- | --- |
| Studio 网页与轨迹演示 | 是 | Python 3.10–3.12 |
| 自定义数据验证与离线管线 | 是 | CPU 与本地磁盘 |
| 本地文档搜索 | 是 | 无需模型或 API Key |
| 带引用的 AI 问答 | 可选 | OpenAI 兼容模型接口 |
| Search-R1 在线 Rollout | 可选 | vLLM/SGLang 或兼容模型服务 |
| 真正更新模型权重 | 否 | GPU、PyTorch、verl 和模型权重 |

项目不会把“生成了训练数据”包装成“已经训练了模型”。Studio 会分别显示核心运行环境和
GPU 训练环境是否就绪。

## 项目结构

```text
src/agentic_rl_forge/   Python 核心库、CLI、Studio 与 RL 组件
configs/                实验和运行配置
examples/               可直接运行的样例与小型数据
recipes/                verl 等训练配方
docs/                   使用、API、部署和高级功能文档
scripts/                Windows/Linux/macOS 安装、启动和检查脚本
tests/                  单元、集成、发布与失败恢复测试
```

## 质量与安全

- CI 覆盖 Python 3.10、3.11、3.12，以及 Linux、Windows、macOS 的 wheel 冒烟测试。
- Docker 启动、文件上传、索引和搜索有自动化验证。
- Ruff、Mypy、Pytest、依赖审计和发布包检查已集成。
- Studio 默认只监听本机地址，并提供同源、CSRF、上传限制和密钥脱敏保护。
- 本地工作流不要求账号或 API Key；连接在线模型时，命中的资料片段会发送给所选服务。

安全问题请遵循 [`SECURITY.md`](SECURITY.md)，贡献方式见
[`CONTRIBUTING.md`](CONTRIBUTING.md)，版本变化见 [`CHANGELOG.md`](CHANGELOG.md)。

## 项目状态

当前版本为 `0.3.0`，定位为可实际运行的 Alpha 版本。CPU 本地流程已可完整验证；GPU 权重
训练需要用户自行准备外部训练环境和模型资源。首次使用建议先运行 Studio 内置演示和离线
管线，再接入大型模型或真实数据。

## 许可证

[Apache License 2.0](LICENSE)
