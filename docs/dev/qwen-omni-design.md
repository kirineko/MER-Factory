---
layout: default
title: Qwen-Omni Provider 设计文档
description: 为 MER-Factory 添加阿里云百炼 qwen3.5-omni-plus 多模态模型支持
lang: zh
---

# Qwen-Omni Provider 设计文档

## 1. 概述

本文档描述了为 MER-Factory 项目新增阿里云百炼平台 `qwen3.5-omni-plus` 多模态模型 Provider 通道的设计方案。

### 1.1 背景

MER-Factory 当前已支持以下 API Provider：

| Provider | 模型示例 | 音频 | 图像 | 视频 | 文本 |
|----------|---------|------|------|------|------|
| ChatGPT | gpt-4o | ✅ | ✅ | ❌ | ✅ |
| Gemini | gemini-2.5-flash | ✅ | ✅ | ✅ | ✅ |
| Kimi | kimi-k2.5 | ⚠️ (Gemini 回退) | ✅ | ✅ | ✅ |
| Ollama | 本地模型 | ❌ | ✅ | ❌ | ✅ |

`qwen3.5-omni-plus` 是阿里云通义千问最新一代全模态模型，**原生支持文本、音频、图像、视频**的输入理解，无需任何回退方案。

### 1.2 目标

- 新增 `QwenOmniModel` Provider 类，实现项目统一的 5 个异步接口方法
- 集成阿里云 DashScope OpenAI 兼容模式 API
- 支持 CLI 参数 `--qwen-omni-model` / `-qom` 选择模型
- 保持与现有 Provider 架构的一致性

---

## 2. 模型能力

### 2.1 qwen3.5-omni-plus 规格

| 特性 | 规格 |
|------|------|
| 模型 ID | `qwen3.5-omni-plus` |
| 输入模态 | 文本 + 图像 + 音频 + 视频 |
| 输出模态 | 文本 + 语音（55 种音色，36 种语言） |
| 音频输入上限 | 2GB / 3 小时 |
| 视频输入上限 | 2GB / 1 小时 |
| 图像输入上限 | URL 最多 2,048 张 / Base64 最多 250 张 |
| 输入语言 | 113 种（92 种语言 + 21 种方言） |
| 特殊功能 | 联网搜索（`search_strategy: "agent"`） |

### 2.2 API 关键约束

> **⚠️ 重要：DashScope Omni API 强制要求 `stream=True`，不支持非流式调用。**

---

## 3. 技术架构

### 3.1 API 接入方式

DashScope 提供 OpenAI 兼容模式，可直接使用 `langchain-openai` 的 `ChatOpenAI` 类：

```
API Base URL: https://dashscope.aliyuncs.com/compatible-mode/v1
认证方式:     Bearer {DASHSCOPE_API_KEY}
协议:        OpenAI Chat Completions（兼容模式）
```

### 3.2 流式处理策略

由于 API 强制流式调用，采用以下方案：

**方案：LangChain `ChatOpenAI(streaming=True)` + `ainvoke()`**

LangChain 的 `ChatOpenAI` 在构造时设置 `streaming=True`，底层 HTTP 请求会自动携带 `stream=True`。而 `ainvoke()` 方法会自动收集所有流式 chunk 并返回完整响应。

优势：
- 保持与现有 Provider（ChatGPT/Kimi/Gemini）完全一致的调用模式
- 无需修改 `StrOutputParser` 链式调用逻辑
- 缓存系统可无缝适配

```python
self.model = ChatOpenAI(
    model="qwen3.5-omni-plus",
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    streaming=True,           # 满足 DashScope 强制流式要求
    temperature=0,
    max_tokens=1024,
    model_kwargs={"modalities": ["text"]},  # 仅输出文本（不生成语音）
)
```

### 3.3 多模态消息格式

遵循 OpenAI 兼容的多模态内容数组格式：

**音频输入：**
```python
HumanMessage(content=[
    {"type": "text", "text": prompt},
    {"type": "input_audio", "input_audio": {"data": base64_data, "format": "wav"}},
])
```

**图像输入：**
```python
HumanMessage(content=[
    {"type": "text", "text": prompt},
    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{data}"}},
])
```

**视频输入：**
```python
HumanMessage(content=[
    {"type": "text", "text": prompt},
    {"type": "video_url", "video_url": {"url": f"data:{mime_type};base64,{data}"}},
])
```

---

## 4. 实现方案

### 4.1 新建文件

#### `mer_factory/models/api_models/qwen_omni.py`

```python
class QwenOmniModel:
    """阿里云百炼 Qwen-Omni 全模态模型 Provider"""

    def __init__(
        self,
        api_key: str,
        model_name: str = "qwen3.5-omni-plus",
        verbose: bool = True,
    ):
        # 单一模型实例，原生支持所有模态
        self.model = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            streaming=True,
            temperature=0,
            max_tokens=1024,
            model_kwargs={"modalities": ["text"]},
        )

    # 实现 5 个标准接口方法：
    async def analyze_audio(audio_path, prompt) -> dict
    async def describe_image(image_path, prompt) -> str
    async def describe_video(video_path, prompt) -> str
    async def describe_facial_expression(prompt) -> str
    async def synthesize_summary(prompt) -> str
```

**与 Kimi Provider 的关键差异：**

| 对比项 | KimiModel | QwenOmniModel |
|--------|-----------|---------------|
| 音频支持 | 回退到 Gemini | 原生支持 |
| 流式调用 | 非流式 | 强制流式 (`streaming=True`) |
| 模型实例数 | 3 个（主模型 + 摘要模型 + 音频模型） | 1 个（全模态统一） |
| 额外依赖 | 需要 `google_api_key` | 无 |

### 4.2 修改文件

#### `utils/config.py` — 配置层

```python
# 新增字段
qwen_omni_model: Optional[str] = None

# 新增环境变量
dashscope_api_key: Optional[str] = Field(
    default=os.getenv("DASHSCOPE_API_KEY"), repr=False
)

# 更新 api_key 属性路由
@property
def api_key(self) -> Optional[str]:
    if self.qwen_omni_model:
        return self.dashscope_api_key
    if self.chatgpt_model:
        return self.openai_api_key
    if self.kimi_model:
        return self.moonshot_api_key
    return self.google_api_key
```

#### `mer_factory/models/__init__.py` — 工厂注册

```python
from .api_models.qwen_omni import QwenOmniModel

# 在 model_factory 字典中新增：
"qwen_omni": {
    "condition": qwen_omni_model_name and api_key,
    "class": QwenOmniModel,
    "args": {
        "api_key": api_key,
        "model_name": qwen_omni_model_name,
        "verbose": verbose,
    },
},
```

#### `main.py` — CLI 入口

```python
# 新增 CLI 参数
qwen_omni_model: str = typer.Option(
    None,
    "--qwen-omni-model",
    "-qom",
    help="Qwen Omni model name (e.g., qwen3.5-omni-plus).",
),
```

---

## 5. 配置与使用

### 5.1 环境变量

在 `.env` 文件中添加：

```bash
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
```

### 5.2 CLI 使用示例

```bash
# 全模态情感识别
python main.py ./input ./output -t MER -qom qwen3.5-omni-plus

# 仅音频分析
python main.py ./input ./output -t audio -qom qwen3.5-omni-plus

# 仅视频分析
python main.py ./input ./output -t video -qom qwen3.5-omni-plus

# 仅图像分析
python main.py ./input ./output -t image -qom qwen3.5-omni-plus
```

---

## 6. 依赖项

**无需安装新依赖。** 项目已有的 `langchain-openai` 包含 `ChatOpenAI` 类和底层 `openai` SDK，完全满足 DashScope OpenAI 兼容模式的需求。

---

## 7. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| LangChain `streaming=True` 不正确转发多模态内容 | 模型调用失败 | 降级方案：直接使用 `openai.AsyncOpenAI` 进行原始流式调用 |
| 大文件 Base64 编码内存压力（视频最大 2GB） | OOM | 与现有 Kimi/Gemini Provider 相同风险，可考虑分块编码 |
| `modalities: ["text"]` 参数未被 LangChain 正确透传 | 响应包含不必要的音频数据 | 通过 `model_kwargs` 传递，或在后处理中忽略音频输出 |
| DashScope 视频 content type 格式与 OpenAI 标准不同 | 视频分析失败 | 测试验证 `video_url` 格式，必要时调整为 DashScope 文档格式 |

---

## 8. 文件变更清单

| 操作 | 文件路径 | 说明 |
|------|---------|------|
| 新建 | `mer_factory/models/api_models/qwen_omni.py` | QwenOmniModel 实现 |
| 修改 | `utils/config.py` | 添加配置字段和 API Key 路由 |
| 修改 | `mer_factory/models/__init__.py` | 注册 Provider 到工厂 |
| 修改 | `main.py` | 添加 CLI 参数 |

---

## 9. 验证计划

1. **CLI 验证**：`python main.py --help` 确认 `--qwen-omni-model` 参数出现
2. **单模态测试**：分别用 `-t audio`、`-t video`、`-t image` 测试各模态
3. **全流程测试**：`-t MER` 完整多模态情感识别流程
4. **缓存兼容**：`--cache` 模式下验证 LLM 响应缓存正常工作
5. **错误处理**：未配置 `DASHSCOPE_API_KEY` 时验证错误提示
