# MER-Factory Startup 操作手册

<p align="left">
  中文 &nbsp;|&nbsp; <a href="README_en.md">English</a>
</p>

这份 README 只做一件事：帮你把 MER-Factory 从零启动起来，并完成第一次 MER 运行。内容按实际启动顺序组织，覆盖安装依赖、使用 `uv` 初始化环境、下载数据集、初始化 `.env`、运行测试和执行 MER。

## 1. 启动前你需要什么

- Python `3.12`
- `uv`
- `FFmpeg`
- `OpenFace`
- 至少一种可用模型配置
  - Gemini：`GOOGLE_API_KEY`
  - Kimi：`MOONSHOT_API_KEY` 和 `--kimi-model`
  - Qwen Omni：`DASHSCOPE_API_KEY` 和 `--qwen-omni-model`
  - ChatGPT：`OPENAI_API_KEY` 和 `--chatgpt-model`
  - Ollama：本地模型服务
  - Hugging Face：`--huggingface-model`，必要时配合本地 HF API Server

`MER` 和 `AU` 流程依赖 `OpenFace`。如果你只跑 `audio` 或 `video`，可以不先配置 `OpenFace`。

## 2. 克隆项目

```bash
git clone https://github.com/Lum1104/MER-Factory.git
cd MER-Factory
```

如果后续需要训练部分，再初始化子模块：

```bash
git submodule update --init --recursive
```

## 3. 用 uv 初始化 Python 环境

当前仓库还是 `requirements.txt` 依赖管理，因此最稳妥的 `uv` 用法是 `uv venv + uv pip install`：

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Windows PowerShell：

```powershell
uv venv --python 3.12 .venv
.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
```

验证环境：

```bash
python --version
python main.py --help
python export.py --help
```

## 4. 安装系统依赖

### 4.1 安装 FFmpeg

macOS：

```bash
brew install ffmpeg
```

Ubuntu / Debian：

```bash
sudo apt update
sudo apt install -y ffmpeg
```

验证：

```bash
ffmpeg -version
ffprobe -version
```

### 4.2 安装 OpenFace

`MER` 和 `AU` 流程都需要 `OpenFace` 的 `FeatureExtraction` 可执行文件。

macOS / Linux：

- 按 OpenFace 官方说明或仓库示例笔记本编译
- 记下最终可执行文件绝对路径，通常类似：
  - `/path/to/OpenFace/build/bin/FeatureExtraction`

Windows：

1. 从 OpenFace Releases 下载预编译版本
2. 运行 `download_models.ps1`
3. 记下 `FeatureExtraction.exe` 的绝对路径

验证时你只需要确认这个文件真实存在：

```bash
ls /path/to/OpenFace/build/bin/FeatureExtraction
```

## 5. 下载并准备数据集

项目可以直接处理单个文件，也可以递归扫描目录中的媒体文件，所以不强制要求特定数据集目录结构。为了快速启动，推荐先拿 CH-SIMS 的原始视频片段做第一轮验证。

### 5.1 下载 CH-SIMS

1. 打开 CH-SIMS / CH-SIMS v2.0 页面
2. 下载原始视频和配套 CSV
3. 解压到项目目录下，例如：

```text
dataset/
  video_0001/
    0001.mp4
    0002.mp4
  video_0002/
    0001.mp4
  ch_sims_metadata.csv
```

### 5.2 MER-Factory 推荐目录

为了后续命令统一，建议保留一个清晰的工作区：

```text
MER-Factory/
  dataset/      # 原始视频
  input/        # 可选：想先单独跑的小样本
  output/       # MER 输出
  test_output/  # 测试脚本输出
```

如果你只是想先跑一个最小样本：

```bash
mkdir -p input
cp dataset/video_0001/0001.mp4 input/
```

### 5.3 关于 `--label-file`

第一次启动不建议先接标签文件，直接跑通流程即可。原因是仓库里的 `--label-file` 读取器要求 CSV 至少有两列：

```csv
name,label
0001,Negative
0002,Positive
```

CH-SIMS 原始 CSV 通常包含更多列，不能直接原样传给 `--label-file`。如果你后面确实要加标签监督，再单独整理成 `name,label` 两列格式。

## 6. 初始化 .env

复制模板：

```bash
cp .env.example .env
```

最小可用配置示例：

```env
# 默认 Gemini
GOOGLE_API_KEY=your_google_api_key

# 使用 Kimi 时需要
MOONSHOT_API_KEY=your_kimi_key

# 使用 Qwen Omni 时需要
DASHSCOPE_API_KEY=your_dashscope_key

# 使用 ChatGPT 时需要
OPENAI_API_KEY=your_openai_key

# MER / AU 必填
OPENFACE_EXECUTABLE=/absolute/path/to/OpenFace/build/bin/FeatureExtraction

# 仅在 Hugging Face API Server 模式下需要
HF_API_BASE_URL="http://localhost:7860/"
```

几条关键映射关系：

- 不传模型参数时，项目默认走 Gemini，需要 `GOOGLE_API_KEY`
- 使用 Kimi 时要传 `--kimi-model`，并在 `.env` 里设置 `MOONSHOT_API_KEY`
- 使用 Qwen Omni 时要传 `--qwen-omni-model`，并在 `.env` 里设置 `DASHSCOPE_API_KEY`
- Kimi 的音频分析在当前实现里会回退到 Gemini，所以跑带音频的 MER 时，建议同时保留 `GOOGLE_API_KEY`
- 使用 ChatGPT 时要传 `--chatgpt-model`，并在 `.env` 里设置 `OPENAI_API_KEY`
- `OPENFACE_EXECUTABLE` 必须是绝对路径

### 6.1 如需离线预下载 Hugging Face 模型

如果你准备使用 `--huggingface-model`，或者你的网络环境不方便在首次运行时在线拉取权重，建议先把需要的模型下载到 Hugging Face 默认缓存目录。这里不要再使用旧的 `huggingface-cli download`，现在统一用 `hf download`，也不需要传 `--local-dir`。

先安装 CLI：

```bash
uv pip install --python .venv/bin/python "huggingface_hub[cli]"
```

如果你在受限网络环境，也可以先设置镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

MER-Factory 当前注册的 Hugging Face 推理模型有：

- `google/gemma-3n-E4B-it`
- `google/gemma-3n-E2B-it`
- `Qwen/Qwen2-Audio-7B-Instruct`
- `zhifeixie/Audio-Reasoner`
- `Qwen/Qwen2.5-Omni-7B`
- `Qwen/Qwen2.5-Omni-3B`
- `openai/whisper-base`
- `openai/whisper-small`

按需预下载示例：

```bash
hf download google/gemma-3n-E4B-it
hf download Qwen/Qwen2.5-Omni-7B
hf download Qwen/Qwen2-Audio-7B-Instruct
hf download zhifeixie/Audio-Reasoner
hf download openai/whisper-base
```

如果你还要跑评估，建议额外把这些评估依赖模型也先下好：

- `laion/CLIP-ViT-B-32-laion2B-s34B-b79K`
- `lukewys/laion_clap`
- `roberta-base`
- `microsoft/deberta-large-mnli`
- `openai/whisper-base`

对应命令：

```bash
hf download laion/CLIP-ViT-B-32-laion2B-s34B-b79K
hf download lukewys/laion_clap
hf download roberta-base
hf download microsoft/deberta-large-mnli
hf download openai/whisper-base
```

这些命令默认会下载到 Hugging Face 缓存目录，通常是 `~/.cache/huggingface/`。后续 MER-Factory 和评估脚本都会直接复用这里的缓存。

## 7. 运行安装测试

先用一段真实视频验证 `FFmpeg` 和 `OpenFace` 是否工作正常。下面的命令可以直接替换成你的文件路径：

```bash
python test/test_ffmpeg.py dataset/video_0001/0001.mp4 test_output/
python test/test_openface.py dataset/video_0001/0001.mp4 test_output/
```

预期结果：

- `test_output/0001_audio.wav`
- `test_output/0001_middle_frame.png`
- `test_output/0001.csv`

如果这两步都通过，说明本机的系统依赖已经就绪。

## 8. 运行第一次 MER

### 8.1 跑单个视频

默认 Gemini：

```bash
python main.py input/0001.mp4 output/ --type MER --silent
```

显式指定 Gemini 模型：

```bash
python main.py input/0001.mp4 output/ --type MER --gemini-model gemini-3.1-flash-lite-preview --silent
```

使用 Kimi：

```bash
python main.py input/0001.mp4 output/ --type MER --kimi-model kimi-k2.5 --silent
```

使用 Qwen Omni：

```bash
python main.py input/0001.mp4 output/ --type MER --qwen-omni-model qwen3.5-omni-plus --silent
```

使用 ChatGPT：

```bash
python main.py input/0001.mp4 output/ --type MER --chatgpt-model gpt-4o --silent
```

### 8.2 跑整个目录

项目会递归扫描目录中的视频、音频和图像文件，所以 CH-SIMS 这种多层目录可以直接跑：

```bash
python main.py dataset/ output/ --type MER --silent
```

如果你只想跑某一个子目录：

```bash
python main.py dataset/video_0001/ output/ --type MER --silent
```

### 8.3 常用参数

```bash
python main.py dataset/ output/ \
  --type MER \
  --threshold 0.8 \
  --concurrency 4 \
  --cache \
  --silent
```

常见参数含义：

- `--type MER`：完整多模态流程
- `--threshold`：情绪检测阈值，默认 `0.8`
- `--concurrency`：异步并发文件数，默认 `4`
- `--cache`：复用已有中间结果并缓存 LLM 调用
- `--silent`：减少日志输出

## 9. 结果怎么看

运行结束后，`output/` 下通常会看到：

- `*_merr_data.json`：最终 MER 结果
- `*_au_analysis.json`：面部 AU 分析
- `*_audio_analysis.json`：音频分析
- `*_video_analysis.json`：视频描述
- `error_logs/`：失败样本的错误日志

你也可能看到中间产物，例如提取出的音频或关键帧。

## 10. 常见启动路径

### 10.1 只验证环境

```bash
python test/test_ffmpeg.py input/0001.mp4 test_output/
python test/test_openface.py input/0001.mp4 test_output/
```

### 10.2 先跑一个样本，再跑整目录

```bash
python main.py input/0001.mp4 output/ --type MER --silent
python main.py dataset/ output/ --type MER --cache --silent
```

### 10.3 导出结果

导出为 CSV：

```bash
python export.py --output_folder output/ --file_type mer --export_csv --export_path output/mer.csv
```

导出为 ShareGPT 格式：

```bash
python export.py --output_folder output/ --file_type mer --export_format sharegpt --export_path output/mer_sharegpt.json
```

### 10.4 运行评估前先做准备

不要在新环境里直接跑下面这条命令：

```bash
python tools/evaluate.py output/ --export-csv output/evaluation_summary.csv
```

评估不是零依赖命令。第一次运行前，至少先确认：

- 已激活 `.venv` 并完成 `uv pip install -r requirements.txt`
- `torch` 和 `torchaudio` 可以一起正常导入
- 可以访问 Hugging Face，或者已经配置 `HF_ENDPOINT=https://hf-mirror.com`
- 接受首次会下载评估模型，不是纯本地秒开

评估依赖的主要模型包括：

- `laion/CLIP-ViT-B-32-laion2B-s34B-b79K`：用于 `CLIP` 图像分数
- `lukewys/laion_clap`：用于 `CLAP` 音频主检查点（`630k-audioset-best.pt`）
- `roberta-base`：`LAION-CLAP` 的文本编码器依赖
- `microsoft/deberta-large-mnli`：用于 `NLI` 一致性
- `openai/whisper-base`：用于 `ASR WER`

如果这些依赖没有准备好，脚本可能还能结束，但 `CLAP=0.5` 或 `WER=0.0` 这类结果很可能只是 fallback 值，不代表真实质量。完整前置步骤、镜像用法和环境校验请先看 [tools/evaluate/README.md](/Users/kirinekoclaw/github/MER-Factory/tools/evaluate/README.md)。

## 11. 排查建议

- `ffmpeg` / `ffprobe` 找不到：先检查 PATH 和 `ffmpeg -version`
- `OpenFace executable not found`：检查 `.env` 中的 `OPENFACE_EXECUTABLE` 是否为绝对路径
- 直接跑 `MER` 时报 API 相关错误：检查你选择的模型和 `.env` 中的 key 是否对应
- 直接跑评估结果异常“整齐”：先检查 `torch` / `torchaudio` / Hugging Face 模型是否真的准备完成
- 传了 `--label-file` 却报 CSV 列不对：把标签文件整理成 `name,label`
- 目录能打开但没有处理任何文件：确认目录下是 `mp4/avi/mov/mkv/flv/wmv/jpg/jpeg/png/bmp/wav/mp3/flac/m4a`

## 12. 参考资料

- 官方入门文档：https://lum.is-a.dev/MER-Factory/zh/docs/getting-started
- CH-SIMS / CH-SIMS v2.0 数据集页面：https://thuiar.github.io/sims.github.io/chsims
- OpenFace：https://github.com/TadasBaltrusaitis/OpenFace/wiki
