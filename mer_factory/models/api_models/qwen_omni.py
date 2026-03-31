from pathlib import Path
from rich.console import Console
import asyncio
import base64
import mimetypes

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI


console = Console(stderr=True)


class QwenOmniModel:
    """
    A provider for DashScope Qwen Omni models via the OpenAI-compatible API.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "qwen3.5-omni-plus",
        verbose: bool = True,
    ):
        self.verbose = verbose
        self.model_name = model_name
        extra_body = {}

        if not api_key:
            raise ValueError("DashScope API key is required for QwenOmniModel.")

        # Qwen3-Omni-Flash is a hybrid thinking model; multimodal calls should
        # use non-thinking mode unless the caller explicitly implements both paths.
        if model_name.startswith("qwen3-omni-flash"):
            extra_body["enable_thinking"] = False

        self.model = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            streaming=True,
            temperature=0,
            max_tokens=1024,
            model_kwargs={"modalities": ["text"]},
            extra_body=extra_body or None,
        )

        if self.verbose:
            console.log(f"Qwen Omni model initialized: {self.model_name}")

    async def _encode_file(self, file_path: Path) -> str:
        return await asyncio.to_thread(
            lambda: base64.b64encode(file_path.read_bytes()).decode("utf-8")
        )

    async def _to_data_url(self, file_path: Path, fallback_mime_type: str) -> str:
        file_data = await self._encode_file(file_path)
        mime_type = mimetypes.guess_type(file_path)[0] or fallback_mime_type
        return f"data:{mime_type};base64,{file_data}"

    async def describe_facial_expression(self, prompt: str) -> str:
        """Generates a description from AU text using Qwen Omni."""
        try:
            chain = self.model | StrOutputParser()
            return await chain.ainvoke(prompt)
        except Exception as e:
            console.log(
                f"[bold red]❌ Error describing facial expression: {e}[/bold red]"
            )
            raise

    async def describe_image(self, image_path: Path, prompt: str) -> str:
        """Generates a description for an image file using Qwen Omni."""
        try:
            image_data_url = await self._to_data_url(image_path, "image/png")
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url},
                    },
                ]
            )
            chain = self.model | StrOutputParser()
            return await chain.ainvoke([message])
        except Exception as e:
            console.log(f"[bold red]❌ Error describing image: {e}[/bold red]")
            raise

    async def analyze_audio(self, audio_path: Path, prompt: str) -> dict:
        """Analyzes an audio file using Qwen Omni."""
        try:
            audio_data_url = await self._to_data_url(audio_path, "audio/wav")
            audio_format = audio_path.suffix.lower().lstrip(".") or "wav"
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_data_url,
                            "format": audio_format,
                        },
                    },
                ]
            )
            chain = self.model | StrOutputParser()
            return await chain.ainvoke([message])
        except Exception as e:
            console.log(f"[bold red]❌ Error analyzing audio: {e}[/bold red]")
            return {"transcript": "", "tone_description": ""}

    async def describe_video(self, video_path: Path, prompt: str) -> str:
        """Generates a description for a video using Qwen Omni."""
        try:
            video_data_url = await self._to_data_url(video_path, "video/mp4")
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "video_url",
                        "video_url": {"url": video_data_url},
                    },
                ]
            )
            chain = self.model | StrOutputParser()
            return await chain.ainvoke([message])
        except Exception as e:
            console.log(f"[bold red]❌ Error describing video: {e}[/bold red]")
            raise

    async def synthesize_summary(self, prompt: str) -> str:
        """Synthesizes a final summary from context using Qwen Omni."""
        try:
            chain = self.model | StrOutputParser()
            return await chain.ainvoke(prompt)
        except Exception as e:
            console.log(f"[bold red]❌ Error synthesizing summary: {e}[/bold red]")
            raise
