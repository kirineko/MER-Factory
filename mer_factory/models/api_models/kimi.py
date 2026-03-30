from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from pathlib import Path
from rich.console import Console
import base64
import mimetypes
import asyncio


console = Console(stderr=True)


class KimiModel:
    """
    A class to handle all interactions with the Moonshot Kimi API.
    The API is OpenAI-compatible and uses the Moonshot base URL.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "kimi-k2.5",
        summary_model_name: str = "kimi-k2-turbo-preview",
        google_api_key: str = None,
        audio_model_name: str = "gemini-3.1-flash-lite-preview",
        verbose: bool = True,
    ):
        self.verbose = verbose
        self.model_name = model_name
        self.summary_model_name = summary_model_name
        self.audio_model_name = audio_model_name

        if not api_key:
            raise ValueError("Moonshot API key is required for KimiModel.")

        self.model = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url="https://api.moonshot.cn/v1",
            temperature=1,
            max_tokens=1024,
        )
        self.vision_model = self.model
        self.summary_model = ChatOpenAI(
            model=summary_model_name,
            api_key=api_key,
            base_url="https://api.moonshot.cn/v1",
            temperature=1,
            max_tokens=1024,
        )
        self.audio_model = None

        if google_api_key:
            self.audio_model = ChatGoogleGenerativeAI(
                model=audio_model_name,
                google_api_key=google_api_key,
                temperature=0,
            )

        if self.verbose:
            console.log(f"Kimi model initialized: {self.model_name}")
            console.log(
                f"Kimi summary model initialized: {self.summary_model_name}"
            )
            if self.audio_model:
                console.log(
                    f"Kimi audio fallback initialized with Gemini: {self.audio_model_name}"
                )

    async def describe_facial_expression(self, prompt: str) -> str:
        """Generates a description from AU text using Kimi."""
        try:
            chain = self.model | StrOutputParser()
            return await chain.ainvoke(prompt)
        except Exception as e:
            console.log(
                f"[bold red]❌ Error describing facial expression: {e}[/bold red]"
            )
            raise

    async def describe_image(self, image_path: Path, prompt: str) -> str:
        """Generates a description for an image file using Kimi."""
        try:
            image_data = await asyncio.to_thread(
                lambda: base64.b64encode(image_path.read_bytes()).decode("utf-8")
            )
            suffix = image_path.suffix.lower().lstrip(".") or "png"
            message = HumanMessage(
                content=[
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{suffix};base64,{image_data}"
                        },
                    },
                    {"type": "text", "text": prompt},
                ]
            )
            chain = self.vision_model | StrOutputParser()
            return await chain.ainvoke([message])
        except Exception as e:
            console.log(f"[bold red]❌ Error describing image: {e}[/bold red]")
            raise

    async def analyze_audio(self, audio_path: Path, prompt: str) -> dict:
        """
        Analyzes an audio file using Gemini as a fallback audio model.
        Kimi K2.5 remains responsible for text/image/video/summary tasks.
        """
        if not self.audio_model:
            console.log(
                "[bold red]❌ Error analyzing audio: GOOGLE_API_KEY is required for Kimi audio fallback.[/bold red]"
            )
            return {"transcript": "", "tone_description": ""}
        try:
            audio_data = await asyncio.to_thread(
                lambda: base64.b64encode(audio_path.read_bytes()).decode("utf-8")
            )
            mime_type = mimetypes.guess_type(audio_path)[0] or "audio/wav"
            message = HumanMessage(
                content=[
                    {
                        "type": "media",
                        "data": audio_data,
                        "mime_type": mime_type,
                    },
                    {"type": "text", "text": prompt},
                ]
            )
            chain = self.audio_model | StrOutputParser()
            return await chain.ainvoke([message])
        except Exception as e:
            console.log(f"[bold red]❌ Error analyzing audio: {e}[/bold red]")
            return {"transcript": "", "tone_description": ""}

    async def describe_video(self, video_path: Path, prompt: str) -> str:
        """Generates a description for a video using Kimi."""
        try:
            video_data = await asyncio.to_thread(
                lambda: base64.b64encode(video_path.read_bytes()).decode("utf-8")
            )
            suffix = video_path.suffix.lower().lstrip(".") or "mp4"
            message = HumanMessage(
                content=[
                    {
                        "type": "video_url",
                        "video_url": {
                            "url": f"data:video/{suffix};base64,{video_data}"
                        },
                    },
                    {"type": "text", "text": prompt},
                ]
            )
            chain = self.vision_model | StrOutputParser()
            return await chain.ainvoke([message])
        except Exception as e:
            console.log(f"[bold red]❌ Error describing video: {e}[/bold red]")
            raise

    async def synthesize_summary(self, prompt: str) -> str:
        """Synthesizes a final summary from context using Kimi."""
        try:
            chain = self.summary_model | StrOutputParser()
            return await chain.ainvoke(prompt)
        except Exception as e:
            console.log(f"[bold red]❌ Error synthesizing summary: {e}[/bold red]")
            raise
