import asyncio
from pathlib import Path
from rich.console import Console
import os
import subprocess

console = Console(stderr=True)


def _to_openface_path(path: Path, openface_executable: Path) -> str:
    """Convert paths for Windows OpenFace binaries running under WSL."""
    resolved = path.resolve()
    if os.name != "nt" and openface_executable.suffix.lower() == ".exe":
        converted = subprocess.run(
            ["wslpath", "-w", str(resolved)],
            check=True,
            capture_output=True,
            text=True,
        )
        return converted.stdout.strip()
    return str(resolved)


class OpenFaceAdapter:
    """A wrapper for the OpenFace FeatureExtraction tool using asyncio."""

    @staticmethod
    async def run_feature_extraction(
        video_path: Path, output_dir: Path, verbose: bool = True
    ) -> bool:
        """Runs OpenFace feature extraction asynchronously."""
        output_csv = output_dir / f"{video_path.stem}.csv"
        if output_csv.exists():
            if verbose:
                console.log(
                    f"OpenFace output exists: [cyan]{output_csv}[/cyan]. Skipping."
                )
            return True

        openface_executable = os.getenv("OPENFACE_EXECUTABLE")
        if not openface_executable:
            console.log(
                "[bold red]❌ Error: OPENFACE_EXECUTABLE not set in environment variables.[/bold red]"
            )
            console.log(
                "Please set OPENFACE_EXECUTABLE in your .env file to the correct path."
            )
            return False

        if os.name == "nt" and not openface_executable.endswith(".exe"):
            openface_executable += ".exe"

        if not os.path.exists(openface_executable):
            console.log(
                f"[bold red]❌ Error: OpenFace executable not found at '{openface_executable}'[/bold red]"
            )
            console.log(
                "Please set OPENFACE_EXECUTABLE in your .env file to the correct path."
            )
            return False

        openface_executable_path = Path(openface_executable)
        video_path_arg = _to_openface_path(video_path, openface_executable_path)
        output_dir_arg = _to_openface_path(output_dir, openface_executable_path)
        command = [
            openface_executable,
            "-f",
            video_path_arg,
            "-out_dir",
            output_dir_arg,
            "-aus",  # Extract Action Units
        ]

        if verbose:
            console.log(f"Running OpenFace on [magenta]{video_path.name}[/magenta]...")

        try:
            proc = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(openface_executable_path.parent),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                console.log(f"❌ OpenFace failed to process {video_path.name}.")
                console.log(f"OpenFace Error: {stderr.decode().strip()}")
                return False

            if verbose:
                console.log(
                    f"✅ OpenFace analysis complete. Output in [green]{output_dir}[/green]"
                )
            return True

        except FileNotFoundError:
            console.log(
                f"[bold red]❌ Error: '{openface_executable}' not found.[/bold red]"
            )
            console.log(
                "Please ensure OPENFACE_EXECUTABLE in your .env file points to the correct path."
            )
            return False
