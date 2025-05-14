#!/usr/bin/env python
# coding: utf-8
"""
Entry point for the Interview Application CLI.
Defines Typer commands and invokes the interview flow logic.
"""
import typer
from .interview_flow import start_full_interview
from rich.console import Console

console = Console()
app = typer.Typer(help="Interactive interview application")

@app.command("interview")
def interview_command(
    resume_path: str = typer.Argument(..., help="Path to the candidate's PDF resume file"),
    output_video: str = typer.Option(
        "interview_recording.avi", "-o", "--output-video",
        help="Output video file path for the recorded interview"
    )
):
    """
    Start a full mock interview using the given PDF resume.
    """
    console.print(f"[bold green]Starting interview with resume:[/bold green] {resume_path}")
    try:
        start_full_interview(file_path=resume_path, output_video=output_video)
    except Exception as e:
        console.print(f"[bold red]Error during interview:[/bold red] {e}")
        raise

if __name__ == "__main__":
    app()
