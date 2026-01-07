"""
Voiceover Module - Natural TTS and Closed Captions
Uses Microsoft Edge TTS for high-quality voices
"""

import asyncio
import os
import re
import subprocess
from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass

try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False

try:
    import pysrt
    PYSRT_AVAILABLE = True
except ImportError:
    PYSRT_AVAILABLE = False


@dataclass
class VoiceConfig:
    """Configuration for voice synthesis."""
    voice: str = "en-US-AriaNeural"  # Natural female voice
    rate: str = "+0%"  # Speech rate adjustment
    pitch: str = "+0Hz"  # Pitch adjustment
    volume: str = "+0%"  # Volume adjustment


# Available natural voices
VOICES = {
    # English - US
    "aria": "en-US-AriaNeural",  # Female, natural, warm
    "guy": "en-US-GuyNeural",  # Male, natural
    "jenny": "en-US-JennyNeural",  # Female, casual
    "davis": "en-US-DavisNeural",  # Male, professional
    "amber": "en-US-AmberNeural",  # Female, friendly
    "andrew": "en-US-AndrewNeural",  # Male, conversational
    # English - UK
    "sonia": "en-GB-SoniaNeural",  # Female, British
    "ryan": "en-GB-RyanNeural",  # Male, British
    # English - AU
    "natasha": "en-AU-NatashaNeural",  # Female, Australian
}


@dataclass
class SubtitleEntry:
    """A single subtitle entry."""
    index: int
    start_ms: int
    end_ms: int
    text: str


class VoiceoverEngine:
    """
    Engine for generating natural voice-overs and closed captions.
    """
    
    def __init__(self, config: Optional[VoiceConfig] = None):
        self.config = config or VoiceConfig()
        self.output_dir = Path("./output/audio")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def set_voice(self, voice_name: str):
        """Set voice by name."""
        if voice_name.lower() in VOICES:
            self.config.voice = VOICES[voice_name.lower()]
        else:
            self.config.voice = voice_name
    
    def list_voices(self) -> List[str]:
        """List available voice names."""
        return list(VOICES.keys())
    
    async def generate_audio_async(
        self,
        text: str,
        output_path: Optional[Path] = None
    ) -> Tuple[Path, List[SubtitleEntry]]:
        """
        Generate audio from text with word-level timing for subtitles.
        
        Returns:
            Tuple of (audio_path, subtitle_entries)
        """
        if not EDGE_TTS_AVAILABLE:
            raise RuntimeError("edge-tts not installed. Run: pip install edge-tts")
        
        if output_path is None:
            output_path = self.output_dir / f"voiceover_{hash(text) % 10000}.mp3"
        
        output_path = Path(output_path)
        
        # Create communicate object for streaming subtitles
        communicate = edge_tts.Communicate(
            text,
            self.config.voice,
            rate=self.config.rate,
            pitch=self.config.pitch,
            volume=self.config.volume
        )
        
        # Collect subtitle data and audio in one pass
        subtitles: List[SubtitleEntry] = []
        submaker = edge_tts.SubMaker()
        audio_data = b""
        
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
            elif chunk["type"] == "WordBoundary":
                submaker.feed(chunk)
        
        # Save audio data to file
        with open(output_path, "wb") as f:
            f.write(audio_data)
        
        # Generate VTT subtitles
        vtt_content = submaker.generate_subs()
        
        # Parse VTT to our format
        subtitles = self._parse_vtt(vtt_content)
        
        return output_path, subtitles
    
    def generate_audio(
        self,
        text: str,
        output_path: Optional[Path] = None
    ) -> Tuple[Path, List[SubtitleEntry]]:
        """Synchronous wrapper for audio generation."""
        return asyncio.run(self.generate_audio_async(text, output_path))
    
    def _parse_vtt(self, vtt_content: str) -> List[SubtitleEntry]:
        """Parse VTT content to subtitle entries."""
        entries = []
        lines = vtt_content.strip().split('\n')
        
        i = 0
        index = 1
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Look for timestamp line
            if '-->' in line:
                # Parse timestamps
                parts = line.split(' --> ')
                if len(parts) == 2:
                    start_ms = self._vtt_to_ms(parts[0])
                    end_ms = self._vtt_to_ms(parts[1])
                    
                    # Next line is the text
                    i += 1
                    if i < len(lines):
                        text = lines[i].strip()
                        if text:
                            entries.append(SubtitleEntry(
                                index=index,
                                start_ms=start_ms,
                                end_ms=end_ms,
                                text=text
                            ))
                            index += 1
            i += 1
        
        return entries
    
    def _vtt_to_ms(self, timestamp: str) -> int:
        """Convert VTT timestamp to milliseconds."""
        # Format: HH:MM:SS.mmm or MM:SS.mmm
        parts = timestamp.replace(',', '.').split(':')
        
        if len(parts) == 3:
            h, m, s = parts
            h = int(h)
        else:
            h = 0
            m, s = parts
        
        m = int(m)
        s = float(s)
        
        return int((h * 3600 + m * 60 + s) * 1000)
    
    def generate_srt(
        self,
        entries: List[SubtitleEntry],
        output_path: Path
    ) -> Path:
        """Generate SRT subtitle file."""
        output_path = Path(output_path)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in entries:
                f.write(f"{entry.index}\n")
                f.write(f"{self._ms_to_srt(entry.start_ms)} --> {self._ms_to_srt(entry.end_ms)}\n")
                f.write(f"{entry.text}\n\n")
        
        return output_path
    
    def _ms_to_srt(self, ms: int) -> str:
        """Convert milliseconds to SRT timestamp format."""
        hours = ms // 3600000
        ms %= 3600000
        minutes = ms // 60000
        ms %= 60000
        seconds = ms // 1000
        milliseconds = ms % 1000
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
    
    def merge_audio_video(
        self,
        video_path: Path,
        audio_path: Path,
        output_path: Optional[Path] = None,
        subtitle_path: Optional[Path] = None
    ) -> Path:
        """
        Merge audio with video and optionally burn in subtitles.
        Uses ffmpeg.
        """
        video_path = Path(video_path)
        audio_path = Path(audio_path)
        
        if output_path is None:
            output_path = video_path.parent / f"{video_path.stem}_voiced{video_path.suffix}"
        
        output_path = Path(output_path)
        
        # Build ffmpeg command
        cmd = ["ffmpeg", "-y", "-i", str(video_path), "-i", str(audio_path)]
        
        if subtitle_path and Path(subtitle_path).exists():
            # Burn in subtitles
            cmd.extend([
                "-vf", f"subtitles={subtitle_path}:force_style='FontSize=24,PrimaryColour=&Hffffff,OutlineColour=&H000000,Outline=2'",
            ])
        
        cmd.extend([
            "-c:v", "libx264",
            "-c:a", "aac",
            "-shortest",  # Match shortest stream length
            str(output_path)
        ])
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return output_path
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg error: {e.stderr.decode()}")
    
    def generate_narration_script(
        self,
        manim_code: str,
        description: str
    ) -> str:
        """
        Generate a narration script from Manim code and description.
        This extracts visual elements and creates natural narration text.
        """
        # Extract key elements from code
        elements = []
        
        # Find MathTex/Tex content
        math_pattern = r'(?:MathTex|Tex)\(["\'](.+?)["\']\)'
        for match in re.finditer(math_pattern, manim_code):
            elements.append(f"equation: {match.group(1)}")
        
        # Find Text content
        text_pattern = r'Text\(["\'](.+?)["\']\)'
        for match in re.finditer(text_pattern, manim_code):
            elements.append(match.group(1))
        
        # Find comments that might be narration hints
        comment_pattern = r'#\s*(?:Narration|Voice|Say):\s*(.+)'
        for match in re.finditer(comment_pattern, manim_code, re.IGNORECASE):
            elements.append(match.group(1))
        
        # Build script
        if elements:
            script = f"Let's explore {description}. "
            script += " ".join(elements[:5])  # First few elements
            return script
        
        return f"In this visualization, we will explore {description}."


# Singleton instance
voiceover_engine = VoiceoverEngine()


def generate_voiceover(
    text: str,
    voice: str = "aria",
    output_dir: str = "./output/audio"
) -> Tuple[str, str]:
    """
    Generate voiceover audio and subtitles.
    
    Args:
        text: The text to speak
        voice: Voice name (aria, guy, jenny, etc.)
        output_dir: Directory for output files
        
    Returns:
        Tuple of (audio_path, srt_path)
    """
    engine = VoiceoverEngine()
    engine.set_voice(voice)
    engine.output_dir = Path(output_dir)
    engine.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate audio and get timing
    audio_path, subtitles = engine.generate_audio(text)
    
    # Generate SRT
    srt_path = audio_path.with_suffix('.srt')
    engine.generate_srt(subtitles, srt_path)
    
    return str(audio_path), str(srt_path)
