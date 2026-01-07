"""
Ollama Client - Interface for local LLM communication
"""

import json
import re
from typing import Optional, Generator, Any
from dataclasses import dataclass

import httpx
import ollama
from rich.console import Console

console = Console()


@dataclass
class OllamaConfig:
    """Configuration for Ollama client."""
    host: str = "http://localhost:11434"
    model: str = "llama3.2"
    timeout: int = 300
    temperature: float = 0.7
    top_p: float = 0.9
    num_ctx: int = 8192


class OllamaClient:
    """
    Client for communicating with locally installed Ollama LLMs.
    
    Supports streaming responses and multiple model configurations
    optimized for code generation.
    """
    
    def __init__(self, config: Optional[OllamaConfig] = None):
        self.config = config or OllamaConfig()
        self.client = ollama.Client(host=self.config.host)
        self._verify_connection()
    
    def _verify_connection(self) -> bool:
        """Verify Ollama is running and accessible."""
        try:
            self.client.list()
            return True
        except Exception as e:
            console.print(f"[red]Error connecting to Ollama: {e}[/red]")
            console.print("[yellow]Make sure Ollama is running: ollama serve[/yellow]")
            raise ConnectionError(f"Cannot connect to Ollama at {self.config.host}")
    
    def list_models(self) -> list[dict]:
        """List all available models."""
        response = self.client.list()
        # Handle both dict (old API) and Pydantic model (new API)
        if hasattr(response, 'models'):
            # New Pydantic response
            return [
                {
                    "name": m.model,
                    "size": m.size,
                    "modified_at": str(m.modified_at) if m.modified_at else None,
                    "details": {
                        "family": m.details.family if m.details else None,
                        "parameter_size": m.details.parameter_size if m.details else None,
                    } if m.details else {}
                }
                for m in response.models
            ]
        # Old dict response
        return response.get("models", [])
    
    def pull_model(self, model_name: str) -> None:
        """Pull a model from Ollama registry."""
        console.print(f"[cyan]Pulling model: {model_name}...[/cyan]")
        self.client.pull(model_name)
        console.print(f"[green]Model {model_name} ready![/green]")
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        stream: bool = False
    ) -> str | Generator[str, None, None]:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt for context
            stream: Whether to stream the response
            
        Returns:
            Generated text or generator for streaming
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        options = {
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "num_ctx": self.config.num_ctx,
        }
        
        if stream:
            return self._stream_generate(messages, options)
        
        response = self.client.chat(
            model=self.config.model,
            messages=messages,
            options=options,
        )
        
        return response["message"]["content"]
    
    def _stream_generate(
        self, 
        messages: list[dict], 
        options: dict
    ) -> Generator[str, None, None]:
        """Stream generate responses token by token."""
        stream = self.client.chat(
            model=self.config.model,
            messages=messages,
            options=options,
            stream=True,
        )
        
        for chunk in stream:
            if chunk.get("message", {}).get("content"):
                yield chunk["message"]["content"]
    
    def generate_code(
        self,
        prompt: str,
        language: str = "python",
        context: Optional[str] = None
    ) -> str:
        """
        Generate code with specialized prompting.
        
        Args:
            prompt: Description of code to generate
            language: Programming language
            context: Additional context or examples
            
        Returns:
            Generated code
        """
        system_prompt = f"""You are an expert {language} programmer specializing in 
mathematical and scientific computing. Generate clean, well-documented code.
Always include necessary imports. Return ONLY the code, no explanations.
When writing Manim code, use the Community Edition (manim) syntax."""
        
        if context:
            prompt = f"Context:\n{context}\n\nTask:\n{prompt}"
        
        response = self.generate(prompt, system_prompt=system_prompt)
        return self._extract_code(response, language)
    
    def _extract_code(self, response: str, language: str) -> str:
        """Extract code blocks from response."""
        # Try to find code blocks with language specifier
        pattern = rf"```(?:{language}|python)?\s*\n(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
        
        if matches:
            return matches[0].strip()
        
        # If no code blocks, try to extract based on imports
        lines = response.split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            if line.strip().startswith(('from ', 'import ', 'class ', 'def ')):
                in_code = True
            if in_code:
                code_lines.append(line)
        
        if code_lines:
            return '\n'.join(code_lines)
        
        # Return as-is if no code detected
        return response.strip()
    
    def chat(
        self,
        messages: list[dict[str, str]],
        stream: bool = False
    ) -> str | Generator[str, None, None]:
        """
        Multi-turn chat interface.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            stream: Whether to stream response
            
        Returns:
            Assistant response
        """
        options = {
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "num_ctx": self.config.num_ctx,
        }
        
        if stream:
            return self._stream_chat(messages, options)
        
        response = self.client.chat(
            model=self.config.model,
            messages=messages,
            options=options,
        )
        
        return response["message"]["content"]
    
    def _stream_chat(
        self,
        messages: list[dict],
        options: dict
    ) -> Generator[str, None, None]:
        """Stream chat responses."""
        stream = self.client.chat(
            model=self.config.model,
            messages=messages,
            options=options,
            stream=True,
        )
        
        for chunk in stream:
            if chunk.get("message", {}).get("content"):
                yield chunk["message"]["content"]
    
    def check_model_available(self, model_name: Optional[str] = None) -> bool:
        """Check if a specific model is available."""
        model = model_name or self.config.model
        models = self.list_models()
        return any(m.get("name", "").startswith(model) for m in models)
    
    def set_model(self, model_name: str) -> None:
        """Change the active model."""
        self.config.model = model_name
        console.print(f"[green]Switched to model: {model_name}[/green]")
