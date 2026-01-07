"""
Main Manai Client - High-level interface for the video generation system
"""

import yaml
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm

from manai.ollama_client import OllamaClient, OllamaConfig
from manai.generator import (
    ManimGenerator,
    ManimConfig,
    GenerationResult,
    CalculusGenerator,
    LinearAlgebraGenerator,
    PhysicsGenerator,
    ThreeDGenerator,
)

console = Console()


@dataclass
class ManaiConfig:
    """Combined configuration for Manai client."""
    ollama: OllamaConfig
    manim: ManimConfig
    
    @classmethod
    def from_yaml(cls, path: str | Path) -> "ManaiConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        
        ollama_data = data.get("ollama", {})
        manim_data = data.get("manim", {})
        
        ollama_config = OllamaConfig(
            host=ollama_data.get("host", "http://localhost:11434"),
            model=ollama_data.get("model", "llama3.2"),
            timeout=ollama_data.get("timeout", 300),
            temperature=ollama_data.get("options", {}).get("temperature", 0.7),
            top_p=ollama_data.get("options", {}).get("top_p", 0.9),
            num_ctx=ollama_data.get("options", {}).get("num_ctx", 8192),
        )
        
        manim_config = ManimConfig(
            quality=manim_data.get("quality", "high"),
            format=manim_data.get("format", "mp4"),
            output_dir=manim_data.get("output_dir", "./output"),
            background_color=manim_data.get("background_color", "#1a1a2e"),
        )
        
        return cls(ollama=ollama_config, manim=manim_config)
    
    @classmethod
    def default(cls) -> "ManaiConfig":
        """Create default configuration."""
        return cls(
            ollama=OllamaConfig(),
            manim=ManimConfig()
        )


class ManaiClient:
    """
    Main client for AI-powered math and physics video generation.
    
    This is the primary interface for interacting with the Manai system.
    It provides both high-level convenience methods and access to
    specialized generators for different topics.
    """
    
    def __init__(self, config: Optional[ManaiConfig] = None, config_path: Optional[str] = None):
        """
        Initialize the Manai client.
        
        Args:
            config: ManaiConfig object
            config_path: Path to YAML configuration file
        """
        if config_path:
            self.config = ManaiConfig.from_yaml(config_path)
        elif config:
            self.config = config
        else:
            self.config = ManaiConfig.default()
        
        # Initialize components
        self.ollama = OllamaClient(self.config.ollama)
        self.generator = ManimGenerator(self.ollama, self.config.manim)
        
        # Specialized generators
        self._calculus: Optional[CalculusGenerator] = None
        self._linear_algebra: Optional[LinearAlgebraGenerator] = None
        self._physics: Optional[PhysicsGenerator] = None
        self._threed: Optional[ThreeDGenerator] = None
        
        console.print("[green]✓ Manai client initialized[/green]")
    
    # =========================================================================
    # SPECIALIZED GENERATOR PROPERTIES
    # =========================================================================
    
    @property
    def calculus(self) -> CalculusGenerator:
        """Get calculus-specialized generator."""
        if self._calculus is None:
            self._calculus = CalculusGenerator(self.ollama, self.config.manim)
        return self._calculus
    
    @property
    def linear_algebra(self) -> LinearAlgebraGenerator:
        """Get linear algebra-specialized generator."""
        if self._linear_algebra is None:
            self._linear_algebra = LinearAlgebraGenerator(self.ollama, self.config.manim)
        return self._linear_algebra
    
    @property
    def physics(self) -> PhysicsGenerator:
        """Get physics-specialized generator."""
        if self._physics is None:
            self._physics = PhysicsGenerator(self.ollama, self.config.manim)
        return self._physics
    
    @property
    def threed(self) -> ThreeDGenerator:
        """Get 3D-specialized generator."""
        if self._threed is None:
            self._threed = ThreeDGenerator(self.ollama, self.config.manim)
        return self._threed
    
    # =========================================================================
    # HIGH-LEVEL GENERATION METHODS
    # =========================================================================
    
    def create(
        self,
        description: str,
        topic: Optional[str] = None,
        level: str = "general",
        render: bool = True,
        preview: bool = True,
    ) -> GenerationResult:
        """
        Create a math/physics animation from a description.
        
        This is the main method for generating videos.
        
        Args:
            description: Natural language description of the animation
            topic: Optional topic category (calculus, physics, geometry, etc.)
            level: Education level (elementary, high_school, undergraduate, etc.)
            render: Whether to render the video immediately
            preview: Whether to preview the video after rendering
            
        Returns:
            GenerationResult with code and output path
            
        Example:
            >>> client = ManaiClient()
            >>> result = client.create(
            ...     "Show how the derivative of sin(x) is cos(x)",
            ...     topic="calculus",
            ...     level="high_school"
            ... )
        """
        self.config.manim.preview = preview
        
        if render:
            return self.generator.generate_and_render(description, topic, level)
        else:
            return self.generator.generate(description, topic, level)
    
    def create_from_equation(
        self,
        equation: str,
        title: Optional[str] = None,
        show_derivation: bool = True,
    ) -> GenerationResult:
        """
        Create an animation explaining an equation or formula.
        
        Args:
            equation: LaTeX equation or mathematical expression
            title: Optional title for the animation
            show_derivation: Whether to show step-by-step derivation
            
        Returns:
            GenerationResult
            
        Example:
            >>> client.create_from_equation(
            ...     r"E = mc^2",
            ...     title="Mass-Energy Equivalence"
            ... )
        """
        description = f"""
        Create an animation explaining the equation: {equation}
        {"Title: " + title if title else ""}
        
        {"Show the derivation step by step." if show_derivation else "Focus on the meaning and applications."}
        
        Include:
        1. Display the equation prominently
        2. Explain each variable/term
        3. {"Show how it's derived" if show_derivation else "Show practical examples"}
        4. Highlight key insights
        """
        
        return self.create(description, topic="equation", level="general")
    
    def create_visualization(
        self,
        concept: str,
        visualization_type: str = "auto",
        dimensions: str = "2d",
    ) -> GenerationResult:
        """
        Create a visual explanation of a concept.
        
        Args:
            concept: The concept to visualize
            visualization_type: Type of visualization (graph, geometric, animation, etc.)
            dimensions: "2d" or "3d"
            
        Returns:
            GenerationResult
        """
        is_3d = dimensions.lower() == "3d"
        
        description = f"""
        Create a {"3D" if is_3d else "2D"} visualization of: {concept}
        
        Visualization approach: {visualization_type if visualization_type != "auto" else "Choose the best approach"}
        
        Make it:
        1. Visually intuitive
        2. Properly labeled
        3. Animated to show relationships
        4. Educational and engaging
        """
        
        topic = "3d_visualization" if is_3d else "2d_visualization"
        return self.create(description, topic=topic)
    
    def create_comparison(
        self,
        concept1: str,
        concept2: str,
        highlight_differences: bool = True,
    ) -> GenerationResult:
        """
        Create a side-by-side comparison animation.
        
        Args:
            concept1: First concept
            concept2: Second concept
            highlight_differences: Whether to emphasize differences
            
        Returns:
            GenerationResult
        """
        description = f"""
        Create a side-by-side comparison of:
        1. {concept1}
        2. {concept2}
        
        Show:
        - Both concepts clearly
        - {"Highlight key differences" if highlight_differences else "Show similarities"}
        - Use consistent visual language
        - Add labels and annotations
        """
        
        return self.create(description, topic="comparison")
    
    def create_proof(
        self,
        theorem: str,
        proof_style: str = "visual",
    ) -> GenerationResult:
        """
        Create an animated mathematical proof.
        
        Args:
            theorem: The theorem to prove
            proof_style: Style of proof ("visual", "algebraic", "geometric")
            
        Returns:
            GenerationResult
        """
        description = f"""
        Create an animated {proof_style} proof of:
        {theorem}
        
        Structure:
        1. State the theorem clearly
        2. Set up any required definitions
        3. Present the proof step by step
        4. Conclude with QED
        
        Make each step visually clear with appropriate pauses.
        """
        
        return self.create(description, topic="proof", level="undergraduate")
    
    # =========================================================================
    # QUICK ACCESS METHODS
    # =========================================================================
    
    def quick_function_plot(
        self,
        function: str,
        x_range: tuple[float, float] = (-5, 5),
        show_derivative: bool = False,
        show_integral: bool = False,
    ) -> GenerationResult:
        """
        Quickly create a function plot animation.
        
        Args:
            function: Function expression (e.g., "x**2", "sin(x)")
            x_range: Range for x-axis
            show_derivative: Also show derivative
            show_integral: Also show integral
            
        Returns:
            GenerationResult
        """
        features = []
        if show_derivative:
            features.append("its derivative")
        if show_integral:
            features.append("its integral")
        
        extra = f" along with {' and '.join(features)}" if features else ""
        
        description = f"""
        Create a clean plot of f(x) = {function} on the interval [{x_range[0]}, {x_range[1]}]{extra}.
        
        Include:
        - Labeled axes
        - Function equation displayed
        - Smooth drawing animation
        {"- Color-coded curves for each function" if features else ""}
        """
        
        return self.create(description, topic="calculus")
    
    def quick_vector_field(
        self,
        field: str,
        dimensions: str = "2d",
    ) -> GenerationResult:
        """
        Quickly create a vector field visualization.
        
        Args:
            field: Vector field expression
            dimensions: "2d" or "3d"
            
        Returns:
            GenerationResult
        """
        is_3d = dimensions.lower() == "3d"
        
        description = f"""
        Visualize the {"3D" if is_3d else "2D"} vector field F = {field}
        
        Show:
        - Vectors at sample points
        - Color by magnitude
        - {"Rotate camera for full view" if is_3d else "Clear 2D representation"}
        - Coordinate axes
        """
        
        if is_3d:
            return self.threed.vector_field_3d(field)
        else:
            return self.create(description, topic="vector_field")
    
    def quick_physics_simulation(
        self,
        scenario: str,
        show_forces: bool = True,
        show_equations: bool = True,
    ) -> GenerationResult:
        """
        Quickly create a physics simulation.
        
        Args:
            scenario: Description of physical scenario
            show_forces: Show force vectors
            show_equations: Show relevant equations
            
        Returns:
            GenerationResult
        """
        description = f"""
        Create a physics simulation of: {scenario}
        
        Include:
        {"- Force vectors with labels" if show_forces else ""}
        {"- Relevant equations of motion" if show_equations else ""}
        - Realistic motion
        - Key physical quantities displayed
        """
        
        return self.create(description, topic="physics")
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def list_models(self) -> list[dict]:
        """List available Ollama models."""
        models = self.ollama.list_models()
        
        table = Table(title="Available Models")
        table.add_column("Name", style="cyan")
        table.add_column("Size", style="green")
        table.add_column("Modified", style="yellow")
        
        for model in models:
            name = model.get("name", "Unknown")
            size = f"{model.get('size', 0) / 1e9:.1f} GB"
            modified = model.get("modified_at", "Unknown")[:10]
            table.add_row(name, size, modified)
        
        console.print(table)
        return models
    
    def set_model(self, model_name: str) -> None:
        """Change the active LLM model."""
        self.ollama.set_model(model_name)
    
    def set_quality(self, quality: str) -> None:
        """Set rendering quality (low, medium, high, production)."""
        self.config.manim.quality = quality
        console.print(f"[green]Quality set to: {quality}[/green]")
    
    def set_output_dir(self, path: str) -> None:
        """Set output directory for rendered videos."""
        self.config.manim.output_dir = path
        Path(path).mkdir(parents=True, exist_ok=True)
        console.print(f"[green]Output directory: {path}[/green]")
    
    def get_last_code(self) -> Optional[str]:
        """Get the last generated code."""
        if self.generator.history:
            return self.generator.history[-1].code
        return None
    
    def save_last_code(self, filename: str) -> Optional[Path]:
        """Save the last generated code to a file."""
        code = self.get_last_code()
        if code:
            return self.generator.save_code(code, filename)
        console.print("[yellow]No code to save[/yellow]")
        return None
    
    def preview_last_code(self) -> None:
        """Display the last generated code with syntax highlighting."""
        code = self.get_last_code()
        if code:
            self.generator.preview_code(code)
        else:
            console.print("[yellow]No code to preview[/yellow]")
    
    def render_code(self, code: str) -> GenerationResult:
        """Render arbitrary Manim code."""
        return self.generator.render(code)
    
    def enhance_last(self, request: str) -> GenerationResult:
        """Enhance the last generated code."""
        code = self.get_last_code()
        if code:
            return self.generator.enhance(code, request)
        console.print("[yellow]No code to enhance[/yellow]")
        return GenerationResult(success=False, code="", error="No code to enhance")
    
    def interactive_session(self) -> None:
        """Start an interactive session for generating videos."""
        console.print(Panel.fit(
            "[bold cyan]Manai Interactive Session[/bold cyan]\n"
            "Create math and physics videos with AI!\n"
            "Type 'help' for commands, 'quit' to exit.",
            border_style="cyan"
        ))
        
        while True:
            try:
                user_input = Prompt.ask("\n[bold green]manai[/bold green]")
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    console.print("[yellow]Goodbye![/yellow]")
                    break
                
                if user_input.lower() == 'help':
                    self._print_help()
                    continue
                
                if user_input.lower() == 'models':
                    self.list_models()
                    continue
                
                if user_input.lower() == 'preview':
                    self.preview_last_code()
                    continue
                
                if user_input.lower().startswith('save '):
                    filename = user_input[5:].strip()
                    self.save_last_code(filename)
                    continue
                
                if user_input.lower().startswith('quality '):
                    quality = user_input[8:].strip()
                    self.set_quality(quality)
                    continue
                
                if user_input.lower().startswith('model '):
                    model = user_input[6:].strip()
                    self.set_model(model)
                    continue
                
                if user_input.lower().startswith('enhance '):
                    request = user_input[8:].strip()
                    result = self.enhance_last(request)
                    if result.success:
                        self.generator.preview_code(result.code)
                    continue
                
                # Default: generate video
                result = self.create(user_input)
                
                if result.success and result.output_path:
                    console.print(f"[green]Video saved: {result.output_path}[/green]")
                elif result.success:
                    self.preview_last_code()
                    if Confirm.ask("Render this code?"):
                        result = self.render_code(result.code)
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Use 'quit' to exit[/yellow]")
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
    
    def _print_help(self) -> None:
        """Print help information."""
        help_text = """
[bold cyan]Manai Commands:[/bold cyan]

[green]Generation:[/green]
  <description>     - Generate a video from description
  enhance <request> - Improve the last generated code

[green]Code Management:[/green]
  preview           - Show the last generated code
  save <filename>   - Save code to file

[green]Settings:[/green]
  quality <level>   - Set quality (low/medium/high/production)
  model <name>      - Change LLM model
  models            - List available models

[green]Navigation:[/green]
  help              - Show this help
  quit              - Exit the session

[bold cyan]Tips:[/bold cyan]
- Be specific about what you want to visualize
- Mention the topic (calculus, physics, etc.) for better results
- Specify the education level for appropriate complexity
        """
        console.print(Panel(help_text, title="Help", border_style="cyan"))
