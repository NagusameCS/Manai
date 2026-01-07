"""
Manim Code Generator - Generates and executes Manim animations
"""

import os
import re
import sys
import subprocess
import tempfile
import traceback
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime

from rich.console import Console
from rich.syntax import Syntax
from rich.panel import Panel

from manai.ollama_client import OllamaClient, OllamaConfig
from manai.prompts import (
    get_prompt_for_topic,
    MANIM_EXPERT_SYSTEM,
    GENERATE_FROM_DESCRIPTION,
    FIX_CODE_ERRORS,
    ENHANCE_EXISTING_CODE,
    EXPLAIN_AND_GENERATE,
)

console = Console()


@dataclass
class ManimConfig:
    """Configuration for Manim rendering."""
    quality: str = "high"
    format: str = "mp4"
    output_dir: str = "./output"
    preview: bool = True
    save_last_frame: bool = False
    background_color: str = "#1a1a2e"
    
    # Quality presets
    quality_flags: dict = field(default_factory=lambda: {
        "low": "-ql",
        "medium": "-qm", 
        "high": "-qh",
        "production": "-qp",
        "fourk": "-qk",
    })


@dataclass
class GenerationResult:
    """Result of a code generation attempt."""
    success: bool
    code: str
    output_path: Optional[str] = None
    error: Optional[str] = None
    attempts: int = 1


class ManimGenerator:
    """
    Generates Manim code using LLM and renders it to video.
    
    This class handles the full pipeline from natural language
    description to rendered video output.
    """
    
    def __init__(
        self,
        ollama_client: Optional[OllamaClient] = None,
        manim_config: Optional[ManimConfig] = None
    ):
        self.llm = ollama_client or OllamaClient()
        self.config = manim_config or ManimConfig()
        self.history: list[GenerationResult] = []
        
        # Ensure output directory exists
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def generate(
        self,
        description: str,
        topic: Optional[str] = None,
        level: str = "general",
        scene_name: Optional[str] = None,
    ) -> GenerationResult:
        """
        Generate Manim code from a description.
        
        Args:
            description: Natural language description of the animation
            topic: Optional topic category (calculus, physics, etc.)
            level: Education level (elementary, high_school, undergraduate, etc.)
            scene_name: Optional name for the scene class
            
        Returns:
            GenerationResult with generated code
        """
        console.print(Panel(f"[bold cyan]Generating animation for:[/bold cyan]\n{description}"))
        
        # Get appropriate prompts
        if topic:
            system_prompt, user_prompt = get_prompt_for_topic(topic, level, description)
        else:
            system_prompt = MANIM_EXPERT_SYSTEM
            user_prompt = GENERATE_FROM_DESCRIPTION.format(description=description)
        
        # Generate code
        try:
            code = self.llm.generate(user_prompt, system_prompt=system_prompt)
            code = self._clean_code(code)
            
            # Validate basic structure
            if not self._validate_code_structure(code):
                console.print("[yellow]Warning: Generated code may have structural issues[/yellow]")
            
            result = GenerationResult(success=True, code=code)
            self.history.append(result)
            
            return result
            
        except Exception as e:
            error_msg = str(e)
            console.print(f"[red]Generation error: {error_msg}[/red]")
            result = GenerationResult(success=False, code="", error=error_msg)
            self.history.append(result)
            return result
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for code generation."""
        return MANIM_EXPERT_SYSTEM
    
    def generate_and_render(
        self,
        description: str,
        topic: Optional[str] = None,
        level: str = "general",
        max_retries: int = 3,
    ) -> GenerationResult:
        """
        Generate code and render it, with automatic error correction.
        
        Args:
            description: Natural language description
            topic: Optional topic category
            level: Education level
            max_retries: Maximum attempts to fix errors
            
        Returns:
            GenerationResult with output path if successful
        """
        result = self.generate(description, topic, level)
        
        if not result.success:
            return result
        
        # Try to render, fixing errors if needed
        for attempt in range(max_retries):
            render_result = self.render(result.code)
            
            if render_result.success:
                return render_result
            
            console.print(f"[yellow]Attempt {attempt + 1}/{max_retries} failed. Trying to fix...[/yellow]")
            
            # Try to fix the error
            fixed_result = self.fix_errors(result.code, render_result.error or "Unknown error")
            
            if fixed_result.success:
                result = fixed_result
            else:
                console.print("[red]Could not fix errors[/red]")
                break
        
        return result
    
    def render(
        self,
        code: str,
        scene_name: Optional[str] = None,
        output_name: Optional[str] = None,
    ) -> GenerationResult:
        """
        Render Manim code to video.
        
        Args:
            code: Complete Manim Python code
            scene_name: Optional scene class name (auto-detected if not provided)
            output_name: Optional output filename
            
        Returns:
            GenerationResult with output path
        """
        # Auto-detect scene name if not provided
        if not scene_name:
            scene_name = self._extract_scene_name(code)
        
        if not scene_name:
            return GenerationResult(
                success=False,
                code=code,
                error="Could not detect scene class name in code"
            )
        
        # Create temporary file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_filename = f"manai_scene_{timestamp}.py"
        temp_path = Path(tempfile.gettempdir()) / temp_filename
        
        # Write code to file
        with open(temp_path, 'w') as f:
            f.write(code)
        
        console.print(f"[dim]Rendering scene: {scene_name}[/dim]")
        
        # Build manim command
        quality_flag = self.config.quality_flags.get(self.config.quality, "-qh")
        output_dir = Path(self.config.output_dir).absolute()
        
        cmd = [
            sys.executable, "-m", "manim",
            quality_flag,
            str(temp_path),
            scene_name,
            "--media_dir", str(output_dir),
        ]
        
        if self.config.preview:
            cmd.append("--preview")
        
        if self.config.save_last_frame:
            cmd.append("--save_last_frame")
        
        try:
            # Run manim
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )
            
            if result.returncode == 0:
                # Find output file
                output_path = self._find_output_file(output_dir, scene_name)
                console.print(f"[green]Successfully rendered: {output_path}[/green]")
                
                return GenerationResult(
                    success=True,
                    code=code,
                    output_path=str(output_path) if output_path else None
                )
            else:
                error_msg = result.stderr or result.stdout
                console.print(f"[red]Render error:[/red]\n{error_msg}")
                
                return GenerationResult(
                    success=False,
                    code=code,
                    error=error_msg
                )
                
        except subprocess.TimeoutExpired:
            return GenerationResult(
                success=False,
                code=code,
                error="Rendering timed out after 5 minutes"
            )
        except Exception as e:
            return GenerationResult(
                success=False,
                code=code,
                error=str(e)
            )
        finally:
            # Clean up temp file
            if temp_path.exists():
                temp_path.unlink()
    
    def fix_errors(self, code: str, error: str) -> GenerationResult:
        """
        Use LLM to fix errors in Manim code.
        
        Args:
            code: Original code with errors
            error: Error message from rendering
            
        Returns:
            GenerationResult with fixed code
        """
        prompt = FIX_CODE_ERRORS.format(code=code, error=error)
        
        try:
            fixed_code = self.llm.generate(prompt, system_prompt=MANIM_EXPERT_SYSTEM)
            fixed_code = self._clean_code(fixed_code)
            
            return GenerationResult(success=True, code=fixed_code)
        except Exception as e:
            return GenerationResult(success=False, code=code, error=str(e))
    
    def enhance(self, code: str, request: str) -> GenerationResult:
        """
        Enhance existing Manim code.
        
        Args:
            code: Existing code to enhance
            request: Enhancement request
            
        Returns:
            GenerationResult with enhanced code
        """
        prompt = ENHANCE_EXISTING_CODE.format(code=code, request=request)
        
        try:
            enhanced_code = self.llm.generate(prompt, system_prompt=MANIM_EXPERT_SYSTEM)
            enhanced_code = self._clean_code(enhanced_code)
            
            return GenerationResult(success=True, code=enhanced_code)
        except Exception as e:
            return GenerationResult(success=False, code=code, error=str(e))
    
    def generate_with_narration(
        self,
        topic: str,
        audience: str = "general",
        duration: int = 60
    ) -> GenerationResult:
        """
        Generate animation with narration cues in comments.
        
        Args:
            topic: Topic to explain
            audience: Target audience
            duration: Target duration in seconds
            
        Returns:
            GenerationResult with narration-annotated code
        """
        prompt = EXPLAIN_AND_GENERATE.format(
            topic=topic,
            audience=audience,
            duration=duration
        )
        
        try:
            code = self.llm.generate(prompt, system_prompt=MANIM_EXPERT_SYSTEM)
            code = self._clean_code(code)
            
            return GenerationResult(success=True, code=code)
        except Exception as e:
            return GenerationResult(success=False, code="", error=str(e))
    
    def _clean_code(self, code: str) -> str:
        """Clean and normalize generated code."""
        # Remove markdown code blocks
        code = re.sub(r'^```python\s*', '', code, flags=re.MULTILINE)
        code = re.sub(r'^```\s*$', '', code, flags=re.MULTILINE)
        code = code.strip()
        
        # Ensure proper import
        if 'from manim import' not in code and 'import manim' not in code:
            code = "from manim import *\n\n" + code
        
        return code
    
    def _validate_code_structure(self, code: str) -> bool:
        """Validate basic code structure."""
        has_import = 'manim' in code
        has_class = 'class ' in code and ('Scene' in code or 'ThreeDScene' in code)
        has_construct = 'def construct' in code
        
        return has_import and has_class and has_construct
    
    def _extract_scene_name(self, code: str) -> Optional[str]:
        """Extract scene class name from code."""
        pattern = r'class\s+(\w+)\s*\(\s*(?:Scene|ThreeDScene|MovingCameraScene|ZoomedScene)'
        match = re.search(pattern, code)
        return match.group(1) if match else None
    
    def _find_output_file(self, output_dir: Path, scene_name: str) -> Optional[Path]:
        """Find the rendered output file."""
        # Manim typically outputs to media/videos/{filename}/{quality}/{SceneName}.mp4
        for video_file in output_dir.rglob(f"{scene_name}.*"):
            if video_file.suffix in ['.mp4', '.mov', '.gif', '.webm']:
                return video_file
        return None
    
    def preview_code(self, code: str) -> None:
        """Display code with syntax highlighting."""
        syntax = Syntax(code, "python", theme="monokai", line_numbers=True)
        console.print(Panel(syntax, title="Generated Manim Code"))
    
    def save_code(self, code: str, filename: str) -> Path:
        """Save code to a file."""
        output_path = Path(self.config.output_dir) / f"{filename}.py"
        with open(output_path, 'w') as f:
            f.write(code)
        console.print(f"[green]Code saved to: {output_path}[/green]")
        return output_path


# =============================================================================
# SPECIALIZED GENERATORS
# =============================================================================

class CalculusGenerator(ManimGenerator):
    """Specialized generator for calculus animations."""
    
    def derivative_visualization(
        self,
        function: str,
        interval: tuple[float, float] = (-3, 3)
    ) -> GenerationResult:
        """Generate derivative visualization for a function."""
        description = f"""
        Create a visualization of the derivative of f(x) = {function}:
        1. Plot the original function on interval {interval}
        2. Show a tangent line that moves along the curve
        3. Plot the derivative function below
        4. Show how the slope of tangent matches derivative value
        """
        return self.generate(description, topic="calculus", level="undergraduate")
    
    def integral_visualization(
        self,
        function: str,
        interval: tuple[float, float] = (0, 2)
    ) -> GenerationResult:
        """Generate integral/area visualization."""
        description = f"""
        Create a visualization of the integral of f(x) = {function} from {interval[0]} to {interval[1]}:
        1. Plot the function
        2. Show Riemann sum rectangles increasing in number
        3. Animate the rectangles filling in the area
        4. Show the exact area value
        """
        return self.generate(description, topic="calculus", level="undergraduate")


class LinearAlgebraGenerator(ManimGenerator):
    """Specialized generator for linear algebra animations."""
    
    def transformation_visualization(
        self,
        matrix: list[list[float]],
        show_basis: bool = True
    ) -> GenerationResult:
        """Visualize a linear transformation."""
        description = f"""
        Visualize the linear transformation given by matrix {matrix}:
        1. Show the original coordinate plane with basis vectors
        2. Apply the transformation smoothly
        3. Show how the entire plane transforms
        4. Highlight eigenspaces if they exist
        5. Display the matrix alongside the transformation
        """
        return self.generate(description, topic="linear_algebra", level="undergraduate")
    
    def eigenvalue_visualization(self, matrix: list[list[float]]) -> GenerationResult:
        """Visualize eigenvalues and eigenvectors."""
        description = f"""
        Visualize eigenvalues and eigenvectors of matrix {matrix}:
        1. Show the original matrix
        2. Find and display eigenvectors
        3. Show how eigenvectors only scale (not rotate) under transformation
        4. Display eigenvalues
        5. Show the characteristic equation
        """
        return self.generate(description, topic="linear_algebra", level="undergraduate")


class PhysicsGenerator(ManimGenerator):
    """Specialized generator for physics animations."""
    
    def projectile_motion(
        self,
        initial_velocity: float = 20,
        angle: float = 45,
        gravity: float = 9.8
    ) -> GenerationResult:
        """Generate projectile motion animation."""
        description = f"""
        Create a projectile motion simulation:
        - Initial velocity: {initial_velocity} m/s
        - Launch angle: {angle} degrees
        - Gravity: {gravity} m/s²
        
        Show:
        1. The parabolic trajectory
        2. Velocity vectors at key points
        3. Horizontal and vertical components
        4. Maximum height and range
        5. Equations of motion
        """
        return self.generate(description, topic="mechanics", level="high_school")
    
    def wave_interference(
        self,
        wavelength1: float = 1.0,
        wavelength2: float = 1.2
    ) -> GenerationResult:
        """Generate wave interference pattern."""
        description = f"""
        Create a wave interference visualization:
        - Wave 1 wavelength: {wavelength1}
        - Wave 2 wavelength: {wavelength2}
        
        Show:
        1. Two separate waves
        2. Superposition principle
        3. Constructive and destructive interference
        4. Resulting wave pattern
        5. Beat frequency (if applicable)
        """
        return self.generate(description, topic="waves", level="undergraduate")
    
    def electric_field(self, charges: list[dict]) -> GenerationResult:
        """Generate electric field visualization."""
        description = f"""
        Visualize the electric field for charges: {charges}
        
        Show:
        1. Point charges with their signs
        2. Electric field lines
        3. Equipotential surfaces
        4. Field strength (vector magnitude) at sample points
        5. Superposition of fields from multiple charges
        """
        return self.generate(description, topic="electromagnetism", level="undergraduate")


class ThreeDGenerator(ManimGenerator):
    """Specialized generator for 3D animations."""
    
    def surface_plot(self, function: str) -> GenerationResult:
        """Generate a 3D surface plot."""
        description = f"""
        Create a 3D visualization of the surface z = {function}:
        1. Use ThreeDScene with proper camera setup
        2. Plot the surface with color gradient based on height
        3. Add 3D axes with labels
        4. Rotate the camera around the surface
        5. Show contour lines at the base
        """
        return self.generate(description, topic="3d_surface", level="undergraduate")
    
    def solid_of_revolution(
        self,
        function: str,
        axis: str = "x",
        interval: tuple[float, float] = (0, 2)
    ) -> GenerationResult:
        """Generate solid of revolution visualization."""
        description = f"""
        Create a solid of revolution animation:
        - Function: y = {function}
        - Rotate around the {axis}-axis
        - Interval: {interval}
        
        Show:
        1. The original 2D curve
        2. Animate the rotation to form the solid
        3. Display the resulting 3D solid
        4. Show cross-sections (disks/washers)
        5. Display the volume integral formula
        """
        return self.generate(description, topic="3d_solid", level="undergraduate")
    
    def vector_field_3d(self, field: str) -> GenerationResult:
        """Generate 3D vector field visualization."""
        description = f"""
        Visualize the 3D vector field F = {field}:
        1. Use ThreeDScene
        2. Show vectors at sample points in 3D space
        3. Add 3D axes
        4. Color vectors by magnitude
        5. Show streamlines if appropriate
        6. Rotate camera for full view
        """
        return self.generate(description, topic="3d_vector_field", level="undergraduate")
