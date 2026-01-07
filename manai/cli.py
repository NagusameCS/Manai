"""
Manai CLI - Command Line Interface for video generation
"""

import sys
from pathlib import Path
from typing import Optional

import typer

from manai.client import ManaiClient, ManaiConfig
from manai.ollama_client import OllamaConfig, OllamaClient
from manai.generator import ManimConfig
from manai.ui import ui, console
from manai.agent import Agent, ThinkingMode
from manai.templates_2d import list_2d_templates, get_2d_template
from manai.templates_3d import list_3d_templates, get_3d_template
from manai.transitions import list_transitions

app = typer.Typer(
    name="manai",
    help="AI-powered math and physics video generator using Manim and local LLMs",
    add_completion=False,
)


@app.command()
def ui(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to listen on"),
    reload: bool = typer.Option(False, "--reload", "-r", help="Enable auto-reload"),
):
    """
    Launch the web UI in your browser.
    
    Starts a local web server with a clean interface for generating videos.
    
    Examples:
        manai ui
        manai ui --port 3000
    """
    from manai.web import run_server
    import webbrowser
    
    # Open browser after a short delay
    import threading
    def open_browser():
        import time
        time.sleep(1)
        webbrowser.open(f"http://{host}:{port}")
    
    threading.Thread(target=open_browser, daemon=True).start()
    
    run_server(host=host, port=port, reload=reload)


def get_client(
    model: str = "llama3.2",
    quality: str = "high",
    output_dir: str = "./output",
) -> ManaiClient:
    """Create a configured client."""
    config = ManaiConfig(
        ollama=OllamaConfig(model=model),
        manim=ManimConfig(quality=quality, output_dir=output_dir)
    )
    return ManaiClient(config=config)


@app.command()
def create(
    description: str = typer.Argument(..., help="Description of the video to create"),
    topic: Optional[str] = typer.Option(None, "--topic", "-t", help="Topic category"),
    level: str = typer.Option("general", "--level", "-l", help="Education level"),
    model: str = typer.Option("llama3.2", "--model", "-m", help="Ollama model to use"),
    quality: str = typer.Option("high", "--quality", "-q", help="Video quality"),
    output_dir: str = typer.Option("./output", "--output", "-o", help="Output directory"),
    no_render: bool = typer.Option(False, "--no-render", help="Generate code without rendering"),
    no_preview: bool = typer.Option(False, "--no-preview", help="Don't preview after rendering"),
    save_code: Optional[str] = typer.Option(None, "--save-code", "-s", help="Save code to file"),
):
    """
    Create a math/physics video from a natural language description.
    
    Examples:
        manai create "Show the derivative of sin(x)"
        manai create "Projectile motion simulation" -t physics
        manai create "3D plot of z = x^2 + y^2" -q production
    """
    try:
        client = get_client(model, quality, output_dir)
        
        result = client.create(
            description=description,
            topic=topic,
            level=level,
            render=not no_render,
            preview=not no_preview,
        )
        
        if result.success:
            if result.output_path:
                console.print(f"[green]✓ Video saved to: {result.output_path}[/green]")
            else:
                console.print("[green]✓ Code generated successfully[/green]")
                client.preview_last_code()
            
            if save_code:
                client.save_last_code(save_code)
        else:
            console.print(f"[red]✗ Generation failed: {result.error}[/red]")
            sys.exit(1)
            
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@app.command()
def equation(
    equation: str = typer.Argument(..., help="LaTeX equation to explain"),
    title: Optional[str] = typer.Option(None, "--title", "-t", help="Video title"),
    derivation: bool = typer.Option(True, "--derivation/--no-derivation", help="Show derivation"),
    model: str = typer.Option("llama3.2", "--model", "-m", help="Ollama model"),
    quality: str = typer.Option("high", "--quality", "-q", help="Video quality"),
):
    """
    Create a video explaining a mathematical equation.
    
    Examples:
        manai equation "E = mc^2" --title "Mass-Energy Equivalence"
        manai equation "\\\\int_a^b f(x)dx = F(b) - F(a)"
    """
    try:
        client = get_client(model, quality)
        result = client.create_from_equation(equation, title, derivation)
        
        if result.success:
            console.print("[green]✓ Video created successfully[/green]")
        else:
            console.print(f"[red]✗ Failed: {result.error}[/red]")
            
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@app.command()
def plot(
    function: str = typer.Argument(..., help="Function to plot (e.g., 'x**2', 'sin(x)')"),
    x_min: float = typer.Option(-5, "--x-min", help="Minimum x value"),
    x_max: float = typer.Option(5, "--x-max", help="Maximum x value"),
    derivative: bool = typer.Option(False, "--derivative", "-d", help="Also show derivative"),
    integral: bool = typer.Option(False, "--integral", "-i", help="Also show integral"),
    model: str = typer.Option("llama3.2", "--model", "-m", help="Ollama model"),
    quality: str = typer.Option("high", "--quality", "-q", help="Video quality"),
):
    """
    Create a quick function plot animation.
    
    Examples:
        manai plot "sin(x)" --derivative
        manai plot "x**3 - 2*x" -d -i
    """
    try:
        client = get_client(model, quality)
        result = client.quick_function_plot(
            function,
            x_range=(x_min, x_max),
            show_derivative=derivative,
            show_integral=integral,
        )
        
        if result.success:
            console.print("[green]✓ Plot created successfully[/green]")
        else:
            console.print(f"[red]✗ Failed: {result.error}[/red]")
            
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@app.command()
def physics(
    scenario: str = typer.Argument(..., help="Physics scenario to simulate"),
    forces: bool = typer.Option(True, "--forces/--no-forces", help="Show force vectors"),
    equations: bool = typer.Option(True, "--equations/--no-equations", help="Show equations"),
    model: str = typer.Option("llama3.2", "--model", "-m", help="Ollama model"),
    quality: str = typer.Option("high", "--quality", "-q", help="Video quality"),
):
    """
    Create a physics simulation.
    
    Examples:
        manai physics "Ball rolling down an inclined plane"
        manai physics "Two objects colliding elastically"
    """
    try:
        client = get_client(model, quality)
        result = client.quick_physics_simulation(scenario, forces, equations)
        
        if result.success:
            console.print("[green]✓ Simulation created successfully[/green]")
        else:
            console.print(f"[red]✗ Failed: {result.error}[/red]")
            
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@app.command()
def surface(
    function: str = typer.Argument(..., help="Surface function z = f(x,y)"),
    model: str = typer.Option("llama3.2", "--model", "-m", help="Ollama model"),
    quality: str = typer.Option("high", "--quality", "-q", help="Video quality"),
):
    """
    Create a 3D surface plot.
    
    Examples:
        manai surface "x**2 + y**2"
        manai surface "sin(x) * cos(y)"
    """
    try:
        client = get_client(model, quality)
        result = client.threed.surface_plot(function)
        
        if result.success:
            console.print("[green]✓ 3D surface created successfully[/green]")
        else:
            console.print(f"[red]✗ Failed: {result.error}[/red]")
            
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@app.command()
def render(
    file: Path = typer.Argument(..., help="Python file containing Manim scene"),
    scene: Optional[str] = typer.Option(None, "--scene", "-s", help="Scene class name"),
    quality: str = typer.Option("high", "--quality", "-q", help="Video quality"),
    output_dir: str = typer.Option("./output", "--output", "-o", help="Output directory"),
    preview: bool = typer.Option(True, "--preview/--no-preview", help="Preview after rendering"),
):
    """
    Render an existing Manim scene file.
    
    Examples:
        manai render my_scene.py
        manai render my_scene.py --scene MyScene -q production
    """
    try:
        if not file.exists():
            console.print(f"[red]File not found: {file}[/red]")
            sys.exit(1)
        
        with open(file) as f:
            code = f.read()
        
        client = get_client(quality=quality, output_dir=output_dir)
        client.config.manim.preview = preview
        
        result = client.render_code(code)
        
        if result.success:
            console.print(f"[green]✓ Rendered: {result.output_path}[/green]")
        else:
            console.print(f"[red]✗ Render failed: {result.error}[/red]")
            
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@app.command()
def models():
    """List available Ollama models."""
    try:
        client = get_client()
        client.list_models()
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("[yellow]Make sure Ollama is running: ollama serve[/yellow]")
        sys.exit(1)


@app.command()
def pull(
    model: str = typer.Argument(..., help="Model to pull from Ollama"),
):
    """
    Pull/download an Ollama model.
    
    Examples:
        manai pull llama3.2
        manai pull codellama
        manai pull deepseek-coder
    """
    try:
        client = get_client()
        client.ollama.pull_model(model)
        console.print(f"[green]✓ Model {model} ready![/green]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@app.command()
def interactive():
    """Start an interactive session for creating videos."""
    try:
        client = get_client()
        client.interactive_session()
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@app.command()
def topics():
    """Show supported topics and examples."""
    table = Table(title="Supported Topics", show_header=True)
    table.add_column("Category", style="cyan", width=15)
    table.add_column("Topics", style="green", width=40)
    table.add_column("Example", style="yellow", width=35)
    
    table.add_row(
        "Calculus",
        "derivatives, integrals, limits, series",
        "manai create 'derivative of x^2'"
    )
    table.add_row(
        "Linear Algebra",
        "matrices, vectors, transformations",
        "manai create 'matrix multiplication'"
    )
    table.add_row(
        "Geometry",
        "triangles, circles, proofs",
        "manai create 'Pythagorean theorem'"
    )
    table.add_row(
        "Trigonometry",
        "unit circle, identities, waves",
        "manai create 'unit circle'"
    )
    table.add_row(
        "Mechanics",
        "forces, motion, momentum, energy",
        "manai physics 'pendulum motion'"
    )
    table.add_row(
        "E&M",
        "fields, charges, circuits",
        "manai create 'electric field lines'"
    )
    table.add_row(
        "Waves",
        "interference, diffraction, sound",
        "manai create 'wave superposition'"
    )
    table.add_row(
        "Quantum",
        "wave functions, probabilities",
        "manai create 'particle in a box'"
    )
    table.add_row(
        "3D Graphics",
        "surfaces, solids, volumes",
        "manai surface 'sin(x)*cos(y)'"
    )
    
    console.print(table)
    
    console.print("\n[cyan]Education Levels:[/cyan]")
    console.print("  elementary, middle_school, high_school, undergraduate, graduate, general")


@app.command()
def templates():
    """List all available templates."""
    ui.header("Available Templates")
    
    # 2D Templates
    ui.panel("2D Templates", "\n".join([f"  • {t}" for t in list_2d_templates()]))
    
    # 3D Templates
    ui.panel("3D Templates", "\n".join([f"  • {t}" for t in list_3d_templates()]))
    
    # Transitions
    ui.panel("Transitions", "\n".join([f"  • {t}" for t in list_transitions()]))
    
    console.print("\n[dim]Use: manai template <name> to generate from a template[/dim]")


@app.command()
def template(
    name: str = typer.Argument(..., help="Template name to use"),
    output: str = typer.Option("template_scene.py", "--output", "-o", help="Output file"),
):
    """Generate code from a template."""
    code = get_2d_template(name) or get_3d_template(name)
    
    if not code:
        console.print(f"[red]Template '{name}' not found[/red]")
        console.print("Use 'manai templates' to see available templates")
        sys.exit(1)
    
    Path(output).write_text(code.strip())
    console.print(f"[green]✓ Template saved to {output}[/green]")
    ui.code_panel(code[:500] + "..." if len(code) > 500 else code, "Preview")


@app.command()
def switch_model(
    model: str = typer.Argument(..., help="Model name to switch to"),
):
    """Switch the default Ollama model."""
    try:
        client = OllamaClient()
        models = client.list_models()
        
        model_names = [m.get("name", "").split(":")[0] for m in models]
        
        if model not in model_names and f"{model}:latest" not in [m.get("name") for m in models]:
            console.print(f"[yellow]Model '{model}' not found locally[/yellow]")
            if ui.prompt(f"Pull {model} from Ollama?").lower() == "y":
                console.print(f"[dim]Pulling {model}...[/dim]")
                client.pull_model(model)
                console.print(f"[green]✓ Model {model} pulled successfully[/green]")
            else:
                sys.exit(0)
        
        # Update config.yaml
        import yaml
        config_path = Path("config.yaml")
        if config_path.exists():
            config = yaml.safe_load(config_path.read_text())
            config["ollama"]["model"] = model
            config_path.write_text(yaml.dump(config, default_flow_style=False))
            console.print(f"[green]✓ Default model set to: {model}[/green]")
        else:
            console.print(f"[dim]Note: config.yaml not found, model set for this session only[/dim]")
            
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@app.command()
def agent(
    query: str = typer.Argument(..., help="What to create"),
    thinking: str = typer.Option("standard", "--thinking", "-t", 
                                  help="Thinking mode: quick, standard, deep, verify"),
    model: str = typer.Option("llama3.2", "--model", "-m", help="Ollama model"),
):
    """
    Use the intelligent agent with reasoning and tools.
    
    The agent thinks through your request, uses math verification tools,
    and generates high-quality Manim code.
    
    Examples:
        manai agent "Prove the derivative of x^3 is 3x^2"
        manai agent "Visualize eigenvalue decomposition" --thinking deep
    """
    ui.header("Manai Agent")
    
    try:
        # Parse thinking mode
        mode_map = {
            "quick": ThinkingMode.QUICK,
            "standard": ThinkingMode.STANDARD,
            "deep": ThinkingMode.DEEP,
            "verify": ThinkingMode.VERIFY,
        }
        mode = mode_map.get(thinking, ThinkingMode.STANDARD)
        
        # Create agent
        ollama = OllamaClient(OllamaConfig(model=model))
        agent_instance = Agent(ollama, mode)
        
        # Process query
        result = agent_instance.process(query)
        
        # Show results
        if result["plan"]:
            ui.panel("Plan", "\n".join([f"{i+1}. {step}" for i, step in enumerate(result["plan"])]))
        
        ui.panel("Confidence", f"{result['confidence']*100:.0f}%")
        
        # Generate the code
        client = get_client(model)
        gen_result = client.create(query, render=False)
        
        if gen_result.success:
            ui.code_panel(gen_result.code[:1000] if gen_result.code else "", "Generated Code")
            
            if ui.prompt("Render video? (y/n)").lower() == "y":
                client.render_last()
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@app.command()
def verify(
    expression: str = typer.Argument(..., help="Math expression to verify"),
):
    """Verify a mathematical expression."""
    from manai.math_engine import math_engine
    
    ui.header("Math Verification")
    
    # Try to parse and verify
    result = math_engine.parse(expression)
    
    if result.success:
        ui.verification(True, f"Valid expression: {result.latex}")
        
        # Try simplification
        simplified = math_engine.simplify_expr(expression)
        if simplified.success and str(simplified.result) != expression:
            ui.panel("Simplified", simplified.latex)
    else:
        ui.verification(False, result.error)


@app.command()
def version():
    """Show version information."""
    from manai import __version__
    
    ui.header(f"Manai v{__version__}")
    console.print("AI-powered Math & Physics Video Generator")
    console.print("[dim]Powered by: Manim + Ollama[/dim]")
    console.print()
    
    # Show available models
    try:
        client = OllamaClient()
        models = client.list_models()
        ui.model_table(models)
    except:
        console.print("[dim]Ollama not available[/dim]")


if __name__ == "__main__":
    app()
