# Manai 🎬

**AI-powered Math and Physics Video Generator**

Create beautiful educational videos for mathematics and physics using natural language descriptions. Powered by locally-run LLMs (Ollama) and the Manim animation library.

## ✨ Features

- **Natural Language → Video**: Describe what you want to visualize, get a rendered video
- **100% Local**: Uses Ollama for LLM inference - no API keys or cloud services required
- **Full Manim Support**: Access to all 2D and 3D Manim capabilities
- **Specialized Generators**: Purpose-built generators for calculus, linear algebra, physics, and 3D
- **Auto Error Correction**: Automatically fixes code errors and re-renders
- **Interactive Mode**: Chat-based interface for iterative creation
- **Multiple Quality Levels**: From quick previews to production-quality 4K renders

## 🎯 Supported Topics

### Mathematics
- **Calculus**: Derivatives, integrals, limits, series, differential equations
- **Linear Algebra**: Matrices, transformations, eigenvalues, vector spaces
- **Geometry**: Proofs, constructions, theorems, transformations
- **Trigonometry**: Unit circle, identities, wave functions
- **Algebra**: Equations, inequalities, polynomials
- **Complex Analysis**: Complex functions, mappings
- **Probability & Statistics**: Distributions, visualizations

### Physics
- **Classical Mechanics**: Forces, motion, energy, momentum
- **Electromagnetism**: Fields, charges, circuits, waves
- **Thermodynamics**: PV diagrams, heat flow, entropy
- **Waves & Optics**: Interference, diffraction, refraction
- **Quantum Mechanics**: Wave functions, probability clouds
- **Relativity**: Spacetime diagrams, Lorentz transformations
- **Fluid Dynamics**: Flow visualization, Bernoulli

## 🚀 Quick Start

### Prerequisites

1. **Install Ollama** (for local LLM)
   ```bash
   # macOS
   brew install ollama
   
   # Linux
   curl -fsSL https://ollama.com/install.sh | sh
   
   # Windows
   # Download from https://ollama.com/download
   ```

2. **Start Ollama and pull a model**
   ```bash
   ollama serve  # Start the Ollama server
   ollama pull llama3.2  # Or: codellama, deepseek-coder, qwen2.5-coder
   ```

3. **Install Manim dependencies** (for rendering)
   ```bash
   # macOS
   brew install py3cairo ffmpeg pango scipy
   
   # Ubuntu/Debian
   sudo apt install libcairo2-dev ffmpeg libpango1.0-dev
   
   # See: https://docs.manim.community/en/stable/installation.html
   ```

### Installation

```bash
# Clone the repository
git clone https://github.com/NagusameCS/Manai.git
cd Manai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package
pip install -e .
```

### Your First Video

```bash
# Simple command
manai create "Show how the derivative of sin(x) equals cos(x)"

# With options
manai create "Visualize a 2x2 matrix transformation" --topic linear_algebra --quality high

# Interactive mode
manai interactive
```

## �️ Web UI

Launch the web interface for a visual way to create videos:

```bash
manai ui
```

This opens a clean, modern browser interface at `http://localhost:8000` with:

- **Model Switcher** - Easily change between Ollama models
- **Template Browser** - Quick access to 25+ 2D/3D templates and transitions
- **Thinking Modes** - Quick, Standard, Deep, and Verify modes for different reasoning depths
- **Live Preview** - See generated code and videos in real-time
- **Math Verification** - Built-in equation checking with the math engine

Options:
```bash
manai ui --port 3000          # Custom port
manai ui --host 0.0.0.0       # Expose to network
```

## �📖 Usage

### Command Line Interface

```bash
# Create a video from description
manai create "Explain the Pythagorean theorem with a visual proof"

# Create with specific topic and education level
manai create "Derive the quadratic formula" -t algebra -l high_school

# Quick function plot
manai plot "sin(x)" --derivative --x-min -6.28 --x-max 6.28

# Physics simulation
manai physics "Projectile motion with air resistance"

# 3D surface
manai surface "sin(x)*cos(y)"

# Explain an equation
manai equation "E = mc^2" --title "Mass-Energy Equivalence"

# Render existing file
manai render my_scene.py --quality production
```

### CLI Options

| Option | Description |
|--------|-------------|
| `-t, --topic` | Topic category (calculus, physics, geometry, etc.) |
| `-l, --level` | Education level (elementary, high_school, undergraduate, graduate, general) |
| `-m, --model` | Ollama model to use (default: llama3.2) |
| `-q, --quality` | Video quality (low, medium, high, production) |
| `-o, --output` | Output directory |
| `--no-render` | Generate code only, don't render |
| `--no-preview` | Don't preview after rendering |
| `-s, --save-code` | Save generated code to file |

### Python API

```python
from manai import ManaiClient

# Initialize client
client = ManaiClient()

# Create a video
result = client.create(
    "Visualize the relationship between a function and its derivative",
    topic="calculus",
    level="undergraduate"
)

# Use specialized generators
result = client.calculus.derivative_visualization("x**3")
result = client.physics.projectile_motion(initial_velocity=20, angle=45)
result = client.threed.surface_plot("sin(sqrt(x**2 + y**2))")

# Quick methods
result = client.quick_function_plot("x**2 - 4", show_derivative=True)
result = client.quick_physics_simulation("Pendulum with damping")

# Create comparison
result = client.create_comparison(
    "Differentiation", 
    "Integration",
    highlight_differences=True
)

# Create proof animation
result = client.create_proof("Sum of angles in a triangle equals 180 degrees")

# Work with last generation
client.preview_last_code()
client.save_last_code("my_animation")
client.enhance_last("Add more colors and smoother transitions")
```

### Interactive Session

```bash
manai interactive
```

Commands in interactive mode:
- Type a description to generate a video
- `preview` - Show last generated code
- `save <name>` - Save code to file
- `enhance <request>` - Improve last code
- `quality <level>` - Set quality
- `model <name>` - Switch model
- `models` - List available models
- `help` - Show help
- `quit` - Exit

## ⚙️ Configuration

Create a `config.yaml` file to customize settings:

```yaml
ollama:
  host: "http://localhost:11434"
  model: "llama3.2"
  timeout: 300
  options:
    temperature: 0.7
    num_ctx: 8192

manim:
  quality: "high"
  format: "mp4"
  output_dir: "./output"
  background_color: "#1a1a2e"
```

Load custom config:
```python
client = ManaiClient(config_path="./config.yaml")
```

## 🤖 Recommended Models

| Model | Best For | Speed | Quality |
|-------|----------|-------|---------|
| `llama3.2` | General purpose, balanced | Fast | Good |
| `codellama` | Code-heavy generations | Medium | Excellent |
| `deepseek-coder` | Complex algorithms | Medium | Excellent |
| `qwen2.5-coder` | Detailed visualizations | Medium | Very Good |
| `mixtral` | Creative animations | Slow | Excellent |

Pull models:
```bash
manai pull llama3.2
manai pull codellama
```

## 📂 Project Structure

```
Manai/
├── manai/
│   ├── __init__.py          # Package initialization
│   ├── client.py             # Main client interface
│   ├── cli.py                # Command line interface
│   ├── generator.py          # Manim code generator
│   ├── ollama_client.py      # Ollama LLM client
│   ├── prompts.py            # Specialized prompts
│   ├── utils.py              # Utility functions
│   └── examples.py           # Example scenes
├── config.yaml               # Configuration
├── requirements.txt          # Dependencies
├── pyproject.toml           # Package config
├── README.md                # This file
└── LICENSE                  # MIT License
```

## 🎨 Example Outputs

### Calculus: Derivative Visualization
```bash
manai create "Show the derivative of x^2 with moving tangent line"
```

### Linear Algebra: Matrix Transformation
```bash
manai create "Visualize a 2D shear transformation matrix"
```

### Physics: Wave Interference
```bash
manai physics "Two sine waves interfering constructively and destructively"
```

### 3D: Parametric Surface
```bash
manai surface "sin(sqrt(x**2 + y**2)) / sqrt(x**2 + y**2)"
```

## 🔧 Troubleshooting

### "Cannot connect to Ollama"
```bash
# Make sure Ollama is running
ollama serve
```

### "Model not found"
```bash
# Pull the model first
ollama pull llama3.2
```

### Manim rendering errors
```bash
# Ensure Manim dependencies are installed
pip install manim[all]

# On macOS, ensure Cairo is installed
brew install py3cairo pango
```

### Slow generation
- Use a faster model (`llama3.2` is recommended for speed)
- Reduce quality for previews: `--quality low`
- Use a GPU-enabled Ollama installation

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Manim Community](https://www.manim.community/) - The amazing animation engine
- [Ollama](https://ollama.com/) - Local LLM serving
- [3Blue1Brown](https://www.3blue1brown.com/) - Inspiration for mathematical visualization

---

Made with ❤️ for educators, students, and math/physics enthusiasts