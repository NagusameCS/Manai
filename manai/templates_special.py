"""
Special Topics Templates - Advanced mathematical concepts and visualizations
"""

FOURIER_SERIES = '''
from manim import *
import numpy as np

class FourierSeries(Scene):
    """Visualize Fourier series building a square wave."""
    
    def construct(self):
        title = Text("Fourier Series").scale(0.5).to_edge(UP)
        self.play(Write(title))
        
        # Axes
        axes = Axes(
            x_range=[-PI, PI, PI/2],
            y_range=[-1.5, 1.5, 0.5],
            x_length=8,
            y_length=4,
            axis_config={"include_tip": False}
        ).shift(DOWN*0.5)
        
        self.play(Create(axes))
        
        # Target: square wave
        def square_wave(x):
            return 1 if 0 < x < PI else -1 if -PI < x < 0 else 0
        
        # Build up Fourier terms
        def fourier_approx(x, n_terms):
            result = 0
            for n in range(1, n_terms + 1, 2):  # odd terms only
                result += (4 / (n * PI)) * np.sin(n * x)
            return result
        
        # Show progressive approximation
        colors = [BLUE, GREEN, YELLOW, ORANGE, RED]
        curves = []
        
        for i, n in enumerate([1, 3, 5, 9, 15]):
            curve = axes.plot(lambda x, n=n: fourier_approx(x, n), color=colors[min(i, 4)])
            label = MathTex(f"n = {n}").scale(0.4).to_corner(UR).shift(DOWN*i*0.5)
            
            if curves:
                self.play(Transform(curves[-1], curve), Write(label), run_time=0.8)
            else:
                self.play(Create(curve), Write(label))
            curves.append(curve)
        
        # Formula
        formula = MathTex(r"f(x) = \\sum_{n=1,3,5...} \\frac{4}{n\\pi} \\sin(nx)").scale(0.4).to_edge(DOWN)
        self.play(Write(formula))
        self.wait(2)
'''

EULER_IDENTITY = '''
from manim import *
import numpy as np

class EulerIdentity(Scene):
    """Visualize Euler's identity: e^(iπ) + 1 = 0."""
    
    def construct(self):
        title = Text("Euler's Identity").scale(0.5).to_edge(UP)
        self.play(Write(title))
        
        # Complex plane
        plane = ComplexPlane(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            x_length=5,
            y_length=5
        ).shift(LEFT*2)
        
        plane_labels = plane.get_axis_labels(x_label="Re", y_label="Im")
        
        self.play(Create(plane), Write(plane_labels))
        
        # Unit circle
        circle = Circle(radius=plane.get_x_unit_size(), color=BLUE).move_to(plane.n2p(0))
        self.play(Create(circle))
        
        # Show e^(iθ) = cos(θ) + i*sin(θ)
        angle = ValueTracker(0)
        
        point = always_redraw(lambda: Dot(
            plane.n2p(np.exp(1j * angle.get_value())),
            color=RED
        ))
        
        line = always_redraw(lambda: Line(
            plane.n2p(0),
            plane.n2p(np.exp(1j * angle.get_value())),
            color=YELLOW
        ))
        
        self.play(Create(point), Create(line))
        
        # Rotate to π
        euler_formula = MathTex(r"e^{i\\theta} = \\cos\\theta + i\\sin\\theta").scale(0.4).shift(RIGHT*3 + UP)
        self.play(Write(euler_formula))
        
        self.play(angle.animate.set_value(PI), run_time=2)
        
        # At θ = π
        at_pi = MathTex(r"e^{i\\pi} = \\cos\\pi + i\\sin\\pi = -1").scale(0.4).shift(RIGHT*3)
        self.play(Write(at_pi))
        
        # Final identity
        identity = MathTex(r"e^{i\\pi} + 1 = 0").scale(0.7).shift(RIGHT*3 + DOWN)
        box = SurroundingRectangle(identity, color=GOLD)
        
        self.play(Write(identity), Create(box))
        self.wait(2)
'''

GOLDEN_RATIO = '''
from manim import *

class GoldenRatio(Scene):
    """Visualize the golden ratio and golden spiral."""
    
    def construct(self):
        title = Text("Golden Ratio").scale(0.5).to_edge(UP)
        self.play(Write(title))
        
        # Golden rectangle
        phi = (1 + np.sqrt(5)) / 2
        
        rect = Rectangle(width=phi*2, height=2, color=GOLD).shift(LEFT)
        self.play(Create(rect))
        
        # Divide into square and smaller golden rectangle
        square = Square(side_length=2, color=BLUE).align_to(rect, LEFT).align_to(rect, DOWN)
        smaller = Rectangle(width=phi*2 - 2, height=2, color=GREEN).align_to(rect, RIGHT)
        
        divider = Line(square.get_corner(UR), square.get_corner(DR), color=WHITE)
        
        self.play(Create(divider), Create(square), Create(smaller))
        
        # Labels
        phi_label = MathTex(r"\\phi = \\frac{1 + \\sqrt{5}}{2} \\approx 1.618").scale(0.4).shift(RIGHT*3 + UP*2)
        self.play(Write(phi_label))
        
        # Show ratio
        ratio = MathTex(r"\\frac{a+b}{a} = \\frac{a}{b} = \\phi").scale(0.4).shift(RIGHT*3 + UP)
        self.play(Write(ratio))
        
        # Fibonacci spiral approximation
        spiral = VMobject(color=YELLOW)
        spiral.set_points_as_corners([
            LEFT*1 + DOWN*1,
            LEFT*1 + UP*1,
            RIGHT*1 + UP*1,
            RIGHT*1 + DOWN*0.2,
            LEFT*0.2 + DOWN*0.2,
        ])
        
        # Simple arc spiral
        arc1 = Arc(radius=1, start_angle=PI, angle=PI/2, color=YELLOW).shift(LEFT + DOWN)
        arc2 = Arc(radius=1, start_angle=3*PI/2, angle=PI/2, color=YELLOW).shift(LEFT + UP)
        
        self.play(Create(arc1), Create(arc2))
        
        nature = Text("Found throughout nature!").scale(0.35).to_edge(DOWN)
        self.play(Write(nature))
        self.wait(2)
'''

MANDELBROT = '''
from manim import *
import numpy as np

class MandelbrotIntro(Scene):
    """Introduce the Mandelbrot set concept."""
    
    def construct(self):
        title = Text("Mandelbrot Set").scale(0.5).to_edge(UP)
        self.play(Write(title))
        
        # Show the iteration
        formula = MathTex(r"z_{n+1} = z_n^2 + c").scale(0.6)
        self.play(Write(formula))
        
        explanation = Text("Starting from z₀ = 0, iterate").scale(0.35).shift(DOWN)
        self.play(Write(explanation), formula.animate.shift(UP*0.5))
        
        # Complex plane
        plane = ComplexPlane(
            x_range=[-2.5, 1, 0.5],
            y_range=[-1.5, 1.5, 0.5],
            x_length=7,
            y_length=5
        ).shift(DOWN)
        
        self.play(FadeOut(explanation), Create(plane))
        
        # Show a point inside (bounded)
        c_inside = -0.5
        point_in = Dot(plane.n2p(c_inside), color=BLUE)
        label_in = Text("Bounded", color=BLUE).scale(0.3).next_to(point_in, UP)
        
        self.play(Create(point_in), Write(label_in))
        
        # Show a point outside (escapes)
        c_outside = 1
        point_out = Dot(plane.n2p(c_outside), color=RED)
        label_out = Text("Escapes", color=RED).scale(0.3).next_to(point_out, UP)
        
        self.play(Create(point_out), Write(label_out))
        
        # Rough outline of Mandelbrot
        mandelbrot_outline = Circle(radius=0.8, color=GREEN).shift(plane.n2p(-0.5))
        self.play(Create(mandelbrot_outline))
        
        definition = Text("Set of all c where sequence stays bounded").scale(0.3).to_edge(DOWN)
        self.play(Write(definition))
        self.wait(2)
'''

COMPLEX_MULTIPLICATION = '''
from manim import *
import numpy as np

class ComplexMultiplication(Scene):
    """Visualize complex number multiplication as rotation + scaling."""
    
    def construct(self):
        title = Text("Complex Multiplication").scale(0.5).to_edge(UP)
        self.play(Write(title))
        
        # Complex plane
        plane = ComplexPlane(
            x_range=[-4, 4, 1],
            y_range=[-3, 3, 1],
            x_length=8,
            y_length=6
        ).shift(DOWN*0.3)
        
        self.play(Create(plane))
        
        # Original point z = 1 + i
        z = complex(1, 1)
        z_point = Dot(plane.n2p(z), color=BLUE)
        z_label = MathTex("z = 1 + i").scale(0.4).next_to(z_point, UR)
        
        self.play(Create(z_point), Write(z_label))
        
        # Multiplier w = e^(i*π/4) * √2 = 1 + i
        # Multiply by i (rotate 90°)
        w = complex(0, 1)
        w_label = MathTex("w = i").scale(0.4).shift(RIGHT*3 + UP*2)
        
        self.play(Write(w_label))
        
        # Show rotation
        result = z * w
        result_point = Dot(plane.n2p(result), color=RED)
        result_label = MathTex("z \\cdot w = -1 + i").scale(0.4).next_to(result_point, UL)
        
        # Draw arc showing rotation
        arc = Arc(
            radius=abs(z) * plane.get_x_unit_size(),
            start_angle=np.angle(z),
            angle=PI/2,
            color=YELLOW
        ).move_to(plane.n2p(0))
        
        self.play(Create(arc), Create(result_point), Write(result_label))
        
        # Explanation
        explanation = MathTex(r"\\text{Multiply: Add angles, multiply magnitudes}").scale(0.4).to_edge(DOWN)
        self.play(Write(explanation))
        self.wait(2)
'''

EIGENVALUE_VISUAL = '''
from manim import *
import numpy as np

class EigenvalueVisualization(Scene):
    """Visualize eigenvalues and eigenvectors."""
    
    def construct(self):
        title = Text("Eigenvalues & Eigenvectors").scale(0.5).to_edge(UP)
        self.play(Write(title))
        
        # Grid
        plane = NumberPlane(
            x_range=[-4, 4, 1],
            y_range=[-3, 3, 1],
            x_length=8,
            y_length=6,
            background_line_style={"stroke_opacity": 0.3}
        )
        
        self.play(Create(plane))
        
        # Matrix A = [[2, 1], [1, 2]]
        # Eigenvalues: 3 and 1
        # Eigenvectors: [1, 1] and [1, -1]
        
        matrix = MathTex(r"A = \\begin{bmatrix} 2 & 1 \\\\ 1 & 2 \\end{bmatrix}").scale(0.5).to_corner(UL)
        self.play(Write(matrix))
        
        # Original vectors
        v1 = Arrow(ORIGIN, RIGHT + UP, color=BLUE, buff=0)
        v2 = Arrow(ORIGIN, RIGHT + DOWN, color=GREEN, buff=0)
        
        v1_label = MathTex(r"\\vec{v}_1").scale(0.4).next_to(v1.get_end(), UR)
        v2_label = MathTex(r"\\vec{v}_2").scale(0.4).next_to(v2.get_end(), DR)
        
        self.play(Create(v1), Create(v2), Write(v1_label), Write(v2_label))
        
        # Apply transformation - eigenvectors only scale
        # v1 scales by 3, v2 scales by 1
        v1_new = Arrow(ORIGIN, 3*(RIGHT + UP), color=BLUE, buff=0)
        v2_new = Arrow(ORIGIN, 1*(RIGHT + DOWN), color=GREEN, buff=0)
        
        self.play(Transform(v1, v1_new), Transform(v2, v2_new))
        
        # Eigenvalue equation
        equation = MathTex(r"A\\vec{v} = \\lambda\\vec{v}").scale(0.5).to_edge(DOWN)
        self.play(Write(equation))
        
        eigenvals = VGroup(
            MathTex(r"\\lambda_1 = 3", color=BLUE).scale(0.4),
            MathTex(r"\\lambda_2 = 1", color=GREEN).scale(0.4)
        ).arrange(DOWN).to_corner(UR)
        
        self.play(Write(eigenvals))
        self.wait(2)
'''

LIMITS_VISUAL = '''
from manim import *
import numpy as np

class LimitsVisualization(Scene):
    """Visualize the concept of limits."""
    
    def construct(self):
        title = Text("Limits").scale(0.5).to_edge(UP)
        self.play(Write(title))
        
        # Axes
        axes = Axes(
            x_range=[-1, 5, 1],
            y_range=[-1, 5, 1],
            x_length=7,
            y_length=5,
            axis_config={"include_tip": False}
        ).shift(DOWN*0.5)
        
        self.play(Create(axes))
        
        # Function with a hole at x=2
        def f(x):
            if abs(x - 2) < 0.01:
                return None
            return (x**2 - 4) / (x - 2)  # = x + 2 for x ≠ 2
        
        graph = axes.plot(lambda x: x + 2, x_range=[-0.5, 4.5], color=BLUE)
        self.play(Create(graph))
        
        # Hole at x=2
        hole = Circle(radius=0.08, color=WHITE, fill_color=BLACK, fill_opacity=1).move_to(axes.c2p(2, 4))
        self.play(Create(hole))
        
        # Approaching from left and right
        left_dots = VGroup()
        right_dots = VGroup()
        
        for x in [1.5, 1.7, 1.9, 1.95]:
            dot = Dot(axes.c2p(x, x + 2), color=GREEN, radius=0.05)
            left_dots.add(dot)
        
        for x in [2.5, 2.3, 2.1, 2.05]:
            dot = Dot(axes.c2p(x, x + 2), color=RED, radius=0.05)
            right_dots.add(dot)
        
        self.play(Create(left_dots), Create(right_dots))
        
        # Limit notation
        limit = MathTex(r"\\lim_{x \\to 2} \\frac{x^2 - 4}{x - 2} = 4").scale(0.5).shift(RIGHT*2 + UP*2)
        self.play(Write(limit))
        
        # Target line
        target = DashedLine(axes.c2p(-0.5, 4), axes.c2p(4.5, 4), color=YELLOW)
        self.play(Create(target))
        self.wait(2)
'''

# Registry
SPECIAL_TEMPLATES = {
    "fourier": FOURIER_SERIES,
    "euler_identity": EULER_IDENTITY,
    "golden_ratio": GOLDEN_RATIO,
    "mandelbrot": MANDELBROT,
    "complex_mult": COMPLEX_MULTIPLICATION,
    "eigenvalues": EIGENVALUE_VISUAL,
    "limits": LIMITS_VISUAL,
}

def get_special_template(name: str) -> str:
    return SPECIAL_TEMPLATES.get(name, "")

def list_special_templates() -> list:
    return list(SPECIAL_TEMPLATES.keys())
