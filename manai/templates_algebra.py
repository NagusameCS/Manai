"""
Algebra Templates - Intuitive visualizations for algebraic concepts
"""

QUADRATIC_FORMULA = '''
from manim import *

class QuadraticFormula(Scene):
    """Derive the quadratic formula visually."""
    
    def construct(self):
        title = Text("Quadratic Formula").scale(0.5).to_edge(UP)
        self.play(Write(title))
        
        axes = Axes(x_range=[-4, 4], y_range=[-3, 5], x_length=6, y_length=4).shift(DOWN*0.3)
        
        # Show parabola
        a, b, c = 1, -2, -3
        func = lambda x: a*x**2 + b*x + c
        parabola = axes.plot(func, x_range=[-2, 4], color=BLUE)
        
        self.play(Create(axes), Create(parabola))
        
        # Show roots
        roots = [(-b + np.sqrt(b**2 - 4*a*c))/(2*a), (-b - np.sqrt(b**2 - 4*a*c))/(2*a)]
        root_dots = VGroup(*[Dot(axes.c2p(r, 0), color=RED) for r in roots])
        
        self.play(Create(root_dots))
        
        # Formula
        formula = MathTex(r"x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}").scale(0.5).to_edge(DOWN)
        self.play(Write(formula))
        self.wait(2)
'''

COMPLETING_SQUARE = '''
from manim import *

class CompletingSquare(Scene):
    """Visualize completing the square."""
    
    def construct(self):
        title = Text("Completing the Square").scale(0.5).to_edge(UP)
        self.play(Write(title))
        
        # Start with x^2 + 6x
        expr1 = MathTex("x^2 + 6x").scale(0.7)
        self.play(Write(expr1))
        self.wait()
        
        # Show geometric representation
        x_sq = Square(side_length=2, color=BLUE, fill_opacity=0.3).shift(LEFT*2)
        x_label = MathTex("x").scale(0.5).next_to(x_sq, DOWN)
        
        rect1 = Rectangle(width=0.6, height=2, color=GREEN, fill_opacity=0.3).next_to(x_sq, RIGHT, buff=0)
        rect2 = Rectangle(width=2, height=0.6, color=GREEN, fill_opacity=0.3).next_to(x_sq, DOWN, buff=0)
        
        self.play(expr1.animate.shift(UP*2))
        self.play(Create(x_sq), Write(x_label))
        self.play(Create(rect1), Create(rect2))
        self.wait()
        
        # Complete with corner
        corner = Square(side_length=0.6, color=RED, fill_opacity=0.3).next_to(rect1, DOWN, buff=0)
        self.play(Create(corner))
        
        # Result
        result = MathTex("= (x + 3)^2 - 9").scale(0.6).next_to(expr1, DOWN)
        self.play(Write(result))
        self.wait(2)
'''

LOGARITHM_EXPLAINED = '''
from manim import *

class LogarithmExplained(Scene):
    """Explain what logarithm means visually."""
    
    def construct(self):
        title = Text("What is a Logarithm?").scale(0.5).to_edge(UP)
        self.play(Write(title))
        
        # Show exponential first
        question = MathTex("2^? = 8").scale(0.7)
        self.play(Write(question))
        self.wait()
        
        # Visual: doubling
        dots_1 = VGroup(Dot()).shift(LEFT*3)
        dots_2 = VGroup(*[Dot() for _ in range(2)]).arrange(RIGHT, buff=0.2).shift(LEFT*1.5)
        dots_4 = VGroup(*[Dot() for _ in range(4)]).arrange(RIGHT, buff=0.2).shift(RIGHT*0)
        dots_8 = VGroup(*[Dot() for _ in range(8)]).arrange(RIGHT, buff=0.2).shift(RIGHT*2)
        
        labels = VGroup(
            MathTex("2^0=1").scale(0.4).next_to(dots_1, DOWN),
            MathTex("2^1=2").scale(0.4).next_to(dots_2, DOWN),
            MathTex("2^2=4").scale(0.4).next_to(dots_4, DOWN),
            MathTex("2^3=8").scale(0.4).next_to(dots_8, DOWN),
        )
        
        self.play(question.animate.shift(UP*2))
        self.play(Create(dots_1), Write(labels[0]))
        self.play(Transform(dots_1.copy(), dots_2), Write(labels[1]))
        self.play(Create(dots_4), Write(labels[2]))
        self.play(Create(dots_8), Write(labels[3]))
        
        # Answer
        answer = MathTex(r"\\log_2(8) = 3").scale(0.6).to_edge(DOWN)
        explanation = Text("How many times do we multiply?").scale(0.35).next_to(answer, UP)
        
        self.play(Write(explanation), Write(answer))
        self.wait(2)
'''

EXPONENT_RULES = '''
from manim import *

class ExponentRules(Scene):
    """Visualize exponent rules."""
    
    def construct(self):
        title = Text("Exponent Rules").scale(0.5).to_edge(UP)
        self.play(Write(title))
        
        # Rule 1: x^a * x^b = x^(a+b)
        rule1 = MathTex("x^a \\cdot x^b = x^{a+b}").scale(0.6).shift(UP)
        
        # Visual: 2^2 * 2^3 = 2^5
        example1 = VGroup(
            MathTex("2^2").scale(0.5),
            MathTex("\\cdot").scale(0.5),
            MathTex("2^3").scale(0.5),
            MathTex("=").scale(0.5),
            MathTex("2^5").scale(0.5),
        ).arrange(RIGHT).shift(DOWN*0.5)
        
        dots1 = VGroup(*[Dot(color=BLUE) for _ in range(4)]).arrange(RIGHT, buff=0.1).shift(DOWN*1.5 + LEFT*2)
        dots2 = VGroup(*[Dot(color=RED) for _ in range(8)]).arrange(RIGHT, buff=0.1).shift(DOWN*1.5 + RIGHT)
        
        self.play(Write(rule1))
        self.play(Write(example1))
        self.play(Create(dots1), Create(dots2))
        
        # Show multiplication combining
        combined = VGroup(*[Dot(color=GREEN) for _ in range(32)]).arrange_in_grid(4, 8, buff=0.1).shift(DOWN*2.5)
        self.play(Transform(VGroup(dots1, dots2), combined))
        self.wait(2)
'''

COMPLEX_NUMBERS = '''
from manim import *

class ComplexNumbers(Scene):
    """Visualize complex numbers on the plane."""
    
    def construct(self):
        title = Text("Complex Numbers").scale(0.5).to_edge(UP)
        self.play(Write(title))
        
        # Complex plane
        plane = ComplexPlane(x_range=[-4, 4], y_range=[-3, 3], x_length=7, y_length=5)
        plane.add_coordinates()
        
        self.play(Create(plane))
        
        # Plot a complex number
        z = 3 + 2j
        dot = Dot(plane.n2p(z), color=RED)
        label = MathTex("3 + 2i").scale(0.5).next_to(dot, UR)
        
        # Real and imaginary parts
        real_line = Line(plane.n2p(0), plane.n2p(3), color=BLUE)
        imag_line = Line(plane.n2p(3), plane.n2p(z), color=GREEN)
        
        self.play(Create(real_line), Create(imag_line))
        self.play(Create(dot), Write(label))
        
        # Show magnitude
        mag_line = Line(plane.n2p(0), plane.n2p(z), color=YELLOW)
        mag_label = MathTex("|z| = \\sqrt{13}").scale(0.4).next_to(mag_line, LEFT)
        
        self.play(Create(mag_line), Write(mag_label))
        self.wait(2)
'''

POLYNOMIAL_ROOTS = '''
from manim import *

class PolynomialRoots(Scene):
    """Visualize polynomial roots."""
    
    def construct(self):
        title = Text("Polynomial Roots").scale(0.5).to_edge(UP)
        self.play(Write(title))
        
        axes = Axes(x_range=[-4, 4], y_range=[-5, 5], x_length=7, y_length=4).shift(DOWN*0.3)
        
        # Polynomial with roots at -2, 1, 3
        func = lambda x: 0.2 * (x + 2) * (x - 1) * (x - 3)
        graph = axes.plot(func, x_range=[-3, 4], color=BLUE)
        
        self.play(Create(axes), Create(graph))
        
        # Highlight roots
        roots = [-2, 1, 3]
        for r in roots:
            dot = Dot(axes.c2p(r, 0), color=RED)
            label = MathTex(f"x = {r}").scale(0.4).next_to(dot, DOWN)
            self.play(Create(dot), Write(label))
        
        # Factored form
        factored = MathTex("f(x) = (x+2)(x-1)(x-3)").scale(0.5).to_edge(DOWN)
        self.play(Write(factored))
        self.wait(2)
'''

SEQUENCES_SERIES = '''
from manim import *

class SequencesSeries(Scene):
    """Visualize arithmetic and geometric sequences."""
    
    def construct(self):
        title = Text("Sequences").scale(0.5).to_edge(UP)
        self.play(Write(title))
        
        # Arithmetic sequence: constant difference
        arith_label = Text("Arithmetic: +3 each time").scale(0.35).shift(UP*1.5 + LEFT*3)
        arith_dots = VGroup()
        for i, val in enumerate([2, 5, 8, 11, 14]):
            dot = Dot().shift(LEFT*3 + RIGHT*i*1.2 + UP*0.5)
            label = MathTex(str(val)).scale(0.4).next_to(dot, DOWN)
            arith_dots.add(VGroup(dot, label))
        
        self.play(Write(arith_label))
        for d in arith_dots:
            self.play(Create(d), run_time=0.3)
        
        # Geometric sequence: constant ratio
        geo_label = Text("Geometric: x2 each time").scale(0.35).shift(DOWN*1 + LEFT*3)
        geo_dots = VGroup()
        for i, val in enumerate([1, 2, 4, 8, 16]):
            dot = Dot(color=RED).shift(LEFT*3 + RIGHT*i*1.2 + DOWN*1.5)
            label = MathTex(str(val)).scale(0.4).next_to(dot, DOWN)
            geo_dots.add(VGroup(dot, label))
        
        self.play(Write(geo_label))
        for d in geo_dots:
            self.play(Create(d), run_time=0.3)
        
        self.wait(2)
'''

FACTORING_VISUAL = '''
from manim import *

class FactoringVisual(Scene):
    """Visualize factoring as area."""
    
    def construct(self):
        title = Text("Factoring").scale(0.5).to_edge(UP)
        self.play(Write(title))
        
        # x^2 + 5x + 6 = (x+2)(x+3)
        expr = MathTex("x^2 + 5x + 6").scale(0.6).shift(UP*2)
        self.play(Write(expr))
        
        # Create rectangle grid
        main_sq = Square(side_length=2, color=BLUE, fill_opacity=0.3).shift(LEFT*1.5 + DOWN*0.5)
        main_label = MathTex("x^2").scale(0.4).move_to(main_sq)
        
        rect_right = Rectangle(width=1.5, height=2, color=GREEN, fill_opacity=0.3).next_to(main_sq, RIGHT, buff=0)
        rect_label1 = MathTex("3x").scale(0.4).move_to(rect_right)
        
        rect_bottom = Rectangle(width=2, height=1, color=GREEN, fill_opacity=0.3).next_to(main_sq, DOWN, buff=0)
        rect_label2 = MathTex("2x").scale(0.4).move_to(rect_bottom)
        
        corner = Rectangle(width=1.5, height=1, color=YELLOW, fill_opacity=0.3).next_to(rect_right, DOWN, buff=0)
        corner_label = MathTex("6").scale(0.4).move_to(corner)
        
        self.play(Create(main_sq), Write(main_label))
        self.play(Create(rect_right), Write(rect_label1))
        self.play(Create(rect_bottom), Write(rect_label2))
        self.play(Create(corner), Write(corner_label))
        
        # Show factors on sides
        result = MathTex("= (x+2)(x+3)").scale(0.6).next_to(expr, DOWN)
        self.play(Write(result))
        self.wait(2)
'''

# Registry
ALGEBRA_TEMPLATES = {
    "quadratic": QUADRATIC_FORMULA,
    "completing_square": COMPLETING_SQUARE,
    "logarithm": LOGARITHM_EXPLAINED,
    "exponent_rules": EXPONENT_RULES,
    "complex_numbers": COMPLEX_NUMBERS,
    "polynomial_roots": POLYNOMIAL_ROOTS,
    "sequences": SEQUENCES_SERIES,
    "factoring": FACTORING_VISUAL,
}

def get_algebra_template(name: str) -> str:
    return ALGEBRA_TEMPLATES.get(name, "")

def list_algebra_templates() -> list:
    return list(ALGEBRA_TEMPLATES.keys())
