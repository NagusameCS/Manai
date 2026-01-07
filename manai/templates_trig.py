"""
Trigonometry Templates - Visual explanations for trig concepts
"""

UNIT_CIRCLE = '''
from manim import *

class UnitCircle(Scene):
    """Visualize the unit circle and trig functions."""
    
    def construct(self):
        title = Text("Unit Circle").scale(0.5).to_edge(UP)
        self.play(Write(title))
        
        # Create unit circle
        circle = Circle(radius=2, color=WHITE)
        axes = Axes(x_range=[-2.5, 2.5], y_range=[-2.5, 2.5], x_length=5, y_length=5)
        
        self.play(Create(axes), Create(circle))
        
        # Rotating angle
        angle = ValueTracker(0)
        
        radius = always_redraw(lambda: Line(
            ORIGIN, 2 * np.array([np.cos(angle.get_value()), np.sin(angle.get_value()), 0]),
            color=YELLOW
        ))
        
        point = always_redraw(lambda: Dot(
            2 * np.array([np.cos(angle.get_value()), np.sin(angle.get_value()), 0]),
            color=RED
        ))
        
        # Sin and cos lines
        cos_line = always_redraw(lambda: Line(
            ORIGIN, [2*np.cos(angle.get_value()), 0, 0], color=BLUE
        ))
        sin_line = always_redraw(lambda: Line(
            [2*np.cos(angle.get_value()), 0, 0],
            [2*np.cos(angle.get_value()), 2*np.sin(angle.get_value()), 0],
            color=GREEN
        ))
        
        cos_label = always_redraw(lambda: MathTex("\\cos", color=BLUE).scale(0.4).next_to(cos_line, DOWN, buff=0.1))
        sin_label = always_redraw(lambda: MathTex("\\sin", color=GREEN).scale(0.4).next_to(sin_line, RIGHT, buff=0.1))
        
        self.play(Create(radius), Create(point))
        self.play(Create(cos_line), Create(sin_line))
        self.play(Write(cos_label), Write(sin_label))
        
        # Animate rotation
        self.play(angle.animate.set_value(2*PI), run_time=6, rate_func=linear)
        self.wait(2)
'''

TRIG_IDENTITIES = '''
from manim import *

class TrigIdentities(Scene):
    """Visualize Pythagorean identity."""
    
    def construct(self):
        title = Text("Pythagorean Identity").scale(0.5).to_edge(UP)
        self.play(Write(title))
        
        # Unit circle
        circle = Circle(radius=2, color=WHITE).shift(LEFT*2)
        
        angle = PI/4
        point = Dot(2 * np.array([np.cos(angle), np.sin(angle), 0]) + LEFT*2, color=RED)
        
        # Right triangle
        cos_line = Line(LEFT*2, LEFT*2 + RIGHT*2*np.cos(angle), color=BLUE)
        sin_line = Line(
            LEFT*2 + RIGHT*2*np.cos(angle),
            LEFT*2 + RIGHT*2*np.cos(angle) + UP*2*np.sin(angle),
            color=GREEN
        )
        hyp = Line(LEFT*2, point.get_center(), color=YELLOW)
        
        self.play(Create(circle))
        self.play(Create(cos_line), Create(sin_line), Create(hyp), Create(point))
        
        # Labels
        cos_label = MathTex("\\cos\\theta", color=BLUE).scale(0.4).next_to(cos_line, DOWN)
        sin_label = MathTex("\\sin\\theta", color=GREEN).scale(0.4).next_to(sin_line, RIGHT)
        hyp_label = MathTex("1", color=YELLOW).scale(0.4).next_to(hyp, UL, buff=0.1)
        
        self.play(Write(cos_label), Write(sin_label), Write(hyp_label))
        
        # Pythagorean theorem
        identity = MathTex("\\sin^2\\theta + \\cos^2\\theta = 1").scale(0.6).shift(RIGHT*2)
        self.play(Write(identity))
        self.wait(2)
'''

LAW_OF_SINES = '''
from manim import *

class LawOfSines(Scene):
    """Visualize the Law of Sines."""
    
    def construct(self):
        title = Text("Law of Sines").scale(0.5).to_edge(UP)
        self.play(Write(title))
        
        # Create triangle
        A = np.array([-2, -1, 0])
        B = np.array([2, -1, 0])
        C = np.array([0.5, 1.5, 0])
        
        triangle = Polygon(A, B, C, color=BLUE)
        
        # Labels for vertices
        label_A = MathTex("A").scale(0.5).next_to(A, DL, buff=0.1)
        label_B = MathTex("B").scale(0.5).next_to(B, DR, buff=0.1)
        label_C = MathTex("C").scale(0.5).next_to(C, UP, buff=0.1)
        
        # Side labels
        a = Line(B, C, color=RED)
        b = Line(A, C, color=GREEN)
        c = Line(A, B, color=YELLOW)
        
        a_label = MathTex("a", color=RED).scale(0.4).next_to(a, RIGHT)
        b_label = MathTex("b", color=GREEN).scale(0.4).next_to(b, LEFT)
        c_label = MathTex("c", color=YELLOW).scale(0.4).next_to(c, DOWN)
        
        self.play(Create(triangle))
        self.play(Write(label_A), Write(label_B), Write(label_C))
        self.play(Create(a), Create(b), Create(c))
        self.play(Write(a_label), Write(b_label), Write(c_label))
        
        # Law of Sines
        law = MathTex(r"\\frac{a}{\\sin A} = \\frac{b}{\\sin B} = \\frac{c}{\\sin C}").scale(0.5).to_edge(DOWN)
        self.play(Write(law))
        self.wait(2)
'''

LAW_OF_COSINES = '''
from manim import *

class LawOfCosines(Scene):
    """Visualize the Law of Cosines."""
    
    def construct(self):
        title = Text("Law of Cosines").scale(0.5).to_edge(UP)
        self.play(Write(title))
        
        # Create triangle
        A = np.array([-2, -1, 0])
        B = np.array([2, -1, 0])
        C = np.array([0, 1.5, 0])
        
        triangle = Polygon(A, B, C, color=BLUE, fill_opacity=0.2)
        
        self.play(Create(triangle))
        
        # Show sides
        side_c = Line(A, B, color=YELLOW)
        side_a = Line(B, C, color=RED)
        side_b = Line(A, C, color=GREEN)
        
        self.play(Create(side_a), Create(side_b), Create(side_c))
        
        # Angle at C
        angle_arc = Arc(radius=0.3, start_angle=PI + 0.4, angle=-1.2, color=PURPLE).shift(C)
        angle_label = MathTex("C", color=PURPLE).scale(0.4).next_to(angle_arc, DOWN, buff=0.2)
        
        self.play(Create(angle_arc), Write(angle_label))
        
        # Formula
        formula = MathTex("c^2 = a^2 + b^2 - 2ab\\cos C").scale(0.5).to_edge(DOWN)
        self.play(Write(formula))
        self.wait(2)
'''

INVERSE_TRIG = '''
from manim import *

class InverseTrig(Scene):
    """Visualize inverse trig functions."""
    
    def construct(self):
        title = Text("Inverse Trig Functions").scale(0.5).to_edge(UP)
        self.play(Write(title))
        
        axes = Axes(x_range=[-2, 2], y_range=[-2, 2], x_length=4, y_length=4).shift(LEFT*3)
        
        # sin function (restricted)
        sin_graph = axes.plot(np.sin, x_range=[-PI/2, PI/2], color=BLUE)
        sin_label = MathTex("y = \\sin x").scale(0.4).next_to(axes, DOWN)
        
        self.play(Create(axes), Create(sin_graph), Write(sin_label))
        
        # arcsin on right
        axes2 = Axes(x_range=[-2, 2], y_range=[-2, 2], x_length=4, y_length=4).shift(RIGHT*3)
        arcsin_graph = axes2.plot(np.arcsin, x_range=[-0.99, 0.99], color=RED)
        arcsin_label = MathTex("y = \\arcsin x").scale(0.4).next_to(axes2, DOWN)
        
        self.play(Create(axes2), Create(arcsin_graph), Write(arcsin_label))
        
        # Show reflection line
        reflection = Text("Reflected over y = x").scale(0.35).shift(UP*2)
        self.play(Write(reflection))
        self.wait(2)
'''

POLAR_COORDINATES = '''
from manim import *

class PolarCoordinates(Scene):
    """Visualize polar coordinate system."""
    
    def construct(self):
        title = Text("Polar Coordinates").scale(0.5).to_edge(UP)
        self.play(Write(title))
        
        # Create polar grid
        circles = VGroup(*[Circle(radius=r, color=GRAY, stroke_opacity=0.3) for r in [0.5, 1, 1.5, 2, 2.5]])
        rays = VGroup(*[Line(ORIGIN, 3*np.array([np.cos(a), np.sin(a), 0]), color=GRAY, stroke_opacity=0.3) 
                        for a in np.linspace(0, 2*PI, 12, endpoint=False)])
        
        self.play(Create(circles), Create(rays))
        
        # Plot a point
        r, theta = 2, PI/3
        point = Dot(r * np.array([np.cos(theta), np.sin(theta), 0]), color=RED)
        
        radius_line = Line(ORIGIN, point.get_center(), color=YELLOW)
        angle_arc = Arc(radius=0.5, start_angle=0, angle=theta, color=GREEN)
        
        r_label = MathTex("r=2").scale(0.4).next_to(radius_line, UL, buff=0.1)
        theta_label = MathTex(r"\\theta=\\frac{\\pi}{3}").scale(0.4).next_to(angle_arc, RIGHT)
        
        self.play(Create(radius_line), Create(angle_arc))
        self.play(Create(point), Write(r_label), Write(theta_label))
        
        # Conversion formulas
        formulas = MathTex(r"x = r\\cos\\theta, \\quad y = r\\sin\\theta").scale(0.5).to_edge(DOWN)
        self.play(Write(formulas))
        self.wait(2)
'''

WAVE_PROPERTIES = '''
from manim import *

class WaveProperties(Scene):
    """Visualize sine wave properties."""
    
    def construct(self):
        title = Text("Sine Wave Properties").scale(0.5).to_edge(UP)
        self.play(Write(title))
        
        axes = Axes(x_range=[0, 4*PI], y_range=[-2.5, 2.5], x_length=10, y_length=4)
        
        # Standard sine
        A, omega = 1.5, 1
        wave = axes.plot(lambda x: A * np.sin(omega * x), color=BLUE)
        
        self.play(Create(axes), Create(wave))
        
        # Show amplitude
        amp_line = DoubleArrow(axes.c2p(PI/2, 0), axes.c2p(PI/2, A), color=RED, buff=0)
        amp_label = MathTex("A", color=RED).scale(0.5).next_to(amp_line, RIGHT)
        
        self.play(Create(amp_line), Write(amp_label))
        
        # Show period
        period_line = DoubleArrow(axes.c2p(0, -2), axes.c2p(2*PI, -2), color=GREEN, buff=0)
        period_label = MathTex("T = 2\\pi", color=GREEN).scale(0.5).next_to(period_line, DOWN)
        
        self.play(Create(period_line), Write(period_label))
        
        # Formula
        formula = MathTex("y = A\\sin(\\omega x)").scale(0.5).to_corner(UR)
        self.play(Write(formula))
        self.wait(2)
'''

# Registry
TRIG_TEMPLATES = {
    "unit_circle": UNIT_CIRCLE,
    "trig_identities": TRIG_IDENTITIES,
    "law_of_sines": LAW_OF_SINES,
    "law_of_cosines": LAW_OF_COSINES,
    "inverse_trig": INVERSE_TRIG,
    "polar_coords": POLAR_COORDINATES,
    "wave_properties": WAVE_PROPERTIES,
}

def get_trig_template(name: str) -> str:
    return TRIG_TEMPLATES.get(name, "")

def list_trig_templates() -> list:
    return list(TRIG_TEMPLATES.keys())
