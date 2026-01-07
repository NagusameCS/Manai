"""
Physics Templates - Visual demonstrations of physics concepts
"""

NEWTONS_LAWS = '''
from manim import *

class NewtonsLaws(Scene):
    """Visualize Newton's Laws of Motion."""
    
    def construct(self):
        title = Text("Newton's Laws").scale(0.5).to_edge(UP)
        self.play(Write(title))
        
        # First Law - Object at rest
        law1 = Text("1st: Objects at rest stay at rest").scale(0.35).shift(UP*2)
        box1 = Square(side_length=0.5, color=BLUE, fill_opacity=0.5).shift(LEFT*3 + UP*0.5)
        
        self.play(Write(law1), Create(box1))
        self.wait(1)
        
        # Second Law - F = ma
        law2 = Text("2nd: F = ma").scale(0.35).shift(UP*0.5)
        box2 = Square(side_length=0.5, color=GREEN, fill_opacity=0.5).shift(LEFT*3 + DOWN*0.5)
        force_arrow = Arrow(box2.get_left(), box2.get_left() + LEFT, color=RED)
        
        self.play(Write(law2), Create(box2), Create(force_arrow))
        self.play(box2.animate.shift(RIGHT*3), run_time=2)
        
        # Third Law - Action/Reaction
        law3 = Text("3rd: Action = Reaction").scale(0.35).shift(DOWN*1.5)
        box3a = Square(side_length=0.5, color=BLUE, fill_opacity=0.5).shift(LEFT*1 + DOWN*2.5)
        box3b = Square(side_length=0.5, color=RED, fill_opacity=0.5).shift(RIGHT*1 + DOWN*2.5)
        
        arrow_ab = Arrow(box3a.get_right(), box3b.get_left(), color=YELLOW, buff=0.1)
        arrow_ba = Arrow(box3b.get_left(), box3a.get_right(), color=YELLOW, buff=0.1).shift(DOWN*0.3)
        
        self.play(Write(law3), Create(box3a), Create(box3b))
        self.play(Create(arrow_ab), Create(arrow_ba))
        self.wait(2)
'''

MOMENTUM_CONSERVATION = '''
from manim import *

class MomentumConservation(Scene):
    """Visualize conservation of momentum in collision."""
    
    def construct(self):
        title = Text("Conservation of Momentum").scale(0.5).to_edge(UP)
        self.play(Write(title))
        
        # Two balls
        ball1 = Circle(radius=0.4, color=BLUE, fill_opacity=0.8).shift(LEFT*4)
        ball2 = Circle(radius=0.4, color=RED, fill_opacity=0.8).shift(RIGHT*1)
        
        v1_arrow = Arrow(ball1.get_right(), ball1.get_right() + RIGHT*1.5, color=YELLOW, buff=0)
        v2_arrow = Arrow(ball2.get_left(), ball2.get_left() + LEFT*0.5, color=YELLOW, buff=0)
        
        m1_label = MathTex("m_1").scale(0.4).next_to(ball1, UP)
        m2_label = MathTex("m_2").scale(0.4).next_to(ball2, UP)
        
        self.play(Create(ball1), Create(ball2))
        self.play(Write(m1_label), Write(m2_label))
        self.play(Create(v1_arrow), Create(v2_arrow))
        
        # Before momentum
        before = MathTex("p_{before} = m_1 v_1 + m_2 v_2").scale(0.5).shift(DOWN*2)
        self.play(Write(before))
        
        # Collision
        self.play(
            ball1.animate.shift(RIGHT*3),
            ball2.animate.shift(LEFT*1),
            v1_arrow.animate.shift(RIGHT*3),
            v2_arrow.animate.shift(LEFT*1),
            run_time=1.5
        )
        
        # After collision - balls bounce
        self.play(
            ball1.animate.shift(LEFT*1),
            ball2.animate.shift(RIGHT*2),
            run_time=1
        )
        
        after = MathTex("p_{after} = p_{before}").scale(0.5).next_to(before, DOWN)
        self.play(Write(after))
        self.wait(2)
'''

ENERGY_CONSERVATION = '''
from manim import *

class EnergyConservation(Scene):
    """Visualize conservation of energy."""
    
    def construct(self):
        title = Text("Conservation of Energy").scale(0.5).to_edge(UP)
        self.play(Write(title))
        
        # Pendulum or falling ball
        ground = Line(LEFT*4, RIGHT*4, color=WHITE).shift(DOWN*2)
        self.play(Create(ground))
        
        # Ball falling
        height = ValueTracker(2)
        
        ball = always_redraw(lambda: Circle(
            radius=0.3, color=BLUE, fill_opacity=0.8
        ).shift(UP*height.get_value()))
        
        # Energy bars
        def get_pe_bar():
            h = height.get_value()
            return Rectangle(width=0.5, height=h, color=GREEN, fill_opacity=0.7).move_to(
                RIGHT*3 + UP*(h/2 - 2)
            )
        
        def get_ke_bar():
            h = height.get_value()
            ke_height = 2 - h  # Converts PE to KE
            return Rectangle(width=0.5, height=max(ke_height, 0.01), color=RED, fill_opacity=0.7).move_to(
                RIGHT*4 + UP*(max(ke_height, 0.01)/2 - 2)
            )
        
        pe_bar = always_redraw(get_pe_bar)
        ke_bar = always_redraw(get_ke_bar)
        
        pe_label = Text("PE", color=GREEN).scale(0.3).next_to(pe_bar, UP)
        ke_label = Text("KE", color=RED).scale(0.3).next_to(ke_bar, UP)
        
        self.play(Create(ball))
        self.play(Create(pe_bar), Create(ke_bar))
        
        # Drop the ball
        self.play(height.animate.set_value(0), run_time=2, rate_func=rate_functions.ease_in_quad)
        
        # Energy equation
        equation = MathTex("PE + KE = constant").scale(0.5).shift(DOWN*3)
        self.play(Write(equation))
        self.wait(2)
'''

FRICTION_VISUAL = '''
from manim import *

class FrictionVisual(Scene):
    """Visualize friction forces."""
    
    def construct(self):
        title = Text("Friction").scale(0.5).to_edge(UP)
        self.play(Write(title))
        
        # Surface
        surface = Line(LEFT*4, RIGHT*4, color=GRAY).shift(DOWN*1)
        self.play(Create(surface))
        
        # Block
        block = Square(side_length=1, color=BLUE, fill_opacity=0.5).shift(DOWN*0.5)
        self.play(Create(block))
        
        # Forces
        weight = Arrow(block.get_center(), block.get_center() + DOWN*1.2, color=RED, buff=0)
        normal = Arrow(block.get_bottom(), block.get_bottom() + UP*1.2, color=GREEN, buff=0)
        
        w_label = MathTex("W = mg", color=RED).scale(0.4).next_to(weight, RIGHT)
        n_label = MathTex("N", color=GREEN).scale(0.4).next_to(normal, LEFT)
        
        self.play(Create(weight), Create(normal))
        self.play(Write(w_label), Write(n_label))
        
        # Applied force and friction
        applied = Arrow(block.get_left() + LEFT*0.5, block.get_left(), color=YELLOW, buff=0)
        friction = Arrow(block.get_right(), block.get_right() + RIGHT*0.7, color=PURPLE, buff=0)
        
        f_label = MathTex("F", color=YELLOW).scale(0.4).next_to(applied, LEFT)
        fr_label = MathTex("f = \\mu N", color=PURPLE).scale(0.4).next_to(friction, RIGHT)
        
        self.play(Create(applied), Write(f_label))
        self.play(Create(friction), Write(fr_label))
        self.wait(2)
'''

PENDULUM_MOTION = '''
from manim import *

class PendulumMotion(Scene):
    """Visualize pendulum motion."""
    
    def construct(self):
        title = Text("Pendulum").scale(0.5).to_edge(UP)
        self.play(Write(title))
        
        pivot = Dot(UP*2, color=WHITE)
        self.play(Create(pivot))
        
        # Pendulum
        L = 2.5
        theta = ValueTracker(PI/4)
        
        def get_pendulum():
            angle = theta.get_value()
            bob_pos = pivot.get_center() + L * np.array([np.sin(angle), -np.cos(angle), 0])
            rod = Line(pivot.get_center(), bob_pos, color=GRAY)
            bob = Circle(radius=0.2, color=BLUE, fill_opacity=0.8).move_to(bob_pos)
            return VGroup(rod, bob)
        
        pendulum = always_redraw(get_pendulum)
        self.play(Create(pendulum))
        
        # Swing back and forth
        for _ in range(3):
            self.play(theta.animate.set_value(-PI/4), run_time=1, rate_func=rate_functions.ease_in_out_sine)
            self.play(theta.animate.set_value(PI/4), run_time=1, rate_func=rate_functions.ease_in_out_sine)
        
        # Period formula
        formula = MathTex(r"T = 2\\pi\\sqrt{\\frac{L}{g}}").scale(0.5).to_edge(DOWN)
        self.play(Write(formula))
        self.wait(2)
'''

CIRCULAR_MOTION = '''
from manim import *

class CircularMotion(Scene):
    """Visualize uniform circular motion."""
    
    def construct(self):
        title = Text("Circular Motion").scale(0.5).to_edge(UP)
        self.play(Write(title))
        
        # Circle path
        radius = 2
        circle = Circle(radius=radius, color=GRAY, stroke_opacity=0.5)
        center = Dot(ORIGIN, color=WHITE)
        
        self.play(Create(circle), Create(center))
        
        # Moving object
        angle = ValueTracker(0)
        
        obj = always_redraw(lambda: Dot(
            radius * np.array([np.cos(angle.get_value()), np.sin(angle.get_value()), 0]),
            color=BLUE
        ))
        
        # Velocity vector (tangent)
        velocity = always_redraw(lambda: Arrow(
            obj.get_center(),
            obj.get_center() + 0.8 * np.array([-np.sin(angle.get_value()), np.cos(angle.get_value()), 0]),
            color=GREEN, buff=0
        ))
        
        # Centripetal acceleration (toward center)
        accel = always_redraw(lambda: Arrow(
            obj.get_center(),
            obj.get_center() - 0.6 * np.array([np.cos(angle.get_value()), np.sin(angle.get_value()), 0]),
            color=RED, buff=0
        ))
        
        v_label = MathTex("v", color=GREEN).scale(0.4).shift(RIGHT*3 + UP)
        a_label = MathTex("a_c", color=RED).scale(0.4).next_to(v_label, DOWN)
        
        self.play(Create(obj), Create(velocity), Create(accel))
        self.play(Write(v_label), Write(a_label))
        
        self.play(angle.animate.set_value(2*PI), run_time=4, rate_func=linear)
        
        formula = MathTex(r"a_c = \\frac{v^2}{r}").scale(0.5).to_edge(DOWN)
        self.play(Write(formula))
        self.wait(2)
'''

ELECTRIC_FIELD = '''
from manim import *

class ElectricField(Scene):
    """Visualize electric field lines."""
    
    def construct(self):
        title = Text("Electric Field").scale(0.5).to_edge(UP)
        self.play(Write(title))
        
        # Positive charge
        pos_charge = Circle(radius=0.3, color=RED, fill_opacity=0.8).shift(LEFT*2)
        plus = MathTex("+").scale(0.6).move_to(pos_charge)
        
        # Negative charge
        neg_charge = Circle(radius=0.3, color=BLUE, fill_opacity=0.8).shift(RIGHT*2)
        minus = MathTex("-").scale(0.6).move_to(neg_charge)
        
        self.play(Create(pos_charge), Write(plus), Create(neg_charge), Write(minus))
        
        # Field lines from positive to negative
        field_lines = VGroup()
        for angle in np.linspace(-PI/3, PI/3, 5):
            start = pos_charge.get_center() + 0.3 * np.array([np.cos(angle), np.sin(angle), 0])
            end = neg_charge.get_center() - 0.3 * np.array([np.cos(angle), np.sin(angle), 0])
            
            # Curved line
            ctrl1 = start + RIGHT*1.5 + np.array([0, np.sin(angle)*1.5, 0])
            ctrl2 = end + LEFT*1.5 + np.array([0, np.sin(angle)*1.5, 0])
            
            line = CubicBezier(start, ctrl1, ctrl2, end, color=YELLOW)
            arrow = Arrow(end + LEFT*0.3, end, color=YELLOW, buff=0, stroke_width=2)
            field_lines.add(line, arrow)
        
        self.play(Create(field_lines), run_time=2)
        
        formula = MathTex(r"E = \\frac{kq}{r^2}").scale(0.5).to_edge(DOWN)
        self.play(Write(formula))
        self.wait(2)
'''

DOPPLER_EFFECT = '''
from manim import *

class DopplerEffect(Scene):
    """Visualize the Doppler effect."""
    
    def construct(self):
        title = Text("Doppler Effect").scale(0.5).to_edge(UP)
        self.play(Write(title))
        
        # Moving source
        source = Dot(color=RED).shift(LEFT*4)
        source_label = Text("Source", color=RED).scale(0.3).next_to(source, UP)
        
        self.play(Create(source), Write(source_label))
        
        # Wave circles
        waves = VGroup()
        centers = []
        
        for i in range(5):
            center = LEFT*(4 - i*0.8)
            circle = Circle(radius=0.3 + i*0.4, color=BLUE, stroke_opacity=0.5)
            circle.move_to(center)
            waves.add(circle)
            centers.append(center)
        
        self.play(Create(waves), run_time=2)
        
        # Labels
        compressed = Text("Compressed", color=GREEN).scale(0.3).shift(RIGHT*2 + UP)
        stretched = Text("Stretched", color=ORANGE).scale(0.3).shift(LEFT*2 + UP)
        
        self.play(Write(compressed), Write(stretched))
        
        # Move source
        self.play(source.animate.shift(RIGHT*4), source_label.animate.shift(RIGHT*4), run_time=2)
        
        explanation = Text("Higher pitch ahead, lower behind").scale(0.35).to_edge(DOWN)
        self.play(Write(explanation))
        self.wait(2)
'''

# Registry
PHYSICS_TEMPLATES = {
    "newtons_laws": NEWTONS_LAWS,
    "momentum": MOMENTUM_CONSERVATION,
    "energy": ENERGY_CONSERVATION,
    "friction": FRICTION_VISUAL,
    "pendulum": PENDULUM_MOTION,
    "circular_motion": CIRCULAR_MOTION,
    "electric_field": ELECTRIC_FIELD,
    "doppler": DOPPLER_EFFECT,
}

def get_physics_template(name: str) -> str:
    return PHYSICS_TEMPLATES.get(name, "")

def list_physics_templates() -> list:
    return list(PHYSICS_TEMPLATES.keys())
