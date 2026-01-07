"""
2D Templates - Extensive collection of 2D animation templates
"""

# =============================================================================
# CALCULUS TEMPLATES
# =============================================================================

DERIVATIVE_VISUALIZATION = '''
from manim import *

class DerivativeVisualization(Scene):
    """Visualize the derivative as slope of tangent line."""
    
    def construct(self):
        # Setup axes
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-2, 8, 2],
            x_length=8,
            y_length=5,
            axis_config={"color": GRAY_B}
        )
        labels = axes.get_axis_labels(x_label="x", y_label="y")
        
        # Function and its derivative
        func = lambda x: x**2
        deriv = lambda x: 2*x
        
        graph = axes.plot(func, color=WHITE)
        graph_label = MathTex("f(x) = x^2").next_to(graph, UR)
        
        # Value tracker for tangent point
        x_val = ValueTracker(-2)
        
        # Tangent line (always redrawn)
        def get_tangent():
            x = x_val.get_value()
            slope = deriv(x)
            point = axes.c2p(x, func(x))
            
            # Tangent line equation: y - y0 = m(x - x0)
            line = axes.plot(
                lambda t: slope * (t - x) + func(x),
                x_range=[x - 1.5, x + 1.5],
                color=GRAY_A
            )
            return line
        
        tangent = always_redraw(get_tangent)
        
        # Moving dot on curve
        dot = always_redraw(
            lambda: Dot(
                axes.c2p(x_val.get_value(), func(x_val.get_value())),
                color=WHITE
            )
        )
        
        # Slope display
        slope_text = always_redraw(
            lambda: MathTex(
                f"\\\\text{{slope}} = {deriv(x_val.get_value()):.1f}"
            ).to_corner(UR)
        )
        
        # Animation sequence
        self.play(Create(axes), Write(labels))
        self.play(Create(graph), Write(graph_label))
        self.wait()
        
        self.play(Create(tangent), Create(dot), Write(slope_text))
        self.wait()
        
        # Animate x value moving
        self.play(x_val.animate.set_value(2), run_time=4, rate_func=smooth)
        self.wait()
        
        # Highlight derivative formula
        deriv_formula = MathTex("f'(x) = 2x").to_corner(UL)
        self.play(Write(deriv_formula))
        self.wait(2)
'''

INTEGRAL_AREA = '''
from manim import *

class IntegralArea(Scene):
    """Visualize definite integral as area under curve."""
    
    def construct(self):
        axes = Axes(
            x_range=[-1, 5, 1],
            y_range=[-1, 5, 1],
            x_length=8,
            y_length=5,
            axis_config={"color": GRAY_B}
        )
        labels = axes.get_axis_labels()
        
        # Function
        func = lambda x: 0.25 * x**2
        graph = axes.plot(func, x_range=[0, 4], color=WHITE)
        
        # Riemann sum rectangles
        n_rects = ValueTracker(4)
        
        def get_riemann_rects():
            return axes.get_riemann_rectangles(
                graph,
                x_range=[0, 4],
                dx=4 / n_rects.get_value(),
                color=[GRAY_C, GRAY_B],
                fill_opacity=0.6,
                stroke_width=1,
                stroke_color=WHITE
            )
        
        rects = always_redraw(get_riemann_rects)
        
        # Show integral notation
        integral = MathTex(
            r"\\int_0^4 \\frac{x^2}{4} \\, dx"
        ).to_corner(UR)
        
        # Animation
        self.play(Create(axes), Write(labels))
        self.play(Create(graph))
        self.wait()
        
        self.play(Write(integral))
        self.play(Create(rects))
        self.wait()
        
        # Increase rectangles
        for n in [8, 16, 32]:
            self.play(n_rects.animate.set_value(n), run_time=1.5)
            self.wait(0.5)
        
        # Show result
        result = MathTex(r"= \\frac{16}{3}").next_to(integral, DOWN)
        self.play(Write(result))
        self.wait(2)
'''

LIMIT_CONCEPT = '''
from manim import *

class LimitConcept(Scene):
    """Visualize limits approaching a value."""
    
    def construct(self):
        axes = Axes(
            x_range=[-1, 5, 1],
            y_range=[-1, 4, 1],
            x_length=8,
            y_length=5,
            axis_config={"color": GRAY_B}
        )
        
        # Function with a hole at x=2
        def func(x):
            if abs(x - 2) < 0.01:
                return 3  # The limit value
            return (x**2 - 4) / (x - 2)  # = x + 2
        
        # Plot in two pieces to show the hole
        graph_left = axes.plot(func, x_range=[0, 1.95], color=WHITE)
        graph_right = axes.plot(func, x_range=[2.05, 4], color=WHITE)
        
        # Hollow circle at the hole
        hole = Circle(radius=0.08, color=WHITE, fill_opacity=0).move_to(
            axes.c2p(2, 4)
        )
        
        # Limit notation
        limit_tex = MathTex(
            r"\\lim_{x \\to 2} \\frac{x^2 - 4}{x - 2}"
        ).to_corner(UR)
        
        # Approaching arrows
        left_arrow = Arrow(
            axes.c2p(0.5, func(0.5)),
            axes.c2p(1.8, func(1.8)),
            color=GRAY_A,
            buff=0.1
        )
        right_arrow = Arrow(
            axes.c2p(3.5, func(3.5)),
            axes.c2p(2.2, func(2.2)),
            color=GRAY_A,
            buff=0.1
        )
        
        # Animation
        self.play(Create(axes))
        self.play(Create(graph_left), Create(graph_right))
        self.play(Create(hole))
        self.wait()
        
        self.play(Write(limit_tex))
        self.play(Create(left_arrow), Create(right_arrow))
        self.wait()
        
        # Show the limit value
        result = MathTex("= 4").next_to(limit_tex, DOWN)
        limit_point = Dot(axes.c2p(2, 4), color=WHITE, fill_opacity=0.5)
        
        self.play(Write(result), FadeIn(limit_point))
        self.wait(2)
'''

TAYLOR_SERIES = '''
from manim import *
import numpy as np

class TaylorSeries(Scene):
    """Show Taylor series approximation."""
    
    def construct(self):
        axes = Axes(
            x_range=[-4, 4, 1],
            y_range=[-2, 2, 1],
            x_length=10,
            y_length=5,
            axis_config={"color": GRAY_B}
        )
        
        # Original function: sin(x)
        sin_graph = axes.plot(np.sin, x_range=[-4, 4], color=WHITE)
        sin_label = MathTex(r"\\sin(x)").to_corner(UR)
        
        # Taylor approximations
        taylor_1 = lambda x: x
        taylor_3 = lambda x: x - x**3/6
        taylor_5 = lambda x: x - x**3/6 + x**5/120
        taylor_7 = lambda x: x - x**3/6 + x**5/120 - x**7/5040
        
        taylor_graphs = [
            (taylor_1, "x", GRAY_C),
            (taylor_3, r"x - \\frac{x^3}{6}", GRAY_B),
            (taylor_5, r"x - \\frac{x^3}{6} + \\frac{x^5}{120}", GRAY_A),
            (taylor_7, r"...", WHITE),
        ]
        
        # Animation
        self.play(Create(axes))
        self.play(Create(sin_graph), Write(sin_label))
        self.wait()
        
        current_graph = None
        for func, label, color in taylor_graphs:
            new_graph = axes.plot(func, x_range=[-3.5, 3.5], color=color)
            taylor_label = MathTex(label, color=color).next_to(sin_label, DOWN)
            
            if current_graph:
                self.play(
                    Transform(current_graph, new_graph),
                    FadeOut(prev_label),
                    Write(taylor_label)
                )
            else:
                current_graph = new_graph
                self.play(Create(current_graph), Write(taylor_label))
            
            prev_label = taylor_label
            self.wait()
        
        self.wait(2)
'''

# =============================================================================
# LINEAR ALGEBRA TEMPLATES
# =============================================================================

VECTOR_ADDITION = '''
from manim import *

class VectorAddition(Scene):
    """Visualize vector addition."""
    
    def construct(self):
        plane = NumberPlane(
            x_range=[-5, 5, 1],
            y_range=[-4, 4, 1],
            background_line_style={"stroke_color": GRAY_D, "stroke_opacity": 0.5}
        )
        
        # Vectors
        v1 = Arrow(ORIGIN, [2, 1, 0], buff=0, color=GRAY_B)
        v2 = Arrow(ORIGIN, [1, 2, 0], buff=0, color=GRAY_A)
        v_sum = Arrow(ORIGIN, [3, 3, 0], buff=0, color=WHITE)
        
        # Labels
        v1_label = MathTex(r"\\vec{a}", color=GRAY_B).next_to(v1.get_center(), DOWN)
        v2_label = MathTex(r"\\vec{b}", color=GRAY_A).next_to(v2.get_center(), LEFT)
        sum_label = MathTex(r"\\vec{a} + \\vec{b}", color=WHITE)
        
        # Animation
        self.play(Create(plane))
        self.play(GrowArrow(v1), Write(v1_label))
        self.play(GrowArrow(v2), Write(v2_label))
        self.wait()
        
        # Move v2 to tip of v1
        v2_copy = v2.copy()
        self.play(v2_copy.animate.shift([2, 1, 0]))
        self.wait()
        
        # Draw sum vector
        sum_label.next_to(v_sum.get_center(), RIGHT)
        self.play(GrowArrow(v_sum), Write(sum_label))
        self.wait(2)
'''

MATRIX_TRANSFORMATION = '''
from manim import *

class MatrixTransformation(Scene):
    """Visualize 2D linear transformation."""
    
    def construct(self):
        plane = NumberPlane(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            background_line_style={"stroke_color": GRAY_D, "stroke_opacity": 0.4}
        )
        
        # Transformation matrix (rotation + scale)
        matrix = [[1, 1], [0, 1]]  # Shear
        
        # Show matrix
        matrix_tex = MathTex(
            r"A = \\begin{bmatrix} 1 & 1 \\\\ 0 & 1 \\end{bmatrix}"
        ).to_corner(UR)
        
        # Unit vectors
        i_hat = Arrow(ORIGIN, [1, 0, 0], buff=0, color=GRAY_B)
        j_hat = Arrow(ORIGIN, [0, 1, 0], buff=0, color=GRAY_A)
        i_label = MathTex(r"\\hat{i}", color=GRAY_B).next_to(i_hat, DOWN)
        j_label = MathTex(r"\\hat{j}", color=GRAY_A).next_to(j_hat, LEFT)
        
        # Animation
        self.play(Create(plane))
        self.play(GrowArrow(i_hat), GrowArrow(j_hat))
        self.play(Write(i_label), Write(j_label))
        self.play(Write(matrix_tex))
        self.wait()
        
        # Apply transformation
        self.play(
            plane.animate.apply_matrix(matrix),
            i_hat.animate.put_start_and_end_on(ORIGIN, [1, 0, 0]).apply_matrix(matrix),
            j_hat.animate.put_start_and_end_on(ORIGIN, [0, 1, 0]).apply_matrix(matrix),
            run_time=2
        )
        self.wait(2)
'''

EIGENVALUE_VISUALIZATION = '''
from manim import *
import numpy as np

class EigenvalueVisualization(Scene):
    """Visualize eigenvectors and eigenvalues."""
    
    def construct(self):
        plane = NumberPlane(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            background_line_style={"stroke_color": GRAY_D, "stroke_opacity": 0.4}
        )
        
        # Matrix with real eigenvalues
        matrix = [[2, 1], [1, 2]]  # eigenvalues: 3, 1
        
        # Eigenvectors
        ev1 = [1/np.sqrt(2), 1/np.sqrt(2)]  # eigenvalue 3
        ev2 = [-1/np.sqrt(2), 1/np.sqrt(2)]  # eigenvalue 1
        
        # Draw eigenvector lines
        eigen_line1 = Line(
            [-3*ev1[0], -3*ev1[1], 0],
            [3*ev1[0], 3*ev1[1], 0],
            color=GRAY_B
        )
        eigen_line2 = Line(
            [-3*ev2[0], -3*ev2[1], 0],
            [3*ev2[0], 3*ev2[1], 0],
            color=GRAY_A
        )
        
        # Arrows on eigenvector directions
        ev1_arrow = Arrow(ORIGIN, [ev1[0], ev1[1], 0], buff=0, color=WHITE)
        ev2_arrow = Arrow(ORIGIN, [ev2[0], ev2[1], 0], buff=0, color=WHITE)
        
        # Labels
        eigen_label = MathTex(
            r"\\lambda_1 = 3, \\lambda_2 = 1"
        ).to_corner(UR)
        
        # Animation
        self.play(Create(plane))
        self.play(Create(eigen_line1), Create(eigen_line2))
        self.play(GrowArrow(ev1_arrow), GrowArrow(ev2_arrow))
        self.play(Write(eigen_label))
        self.wait()
        
        # Apply transformation - eigenvectors only scale
        self.play(
            plane.animate.apply_matrix(matrix),
            ev1_arrow.animate.put_start_and_end_on(
                ORIGIN, [3*ev1[0], 3*ev1[1], 0]
            ),
            ev2_arrow.animate.put_start_and_end_on(
                ORIGIN, [1*ev2[0], 1*ev2[1], 0]
            ),
            run_time=2
        )
        self.wait(2)
'''

# =============================================================================
# GEOMETRY TEMPLATES
# =============================================================================

PYTHAGOREAN_THEOREM = '''
from manim import *

class PythagoreanTheorem(Scene):
    """Visual proof of Pythagorean theorem."""
    
    def construct(self):
        # Right triangle
        a, b = 2, 1.5
        c = (a**2 + b**2)**0.5
        
        triangle = Polygon(
            ORIGIN, [a, 0, 0], [a, b, 0],
            color=WHITE,
            fill_opacity=0.2
        ).shift(LEFT * 2)
        
        # Side labels
        a_label = MathTex("a").next_to(triangle, DOWN)
        b_label = MathTex("b").next_to(triangle, RIGHT)
        c_label = MathTex("c").move_to(triangle.get_center() + UP*0.3 + LEFT*0.3)
        
        # Squares on each side
        sq_a = Square(side_length=a, color=GRAY_B, fill_opacity=0.3)
        sq_a.next_to(triangle, DOWN, buff=0)
        
        sq_b = Square(side_length=b, color=GRAY_A, fill_opacity=0.3)
        sq_b.next_to(triangle, RIGHT, buff=0)
        
        # Animation
        self.play(Create(triangle))
        self.play(Write(a_label), Write(b_label), Write(c_label))
        self.wait()
        
        self.play(Create(sq_a), Create(sq_b))
        
        # Area labels
        area_a = MathTex("a^2", color=GRAY_B).move_to(sq_a)
        area_b = MathTex("b^2", color=GRAY_A).move_to(sq_b)
        self.play(Write(area_a), Write(area_b))
        self.wait()
        
        # The theorem
        theorem = MathTex("a^2 + b^2 = c^2").to_edge(UP)
        self.play(Write(theorem))
        self.wait(2)
'''

CIRCLE_THEOREMS = '''
from manim import *
import numpy as np

class CircleTheorems(Scene):
    """Inscribed angle theorem."""
    
    def construct(self):
        circle = Circle(radius=2, color=WHITE)
        center = Dot(ORIGIN, color=GRAY_A)
        center_label = MathTex("O").next_to(center, DOWN, buff=0.1)
        
        # Points on circle
        angle1, angle2, angle3 = 0.5, 2.5, 4.5
        A = 2 * np.array([np.cos(angle1), np.sin(angle1), 0])
        B = 2 * np.array([np.cos(angle2), np.sin(angle2), 0])
        C = 2 * np.array([np.cos(angle3), np.sin(angle3), 0])
        
        points = [Dot(p, color=WHITE) for p in [A, B, C]]
        labels = [
            MathTex("A").next_to(A, A/2, buff=0.1),
            MathTex("B").next_to(B, B/2, buff=0.1),
            MathTex("C").next_to(C, C/2, buff=0.1),
        ]
        
        # Inscribed angle (from C)
        inscribed = Polygon(A, C, B, color=GRAY_B, fill_opacity=0)
        
        # Central angle (from O)
        central = Polygon(A, ORIGIN, B, color=GRAY_A, fill_opacity=0)
        
        # Angle arcs
        inscribed_arc = Angle(
            Line(C, A), Line(C, B),
            radius=0.4, color=GRAY_B
        )
        central_arc = Angle(
            Line(ORIGIN, A), Line(ORIGIN, B),
            radius=0.5, color=GRAY_A
        )
        
        # Animation
        self.play(Create(circle), Create(center), Write(center_label))
        self.play(*[Create(p) for p in points], *[Write(l) for l in labels])
        self.wait()
        
        self.play(Create(inscribed), Create(inscribed_arc))
        theta = MathTex(r"\\theta", color=GRAY_B).next_to(inscribed_arc, DOWN, buff=0.1)
        self.play(Write(theta))
        self.wait()
        
        self.play(Create(central), Create(central_arc))
        two_theta = MathTex(r"2\\theta", color=GRAY_A).next_to(central_arc, LEFT, buff=0.1)
        self.play(Write(two_theta))
        self.wait()
        
        # Theorem statement
        theorem = MathTex(
            r"\\text{Inscribed angle} = \\frac{1}{2} \\times \\text{Central angle}"
        ).to_edge(UP)
        self.play(Write(theorem))
        self.wait(2)
'''

CONIC_SECTIONS = '''
from manim import *
import numpy as np

class ConicSections(Scene):
    """Ellipse, parabola, hyperbola."""
    
    def construct(self):
        title = Text("Conic Sections", font_size=36).to_edge(UP)
        self.play(Write(title))
        
        # Ellipse
        ellipse = Ellipse(width=3, height=2, color=WHITE)
        ellipse_label = MathTex(r"\\frac{x^2}{a^2} + \\frac{y^2}{b^2} = 1")
        ellipse_label.scale(0.6).next_to(ellipse, DOWN)
        ellipse_group = VGroup(ellipse, ellipse_label).shift(LEFT * 4)
        
        # Parabola
        parabola = axes_parabola = Axes(
            x_range=[-2, 2, 1], y_range=[0, 4, 1],
            x_length=2, y_length=2.5,
            axis_config={"include_ticks": False, "include_tip": False}
        ).shift(ORIGIN)
        parabola_curve = parabola.plot(lambda x: x**2, color=WHITE)
        parabola_label = MathTex("y = x^2").scale(0.6)
        parabola_label.next_to(parabola, DOWN)
        parabola_group = VGroup(parabola, parabola_curve, parabola_label)
        
        # Hyperbola (parametric)
        t_range = np.linspace(-1.5, 1.5, 100)
        hyperbola_right = [
            [np.cosh(t), np.sinh(t), 0] for t in t_range
        ]
        hyperbola_left = [
            [-np.cosh(t), np.sinh(t), 0] for t in t_range
        ]
        
        hyp_r = VMobject(color=WHITE)
        hyp_r.set_points_smoothly([np.array(p) for p in hyperbola_right])
        hyp_l = VMobject(color=WHITE)
        hyp_l.set_points_smoothly([np.array(p) for p in hyperbola_left])
        hyperbola = VGroup(hyp_r, hyp_l).scale(0.8).shift(RIGHT * 4)
        hyperbola_label = MathTex(r"\\frac{x^2}{a^2} - \\frac{y^2}{b^2} = 1")
        hyperbola_label.scale(0.6).next_to(hyperbola, DOWN)
        hyperbola_group = VGroup(hyperbola, hyperbola_label)
        
        # Animation
        self.play(Create(ellipse_group))
        self.wait()
        self.play(Create(parabola_group))
        self.wait()
        self.play(Create(hyperbola_group))
        self.wait(2)
'''

# =============================================================================
# PHYSICS TEMPLATES (2D)
# =============================================================================

PROJECTILE_MOTION = '''
from manim import *
import numpy as np

class ProjectileMotion(Scene):
    """2D projectile motion animation."""
    
    def construct(self):
        # Parameters
        v0 = 5
        angle = PI / 4
        g = 9.8
        
        # Time of flight
        t_max = 2 * v0 * np.sin(angle) / g
        
        # Trajectory function
        def trajectory(t):
            x = v0 * np.cos(angle) * t
            y = v0 * np.sin(angle) * t - 0.5 * g * t**2
            return np.array([x, y, 0])
        
        # Setup
        axes = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 2, 0.5],
            x_length=8,
            y_length=4,
            axis_config={"color": GRAY_B}
        )
        labels = axes.get_axis_labels(x_label="x", y_label="y")
        
        # Scale trajectory to axes
        scale_x = 8 / (v0 * np.cos(angle) * t_max)
        scale_y = 4 / (v0**2 * np.sin(angle)**2 / (2 * g))
        
        # Path
        path = VMobject(color=GRAY_A)
        points = [axes.c2p(
            v0 * np.cos(angle) * t,
            v0 * np.sin(angle) * t - 0.5 * g * t**2
        ) for t in np.linspace(0, t_max, 50)]
        path.set_points_smoothly(points)
        
        # Ball
        ball = Dot(axes.c2p(0, 0), color=WHITE, radius=0.1)
        
        # Velocity vector (changes over time)
        t_tracker = ValueTracker(0)
        
        def get_velocity_arrow():
            t = t_tracker.get_value()
            pos = axes.c2p(
                v0 * np.cos(angle) * t,
                max(0, v0 * np.sin(angle) * t - 0.5 * g * t**2)
            )
            vx = v0 * np.cos(angle) * 0.15
            vy = (v0 * np.sin(angle) - g * t) * 0.15
            return Arrow(
                pos, 
                pos + np.array([vx, vy, 0]),
                buff=0, color=GRAY_B
            )
        
        velocity = always_redraw(get_velocity_arrow)
        
        # Equations
        equations = VGroup(
            MathTex(r"x = v_0 \\cos(\\theta) t"),
            MathTex(r"y = v_0 \\sin(\\theta) t - \\frac{1}{2}gt^2"),
        ).arrange(DOWN, aligned_edge=LEFT).scale(0.7).to_corner(UR)
        
        # Animation
        self.play(Create(axes), Write(labels))
        self.play(Write(equations))
        self.play(Create(ball), Create(velocity))
        self.wait()
        
        # Animate projectile
        self.play(
            MoveAlongPath(ball, path),
            t_tracker.animate.set_value(t_max),
            Create(path),
            run_time=3,
            rate_func=linear
        )
        self.wait(2)
'''

WAVE_VISUALIZATION = '''
from manim import *
import numpy as np

class WaveVisualization(Scene):
    """Traveling wave animation."""
    
    def construct(self):
        axes = Axes(
            x_range=[0, 4*PI, PI],
            y_range=[-2, 2, 1],
            x_length=10,
            y_length=4,
            axis_config={"color": GRAY_B}
        )
        labels = axes.get_axis_labels(x_label="x", y_label="y")
        
        # Wave parameters
        A = 1.5  # Amplitude
        k = 1    # Wave number
        omega = 2  # Angular frequency
        
        t = ValueTracker(0)
        
        # Wave equation: y = A sin(kx - ωt)
        def get_wave():
            return axes.plot(
                lambda x: A * np.sin(k * x - omega * t.get_value()),
                color=WHITE
            )
        
        wave = always_redraw(get_wave)
        
        # Wave equation display
        equation = MathTex(
            r"y = A \\sin(kx - \\omega t)"
        ).to_corner(UR)
        
        # Wavelength indicator
        wavelength_line = always_redraw(lambda: 
            Line(
                axes.c2p(omega * t.get_value() / k, -1.8),
                axes.c2p(omega * t.get_value() / k + 2*PI, -1.8),
                color=GRAY_A
            )
        )
        lambda_label = MathTex(r"\\lambda", color=GRAY_A).next_to(wavelength_line, DOWN)
        
        # Animation
        self.play(Create(axes), Write(labels))
        self.play(Create(wave), Write(equation))
        self.play(Create(wavelength_line))
        self.wait()
        
        # Animate wave motion
        self.play(
            t.animate.set_value(4 * PI),
            run_time=4,
            rate_func=linear
        )
        self.wait()
'''

SIMPLE_HARMONIC_MOTION = '''
from manim import *
import numpy as np

class SimpleHarmonicMotion(Scene):
    """Mass-spring system visualization."""
    
    def construct(self):
        # Wall and spring anchor
        wall = Line(UP * 2, DOWN * 2, color=GRAY_B).shift(LEFT * 4)
        
        # Time tracker
        t = ValueTracker(0)
        omega = 2
        A = 1.5
        
        # Mass position
        def get_mass_pos():
            return A * np.cos(omega * t.get_value())
        
        # Spring (simplified as zigzag)
        def get_spring():
            x = get_mass_pos()
            # Create zigzag
            n_coils = 10
            points = [np.array([-4, 0, 0])]
            for i in range(n_coils):
                progress = (i + 1) / (n_coils + 1)
                x_pos = -4 + progress * (x + 4)
                y_offset = 0.3 * (1 if i % 2 == 0 else -1)
                points.append(np.array([x_pos, y_offset, 0]))
            points.append(np.array([x, 0, 0]))
            
            spring = VMobject(color=GRAY_A)
            spring.set_points_as_corners(points)
            return spring
        
        spring = always_redraw(get_spring)
        
        # Mass
        mass = always_redraw(lambda:
            Square(side_length=0.6, color=WHITE, fill_opacity=0.5).move_to(
                [get_mass_pos(), 0, 0]
            )
        )
        
        # Position graph
        graph_axes = Axes(
            x_range=[0, 4*PI, PI],
            y_range=[-2, 2, 1],
            x_length=6,
            y_length=2,
            axis_config={"color": GRAY_B}
        ).shift(DOWN * 2.5 + RIGHT)
        
        # Trace position
        trace = always_redraw(lambda:
            graph_axes.plot(
                lambda x: A * np.cos(omega * x),
                x_range=[0, min(t.get_value(), 4*PI)],
                color=WHITE
            )
        )
        
        # Equation
        equation = MathTex(r"x(t) = A\\cos(\\omega t)").to_corner(UR)
        
        # Animation
        self.play(Create(wall), Create(spring), Create(mass))
        self.play(Create(graph_axes), Write(equation))
        self.add(trace)
        self.wait()
        
        self.play(
            t.animate.set_value(4 * PI),
            run_time=6,
            rate_func=linear
        )
        self.wait()
'''

# =============================================================================
# TEMPLATE REGISTRY
# =============================================================================

TEMPLATES_2D = {
    # Calculus
    "derivative": DERIVATIVE_VISUALIZATION,
    "integral": INTEGRAL_AREA,
    "limit": LIMIT_CONCEPT,
    "taylor": TAYLOR_SERIES,
    
    # Linear Algebra
    "vector_addition": VECTOR_ADDITION,
    "matrix_transform": MATRIX_TRANSFORMATION,
    "eigenvalue": EIGENVALUE_VISUALIZATION,
    
    # Geometry
    "pythagorean": PYTHAGOREAN_THEOREM,
    "circle_theorem": CIRCLE_THEOREMS,
    "conic": CONIC_SECTIONS,
    
    # Physics (basic)
    "projectile": PROJECTILE_MOTION,
    "wave": WAVE_VISUALIZATION,
    "harmonic": SIMPLE_HARMONIC_MOTION,
}

# Import and merge templates from specialized modules
try:
    from .templates_calculus import CALCULUS_TEMPLATES
    TEMPLATES_2D.update(CALCULUS_TEMPLATES)
except ImportError:
    pass

try:
    from .templates_algebra import ALGEBRA_TEMPLATES
    TEMPLATES_2D.update(ALGEBRA_TEMPLATES)
except ImportError:
    pass

try:
    from .templates_trig import TRIG_TEMPLATES
    TEMPLATES_2D.update(TRIG_TEMPLATES)
except ImportError:
    pass

try:
    from .templates_physics import PHYSICS_TEMPLATES
    TEMPLATES_2D.update(PHYSICS_TEMPLATES)
except ImportError:
    pass

try:
    from .templates_stats import STATS_TEMPLATES
    TEMPLATES_2D.update(STATS_TEMPLATES)
except ImportError:
    pass

try:
    from .templates_special import SPECIAL_TEMPLATES
    TEMPLATES_2D.update(SPECIAL_TEMPLATES)
except ImportError:
    pass

# Placeholder for templates that need to be added


def get_2d_template(name: str) -> str:
    """Get a 2D template by name."""
    return TEMPLATES_2D.get(name, "")


def list_2d_templates() -> list[str]:
    """List all available 2D templates."""
    return list(TEMPLATES_2D.keys())
