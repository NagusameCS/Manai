"""
Example Manim Scenes - Reference implementations for common visualizations
These examples demonstrate proper Manim code that can be used as templates.
"""

from manim import *
import numpy as np


# =============================================================================
# CALCULUS EXAMPLES
# =============================================================================

class DerivativeVisualization(Scene):
    """Visualize the derivative of a function with tangent lines."""
    
    def construct(self):
        # Title
        title = Title("Understanding Derivatives")
        self.play(Write(title))
        self.wait()
        
        # Create axes
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-1, 9, 2],
            x_length=8,
            y_length=5,
            axis_config={"include_tip": True, "include_numbers": True},
        )
        axes_labels = axes.get_axis_labels(x_label="x", y_label="y")
        
        # Function: f(x) = x^2
        func = axes.plot(lambda x: x**2, color=BLUE, x_range=[-2.5, 2.5])
        func_label = MathTex("f(x) = x^2", color=BLUE).to_corner(UR)
        
        self.play(
            FadeOut(title),
            Create(axes),
            Write(axes_labels),
            run_time=2
        )
        self.play(Create(func), Write(func_label))
        self.wait()
        
        # Tangent line tracker
        x_tracker = ValueTracker(-2)
        
        def get_tangent_line():
            x = x_tracker.get_value()
            slope = 2 * x  # derivative of x^2
            y = x**2
            
            # Create tangent line
            line = axes.plot(
                lambda t: slope * (t - x) + y,
                x_range=[x - 1.5, x + 1.5],
                color=YELLOW
            )
            return line
        
        def get_point():
            x = x_tracker.get_value()
            return Dot(axes.c2p(x, x**2), color=RED)
        
        def get_slope_label():
            x = x_tracker.get_value()
            slope = 2 * x
            label = MathTex(f"\\text{{slope}} = {slope:.1f}", color=YELLOW)
            label.to_corner(UL)
            return label
        
        tangent = always_redraw(get_tangent_line)
        point = always_redraw(get_point)
        slope_label = always_redraw(get_slope_label)
        
        self.play(Create(tangent), Create(point), Write(slope_label))
        
        # Animate tangent line moving along curve
        self.play(x_tracker.animate.set_value(2), run_time=6, rate_func=smooth)
        self.wait()
        
        # Show derivative function
        derivative_text = MathTex("f'(x) = 2x", color=RED).next_to(func_label, DOWN)
        self.play(Write(derivative_text))
        self.wait(2)


class IntegralAreaVisualization(Scene):
    """Visualize the integral as area under a curve with Riemann sums."""
    
    def construct(self):
        # Create axes
        axes = Axes(
            x_range=[0, 4, 1],
            y_range=[0, 10, 2],
            x_length=10,
            y_length=5,
            axis_config={"include_numbers": True},
        )
        labels = axes.get_axis_labels(x_label="x", y_label="y")
        
        # Function
        func = axes.plot(lambda x: 0.5 * x**2 + 1, color=BLUE, x_range=[0, 4])
        func_label = MathTex("f(x) = \\frac{1}{2}x^2 + 1", color=BLUE).to_corner(UR)
        
        self.play(Create(axes), Write(labels))
        self.play(Create(func), Write(func_label))
        self.wait()
        
        # Show Riemann sums with increasing rectangles
        for n in [4, 8, 16, 32]:
            rects = axes.get_riemann_rectangles(
                func,
                x_range=[1, 3],
                dx=2/n,
                color=YELLOW,
                fill_opacity=0.5,
                stroke_width=1,
            )
            
            if n == 4:
                self.play(Create(rects))
                n_label = MathTex(f"n = {n}").to_corner(UL)
                self.play(Write(n_label))
            else:
                new_label = MathTex(f"n = {n}").to_corner(UL)
                self.play(
                    Transform(rects, rects),
                    Transform(n_label, new_label)
                )
            
            self.wait()
        
        # Show exact area
        area = axes.get_area(func, x_range=[1, 3], color=GREEN, opacity=0.5)
        
        integral = MathTex(
            "\\int_1^3 \\left(\\frac{1}{2}x^2 + 1\\right)dx = \\frac{16}{3}",
            color=GREEN
        ).to_edge(DOWN)
        
        self.play(
            FadeOut(rects),
            FadeIn(area),
            Write(integral)
        )
        self.wait(2)


class LimitConcept(Scene):
    """Visualize the concept of limits."""
    
    def construct(self):
        # Title
        title = Tex("The Concept of Limits").scale(1.2)
        self.play(Write(title))
        self.wait()
        self.play(title.animate.to_edge(UP))
        
        # Create axes
        axes = Axes(
            x_range=[-1, 5, 1],
            y_range=[-1, 5, 1],
            x_length=8,
            y_length=5,
        )
        
        # Function with a hole at x=2
        func_left = axes.plot(lambda x: x + 1, x_range=[-0.5, 1.95], color=BLUE)
        func_right = axes.plot(lambda x: x + 1, x_range=[2.05, 4.5], color=BLUE)
        hole = Dot(axes.c2p(2, 3), color=BLUE, fill_opacity=0)
        hole.set_stroke(BLUE, width=2)
        
        func_label = MathTex("f(x) = x + 1, x \\neq 2").to_corner(UR)
        
        self.play(Create(axes))
        self.play(Create(func_left), Create(func_right), Create(hole))
        self.play(Write(func_label))
        self.wait()
        
        # Approaching from left
        left_tracker = ValueTracker(0.5)
        left_dot = always_redraw(
            lambda: Dot(
                axes.c2p(left_tracker.get_value(), left_tracker.get_value() + 1),
                color=RED
            )
        )
        left_arrow = always_redraw(
            lambda: Arrow(
                axes.c2p(left_tracker.get_value() - 0.3, 0),
                axes.c2p(left_tracker.get_value(), 0),
                color=RED
            )
        )
        
        self.play(Create(left_dot), Create(left_arrow))
        self.play(left_tracker.animate.set_value(1.9), run_time=2)
        
        # Approaching from right
        right_tracker = ValueTracker(4)
        right_dot = always_redraw(
            lambda: Dot(
                axes.c2p(right_tracker.get_value(), right_tracker.get_value() + 1),
                color=GREEN
            )
        )
        right_arrow = always_redraw(
            lambda: Arrow(
                axes.c2p(right_tracker.get_value() + 0.3, 0),
                axes.c2p(right_tracker.get_value(), 0),
                color=GREEN
            )
        )
        
        self.play(Create(right_dot), Create(right_arrow))
        self.play(right_tracker.animate.set_value(2.1), run_time=2)
        
        # Show limit
        limit_eq = MathTex(
            "\\lim_{x \\to 2} f(x) = 3",
            color=YELLOW
        ).to_edge(DOWN)
        
        self.play(Write(limit_eq))
        self.wait(2)


# =============================================================================
# LINEAR ALGEBRA EXAMPLES
# =============================================================================

class LinearTransformation2D(Scene):
    """Visualize a 2D linear transformation."""
    
    def construct(self):
        # Create plane
        plane = NumberPlane(
            x_range=[-5, 5, 1],
            y_range=[-5, 5, 1],
            background_line_style={
                "stroke_color": BLUE_D,
                "stroke_width": 1,
                "stroke_opacity": 0.5
            }
        )
        
        # Basis vectors
        i_hat = Arrow(ORIGIN, RIGHT, color=GREEN, buff=0)
        j_hat = Arrow(ORIGIN, UP, color=RED, buff=0)
        i_label = MathTex("\\hat{i}", color=GREEN).next_to(i_hat, DOWN)
        j_label = MathTex("\\hat{j}", color=RED).next_to(j_hat, LEFT)
        
        basis = VGroup(i_hat, j_hat, i_label, j_label)
        
        # Matrix
        matrix = [[2, 1], [1, 2]]
        matrix_tex = MathTex(
            "A = \\begin{bmatrix} 2 & 1 \\\\ 1 & 2 \\end{bmatrix}"
        ).to_corner(UL).set_color(YELLOW)
        
        self.play(Create(plane))
        self.play(Create(i_hat), Create(j_hat))
        self.play(Write(i_label), Write(j_label))
        self.play(Write(matrix_tex))
        self.wait()
        
        # Apply transformation
        self.play(
            plane.animate.apply_matrix(matrix),
            i_hat.animate.put_start_and_end_on(ORIGIN, 2*RIGHT + UP),
            j_hat.animate.put_start_and_end_on(ORIGIN, RIGHT + 2*UP),
            run_time=3
        )
        
        # Update labels
        new_i_label = MathTex("A\\hat{i}", color=GREEN).next_to(2*RIGHT + UP, DOWN)
        new_j_label = MathTex("A\\hat{j}", color=RED).next_to(RIGHT + 2*UP, LEFT)
        
        self.play(
            Transform(i_label, new_i_label),
            Transform(j_label, new_j_label)
        )
        self.wait(2)


class EigenvectorVisualization(Scene):
    """Visualize eigenvectors and how they behave under transformation."""
    
    def construct(self):
        # Title
        title = Title("Eigenvectors and Eigenvalues")
        self.play(Write(title))
        self.wait()
        self.play(FadeOut(title))
        
        # Create plane
        plane = NumberPlane(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
        )
        
        # Matrix: [[3, 1], [0, 2]]
        # Eigenvalues: 3, 2
        # Eigenvectors: [1, 0], [1, -1]
        
        matrix_tex = MathTex(
            "A = \\begin{bmatrix} 3 & 1 \\\\ 0 & 2 \\end{bmatrix}"
        ).to_corner(UL).set_color(YELLOW)
        
        # Eigenvectors
        ev1 = Arrow(ORIGIN, 2*RIGHT, color=GREEN, buff=0, stroke_width=4)
        ev2 = Arrow(ORIGIN, RIGHT - UP, color=RED, buff=0, stroke_width=4)
        
        ev1_label = MathTex("\\vec{v}_1", color=GREEN).next_to(ev1, UP)
        ev2_label = MathTex("\\vec{v}_2", color=RED).next_to(ev2, DOWN)
        
        # Other vectors
        other_vec = Arrow(ORIGIN, UP + RIGHT, color=BLUE, buff=0)
        
        self.play(Create(plane))
        self.play(Write(matrix_tex))
        self.play(
            Create(ev1), Write(ev1_label),
            Create(ev2), Write(ev2_label),
            Create(other_vec)
        )
        self.wait()
        
        # Eigenvalue info
        eigen_info = VGroup(
            MathTex("\\lambda_1 = 3", color=GREEN),
            MathTex("\\lambda_2 = 2", color=RED),
        ).arrange(DOWN).to_corner(UR)
        
        self.play(Write(eigen_info))
        
        # Apply transformation
        matrix = [[3, 1], [0, 2]]
        
        self.play(
            plane.animate.apply_matrix(matrix),
            ev1.animate.put_start_and_end_on(ORIGIN, 6*RIGHT),  # scaled by 3
            ev2.animate.put_start_and_end_on(ORIGIN, 2*(RIGHT - UP)),  # scaled by 2
            other_vec.animate.put_start_and_end_on(ORIGIN, 3*RIGHT + 2*UP + RIGHT),  # rotates
            run_time=3
        )
        
        # Explanation
        explanation = Tex(
            "Eigenvectors only scale, they don't rotate!",
            color=YELLOW
        ).to_edge(DOWN)
        
        self.play(Write(explanation))
        self.wait(2)


# =============================================================================
# PHYSICS EXAMPLES
# =============================================================================

class ProjectileMotion(Scene):
    """Simulate and visualize projectile motion."""
    
    def construct(self):
        # Title
        title = Title("Projectile Motion")
        self.play(Write(title))
        self.wait()
        self.play(title.animate.scale(0.5).to_corner(UL))
        
        # Ground and axes
        ground = Line(LEFT * 6, RIGHT * 6, color=GREEN).shift(DOWN * 2.5)
        
        # Parameters
        v0 = 5  # initial velocity
        theta = PI / 4  # 45 degrees
        g = 10  # gravity
        
        # Time of flight
        t_max = 2 * v0 * np.sin(theta) / g
        
        # Create trajectory
        trajectory = ParametricFunction(
            lambda t: np.array([
                v0 * np.cos(theta) * t - 5,
                v0 * np.sin(theta) * t - 0.5 * g * t**2 + ground.get_center()[1] + 0.5,
                0
            ]),
            t_range=[0, t_max],
            color=YELLOW
        )
        
        # Ball
        ball = Dot(color=RED).move_to(trajectory.get_start())
        
        # Equations
        equations = VGroup(
            MathTex("x(t) = v_0 \\cos(\\theta) \\cdot t"),
            MathTex("y(t) = v_0 \\sin(\\theta) \\cdot t - \\frac{1}{2}gt^2"),
        ).arrange(DOWN).scale(0.6).to_corner(UR)
        
        self.play(Create(ground))
        self.play(Write(equations))
        self.play(Create(ball))
        
        # Animate the ball along trajectory
        self.play(
            MoveAlongPath(ball, trajectory),
            Create(trajectory),
            run_time=3
        )
        
        # Show max height
        max_height = v0**2 * np.sin(theta)**2 / (2 * g)
        max_height_line = DashedLine(
            start=LEFT * 5 + UP * (max_height + ground.get_center()[1] - 2),
            end=RIGHT * 5 + UP * (max_height + ground.get_center()[1] - 2),
            color=BLUE
        )
        max_height_label = MathTex(
            f"h_{{max}} = \\frac{{v_0^2 \\sin^2\\theta}}{{2g}}"
        ).scale(0.6).next_to(max_height_line, UP)
        
        self.play(Create(max_height_line), Write(max_height_label))
        self.wait(2)


class SimpleHarmonicMotion(Scene):
    """Visualize simple harmonic motion with a spring-mass system."""
    
    def construct(self):
        # Title
        title = Title("Simple Harmonic Motion")
        self.play(Write(title))
        self.wait()
        
        # Spring anchor
        wall = Rectangle(width=0.3, height=2, color=GRAY).to_edge(LEFT).shift(RIGHT)
        
        # Mass
        mass = Square(side_length=0.6, color=BLUE, fill_opacity=1)
        
        # Time tracker
        time = ValueTracker(0)
        
        # Amplitude and frequency
        A = 2  # amplitude
        omega = 2  # angular frequency
        
        # Spring (simplified as zigzag line)
        def get_spring():
            t = time.get_value()
            x = A * np.cos(omega * t)
            mass.move_to(RIGHT * x)
            
            # Create spring points
            start = wall.get_right()
            end = mass.get_left()
            
            spring = VMobject(color=WHITE)
            n_coils = 8
            points = [start]
            
            for i in range(1, n_coils * 2 + 1):
                frac = i / (n_coils * 2 + 1)
                point = start + (end - start) * frac
                offset = UP * 0.2 * ((-1) ** i)
                if i == 1 or i == n_coils * 2:
                    offset *= 0
                points.append(point + offset)
            
            points.append(end)
            spring.set_points_as_corners(points)
            
            return spring
        
        spring = always_redraw(get_spring)
        
        # Position graph
        axes = Axes(
            x_range=[0, 8, 2],
            y_range=[-3, 3, 1],
            x_length=5,
            y_length=2,
        ).to_edge(DOWN)
        axes_labels = axes.get_axis_labels(x_label="t", y_label="x")
        
        # Traced curve
        def get_curve():
            t = time.get_value()
            curve = axes.plot(
                lambda s: A * np.cos(omega * s),
                x_range=[0, min(t, 8)],
                color=YELLOW
            )
            return curve
        
        curve = always_redraw(get_curve)
        
        # Equation
        equation = MathTex("x(t) = A\\cos(\\omega t)").to_corner(UR)
        
        self.play(FadeOut(title))
        self.play(Create(wall), Create(spring), Create(mass))
        self.play(Create(axes), Write(axes_labels), Write(equation))
        self.add(curve)
        
        # Animate
        self.play(time.animate.set_value(8), run_time=8, rate_func=linear)
        self.wait(2)


class ElectricFieldLines(Scene):
    """Visualize electric field lines from point charges."""
    
    def construct(self):
        # Title
        title = Title("Electric Field Lines")
        self.play(Write(title))
        self.wait()
        self.play(title.animate.scale(0.6).to_corner(UL))
        
        # Positive charge
        positive = VGroup(
            Circle(radius=0.3, color=RED, fill_opacity=1),
            MathTex("+", color=WHITE)
        ).move_to(LEFT * 2)
        
        # Negative charge
        negative = VGroup(
            Circle(radius=0.3, color=BLUE, fill_opacity=1),
            MathTex("-", color=WHITE)
        ).move_to(RIGHT * 2)
        
        self.play(Create(positive), Create(negative))
        
        # Field lines from positive to negative
        field_lines = VGroup()
        n_lines = 8
        
        for i in range(n_lines):
            angle = i * 2 * PI / n_lines
            
            # Start from positive charge
            start = positive.get_center() + 0.35 * np.array([np.cos(angle), np.sin(angle), 0])
            
            # End at negative charge (opposite angle)
            end_angle = PI - angle
            end = negative.get_center() + 0.35 * np.array([np.cos(end_angle), np.sin(end_angle), 0])
            
            # Create curved arrow
            line = CurvedArrow(
                start, end,
                color=YELLOW,
                angle=0.3 * np.sin(angle)
            )
            field_lines.add(line)
        
        self.play(Create(field_lines), run_time=2)
        
        # Equation
        equation = MathTex(
            "\\vec{E} = \\frac{1}{4\\pi\\epsilon_0} \\frac{q}{r^2} \\hat{r}"
        ).to_edge(DOWN)
        
        self.play(Write(equation))
        self.wait(2)


# =============================================================================
# 3D EXAMPLES
# =============================================================================

class Surface3DPlot(ThreeDScene):
    """Create a 3D surface plot."""
    
    def construct(self):
        # Set up camera
        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)
        
        # Create 3D axes
        axes = ThreeDAxes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            z_range=[-1, 5, 1],
            x_length=6,
            y_length=6,
            z_length=4,
        )
        
        # Surface z = x^2 + y^2
        surface = Surface(
            lambda u, v: axes.c2p(u, v, u**2 + v**2),
            u_range=[-2, 2],
            v_range=[-2, 2],
            resolution=(30, 30),
            fill_opacity=0.7,
            checkerboard_colors=[BLUE_D, BLUE_E],
        )
        
        # Labels
        labels = axes.get_axis_labels(
            x_label=MathTex("x"),
            y_label=MathTex("y"),
            z_label=MathTex("z")
        )
        
        equation = MathTex("z = x^2 + y^2").to_corner(UL)
        self.add_fixed_in_frame_mobjects(equation)
        
        self.play(Create(axes), Write(labels))
        self.play(Create(surface), run_time=2)
        self.play(Write(equation))
        
        # Rotate camera
        self.begin_ambient_camera_rotation(rate=0.2)
        self.wait(8)
        self.stop_ambient_camera_rotation()


class Vector3DVisualization(ThreeDScene):
    """Visualize 3D vectors and operations."""
    
    def construct(self):
        # Camera setup
        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)
        
        # 3D axes
        axes = ThreeDAxes(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            z_range=[-4, 4, 1],
            x_length=6,
            y_length=6,
            z_length=6,
        )
        
        # Vectors
        vec_a = Arrow3D(
            start=axes.c2p(0, 0, 0),
            end=axes.c2p(2, 1, 2),
            color=RED
        )
        vec_b = Arrow3D(
            start=axes.c2p(0, 0, 0),
            end=axes.c2p(1, 2, 1),
            color=BLUE
        )
        
        # Cross product
        cross = Arrow3D(
            start=axes.c2p(0, 0, 0),
            end=axes.c2p(-3, 0, 3),  # a × b
            color=GREEN
        )
        
        # Labels (fixed in frame)
        label_a = MathTex("\\vec{a}", color=RED).to_corner(UR)
        label_b = MathTex("\\vec{b}", color=BLUE).next_to(label_a, DOWN)
        label_cross = MathTex("\\vec{a} \\times \\vec{b}", color=GREEN).next_to(label_b, DOWN)
        
        self.add_fixed_in_frame_mobjects(label_a, label_b, label_cross)
        
        self.play(Create(axes))
        self.play(Create(vec_a), Write(label_a))
        self.play(Create(vec_b), Write(label_b))
        self.wait()
        self.play(Create(cross), Write(label_cross))
        
        # Rotate to show perpendicularity
        self.move_camera(phi=45 * DEGREES, theta=-90 * DEGREES, run_time=2)
        self.wait()
        self.move_camera(phi=90 * DEGREES, theta=-45 * DEGREES, run_time=2)
        self.wait(2)


class SolidOfRevolution(ThreeDScene):
    """Visualize a solid of revolution."""
    
    def construct(self):
        # Camera setup
        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)
        
        # 3D axes
        axes = ThreeDAxes(
            x_range=[-1, 3, 1],
            y_range=[-2, 2, 1],
            z_range=[-2, 2, 1],
            x_length=6,
            y_length=4,
            z_length=4,
        )
        
        # Function y = sqrt(x) from 0 to 2
        curve = ParametricFunction(
            lambda t: axes.c2p(t, np.sqrt(t), 0),
            t_range=[0.1, 2],
            color=YELLOW
        )
        
        # Create the solid of revolution
        surface = Surface(
            lambda u, v: axes.c2p(
                u,
                np.sqrt(u) * np.cos(v),
                np.sqrt(u) * np.sin(v)
            ),
            u_range=[0.1, 2],
            v_range=[0, 2 * PI],
            resolution=(20, 40),
            fill_opacity=0.6,
            checkerboard_colors=[BLUE_D, BLUE_E],
        )
        
        # Equation
        equation = MathTex("y = \\sqrt{x}").to_corner(UL)
        volume = MathTex(
            "V = \\pi \\int_0^2 x \\, dx = 2\\pi"
        ).to_corner(UR)
        
        self.add_fixed_in_frame_mobjects(equation, volume)
        
        self.play(Create(axes))
        self.play(Create(curve), Write(equation))
        self.wait()
        
        # Show rotation
        self.begin_ambient_camera_rotation(rate=0.3)
        self.play(Create(surface), run_time=3)
        self.play(Write(volume))
        self.wait(4)
        self.stop_ambient_camera_rotation()


class WaveFunction3D(ThreeDScene):
    """Visualize a 3D wave function."""
    
    def construct(self):
        # Camera setup
        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)
        
        # 3D axes
        axes = ThreeDAxes(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            z_range=[-2, 2, 1],
            x_length=8,
            y_length=8,
            z_length=4,
        )
        
        # Time tracker
        time = ValueTracker(0)
        
        # Wave surface
        def get_wave_surface():
            t = time.get_value()
            surface = Surface(
                lambda u, v: axes.c2p(
                    u, v,
                    np.sin(u - t) * np.cos(v - t) * np.exp(-(u**2 + v**2) / 8)
                ),
                u_range=[-3, 3],
                v_range=[-3, 3],
                resolution=(30, 30),
                fill_opacity=0.8,
                checkerboard_colors=[BLUE, TEAL],
            )
            return surface
        
        surface = always_redraw(get_wave_surface)
        
        # Title
        title = Tex("2D Wave Propagation").to_corner(UL)
        self.add_fixed_in_frame_mobjects(title)
        
        self.play(Create(axes), Write(title))
        self.add(surface)
        
        # Animate wave
        self.begin_ambient_camera_rotation(rate=0.1)
        self.play(time.animate.set_value(4 * PI), run_time=10, rate_func=linear)
        self.stop_ambient_camera_rotation()


# =============================================================================
# GEOMETRY EXAMPLES
# =============================================================================

class PythagoreanTheorem(Scene):
    """Animated proof of the Pythagorean theorem."""
    
    def construct(self):
        # Title
        title = Title("Pythagorean Theorem")
        self.play(Write(title))
        self.wait()
        self.play(FadeOut(title))
        
        # Right triangle
        a, b, c = 2, 1.5, 2.5  # 3-4-5 scaled
        
        triangle = Polygon(
            ORIGIN,
            RIGHT * a,
            RIGHT * a + UP * b,
            color=WHITE,
            fill_opacity=0.3,
            fill_color=BLUE
        ).move_to(ORIGIN)
        
        # Right angle marker
        right_angle = Square(side_length=0.2, color=WHITE).move_to(
            triangle.get_vertices()[1] + UP * 0.1 + LEFT * 0.1
        )
        
        # Labels
        label_a = MathTex("a").next_to(triangle, DOWN)
        label_b = MathTex("b").next_to(triangle, RIGHT)
        label_c = MathTex("c").move_to(
            (triangle.get_vertices()[0] + triangle.get_vertices()[2]) / 2 + LEFT * 0.3 + UP * 0.3
        )
        
        self.play(Create(triangle), Create(right_angle))
        self.play(Write(label_a), Write(label_b), Write(label_c))
        self.wait()
        
        # Create squares on each side
        self.play(triangle.animate.scale(0.7).to_edge(LEFT).shift(DOWN))
        
        # Square on a
        sq_a = Square(side_length=a * 0.7, color=RED, fill_opacity=0.5).next_to(
            triangle, DOWN, buff=0
        ).align_to(triangle, LEFT)
        
        # Square on b
        sq_b = Square(side_length=b * 0.7, color=GREEN, fill_opacity=0.5).next_to(
            triangle, RIGHT, buff=0
        ).align_to(triangle, DOWN)
        
        # Square on c
        sq_c = Square(side_length=c * 0.7, color=YELLOW, fill_opacity=0.5).rotate(
            np.arctan(b/a)
        ).move_to(triangle.get_center() + LEFT * 0.8 + UP * 0.5)
        
        sq_a_label = MathTex("a^2", color=RED).move_to(sq_a)
        sq_b_label = MathTex("b^2", color=GREEN).move_to(sq_b)
        sq_c_label = MathTex("c^2", color=YELLOW).move_to(sq_c)
        
        self.play(Create(sq_a), Write(sq_a_label))
        self.play(Create(sq_b), Write(sq_b_label))
        self.play(Create(sq_c), Write(sq_c_label))
        self.wait()
        
        # Theorem
        theorem = MathTex("a^2 + b^2 = c^2", color=WHITE).scale(1.5).to_edge(RIGHT)
        box = SurroundingRectangle(theorem, color=YELLOW)
        
        self.play(Write(theorem))
        self.play(Create(box))
        self.wait(2)


class CircleTheorems(Scene):
    """Visualize key circle theorems."""
    
    def construct(self):
        # Title
        title = Title("Inscribed Angle Theorem")
        self.play(Write(title))
        self.wait()
        self.play(title.animate.scale(0.6).to_corner(UL))
        
        # Circle
        circle = Circle(radius=2, color=WHITE)
        center = Dot(ORIGIN, color=WHITE)
        
        # Points on circle
        A = circle.point_at_angle(PI / 6)
        B = circle.point_at_angle(5 * PI / 6)
        P = circle.point_at_angle(3 * PI / 2)
        
        dot_A = Dot(A, color=RED)
        dot_B = Dot(B, color=RED)
        dot_P = Dot(P, color=BLUE)
        
        label_A = MathTex("A").next_to(dot_A, UR, buff=0.1)
        label_B = MathTex("B").next_to(dot_B, UL, buff=0.1)
        label_P = MathTex("P").next_to(dot_P, DOWN, buff=0.1)
        
        self.play(Create(circle), Create(center))
        self.play(
            Create(dot_A), Create(dot_B), Create(dot_P),
            Write(label_A), Write(label_B), Write(label_P)
        )
        
        # Central angle
        central_angle_line1 = Line(ORIGIN, A, color=YELLOW)
        central_angle_line2 = Line(ORIGIN, B, color=YELLOW)
        central_arc = Arc(
            radius=0.5,
            start_angle=PI / 6,
            angle=4 * PI / 6,
            color=YELLOW
        )
        central_label = MathTex("2\\theta", color=YELLOW).move_to(UP * 0.8)
        
        # Inscribed angle
        inscribed_line1 = Line(P, A, color=GREEN)
        inscribed_line2 = Line(P, B, color=GREEN)
        inscribed_arc = Arc(
            radius=0.5,
            start_angle=PI / 6 + PI / 2,
            angle=2 * PI / 6,
            color=GREEN
        ).shift(P)
        inscribed_label = MathTex("\\theta", color=GREEN).next_to(P, UP, buff=0.5)
        
        self.play(
            Create(central_angle_line1), Create(central_angle_line2),
            Create(central_arc), Write(central_label)
        )
        self.wait()
        
        self.play(
            Create(inscribed_line1), Create(inscribed_line2),
            Create(inscribed_arc), Write(inscribed_label)
        )
        self.wait()
        
        # Theorem statement
        theorem = MathTex(
            "\\text{Inscribed angle} = \\frac{1}{2} \\times \\text{Central angle}"
        ).to_edge(DOWN)
        
        self.play(Write(theorem))
        self.wait(2)


# =============================================================================
# Run examples directly
# =============================================================================

if __name__ == "__main__":
    print("Example scenes available:")
    print("  - DerivativeVisualization")
    print("  - IntegralAreaVisualization")
    print("  - LimitConcept")
    print("  - LinearTransformation2D")
    print("  - EigenvectorVisualization")
    print("  - ProjectileMotion")
    print("  - SimpleHarmonicMotion")
    print("  - ElectricFieldLines")
    print("  - Surface3DPlot")
    print("  - Vector3DVisualization")
    print("  - SolidOfRevolution")
    print("  - WaveFunction3D")
    print("  - PythagoreanTheorem")
    print("  - CircleTheorems")
    print()
    print("Run with: manim -qh examples.py <SceneName>")
