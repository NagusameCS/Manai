"""
3D Templates - Extensive collection of 3D animation templates
"""

# =============================================================================
# 3D SURFACES AND CALCULUS
# =============================================================================

SURFACE_PLOT = '''
from manim import *
import numpy as np

class SurfacePlot(ThreeDScene):
    """3D surface plot of z = f(x, y)."""
    
    def construct(self):
        # Camera setup
        self.set_camera_orientation(phi=70 * DEGREES, theta=-45 * DEGREES)
        
        # 3D Axes
        axes = ThreeDAxes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            z_range=[-2, 2, 1],
            x_length=6,
            y_length=6,
            z_length=4,
        )
        
        # Surface: z = sin(x) * cos(y)
        surface = Surface(
            lambda u, v: axes.c2p(u, v, np.sin(u) * np.cos(v)),
            u_range=[-3, 3],
            v_range=[-3, 3],
            resolution=(30, 30),
            fill_opacity=0.7,
            checkerboard_colors=[GRAY_C, GRAY_B],
            stroke_color=WHITE,
            stroke_width=0.5,
        )
        
        # Equation
        equation = MathTex(
            r"z = \\sin(x) \\cos(y)"
        ).to_corner(UL).set_z_index(100)
        self.add_fixed_in_frame_mobjects(equation)
        
        # Animation
        self.play(Create(axes))
        self.play(Create(surface), run_time=2)
        self.wait()
        
        # Rotate camera
        self.begin_ambient_camera_rotation(rate=0.15)
        self.wait(4)
        self.stop_ambient_camera_rotation()
        self.wait()
'''

PARTIAL_DERIVATIVES = '''
from manim import *
import numpy as np

class PartialDerivatives(ThreeDScene):
    """Visualize partial derivatives as slope in x and y directions."""
    
    def construct(self):
        self.set_camera_orientation(phi=60 * DEGREES, theta=-60 * DEGREES)
        
        axes = ThreeDAxes(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            z_range=[0, 4, 1],
            x_length=5,
            y_length=5,
            z_length=4,
        )
        
        # Surface: z = x^2 + y^2
        surface = Surface(
            lambda u, v: axes.c2p(u, v, u**2 + v**2),
            u_range=[-2, 2],
            v_range=[-2, 2],
            resolution=(20, 20),
            fill_opacity=0.5,
            checkerboard_colors=[GRAY_D, GRAY_C],
            stroke_width=0.3,
        )
        
        # Point of tangent
        x0, y0 = 1, 0.5
        z0 = x0**2 + y0**2
        point = Dot3D(axes.c2p(x0, y0, z0), color=WHITE)
        
        # Tangent in x direction (partial x)
        tangent_x = Line3D(
            axes.c2p(x0 - 0.5, y0, z0 - 2*x0*0.5),
            axes.c2p(x0 + 0.5, y0, z0 + 2*x0*0.5),
            color=GRAY_B
        )
        
        # Tangent in y direction (partial y)
        tangent_y = Line3D(
            axes.c2p(x0, y0 - 0.5, z0 - 2*y0*0.5),
            axes.c2p(x0, y0 + 0.5, z0 + 2*y0*0.5),
            color=GRAY_A
        )
        
        # Labels
        eq = MathTex(r"z = x^2 + y^2").to_corner(UL)
        partial_x = MathTex(r"\\frac{\\partial z}{\\partial x} = 2x", color=GRAY_B)
        partial_y = MathTex(r"\\frac{\\partial z}{\\partial y} = 2y", color=GRAY_A)
        partials = VGroup(partial_x, partial_y).arrange(DOWN).to_corner(UR)
        
        self.add_fixed_in_frame_mobjects(eq, partials)
        
        # Animation
        self.play(Create(axes))
        self.play(Create(surface))
        self.wait()
        
        self.play(Create(point))
        self.play(Create(tangent_x), Indicate(partial_x))
        self.wait()
        self.play(Create(tangent_y), Indicate(partial_y))
        self.wait(2)
'''

GRADIENT_FIELD = '''
from manim import *
import numpy as np

class GradientField(ThreeDScene):
    """Gradient vector field on a surface."""
    
    def construct(self):
        self.set_camera_orientation(phi=65 * DEGREES, theta=-45 * DEGREES)
        
        axes = ThreeDAxes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            z_range=[0, 5, 1],
            x_length=6,
            y_length=6,
            z_length=4,
        )
        
        # Surface: z = exp(-(x^2 + y^2)/4)
        def f(x, y):
            return 3 * np.exp(-(x**2 + y**2) / 4)
        
        surface = Surface(
            lambda u, v: axes.c2p(u, v, f(u, v)),
            u_range=[-3, 3],
            v_range=[-3, 3],
            resolution=(25, 25),
            fill_opacity=0.6,
            checkerboard_colors=[GRAY_C, GRAY_B],
        )
        
        # Gradient vectors on xy-plane
        arrows = VGroup()
        for x in np.linspace(-2, 2, 5):
            for y in np.linspace(-2, 2, 5):
                if x == 0 and y == 0:
                    continue
                # Gradient of f = (-x*f/2, -y*f/2)
                grad_x = -x * f(x, y) / 2
                grad_y = -y * f(x, y) / 2
                mag = np.sqrt(grad_x**2 + grad_y**2)
                if mag > 0.1:
                    scale = 0.4 / mag
                    arrow = Arrow3D(
                        axes.c2p(x, y, 0.01),
                        axes.c2p(x + grad_x*scale, y + grad_y*scale, 0.01),
                        color=GRAY_A
                    )
                    arrows.add(arrow)
        
        # Labels
        eq = MathTex(r"f(x,y) = e^{-(x^2+y^2)/4}").to_corner(UL)
        grad_eq = MathTex(r"\\nabla f = \\left(-\\frac{x}{2}f, -\\frac{y}{2}f\\right)")
        grad_eq.to_corner(UR)
        self.add_fixed_in_frame_mobjects(eq, grad_eq)
        
        # Animation
        self.play(Create(axes))
        self.play(Create(surface))
        self.wait()
        self.play(Create(arrows))
        self.wait(2)
'''

DOUBLE_INTEGRAL = '''
from manim import *
import numpy as np

class DoubleIntegral(ThreeDScene):
    """Volume under surface as double integral."""
    
    def construct(self):
        self.set_camera_orientation(phi=70 * DEGREES, theta=-50 * DEGREES)
        
        axes = ThreeDAxes(
            x_range=[0, 3, 1],
            y_range=[0, 3, 1],
            z_range=[0, 4, 1],
            x_length=5,
            y_length=5,
            z_length=4,
        )
        
        # Surface
        def f(x, y):
            return 0.5 * (x + y)
        
        surface = Surface(
            lambda u, v: axes.c2p(u, v, f(u, v)),
            u_range=[0, 2],
            v_range=[0, 2],
            resolution=(15, 15),
            fill_opacity=0.6,
            checkerboard_colors=[GRAY_B, GRAY_A],
        )
        
        # Volume blocks
        blocks = VGroup()
        n = 8
        dx = 2 / n
        for i in range(n):
            for j in range(n):
                x = i * dx
                y = j * dx
                z = f(x + dx/2, y + dx/2)
                block = Prism(
                    dimensions=[dx*0.9, dx*0.9, z],
                    fill_opacity=0.4,
                    stroke_width=0.5,
                    stroke_color=WHITE,
                ).move_to(axes.c2p(x + dx/2, y + dx/2, z/2))
                blocks.add(block)
        
        # Integral notation
        integral = MathTex(
            r"\\iint_R f(x,y) \\, dA"
        ).to_corner(UL)
        self.add_fixed_in_frame_mobjects(integral)
        
        # Animation
        self.play(Create(axes))
        self.play(Create(surface))
        self.wait()
        
        self.play(LaggedStart(*[Create(b) for b in blocks], lag_ratio=0.02))
        self.wait(2)
'''

# =============================================================================
# 3D GEOMETRY
# =============================================================================

PLATONIC_SOLIDS = '''
from manim import *

class PlatonicSolids(ThreeDScene):
    """The five Platonic solids."""
    
    def construct(self):
        self.set_camera_orientation(phi=70 * DEGREES, theta=-45 * DEGREES)
        
        # Create solids
        tetrahedron = Tetrahedron(edge_length=1.5)
        cube = Cube(side_length=1.2)
        octahedron = Octahedron(edge_length=1.5)
        dodecahedron = Dodecahedron(edge_length=0.8)
        icosahedron = Icosahedron(edge_length=0.8)
        
        solids = [tetrahedron, cube, octahedron, dodecahedron, icosahedron]
        names = ["Tetrahedron", "Cube", "Octahedron", "Dodecahedron", "Icosahedron"]
        faces = [4, 6, 8, 12, 20]
        
        # Style
        for s in solids:
            s.set_fill(GRAY_C, opacity=0.7)
            s.set_stroke(WHITE, width=1)
        
        # Arrange
        positions = [
            LEFT * 4 + UP,
            LEFT * 2 + DOWN,
            ORIGIN + UP,
            RIGHT * 2 + DOWN,
            RIGHT * 4 + UP,
        ]
        
        for s, pos in zip(solids, positions):
            s.move_to(pos)
        
        # Title
        title = Text("Platonic Solids", font_size=36).to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)
        
        # Animation
        for solid, name, f in zip(solids, names, faces):
            label = Text(f"{name}\\n{f} faces", font_size=20)
            label.next_to(solid, DOWN)
            self.add_fixed_in_frame_mobjects(label)
            self.play(Create(solid), Write(label))
            self.play(Rotate(solid, PI/2, axis=UP), run_time=0.5)
        
        self.begin_ambient_camera_rotation(rate=0.1)
        self.wait(4)
'''

PARAMETRIC_CURVE_3D = '''
from manim import *
import numpy as np

class ParametricCurve3D(ThreeDScene):
    """3D parametric curves - helix, torus knot."""
    
    def construct(self):
        self.set_camera_orientation(phi=70 * DEGREES, theta=-45 * DEGREES)
        
        axes = ThreeDAxes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            z_range=[-3, 3, 1],
            x_length=6,
            y_length=6,
            z_length=6,
        )
        
        # Helix: (cos(t), sin(t), t/3)
        helix = ParametricFunction(
            lambda t: axes.c2p(
                np.cos(t),
                np.sin(t),
                t / 3
            ),
            t_range=[0, 6*PI],
            color=WHITE,
        )
        
        helix_eq = MathTex(
            r"\\vec{r}(t) = (\\cos t, \\sin t, t/3)"
        ).to_corner(UL)
        self.add_fixed_in_frame_mobjects(helix_eq)
        
        # Animation
        self.play(Create(axes))
        self.play(Create(helix), run_time=3)
        self.wait()
        
        self.begin_ambient_camera_rotation(rate=0.2)
        self.wait(4)
        self.stop_ambient_camera_rotation()
        
        # Transform to torus knot
        torus_knot = ParametricFunction(
            lambda t: axes.c2p(
                (2 + np.cos(3*t)) * np.cos(2*t),
                (2 + np.cos(3*t)) * np.sin(2*t),
                np.sin(3*t)
            ),
            t_range=[0, 2*PI],
            color=GRAY_A,
        )
        
        knot_eq = MathTex(
            r"\\text{Torus Knot } (2,3)"
        ).to_corner(UL)
        
        self.add_fixed_in_frame_mobjects(knot_eq)
        self.play(
            Transform(helix, torus_knot),
            FadeOut(helix_eq),
            run_time=2
        )
        self.begin_ambient_camera_rotation(rate=0.15)
        self.wait(4)
'''

VECTOR_FIELD_3D = '''
from manim import *
import numpy as np

class VectorField3D(ThreeDScene):
    """3D vector field visualization."""
    
    def construct(self):
        self.set_camera_orientation(phi=60 * DEGREES, theta=-60 * DEGREES)
        
        axes = ThreeDAxes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            z_range=[-3, 3, 1],
            x_length=6,
            y_length=6,
            z_length=6,
        )
        
        # Vector field: F = (-y, x, 0) - rotation about z
        arrows = VGroup()
        for x in np.linspace(-2, 2, 5):
            for y in np.linspace(-2, 2, 5):
                for z in [-1, 0, 1]:
                    if x == 0 and y == 0:
                        continue
                    # Field vector
                    fx, fy, fz = -y * 0.3, x * 0.3, 0
                    arrow = Arrow3D(
                        axes.c2p(x, y, z),
                        axes.c2p(x + fx, y + fy, z + fz),
                        color=GRAY_A,
                    )
                    arrows.add(arrow)
        
        # Label
        field_eq = MathTex(
            r"\\vec{F} = (-y, x, 0)"
        ).to_corner(UL)
        self.add_fixed_in_frame_mobjects(field_eq)
        
        # Animation
        self.play(Create(axes))
        self.play(LaggedStart(*[Create(a) for a in arrows], lag_ratio=0.02))
        self.wait()
        
        self.begin_ambient_camera_rotation(rate=0.1)
        self.wait(4)
'''

# =============================================================================
# 3D PHYSICS
# =============================================================================

ELECTROMAGNETIC_WAVE = '''
from manim import *
import numpy as np

class ElectromagneticWave(ThreeDScene):
    """3D electromagnetic wave with E and B fields."""
    
    def construct(self):
        self.set_camera_orientation(phi=70 * DEGREES, theta=-60 * DEGREES, zoom=0.8)
        
        axes = ThreeDAxes(
            x_range=[0, 4*PI, PI],
            y_range=[-2, 2, 1],
            z_range=[-2, 2, 1],
            x_length=10,
            y_length=4,
            z_length=4,
        )
        
        t = ValueTracker(0)
        
        # Electric field (oscillates in y)
        def get_e_field():
            wave = ParametricFunction(
                lambda x: axes.c2p(x, np.sin(x - t.get_value()), 0),
                t_range=[0, 4*PI],
                color=GRAY_B,
            )
            return wave
        
        # Magnetic field (oscillates in z)
        def get_b_field():
            wave = ParametricFunction(
                lambda x: axes.c2p(x, 0, np.sin(x - t.get_value())),
                t_range=[0, 4*PI],
                color=GRAY_A,
            )
            return wave
        
        e_wave = always_redraw(get_e_field)
        b_wave = always_redraw(get_b_field)
        
        # Direction of propagation
        k_arrow = Arrow3D(
            axes.c2p(0, 0, 0),
            axes.c2p(2, 0, 0),
            color=WHITE
        )
        
        # Labels
        labels = VGroup(
            MathTex(r"\\vec{E}", color=GRAY_B),
            MathTex(r"\\vec{B}", color=GRAY_A),
            MathTex(r"\\vec{k}", color=WHITE),
        ).arrange(DOWN).to_corner(UL)
        self.add_fixed_in_frame_mobjects(labels)
        
        # Animation
        self.play(Create(axes))
        self.play(Create(e_wave), Create(b_wave), Create(k_arrow))
        self.wait()
        
        # Animate wave propagation
        self.play(
            t.animate.set_value(4*PI),
            run_time=4,
            rate_func=linear
        )
        self.wait()
'''

ORBITAL_MECHANICS = '''
from manim import *
import numpy as np

class OrbitalMechanics(ThreeDScene):
    """Kepler orbits in 3D."""
    
    def construct(self):
        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES, zoom=0.9)
        
        # Central body (sun)
        sun = Sphere(radius=0.3, color=WHITE).set_opacity(0.8)
        
        # Orbital parameters
        a = 2  # Semi-major axis
        e = 0.5  # Eccentricity
        
        # Elliptical orbit
        def orbit_point(theta):
            r = a * (1 - e**2) / (1 + e * np.cos(theta))
            return np.array([r * np.cos(theta), r * np.sin(theta), 0])
        
        orbit_path = ParametricFunction(
            orbit_point,
            t_range=[0, 2*PI],
            color=GRAY_A,
        )
        
        # Planet
        planet = Sphere(radius=0.15, color=GRAY_B).set_opacity(0.9)
        planet.move_to(orbit_point(0))
        
        # Velocity vector (always redrawn)
        t = ValueTracker(0)
        
        def get_planet():
            theta = t.get_value()
            return Sphere(radius=0.15, color=GRAY_B).set_opacity(0.9).move_to(orbit_point(theta))
        
        planet = always_redraw(get_planet)
        
        # Labels
        title = Text("Kepler Orbit (e=0.5)", font_size=28).to_corner(UL)
        self.add_fixed_in_frame_mobjects(title)
        
        # Animation
        self.play(Create(sun))
        self.play(Create(orbit_path))
        self.play(Create(planet))
        self.wait()
        
        # Orbit animation
        self.play(
            t.animate.set_value(2*PI),
            run_time=5,
            rate_func=linear
        )
        self.wait()
'''

WAVE_EQUATION_3D = '''
from manim import *
import numpy as np

class WaveEquation3D(ThreeDScene):
    """2D wave on a membrane."""
    
    def construct(self):
        self.set_camera_orientation(phi=70 * DEGREES, theta=-45 * DEGREES)
        
        t = ValueTracker(0)
        
        # Wave on 2D membrane
        def get_surface():
            return Surface(
                lambda u, v: np.array([
                    u, v,
                    0.3 * np.sin(2*u) * np.sin(2*v) * np.cos(3*t.get_value()) +
                    0.2 * np.sin(3*u) * np.sin(3*v) * np.cos(4*t.get_value())
                ]),
                u_range=[-PI, PI],
                v_range=[-PI, PI],
                resolution=(30, 30),
                fill_opacity=0.7,
                checkerboard_colors=[GRAY_C, GRAY_B],
                stroke_width=0.5,
            )
        
        surface = always_redraw(get_surface)
        
        # Equation
        eq = MathTex(
            r"\\frac{\\partial^2 u}{\\partial t^2} = c^2 \\nabla^2 u"
        ).to_corner(UL)
        self.add_fixed_in_frame_mobjects(eq)
        
        # Animation
        self.play(Create(surface))
        self.wait()
        
        self.play(
            t.animate.set_value(4*PI),
            run_time=6,
            rate_func=linear
        )
        self.wait()
'''

SPHERICAL_HARMONICS = '''
from manim import *
import numpy as np
from scipy.special import sph_harm

class SphericalHarmonics(ThreeDScene):
    """Spherical harmonic visualization."""
    
    def construct(self):
        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)
        
        # Y_l^m spherical harmonic
        l, m = 2, 1
        
        def spherical_harmonic_surface(u, v):
            # u = theta (0 to pi), v = phi (0 to 2pi)
            Y = sph_harm(m, l, v, u)
            r = np.abs(Y)
            x = r * np.sin(u) * np.cos(v)
            y = r * np.sin(u) * np.sin(v)
            z = r * np.cos(u)
            return np.array([x, y, z])
        
        surface = Surface(
            spherical_harmonic_surface,
            u_range=[0, PI],
            v_range=[0, 2*PI],
            resolution=(40, 40),
            fill_opacity=0.8,
            stroke_width=0.3,
        )
        
        # Color by sign (positive/negative)
        surface.set_fill_by_checkerboard(GRAY_B, GRAY_D)
        
        # Label
        label = MathTex(f"Y_{l}^{m}").to_corner(UL)
        self.add_fixed_in_frame_mobjects(label)
        
        # Animation
        self.play(Create(surface), run_time=2)
        self.begin_ambient_camera_rotation(rate=0.15)
        self.wait(5)
'''

COORDINATE_SYSTEMS = '''
from manim import *
import numpy as np

class CoordinateSystems(ThreeDScene):
    """Cartesian, cylindrical, spherical coordinates."""
    
    def construct(self):
        self.set_camera_orientation(phi=65 * DEGREES, theta=-50 * DEGREES)
        
        # Cartesian axes
        axes = ThreeDAxes(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            z_range=[-2, 2, 1],
            x_length=4,
            y_length=4,
            z_length=4,
        )
        
        # A point
        P = np.array([1.5, 1, 1.2])
        point = Dot3D(P, color=WHITE)
        
        # Cartesian coordinates
        cart_lines = VGroup(
            Line3D(ORIGIN, [P[0], 0, 0], color=GRAY_B),
            Line3D([P[0], 0, 0], [P[0], P[1], 0], color=GRAY_B),
            Line3D([P[0], P[1], 0], P, color=GRAY_B),
        )
        
        # Cylindrical (r, theta, z)
        r = np.sqrt(P[0]**2 + P[1]**2)
        theta = np.arctan2(P[1], P[0])
        
        cyl_arc = Arc(
            radius=r,
            start_angle=0,
            angle=theta,
            color=GRAY_A
        )
        cyl_arc_3d = cyl_arc.copy()
        
        # Spherical (rho, theta, phi)
        rho = np.linalg.norm(P)
        phi = np.arccos(P[2] / rho)
        
        sphere_wire = ParametricFunction(
            lambda t: np.array([
                rho * np.sin(t) * np.cos(theta),
                rho * np.sin(t) * np.sin(theta),
                rho * np.cos(t)
            ]),
            t_range=[0, phi],
            color=GRAY_C
        )
        
        # Labels
        cart_label = MathTex("(x, y, z)").to_corner(UL)
        self.add_fixed_in_frame_mobjects(cart_label)
        
        # Animation
        self.play(Create(axes))
        self.play(Create(point))
        self.play(Create(cart_lines))
        self.wait()
        
        cyl_label = MathTex(r"(r, \\theta, z)").to_corner(UL)
        self.add_fixed_in_frame_mobjects(cyl_label)
        self.play(FadeOut(cart_label), FadeIn(cyl_label))
        self.play(Create(cyl_arc))
        self.wait()
        
        sph_label = MathTex(r"(\\rho, \\theta, \\phi)").to_corner(UL)
        self.add_fixed_in_frame_mobjects(sph_label)
        self.play(FadeOut(cyl_label), FadeIn(sph_label))
        self.play(Create(sphere_wire))
        self.wait(2)
'''

# =============================================================================
# TEMPLATE REGISTRY
# =============================================================================

TEMPLATES_3D = {
    # Calculus
    "surface": SURFACE_PLOT,
    "partial_derivative": PARTIAL_DERIVATIVES,
    "gradient": GRADIENT_FIELD,
    "double_integral": DOUBLE_INTEGRAL,
    
    # Geometry
    "platonic": PLATONIC_SOLIDS,
    "parametric_3d": PARAMETRIC_CURVE_3D,
    "vector_field_3d": VECTOR_FIELD_3D,
    "coordinates": COORDINATE_SYSTEMS,
    
    # Physics
    "em_wave": ELECTROMAGNETIC_WAVE,
    "orbital": ORBITAL_MECHANICS,
    "wave_3d": WAVE_EQUATION_3D,
    "spherical_harmonics": SPHERICAL_HARMONICS,
}


def get_3d_template(name: str) -> str:
    """Get a 3D template by name."""
    return TEMPLATES_3D.get(name, "")


def list_3d_templates() -> list[str]:
    """List all available 3D templates."""
    return list(TEMPLATES_3D.keys())
