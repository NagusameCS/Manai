"""
Extended Calculus Templates - Intuitive derivation-focused animations
"""

CHAIN_RULE_VISUAL = '''
from manim import *

class ChainRuleVisual(Scene):
    """Visualize the chain rule with nested functions."""
    
    def construct(self):
        title = Text("Chain Rule").scale(0.6).to_edge(UP)
        self.play(Write(title))
        
        # Show nested function concept
        outer_box = RoundedRectangle(width=4, height=2, color=BLUE).shift(LEFT*2)
        inner_box = RoundedRectangle(width=2, height=1, color=RED).move_to(outer_box)
        
        outer_label = MathTex("f", color=BLUE).next_to(outer_box, UP)
        inner_label = MathTex("g", color=RED).next_to(inner_box, UP, buff=0.1)
        
        self.play(Create(outer_box), Write(outer_label))
        self.play(Create(inner_box), Write(inner_label))
        self.wait()
        
        # Show input flowing through
        x_dot = Dot(color=YELLOW).shift(LEFT*5)
        self.play(Create(x_dot))
        self.play(x_dot.animate.move_to(inner_box.get_center()), run_time=1.5)
        
        g_result = MathTex("g(x)", color=RED).scale(0.5).next_to(inner_box, DOWN)
        self.play(Write(g_result))
        
        self.play(x_dot.animate.move_to(outer_box.get_right() + RIGHT*0.5), run_time=1)
        f_result = MathTex("f(g(x))", color=BLUE).scale(0.5).next_to(outer_box, RIGHT)
        self.play(Write(f_result))
        self.wait()
        
        # The chain rule formula
        self.play(FadeOut(outer_box, inner_box, outer_label, inner_label, x_dot, g_result, f_result, title))
        
        formula = MathTex(
            r"\\frac{d}{dx}[f(g(x))]", "=", r"f\\'(g(x))", r"\\cdot", r"g\\'(x)"
        ).scale(0.7)
        formula[2].set_color(BLUE)
        formula[4].set_color(RED)
        
        self.play(Write(formula))
        self.wait(2)
'''

PRODUCT_RULE_VISUAL = '''
from manim import *

class ProductRuleVisual(Scene):
    """Visualize product rule as area of rectangle."""
    
    def construct(self):
        title = Text("Product Rule").scale(0.6).to_edge(UP)
        self.play(Write(title))
        
        # Rectangle representing f(x) * g(x)
        rect = Rectangle(width=3, height=2, color=BLUE, fill_opacity=0.3).shift(LEFT*2)
        
        f_brace = Brace(rect, DOWN)
        g_brace = Brace(rect, LEFT)
        f_label = MathTex("f(x)").scale(0.5).next_to(f_brace, DOWN)
        g_label = MathTex("g(x)").scale(0.5).next_to(g_brace, LEFT)
        
        self.play(Create(rect), Create(f_brace), Create(g_brace))
        self.play(Write(f_label), Write(g_label))
        self.wait()
        
        # Show the formula
        formula = MathTex(
            r"(f \\cdot g)\\'", "=", r"f\\' \\cdot g", "+", r"f \\cdot g\\'"
        ).scale(0.6).shift(RIGHT*2)
        formula[2].set_color(BLUE)
        formula[4].set_color(RED)
        
        self.play(Write(formula))
        self.wait(2)
'''

RIEMANN_SUM = '''
from manim import *

class RiemannSum(Scene):
    """Visualize Riemann sums approaching integral."""
    
    def construct(self):
        axes = Axes(x_range=[0, 5], y_range=[0, 5], x_length=7, y_length=4).shift(DOWN*0.5)
        
        func = lambda x: 0.2 * x**2 + 0.5
        graph = axes.plot(func, x_range=[0.5, 4.5], color=WHITE)
        
        self.play(Create(axes), Create(graph))
        self.wait()
        
        # Show rectangles with increasing n
        prev_rects = None
        for n in [4, 8, 16, 32]:
            rects = axes.get_riemann_rectangles(
                graph, x_range=[1, 4], dx=(4-1)/n,
                color=[BLUE, GREEN], fill_opacity=0.5
            )
            n_label = MathTex(f"n = {n}").scale(0.6).to_corner(UR)
            
            if prev_rects is None:
                self.play(Create(rects), Write(n_label))
                prev_rects = rects
                prev_label = n_label
            else:
                new_rects = rects
                self.play(
                    ReplacementTransform(prev_rects, new_rects),
                    ReplacementTransform(prev_label, n_label)
                )
                prev_rects = new_rects
                prev_label = n_label
            self.wait(0.5)
        
        integral = MathTex(r"\\int_1^4 f(x)\\, dx").scale(0.6).to_corner(UL)
        self.play(Write(integral))
        self.wait(2)
'''

FUNDAMENTAL_THEOREM = '''
from manim import *

class FundamentalTheorem(Scene):
    """Visualize the Fundamental Theorem of Calculus."""
    
    def construct(self):
        title = Text("Fundamental Theorem").scale(0.5).to_edge(UP)
        self.play(Write(title))
        
        axes = Axes(x_range=[0, 5], y_range=[-1, 4], x_length=6, y_length=3.5).shift(DOWN*0.5)
        
        func = lambda x: 0.3 * x + 0.5
        graph = axes.plot(func, x_range=[0, 5], color=BLUE)
        
        self.play(Create(axes), Create(graph))
        
        # Area accumulator
        x_tracker = ValueTracker(0.5)
        
        area = always_redraw(lambda: axes.get_area(
            graph, x_range=[0.5, x_tracker.get_value()], color=GREEN, opacity=0.4
        ))
        
        self.play(Create(area))
        self.play(x_tracker.animate.set_value(4), run_time=4)
        
        theorem = MathTex(r"\\frac{d}{dx}\\int_a^x f(t)\\,dt = f(x)").scale(0.6).to_edge(DOWN)
        self.play(Write(theorem))
        self.wait(2)
'''

LHOPITAL_RULE = '''
from manim import *

class LHopitalRule(Scene):
    """Visualize L Hopital rule for 0/0."""
    
    def construct(self):
        title = Text("L Hopital Rule").scale(0.5).to_edge(UP)
        self.play(Write(title))
        
        problem = MathTex(r"\\lim_{x \\to 0} \\frac{\\sin x}{x} = \\frac{0}{0}").scale(0.6)
        self.play(Write(problem))
        self.wait()
        self.play(problem.animate.shift(UP*1.5))
        
        axes = Axes(x_range=[-2, 2], y_range=[-2, 2], x_length=5, y_length=3).shift(DOWN*0.5)
        
        sin_graph = axes.plot(np.sin, x_range=[-2, 2], color=BLUE)
        x_graph = axes.plot(lambda x: x, x_range=[-2, 2], color=RED)
        
        self.play(Create(axes), Create(sin_graph), Create(x_graph))
        
        lhopital = MathTex(r"= \\lim_{x \\to 0} \\frac{\\cos x}{1} = 1").scale(0.6).next_to(problem, DOWN)
        self.play(Write(lhopital))
        self.wait(2)
'''

IMPLICIT_DIFF = '''
from manim import *

class ImplicitDiff(Scene):
    """Implicit differentiation on a circle."""
    
    def construct(self):
        title = Text("Implicit Differentiation").scale(0.5).to_edge(UP)
        self.play(Write(title))
        
        axes = Axes(x_range=[-3, 3], y_range=[-3, 3], x_length=5, y_length=5)
        circle = Circle(radius=2, color=BLUE).move_to(axes.c2p(0, 0))
        equation = MathTex("x^2 + y^2 = 4").scale(0.5).to_corner(UR)
        
        self.play(Create(axes), Create(circle), Write(equation))
        
        angle = ValueTracker(PI/4)
        
        point = always_redraw(lambda: Dot(color=RED).move_to(
            axes.c2p(2*np.cos(angle.get_value()), 2*np.sin(angle.get_value()))
        ))
        
        self.play(Create(point))
        
        deriv = MathTex(r"\\frac{dy}{dx} = -\\frac{x}{y}").scale(0.5).to_corner(UL)
        self.play(Write(deriv))
        
        self.play(angle.animate.set_value(2*PI + PI/4), run_time=5, rate_func=linear)
        self.wait(2)
'''

RELATED_RATES = '''
from manim import *

class RelatedRates(Scene):
    """Related rates with expanding circle."""
    
    def construct(self):
        title = Text("Related Rates").scale(0.5).to_edge(UP)
        self.play(Write(title))
        
        r = ValueTracker(0.5)
        
        circle = always_redraw(lambda: Circle(radius=r.get_value(), color=BLUE, fill_opacity=0.3))
        
        r_label = always_redraw(lambda: MathTex(f"r = {r.get_value():.1f}").scale(0.5).shift(DOWN*2))
        
        self.play(Create(circle), Write(r_label))
        
        rate_text = MathTex(r"\\frac{dr}{dt} = 0.5").scale(0.5).to_corner(UL)
        self.play(Write(rate_text))
        
        self.play(r.animate.set_value(2.5), run_time=4, rate_func=linear)
        
        relationship = MathTex(r"\\frac{dA}{dt} = 2\\pi r \\frac{dr}{dt}").scale(0.5).to_edge(DOWN)
        self.play(Write(relationship))
        self.wait(2)
'''

OPTIMIZATION = '''
from manim import *

class Optimization(Scene):
    """Finding max/min with derivatives."""
    
    def construct(self):
        title = Text("Optimization").scale(0.5).to_edge(UP)
        self.play(Write(title))
        
        axes = Axes(x_range=[-1, 5], y_range=[-2, 5], x_length=7, y_length=4).shift(DOWN*0.3)
        
        func = lambda x: -0.3*(x-1)*(x-2)*(x-4) + 2
        graph = axes.plot(func, x_range=[0, 4.5], color=BLUE)
        
        self.play(Create(axes), Create(graph))
        
        # Critical points
        for x in [1.18, 3.15]:
            dot = Dot(color=RED).move_to(axes.c2p(x, func(x)))
            self.play(Create(dot))
        
        derivative_zero = MathTex(r"f\\'(x) = 0").scale(0.5).to_corner(UR)
        self.play(Write(derivative_zero))
        self.wait(2)
'''

ARC_LENGTH = '''
from manim import *

class ArcLength(Scene):
    """Arc length calculation."""
    
    def construct(self):
        title = Text("Arc Length").scale(0.5).to_edge(UP)
        self.play(Write(title))
        
        axes = Axes(x_range=[0, 4], y_range=[0, 3], x_length=6, y_length=4).shift(DOWN*0.3)
        
        func = lambda x: 0.5 * x**1.5
        curve = axes.plot(func, x_range=[0.5, 3.5], color=BLUE)
        
        self.play(Create(axes), Create(curve))
        
        # Show segments
        segments = VGroup()
        x_vals = np.linspace(0.5, 3.5, 9)
        for i in range(8):
            segment = Line(
                axes.c2p(x_vals[i], func(x_vals[i])),
                axes.c2p(x_vals[i+1], func(x_vals[i+1])),
                color=YELLOW
            )
            segments.add(segment)
        
        self.play(Create(segments), run_time=2)
        
        formula = MathTex(r"L = \\int_a^b \\sqrt{1 + (f\\'(x))^2}\\, dx").scale(0.5).to_edge(DOWN)
        self.play(Write(formula))
        self.wait(2)
'''

# Template registry for calculus
CALCULUS_TEMPLATES = {
    "chain_rule": CHAIN_RULE_VISUAL,
    "product_rule": PRODUCT_RULE_VISUAL,
    "riemann_sum": RIEMANN_SUM,
    "fundamental_theorem": FUNDAMENTAL_THEOREM,
    "lhopital": LHOPITAL_RULE,
    "implicit_diff": IMPLICIT_DIFF,
    "related_rates": RELATED_RATES,
    "optimization": OPTIMIZATION,
    "arc_length": ARC_LENGTH,
}

def get_calculus_template(name: str) -> str:
    return CALCULUS_TEMPLATES.get(name, "")

def list_calculus_templates() -> list:
    return list(CALCULUS_TEMPLATES.keys())
