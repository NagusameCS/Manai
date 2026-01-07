"""
Statistics Templates - Visual demonstrations of probability and statistics concepts
"""

NORMAL_DISTRIBUTION = '''
from manim import *
import numpy as np

class NormalDistribution(Scene):
    """Visualize the normal/Gaussian distribution."""
    
    def construct(self):
        title = Text("Normal Distribution").scale(0.5).to_edge(UP)
        self.play(Write(title))
        
        # Axes
        axes = Axes(
            x_range=[-4, 4, 1],
            y_range=[0, 0.5, 0.1],
            x_length=8,
            y_length=4,
            axis_config={"include_tip": False}
        ).shift(DOWN*0.5)
        
        self.play(Create(axes))
        
        # Normal curve
        def normal(x):
            return (1 / np.sqrt(2 * np.pi)) * np.exp(-x**2 / 2)
        
        curve = axes.plot(normal, color=BLUE)
        self.play(Create(curve))
        
        # Mean line
        mean_line = axes.get_vertical_line(axes.c2p(0, normal(0)), color=RED)
        mu_label = MathTex("\\mu").scale(0.5).next_to(mean_line, UP)
        
        self.play(Create(mean_line), Write(mu_label))
        
        # Standard deviation markers
        sigma1 = axes.get_vertical_line(axes.c2p(1, normal(1)), color=GREEN, stroke_opacity=0.5)
        sigma_neg1 = axes.get_vertical_line(axes.c2p(-1, normal(-1)), color=GREEN, stroke_opacity=0.5)
        
        sigma_label = MathTex("\\sigma").scale(0.4).next_to(axes.c2p(1, 0), DOWN)
        
        self.play(Create(sigma1), Create(sigma_neg1), Write(sigma_label))
        
        # 68% area
        area = axes.get_area(curve, x_range=[-1, 1], color=BLUE, opacity=0.3)
        percent_68 = Text("68%").scale(0.4).move_to(axes.c2p(0, 0.15))
        
        self.play(FadeIn(area), Write(percent_68))
        
        # Formula
        formula = MathTex(r"f(x) = \\frac{1}{\\sigma\\sqrt{2\\pi}} e^{-\\frac{(x-\\mu)^2}{2\\sigma^2}}").scale(0.4).to_edge(DOWN)
        self.play(Write(formula))
        self.wait(2)
'''

CENTRAL_LIMIT = '''
from manim import *
import numpy as np

class CentralLimitTheorem(Scene):
    """Visualize the Central Limit Theorem."""
    
    def construct(self):
        title = Text("Central Limit Theorem").scale(0.5).to_edge(UP)
        self.play(Write(title))
        
        # Show dice example
        dice = VGroup()
        for i in range(3):
            d = Square(side_length=0.5, color=WHITE).shift(LEFT*2 + RIGHT*i*0.7)
            dice.add(d)
        
        self.play(Create(dice))
        
        explanation = Text("Sum many random variables...").scale(0.35).shift(UP*2)
        self.play(Write(explanation))
        
        # Show histogram becoming normal
        axes = Axes(
            x_range=[0, 12, 2],
            y_range=[0, 0.2, 0.05],
            x_length=6,
            y_length=3,
            axis_config={"include_tip": False}
        ).shift(DOWN)
        
        self.play(FadeOut(dice), Create(axes))
        
        # Bar chart approximating normal
        bars = VGroup()
        heights = [0.02, 0.05, 0.1, 0.15, 0.18, 0.15, 0.1, 0.05, 0.02]
        for i, h in enumerate(heights):
            bar = Rectangle(
                width=0.5, height=h*15, 
                color=BLUE, fill_opacity=0.7
            ).move_to(axes.c2p(i+2, h*15/2))
            bars.add(bar)
        
        self.play(Create(bars), run_time=2)
        
        # Normal curve overlay
        def normal(x):
            mu, sigma = 6, 1.5
            return 0.15 * np.exp(-((x-mu)**2)/(2*sigma**2))
        
        curve = axes.plot(normal, x_range=[1, 11], color=RED)
        
        result = Text("...approaches normal distribution").scale(0.35).shift(UP*2)
        self.play(Transform(explanation, result), Create(curve))
        self.wait(2)
'''

PROBABILITY_BASICS = '''
from manim import *

class ProbabilityBasics(Scene):
    """Visualize basic probability concepts."""
    
    def construct(self):
        title = Text("Probability").scale(0.5).to_edge(UP)
        self.play(Write(title))
        
        # Sample space
        sample_space = Rectangle(width=5, height=3, color=WHITE).shift(LEFT*2)
        ss_label = Text("Sample Space S").scale(0.3).next_to(sample_space, UP)
        
        self.play(Create(sample_space), Write(ss_label))
        
        # Event A
        event_a = Circle(radius=1, color=BLUE, fill_opacity=0.3).shift(LEFT*2.5)
        a_label = Text("A").scale(0.4).move_to(event_a)
        
        # Event B
        event_b = Circle(radius=1, color=RED, fill_opacity=0.3).shift(LEFT*1.5)
        b_label = Text("B").scale(0.4).move_to(event_b)
        
        self.play(Create(event_a), Write(a_label), Create(event_b), Write(b_label))
        
        # Intersection
        intersection = Intersection(event_a, event_b, color=PURPLE, fill_opacity=0.5)
        self.play(Create(intersection))
        
        # Formulas
        formulas = VGroup(
            MathTex(r"P(A \\cup B) = P(A) + P(B) - P(A \\cap B)").scale(0.4),
            MathTex(r"P(A | B) = \\frac{P(A \\cap B)}{P(B)}").scale(0.4)
        ).arrange(DOWN, buff=0.3).shift(RIGHT*3)
        
        self.play(Write(formulas))
        self.wait(2)
'''

BAYES_THEOREM = '''
from manim import *

class BayesTheorem(Scene):
    """Visualize Bayes' Theorem."""
    
    def construct(self):
        title = Text("Bayes' Theorem").scale(0.5).to_edge(UP)
        self.play(Write(title))
        
        # Prior
        prior = Rectangle(width=4, height=2, color=BLUE, fill_opacity=0.3).shift(LEFT*3)
        prior_label = Text("Prior P(H)").scale(0.3).next_to(prior, UP)
        
        self.play(Create(prior), Write(prior_label))
        
        # Evidence
        evidence = Rectangle(width=4, height=2, color=GREEN, fill_opacity=0.3).shift(ORIGIN)
        evidence_label = Text("Evidence P(E|H)").scale(0.3).next_to(evidence, UP)
        
        self.play(Create(evidence), Write(evidence_label))
        
        # Arrow
        arrow = Arrow(evidence.get_right(), RIGHT*2.5, color=YELLOW)
        self.play(Create(arrow))
        
        # Posterior
        posterior = Rectangle(width=3, height=2, color=RED, fill_opacity=0.3).shift(RIGHT*4)
        posterior_label = Text("Posterior P(H|E)").scale(0.3).next_to(posterior, UP)
        
        self.play(Create(posterior), Write(posterior_label))
        
        # Bayes formula
        formula = MathTex(
            r"P(H|E) = \\frac{P(E|H) \\cdot P(H)}{P(E)}"
        ).scale(0.5).to_edge(DOWN)
        
        self.play(Write(formula))
        
        explanation = Text("Update beliefs with evidence").scale(0.35).shift(DOWN*2)
        self.play(Write(explanation))
        self.wait(2)
'''

CORRELATION = '''
from manim import *
import numpy as np

class CorrelationVisualization(Scene):
    """Visualize correlation between variables."""
    
    def construct(self):
        title = Text("Correlation").scale(0.5).to_edge(UP)
        self.play(Write(title))
        
        # Axes
        axes = Axes(
            x_range=[0, 10, 2],
            y_range=[0, 10, 2],
            x_length=4,
            y_length=4,
            axis_config={"include_tip": False}
        ).shift(LEFT*3)
        
        x_label = Text("X").scale(0.3).next_to(axes.x_axis, RIGHT)
        y_label = Text("Y").scale(0.3).next_to(axes.y_axis, UP)
        
        self.play(Create(axes), Write(x_label), Write(y_label))
        
        # Positive correlation points
        np.random.seed(42)
        points_pos = VGroup()
        for i in range(15):
            x = 1 + i * 0.5 + np.random.uniform(-0.5, 0.5)
            y = 1 + i * 0.5 + np.random.uniform(-0.5, 0.5)
            point = Dot(axes.c2p(x, y), color=BLUE, radius=0.05)
            points_pos.add(point)
        
        self.play(Create(points_pos))
        
        # Trend line
        trend = axes.plot(lambda x: x, x_range=[0, 10], color=RED)
        self.play(Create(trend))
        
        r_label = MathTex("r \\approx +1").scale(0.4).next_to(axes, DOWN)
        self.play(Write(r_label))
        
        # Show formula
        formula = MathTex(r"r = \\frac{\\sum(x_i - \\bar{x})(y_i - \\bar{y})}{\\sqrt{\\sum(x_i - \\bar{x})^2 \\sum(y_i - \\bar{y})^2}}").scale(0.35).shift(RIGHT*2)
        self.play(Write(formula))
        self.wait(2)
'''

REGRESSION = '''
from manim import *
import numpy as np

class LinearRegression(Scene):
    """Visualize linear regression."""
    
    def construct(self):
        title = Text("Linear Regression").scale(0.5).to_edge(UP)
        self.play(Write(title))
        
        # Axes
        axes = Axes(
            x_range=[0, 10, 2],
            y_range=[0, 10, 2],
            x_length=6,
            y_length=5,
            axis_config={"include_tip": False}
        ).shift(DOWN*0.5)
        
        self.play(Create(axes))
        
        # Data points
        np.random.seed(123)
        points = VGroup()
        data = []
        for i in range(10):
            x = i + 0.5
            y = 0.8 * x + 1 + np.random.uniform(-1, 1)
            data.append((x, y))
            point = Dot(axes.c2p(x, y), color=BLUE, radius=0.08)
            points.add(point)
        
        self.play(Create(points), run_time=2)
        
        # Best fit line
        line = axes.plot(lambda x: 0.8*x + 1, color=RED)
        line_label = MathTex("y = mx + b").scale(0.4).next_to(line, UR)
        
        self.play(Create(line), Write(line_label))
        
        # Show residuals
        residuals = VGroup()
        for x, y in data:
            pred_y = 0.8 * x + 1
            residual = DashedLine(axes.c2p(x, y), axes.c2p(x, pred_y), color=GREEN)
            residuals.add(residual)
        
        self.play(Create(residuals))
        
        # Minimize sum of squared residuals
        formula = MathTex(r"\\min \\sum (y_i - \\hat{y}_i)^2").scale(0.4).to_edge(DOWN)
        self.play(Write(formula))
        self.wait(2)
'''

STANDARD_DEVIATION = '''
from manim import *
import numpy as np

class StandardDeviation(Scene):
    """Visualize standard deviation."""
    
    def construct(self):
        title = Text("Standard Deviation").scale(0.5).to_edge(UP)
        self.play(Write(title))
        
        # Number line
        line = NumberLine(x_range=[0, 10, 1], length=8).shift(DOWN)
        self.play(Create(line))
        
        # Data points
        data = [3, 4, 5, 5, 5, 6, 7]
        points = VGroup()
        for val in data:
            dot = Dot(line.n2p(val) + UP*0.3, color=BLUE, radius=0.1)
            points.add(dot)
        
        self.play(Create(points))
        
        # Mean
        mean = np.mean(data)
        mean_line = DashedLine(line.n2p(mean) + UP*1.5, line.n2p(mean) + DOWN*0.5, color=RED)
        mean_label = MathTex(f"\\bar{{x}} = {mean:.1f}").scale(0.4).next_to(mean_line, UP)
        
        self.play(Create(mean_line), Write(mean_label))
        
        # Deviations
        deviations = VGroup()
        for val in data:
            dev = Arrow(line.n2p(mean) + UP*0.3, line.n2p(val) + UP*0.3, color=GREEN, buff=0.1)
            deviations.add(dev)
        
        self.play(Create(deviations))
        
        # Formula
        formula = MathTex(r"\\sigma = \\sqrt{\\frac{\\sum(x_i - \\bar{x})^2}{n}}").scale(0.5).shift(UP*2)
        self.play(Write(formula))
        
        std = np.std(data)
        result = MathTex(f"\\sigma \\approx {std:.2f}").scale(0.4).next_to(formula, DOWN)
        self.play(Write(result))
        self.wait(2)
'''

# Registry
STATS_TEMPLATES = {
    "normal_distribution": NORMAL_DISTRIBUTION,
    "central_limit": CENTRAL_LIMIT,
    "probability": PROBABILITY_BASICS,
    "bayes": BAYES_THEOREM,
    "correlation": CORRELATION,
    "regression": REGRESSION,
    "standard_deviation": STANDARD_DEVIATION,
}

def get_stats_template(name: str) -> str:
    return STATS_TEMPLATES.get(name, "")

def list_stats_templates() -> list:
    return list(STATS_TEMPLATES.keys())
