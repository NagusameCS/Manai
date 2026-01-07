"""
Custom Transitions - Original animation transitions for Manim
"""

from manim import *
import numpy as np


# =============================================================================
# FADE TRANSITIONS
# =============================================================================

class FadeToGray(Animation):
    """Fade mobject to grayscale."""
    
    def __init__(self, mobject, **kwargs):
        self.original_colors = {}
        super().__init__(mobject, **kwargs)
    
    def begin(self):
        for sm in self.mobject.get_family():
            self.original_colors[id(sm)] = sm.get_color()
        super().begin()
    
    def interpolate_mobject(self, alpha):
        for sm in self.mobject.get_family():
            orig = self.original_colors.get(id(sm), WHITE)
            # Interpolate to gray
            gray = interpolate_color(orig, GRAY_B, alpha)
            sm.set_color(gray)


class FadeFromGray(Animation):
    """Fade mobject from grayscale to color."""
    
    def __init__(self, mobject, target_color=WHITE, **kwargs):
        self.target_color = target_color
        super().__init__(mobject, **kwargs)
    
    def interpolate_mobject(self, alpha):
        color = interpolate_color(GRAY_B, self.target_color, alpha)
        self.mobject.set_color(color)


class CrossFadeTransform(Animation):
    """Cross-fade between two mobjects."""
    
    def __init__(self, mobject, target, **kwargs):
        self.target = target.copy()
        super().__init__(mobject, **kwargs)
    
    def begin(self):
        self.target.set_opacity(0)
        self.mobject.add(self.target)
        super().begin()
    
    def interpolate_mobject(self, alpha):
        # Original fades out
        for sm in self.mobject.submobjects[:-1]:
            sm.set_opacity(1 - alpha)
        # Target fades in
        self.target.set_opacity(alpha)


# =============================================================================
# GEOMETRIC TRANSITIONS
# =============================================================================

class CircularReveal(Animation):
    """Reveal mobject with expanding circle."""
    
    def __init__(self, mobject, center=ORIGIN, **kwargs):
        self.center = center
        self.max_radius = max(
            np.linalg.norm(point - center) 
            for point in mobject.get_all_points()
        ) + 0.5
        super().__init__(mobject, **kwargs)
    
    def begin(self):
        self.mobject.set_opacity(0)
        super().begin()
    
    def interpolate_mobject(self, alpha):
        radius = self.max_radius * alpha
        
        for point, orig_point in zip(
            self.mobject.get_all_points(),
            self.starting_mobject.get_all_points()
        ):
            dist = np.linalg.norm(orig_point - self.center)
            if dist <= radius:
                opacity = min(1, (radius - dist) / 0.5)
            else:
                opacity = 0
        
        self.mobject.set_opacity(alpha)


class SpiralIn(Animation):
    """Spiral mobject in from outside."""
    
    def __init__(self, mobject, n_turns=2, **kwargs):
        self.n_turns = n_turns
        self.target_positions = {}
        super().__init__(mobject, **kwargs)
    
    def begin(self):
        for sm in self.mobject.get_family():
            self.target_positions[id(sm)] = sm.get_center()
        super().begin()
    
    def interpolate_mobject(self, alpha):
        angle = (1 - alpha) * self.n_turns * 2 * PI
        radius = (1 - alpha) * 5
        
        for sm in self.mobject.get_family():
            target = self.target_positions.get(id(sm), ORIGIN)
            offset = radius * np.array([np.cos(angle), np.sin(angle), 0])
            sm.move_to(target + offset)
            sm.set_opacity(alpha)


class SpiralOut(Animation):
    """Spiral mobject out to infinity."""
    
    def __init__(self, mobject, n_turns=2, **kwargs):
        self.n_turns = n_turns
        self.start_positions = {}
        super().__init__(mobject, **kwargs)
    
    def begin(self):
        for sm in self.mobject.get_family():
            self.start_positions[id(sm)] = sm.get_center()
        super().begin()
    
    def interpolate_mobject(self, alpha):
        angle = alpha * self.n_turns * 2 * PI
        radius = alpha * 5
        
        for sm in self.mobject.get_family():
            start = self.start_positions.get(id(sm), ORIGIN)
            offset = radius * np.array([np.cos(angle), np.sin(angle), 0])
            sm.move_to(start + offset)
            sm.set_opacity(1 - alpha)


class ShatterEffect(Animation):
    """Shatter mobject into pieces."""
    
    def __init__(self, mobject, n_pieces=20, **kwargs):
        self.n_pieces = n_pieces
        self.velocities = [
            np.array([
                np.random.uniform(-3, 3),
                np.random.uniform(-3, 3),
                0
            ])
            for _ in range(n_pieces)
        ]
        self.rotations = [
            np.random.uniform(-2*PI, 2*PI)
            for _ in range(n_pieces)
        ]
        super().__init__(mobject, **kwargs)
    
    def interpolate_mobject(self, alpha):
        # Simple shatter: just fade and scatter
        self.mobject.set_opacity(1 - alpha)
        
        for i, sm in enumerate(self.mobject.submobjects[:self.n_pieces]):
            if i < len(self.velocities):
                offset = self.velocities[i] * alpha
                sm.shift(offset * 0.1)
                sm.rotate(self.rotations[i] * alpha * 0.1)


# =============================================================================
# WAVE TRANSITIONS
# =============================================================================

class WaveReveal(Animation):
    """Reveal with wave effect from left to right."""
    
    def __init__(self, mobject, wave_width=2, **kwargs):
        self.wave_width = wave_width
        self.x_min = mobject.get_left()[0]
        self.x_max = mobject.get_right()[0]
        super().__init__(mobject, **kwargs)
    
    def begin(self):
        self.mobject.set_opacity(0)
        super().begin()
    
    def interpolate_mobject(self, alpha):
        wave_center = self.x_min + alpha * (self.x_max - self.x_min + self.wave_width)
        
        for point in self.mobject.get_all_points():
            x = point[0]
            if x < wave_center - self.wave_width:
                opacity = 1
            elif x > wave_center:
                opacity = 0
            else:
                opacity = (wave_center - x) / self.wave_width
        
        self.mobject.set_opacity(alpha)


class RippleIn(Animation):
    """Ripple effect revealing mobject."""
    
    def __init__(self, mobject, center=ORIGIN, n_ripples=3, **kwargs):
        self.center = center
        self.n_ripples = n_ripples
        super().__init__(mobject, **kwargs)
    
    def begin(self):
        self.mobject.set_opacity(0)
        super().begin()
    
    def interpolate_mobject(self, alpha):
        # Create ripple pattern
        max_dist = 5
        wave = np.sin(alpha * self.n_ripples * 2 * PI) * 0.5 + 0.5
        
        self.mobject.set_opacity(alpha * wave + alpha * 0.5)


# =============================================================================
# TYPEWRITER TRANSITIONS
# =============================================================================

class TypewriterWrite(Animation):
    """Typewriter effect for text."""
    
    def __init__(self, mobject, cursor=True, **kwargs):
        self.cursor = cursor
        super().__init__(mobject, **kwargs)
    
    def interpolate_mobject(self, alpha):
        # Reveal characters one by one
        n_chars = len(self.mobject.submobjects)
        chars_shown = int(alpha * n_chars)
        
        for i, char in enumerate(self.mobject.submobjects):
            if i < chars_shown:
                char.set_opacity(1)
            elif i == chars_shown:
                # Blinking cursor effect
                char.set_opacity(0.5 + 0.5 * np.sin(alpha * 20))
            else:
                char.set_opacity(0)


class UnwriteReverse(Animation):
    """Reverse typewriter - erase from end."""
    
    def interpolate_mobject(self, alpha):
        n_chars = len(self.mobject.submobjects)
        chars_hidden = int(alpha * n_chars)
        
        for i, char in enumerate(self.mobject.submobjects):
            if i >= n_chars - chars_hidden:
                char.set_opacity(0)
            else:
                char.set_opacity(1)


# =============================================================================
# MORPH TRANSITIONS
# =============================================================================

class MorphToCircle(Animation):
    """Morph any shape to a circle."""
    
    def __init__(self, mobject, **kwargs):
        self.target_circle = Circle(
            radius=max(mobject.width, mobject.height) / 2
        ).move_to(mobject.get_center())
        super().__init__(mobject, **kwargs)
    
    def interpolate_mobject(self, alpha):
        # Interpolate points toward circle
        circle_points = self.target_circle.get_all_points()
        orig_points = self.starting_mobject.get_all_points()
        
        if len(circle_points) != len(orig_points):
            return
        
        new_points = orig_points + alpha * (circle_points - orig_points)
        self.mobject.set_points(new_points)


class ElasticScale(Animation):
    """Scale with elastic/bounce effect."""
    
    def __init__(self, mobject, scale_factor=2, **kwargs):
        self.scale_factor = scale_factor
        super().__init__(mobject, **kwargs)
    
    def interpolate_mobject(self, alpha):
        # Elastic easing
        if alpha < 1:
            elastic = pow(2, -10 * alpha) * np.sin((alpha - 0.1) * 5 * PI) + 1
        else:
            elastic = 1
        
        current_scale = 1 + (self.scale_factor - 1) * elastic
        self.mobject.become(
            self.starting_mobject.copy().scale(current_scale)
        )


class BounceIn(Animation):
    """Bounce in from scale 0."""
    
    def interpolate_mobject(self, alpha):
        # Bounce easing
        if alpha < 4/11:
            scale = 7.5625 * alpha * alpha
        elif alpha < 8/11:
            alpha -= 6/11
            scale = 7.5625 * alpha * alpha + 0.75
        elif alpha < 10/11:
            alpha -= 9/11
            scale = 7.5625 * alpha * alpha + 0.9375
        else:
            alpha -= 21/22
            scale = 7.5625 * alpha * alpha + 0.984375
        
        self.mobject.become(
            self.starting_mobject.copy().scale(scale)
        )
        self.mobject.set_opacity(scale)


# =============================================================================
# GLITCH TRANSITIONS
# =============================================================================

class GlitchEffect(Animation):
    """Digital glitch effect."""
    
    def __init__(self, mobject, intensity=0.3, **kwargs):
        self.intensity = intensity
        super().__init__(mobject, **kwargs)
    
    def interpolate_mobject(self, alpha):
        # Random offset glitches
        if np.random.random() < self.intensity * (1 - alpha):
            offset = np.array([
                np.random.uniform(-0.2, 0.2),
                np.random.uniform(-0.1, 0.1),
                0
            ])
            self.mobject.shift(offset)
            # Color glitch
            if np.random.random() < 0.5:
                self.mobject.set_color(
                    [GRAY_A, GRAY_B, GRAY_C][np.random.randint(3)]
                )
        else:
            self.mobject.become(self.starting_mobject.copy())
        
        self.mobject.set_opacity(alpha)


class ScanlineReveal(Animation):
    """CRT scanline reveal effect."""
    
    def __init__(self, mobject, n_lines=20, **kwargs):
        self.n_lines = n_lines
        super().__init__(mobject, **kwargs)
    
    def begin(self):
        self.mobject.set_opacity(0)
        super().begin()
    
    def interpolate_mobject(self, alpha):
        # Reveal in horizontal scanlines
        lines_shown = int(alpha * self.n_lines)
        
        # Simple fade for now
        self.mobject.set_opacity(alpha)


# =============================================================================
# 3D TRANSITIONS
# =============================================================================

class FlipIn3D(Animation):
    """3D flip reveal (works best in ThreeDScene)."""
    
    def __init__(self, mobject, axis=UP, **kwargs):
        self.axis = axis
        super().__init__(mobject, **kwargs)
    
    def interpolate_mobject(self, alpha):
        angle = (1 - alpha) * PI / 2
        self.mobject.become(
            self.starting_mobject.copy().rotate(angle, axis=self.axis)
        )
        self.mobject.set_opacity(alpha)


class CubeRotateIn(Animation):
    """Rotate in like a 3D cube face."""
    
    def __init__(self, mobject, direction=RIGHT, **kwargs):
        self.direction = direction
        super().__init__(mobject, **kwargs)
    
    def interpolate_mobject(self, alpha):
        angle = (1 - alpha) * PI / 2
        
        # Rotate around edge
        axis = np.cross(self.direction, OUT)
        pivot = self.mobject.get_edge_center(-self.direction)
        
        self.mobject.become(self.starting_mobject.copy())
        self.mobject.rotate(angle, axis=axis, about_point=pivot)
        self.mobject.set_opacity(alpha)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_transition(name: str, mobject, **kwargs) -> Animation:
    """
    Create a transition animation by name.
    
    Args:
        name: Name of the transition
        mobject: Mobject to animate
        **kwargs: Additional arguments
        
    Returns:
        Animation object
    """
    transitions = {
        "fade_to_gray": FadeToGray,
        "fade_from_gray": FadeFromGray,
        "spiral_in": SpiralIn,
        "spiral_out": SpiralOut,
        "shatter": ShatterEffect,
        "wave_reveal": WaveReveal,
        "ripple": RippleIn,
        "typewriter": TypewriterWrite,
        "unwrite": UnwriteReverse,
        "elastic_scale": ElasticScale,
        "bounce_in": BounceIn,
        "glitch": GlitchEffect,
        "scanline": ScanlineReveal,
        "flip_3d": FlipIn3D,
        "cube_rotate": CubeRotateIn,
    }
    
    if name in transitions:
        return transitions[name](mobject, **kwargs)
    else:
        raise ValueError(f"Unknown transition: {name}")


def list_transitions() -> list[str]:
    """List all available transitions."""
    return [
        "fade_to_gray",
        "fade_from_gray",
        "spiral_in",
        "spiral_out",
        "shatter",
        "wave_reveal",
        "ripple",
        "typewriter",
        "unwrite",
        "elastic_scale",
        "bounce_in",
        "glitch",
        "scanline",
        "flip_3d",
        "cube_rotate",
    ]
