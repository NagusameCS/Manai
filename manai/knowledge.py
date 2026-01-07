"""
Knowledge Base - Manim documentation and reference for the agent
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DocEntry:
    """A documentation entry."""
    name: str
    category: str
    description: str
    usage: str
    parameters: str = ""
    example: str = ""


class ManimDocs:
    """Manim documentation database for agent lookup."""
    
    def __init__(self):
        self.entries: list[DocEntry] = self._build_docs()
    
    def _build_docs(self) -> list[DocEntry]:
        """Build the documentation database."""
        return [
            # Mobjects - Basic
            DocEntry("Circle", "mobjects", "A circle shape", 
                     "Circle(radius=1, color=WHITE)", "radius, color, fill_opacity"),
            DocEntry("Square", "mobjects", "A square shape",
                     "Square(side_length=2, color=BLUE)", "side_length, color"),
            DocEntry("Rectangle", "mobjects", "A rectangle shape",
                     "Rectangle(width=4, height=2)", "width, height, color"),
            DocEntry("Line", "mobjects", "A straight line",
                     "Line(start=LEFT, end=RIGHT)", "start, end, color"),
            DocEntry("Arrow", "mobjects", "An arrow with tip",
                     "Arrow(start=LEFT, end=RIGHT)", "start, end, buff, tip_length"),
            DocEntry("Dot", "mobjects", "A small dot",
                     "Dot(point=ORIGIN, radius=0.08)", "point, radius, color"),
            DocEntry("Polygon", "mobjects", "A polygon with vertices",
                     "Polygon(v1, v2, v3)", "vertices, color"),
            DocEntry("RegularPolygon", "mobjects", "Regular n-sided polygon",
                     "RegularPolygon(n=6)", "n, radius, color"),
            DocEntry("Triangle", "mobjects", "An equilateral triangle",
                     "Triangle()", "color, fill_opacity"),
            DocEntry("Ellipse", "mobjects", "An ellipse shape",
                     "Ellipse(width=4, height=2)", "width, height"),
            DocEntry("Arc", "mobjects", "A circular arc",
                     "Arc(radius=1, angle=PI/2)", "radius, start_angle, angle"),
            DocEntry("AnnularSector", "mobjects", "A sector of an annulus",
                     "AnnularSector(inner_radius=1, outer_radius=2)", "inner_radius, outer_radius, angle"),
            
            # Mobjects - Text
            DocEntry("Text", "mobjects", "Plain text",
                     "Text('Hello World', font_size=48)", "text, font_size, color"),
            DocEntry("Tex", "mobjects", "LaTeX text",
                     "Tex(r'$\\int f(x) dx$')", "tex_string, color"),
            DocEntry("MathTex", "mobjects", "LaTeX math mode",
                     "MathTex(r'x^2 + y^2 = r^2')", "tex_string, color"),
            DocEntry("Title", "mobjects", "A title at top of scene",
                     "Title('My Title')", "text, include_underline"),
            DocEntry("BulletedList", "mobjects", "Bulleted list",
                     "BulletedList('Item 1', 'Item 2')", "items, buff"),
            DocEntry("Paragraph", "mobjects", "Multi-line text",
                     "Paragraph('Line 1', 'Line 2')", "lines, alignment"),
            
            # Mobjects - Graphs
            DocEntry("Axes", "mobjects", "2D coordinate axes",
                     "Axes(x_range=[-5, 5], y_range=[-5, 5])", 
                     "x_range, y_range, x_length, y_length, axis_config"),
            DocEntry("NumberPlane", "mobjects", "2D coordinate plane with grid",
                     "NumberPlane(x_range=[-7, 7], y_range=[-4, 4])", 
                     "x_range, y_range, background_line_style"),
            DocEntry("NumberLine", "mobjects", "A number line",
                     "NumberLine(x_range=[-5, 5, 1])", "x_range, length, include_numbers"),
            DocEntry("BarChart", "mobjects", "A bar chart",
                     "BarChart(values=[1, 2, 3], bar_names=['A', 'B', 'C'])", 
                     "values, bar_names, y_range"),
            DocEntry("Table", "mobjects", "A table of values",
                     "Table([['a', 'b'], ['c', 'd']])", "table, row_labels, col_labels"),
            
            # Mobjects - 3D
            DocEntry("Sphere", "mobjects", "A 3D sphere",
                     "Sphere(radius=1)", "radius, resolution, color"),
            DocEntry("Cube", "mobjects", "A 3D cube",
                     "Cube(side_length=2)", "side_length, color"),
            DocEntry("Cylinder", "mobjects", "A 3D cylinder",
                     "Cylinder(radius=1, height=2)", "radius, height, direction"),
            DocEntry("Cone", "mobjects", "A 3D cone",
                     "Cone(base_radius=1, height=2)", "base_radius, height"),
            DocEntry("Torus", "mobjects", "A 3D torus",
                     "Torus(major_radius=2, minor_radius=0.5)", "major_radius, minor_radius"),
            DocEntry("Arrow3D", "mobjects", "A 3D arrow",
                     "Arrow3D(start=ORIGIN, end=OUT)", "start, end"),
            DocEntry("Surface", "mobjects", "A parametric 3D surface",
                     "Surface(lambda u, v: [u, v, u*v])", "func, u_range, v_range"),
            DocEntry("ThreeDAxes", "mobjects", "3D coordinate axes",
                     "ThreeDAxes(x_range=[-5, 5], y_range=[-5, 5], z_range=[-5, 5])",
                     "x_range, y_range, z_range"),
            
            # Animations - Basic
            DocEntry("Create", "animations", "Draw a mobject",
                     "Create(circle)", "mobject, run_time"),
            DocEntry("Write", "animations", "Write text/equations",
                     "Write(tex)", "mobject, run_time"),
            DocEntry("FadeIn", "animations", "Fade in a mobject",
                     "FadeIn(mobject, shift=UP)", "mobject, shift, scale"),
            DocEntry("FadeOut", "animations", "Fade out a mobject",
                     "FadeOut(mobject)", "mobject, shift"),
            DocEntry("GrowFromCenter", "animations", "Grow from center point",
                     "GrowFromCenter(mobject)", "mobject"),
            DocEntry("GrowFromPoint", "animations", "Grow from a point",
                     "GrowFromPoint(mobject, point)", "mobject, point"),
            DocEntry("SpinInFromNothing", "animations", "Spin in while growing",
                     "SpinInFromNothing(mobject)", "mobject, angle"),
            DocEntry("DrawBorderThenFill", "animations", "Draw border then fill",
                     "DrawBorderThenFill(mobject)", "mobject"),
            
            # Animations - Transform
            DocEntry("Transform", "animations", "Transform one mobject to another",
                     "Transform(mob1, mob2)", "mobject, target_mobject"),
            DocEntry("ReplacementTransform", "animations", "Transform and replace",
                     "ReplacementTransform(mob1, mob2)", "mobject, target_mobject"),
            DocEntry("TransformMatchingShapes", "animations", "Transform matching parts",
                     "TransformMatchingShapes(tex1, tex2)", "mobject, target_mobject"),
            DocEntry("MoveToTarget", "animations", "Move to generated target",
                     "MoveToTarget(mobject)", "mobject"),
            DocEntry("ApplyMethod", "animations", "Apply a method as animation",
                     "ApplyMethod(mob.shift, UP)", "method, args"),
            DocEntry("Rotate", "animations", "Rotate a mobject",
                     "Rotate(mobject, angle=PI)", "mobject, angle, axis"),
            DocEntry("ScaleInPlace", "animations", "Scale without moving",
                     "ScaleInPlace(mobject, 2)", "mobject, scale_factor"),
            
            # Animations - Indication
            DocEntry("Indicate", "animations", "Briefly highlight",
                     "Indicate(mobject)", "mobject, scale_factor, color"),
            DocEntry("Flash", "animations", "Flash effect",
                     "Flash(point)", "point, color, line_length"),
            DocEntry("ShowPassingFlash", "animations", "Passing flash along path",
                     "ShowPassingFlash(path)", "mobject, time_width"),
            DocEntry("Circumscribe", "animations", "Draw circle around",
                     "Circumscribe(mobject)", "mobject, shape"),
            DocEntry("Wiggle", "animations", "Wiggle effect",
                     "Wiggle(mobject)", "mobject, scale_value, rotation_angle"),
            DocEntry("FocusOn", "animations", "Focus on a point",
                     "FocusOn(point)", "point, color"),
            
            # Scenes
            DocEntry("Scene", "scenes", "Basic 2D scene",
                     "class MyScene(Scene)", "Background for 2D animations"),
            DocEntry("ThreeDScene", "scenes", "3D scene with camera",
                     "class MyScene(ThreeDScene)", "For 3D animations with camera control"),
            DocEntry("MovingCameraScene", "scenes", "Scene with moving camera",
                     "class MyScene(MovingCameraScene)", "For panning and zooming"),
            DocEntry("ZoomedScene", "scenes", "Scene with zoom capability",
                     "class MyScene(ZoomedScene)", "For zooming into details"),
            
            # Cameras
            DocEntry("ThreeDCamera", "cameras", "3D camera object",
                     "self.camera", "phi, theta, gamma, zoom"),
            DocEntry("set_camera_orientation", "cameras", "Set 3D camera angles",
                     "self.set_camera_orientation(phi=PI/3, theta=PI/4)",
                     "phi, theta, gamma, zoom"),
            DocEntry("begin_ambient_camera_rotation", "cameras", "Start camera rotation",
                     "self.begin_ambient_camera_rotation(rate=0.1)", "rate, about"),
            DocEntry("move_camera", "cameras", "Animate camera movement",
                     "self.move_camera(phi=PI/4)", "phi, theta, gamma, zoom"),
            
            # Utilities
            DocEntry("VGroup", "utilities", "Group of vector mobjects",
                     "VGroup(mob1, mob2, mob3)", "mobjects"),
            DocEntry("Group", "utilities", "Group of any mobjects",
                     "Group(mob1, mob2)", "mobjects"),
            DocEntry("always_redraw", "utilities", "Auto-updating mobject",
                     "always_redraw(lambda: Line(a.get_center(), b.get_center()))", 
                     "func"),
            DocEntry("ValueTracker", "utilities", "Animatable value",
                     "t = ValueTracker(0)", "value"),
            DocEntry("DecimalNumber", "utilities", "Animated number display",
                     "DecimalNumber(0, num_decimal_places=2)", "number, num_decimal_places"),
        ]
    
    def search(self, query: str, category: str = None) -> list[dict]:
        """Search the documentation."""
        query_lower = query.lower()
        results = []
        
        for entry in self.entries:
            # Filter by category if specified
            if category and entry.category != category:
                continue
            
            # Match against name and description
            if (query_lower in entry.name.lower() or 
                query_lower in entry.description.lower()):
                results.append({
                    "name": entry.name,
                    "category": entry.category,
                    "description": entry.description,
                    "usage": entry.usage,
                    "parameters": entry.parameters
                })
        
        return results[:10]  # Limit results
    
    def get_by_name(self, name: str) -> Optional[DocEntry]:
        """Get entry by exact name."""
        for entry in self.entries:
            if entry.name.lower() == name.lower():
                return entry
        return None
    
    def list_category(self, category: str) -> list[str]:
        """List all entries in a category."""
        return [e.name for e in self.entries if e.category == category]


# Singleton instance
manim_docs = ManimDocs()
