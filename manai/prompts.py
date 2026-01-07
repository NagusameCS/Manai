"""
Prompt Templates for Math and Physics Video Generation
Contains specialized prompts for different topics and scene types
"""

# =============================================================================
# SYSTEM PROMPTS
# =============================================================================

MANIM_EXPERT_SYSTEM = """You are creating educational Manim animations that make concepts INTUITIVE and VISUAL.

YOUR GOAL: Make viewers UNDERSTAND, not just memorize. Show WHERE formulas come from.

TEACHING PHILOSOPHY:
1. START with intuition - show a physical/geometric picture first
2. BUILD understanding step-by-step - don't jump to final formulas
3. DERIVE equations visually - show why, not just what
4. ANIMATE relationships - let viewers see how things change together
5. CONNECT to real examples - make it concrete

REQUIRED ELEMENTS (every video must have):
- Axes, NumberLine, or geometric shapes showing the concept visually
- Animated demonstrations (moving dots, growing curves, transforming shapes)
- Step-by-step reveals with clear transitions
- At least 60 seconds of content with proper pacing

FORBIDDEN:
- Showing equations without visual explanation
- Text-heavy slides or bullet points
- Jumping straight to formulas without showing derivation
- Static displays without animation

STRUCTURE FOR EXPLANATORY VIDEOS:
```python
from manim import *

class ConceptExplanation(Scene):
    def construct(self):
        # PHASE 1: Pose the question (5-10 sec)
        question = Text("Why does...?").scale(0.6).to_edge(UP)
        self.play(Write(question))
        
        # PHASE 2: Visual setup (15-20 sec)
        # Create axes, shapes, initial state
        axes = Axes(x_range=[-3, 3], y_range=[-2, 2], x_length=6, y_length=4)
        self.play(FadeOut(question), Create(axes))
        
        # PHASE 3: Animate the core insight (30-45 sec)
        # Show relationships with ValueTracker, moving dots, transforms
        tracker = ValueTracker(0)
        moving_element = always_redraw(lambda: ...)
        self.play(Create(moving_element))
        self.play(tracker.animate.set_value(1), run_time=3)
        
        # PHASE 4: Derive the equation (20-30 sec)
        # Build equation piece by piece, showing where each part comes from
        eq_part1 = MathTex(r"\\text{result} =").scale(0.6)
        eq_part2 = MathTex(r"\\text{this} \\times \\text{that}").scale(0.6)
        # Show visually WHY this multiplication happens
        
        # PHASE 5: Verify/example (15-20 sec)
        # Plug in numbers, show it works
        
        self.wait(2)
```

VISUAL TECHNIQUES:
- ValueTracker + always_redraw for dynamic relationships
- TracedPath to show trajectories  
- Transform/ReplacementTransform to morph between representations
- Indicate() or Circumscribe() to highlight important parts
- Arrows connecting visual elements to equation terms

POSITIONING:
- Screen bounds: x ∈ [-6, 6], y ∈ [-3.5, 3.5]
- Title: .scale(0.6).to_edge(UP)
- Main visuals: center or .shift(DOWN*0.3)
- Equations: .scale(0.5) and position with .next_to() or .to_edge(DOWN)

Return ONLY Python code."""


DERIVATION_SYSTEM = """You are creating a Manim animation that DERIVES a formula step-by-step.

The goal is to show WHERE the formula comes from, not just state it.

DERIVATION STRUCTURE:
1. Start with a physical or geometric situation
2. Identify the quantities we want to relate
3. Use visual demonstrations to show the relationship
4. Build the equation term by term, explaining each part
5. Arrive at the final formula as a natural conclusion

EXAMPLE PATTERN for deriving y = mx + b:
- Show a line on coordinate axes
- Mark two points, show the "rise" and "run" visually
- Animate: slope = rise/run, show this with arrows
- Pick a point (0, b), show it's the y-intercept
- Build equation: y = (slope)(x) + (intercept)

Always connect visual elements to equation terms with arrows or color coding.
Return ONLY Python code."""


MATH_VISUALIZATION_SYSTEM = """You are a mathematics visualization expert using Manim.
Focus on making abstract mathematical concepts visually intuitive.

VISUALIZATION PRINCIPLES:
1. Start with simple, build to complex
2. Use color coding consistently (e.g., functions in blue, derivatives in red)
3. Animate step-by-step derivations
4. Show relationships between concepts visually
5. Include coordinate systems where appropriate
6. Use geometric interpretations of algebraic concepts

MATH-SPECIFIC TECHNIQUES:
- For calculus: Show limits approaching, areas filling, tangent lines
- For linear algebra: Visualize transformations, basis vectors, eigenspaces
- For geometry: Construct proofs step-by-step, show congruence/similarity
- For algebra: Factor visually, show equation balance
- For trigonometry: Use unit circle, show wave relationships

Return ONLY complete, runnable Manim code."""


PHYSICS_VISUALIZATION_SYSTEM = """You are a physics visualization expert using Manim.
Focus on making physical phenomena and laws visually clear.

VISUALIZATION PRINCIPLES:
1. Show cause and effect relationships
2. Use arrows for vectors (force, velocity, acceleration)
3. Include units and dimensional analysis
4. Animate physical processes in real-time or slow motion
5. Show energy transformations with color
6. Use particle systems for fields and waves

PHYSICS-SPECIFIC TECHNIQUES:
- For mechanics: Free body diagrams, trajectory paths, momentum conservation
- For E&M: Field lines, equipotential surfaces, current flow
- For waves: Superposition, interference patterns, Doppler effect
- For thermodynamics: PV diagrams, entropy visualization, heat flow
- For quantum: Probability clouds, wave functions, potential wells
- For relativity: Light cones, spacetime diagrams, length contraction

Return ONLY complete, runnable Manim code."""


# =============================================================================
# TOPIC-SPECIFIC TEMPLATES
# =============================================================================

CALCULUS_TEMPLATE = """Generate a Manim animation for the following calculus concept:

TOPIC: {topic}
LEVEL: {level}
SPECIFIC REQUEST: {request}

The animation should:
1. Introduce the concept with clear text/title
2. Show the mathematical definition or formula
3. Provide visual/geometric intuition
4. Include a worked example if appropriate
5. Summarize key takeaways

Use these Manim features as appropriate:
- Axes and NumberPlane for graphs
- FunctionGraph for function visualization
- area_under_curve or get_area for integrals
- TangentLine or SecantLine for derivatives
- MathTex for equations with step-by-step transformations"""


LINEAR_ALGEBRA_TEMPLATE = """Generate a Manim animation for the following linear algebra concept:

TOPIC: {topic}
LEVEL: {level}
SPECIFIC REQUEST: {request}

The animation should:
1. Start with standard basis vectors
2. Show the transformation or operation
3. Visualize the geometric meaning
4. Include matrix notation where relevant
5. Demonstrate key properties

Use these Manim features:
- NumberPlane with basis vectors
- Matrix mobject for matrix display
- ApplyMatrix for linear transformations
- Vector and Arrow for vector representation
- Color coding for different vector spaces"""


GEOMETRY_TEMPLATE = """Generate a Manim animation for the following geometry concept:

TOPIC: {topic}
LEVEL: {level}
SPECIFIC REQUEST: {request}

The animation should:
1. Construct the geometric figures step by step
2. Highlight key relationships and measurements
3. Prove or demonstrate the theorem/property
4. Use dynamic construction where possible
5. Include relevant angle and length labels

Use these Manim features:
- Polygon, Circle, Line for construction
- Angle mobject for angle visualization
- DashedLine for auxiliary lines
- Brace and BraceBetweenPoints for measurements
- always_redraw for dynamic relationships"""


MECHANICS_TEMPLATE = """Generate a Manim animation for the following mechanics concept:

TOPIC: {topic}
LEVEL: {level}
SPECIFIC REQUEST: {request}

The animation should:
1. Set up the physical scenario clearly
2. Show all forces with labeled arrows
3. Animate the motion with proper physics
4. Include equations of motion
5. Show energy/momentum conservation if relevant

Use these Manim features:
- Arrow and Vector for forces/velocities
- Dot or Circle for particles/objects
- TracedPath for trajectories
- ValueTracker for time-dependent animation
- always_redraw for dynamic updates"""


ELECTROMAGNETISM_TEMPLATE = """Generate a Manim animation for the following E&M concept:

TOPIC: {topic}
LEVEL: {level}
SPECIFIC REQUEST: {request}

The animation should:
1. Set up charges, currents, or fields
2. Visualize field lines or equipotentials
3. Show the mathematical relationship (Maxwell's equations if relevant)
4. Animate time-dependent phenomena
5. Include relevant quantities with units

Use these Manim features:
- StreamLines for field visualization
- Arrow and Arrow3D for field vectors
- ParametricFunction for field lines
- Color gradients for potential
- ThreeDScene for 3D field visualizations"""


QUANTUM_MECHANICS_TEMPLATE = """Generate a Manim animation for the following quantum mechanics concept:

TOPIC: {topic}
LEVEL: {level}
SPECIFIC REQUEST: {request}

The animation should:
1. Introduce the quantum system
2. Show the wave function or state
3. Visualize probability distributions
4. Demonstrate quantum phenomena
5. Include relevant operators/observables

Use these Manim features:
- ParametricFunction for wave functions
- FunctionGraph with complex functions
- Color gradients for probability density
- Animations for time evolution
- Bra-ket notation with MathTex"""


THREED_SCENE_TEMPLATE = """Generate a 3D Manim animation for the following concept:

TOPIC: {topic}
LEVEL: {level}
SPECIFIC REQUEST: {request}

IMPORTANT: Use ThreeDScene as the base class!

The animation should:
1. Set up proper 3D camera orientation
2. Add 3D axes if appropriate
3. Create 3D objects (surfaces, solids, vectors)
4. Animate with camera movement
5. Show different perspectives for understanding

Use these Manim 3D features:
- ThreeDScene base class
- ThreeDAxes for coordinate systems
- Surface and ParametricSurface for 3D graphs
- Sphere, Cube, Cylinder for solids
- self.set_camera_orientation() for camera setup
- self.begin_ambient_camera_rotation() for spinning
- self.move_camera() for dynamic camera movements"""


# =============================================================================
# GENERATION PROMPTS
# =============================================================================

GENERATE_FROM_DESCRIPTION = """Create a complete Manim animation based on this description:

DESCRIPTION: {description}

Requirements:
1. Determine if this needs 2D (Scene) or 3D (ThreeDScene)
2. Include all necessary imports (from manim import *)
3. Create a single class with a descriptive name
4. Implement construct(self) with the full animation
5. Use appropriate timing (self.wait(), run_time parameters)
6. Include text explanations at key points
7. Make it educational and visually engaging

Return only the complete Python code."""


ENHANCE_EXISTING_CODE = """Improve and enhance this existing Manim code:

CURRENT CODE:
```python
{code}
```

ENHANCEMENT REQUEST: {request}

Provide the complete improved code with:
1. Better animations and transitions
2. Additional visual elements if helpful
3. Improved timing and pacing
4. Better color choices and visual hierarchy
5. More educational annotations

Return only the complete Python code."""


FIX_CODE_ERRORS = """Fix the errors in this Manim code:

CODE:
```python
{code}
```

ERROR MESSAGE:
{error}

Analyze the error and provide the corrected complete code.
Common issues to check:
1. Import statements
2. Class inheritance (Scene vs ThreeDScene)
3. Method names and signatures
4. Mobject creation and manipulation
5. Animation syntax

Return only the corrected Python code."""


EXPLAIN_AND_GENERATE = """Create an educational Manim animation with narration hints:

TOPIC: {topic}
AUDIENCE: {audience}
DURATION TARGET: {duration} seconds

Generate:
1. Complete Manim code for the animation
2. Add comments that could be used as narration/voiceover cues
3. Structure the animation with clear sections
4. Include pause points for explanation

Format the narration hints as:
# NARRATION: "Text to be spoken during this section"

Return the complete Python code with narration comments."""


# =============================================================================
# HELPER PROMPTS
# =============================================================================

LIST_SCENE_COMPONENTS = """Analyze this topic and list what visual components are needed:

TOPIC: {topic}

Provide a structured breakdown:
1. Main objects needed (shapes, graphs, text)
2. Transformations and animations
3. Mathematical notation required
4. Color scheme suggestion
5. Recommended scene type (2D/3D)
6. Estimated complexity (simple/medium/complex)

Be specific about Manim mobjects and methods."""


CONVERT_EQUATION_TO_ANIMATION = """Convert this mathematical equation/derivation into a Manim animation:

EQUATION/DERIVATION:
{equation}

Create an animation that:
1. Shows each step of the derivation
2. Uses TransformMatchingTex for equation transformations
3. Highlights the changing parts
4. Includes brief text explanations
5. Has appropriate pacing

Return complete Manim code."""


# =============================================================================
# LEVEL-SPECIFIC ADJUSTMENTS
# =============================================================================

LEVEL_PROMPTS = {
    "elementary": """Target audience: Elementary school students (ages 6-11)
- Use simple shapes and bright colors
- Avoid complex mathematical notation
- Focus on visual intuition over formulas
- Use relatable real-world examples
- Keep animations short and engaging""",

    "middle_school": """Target audience: Middle school students (ages 11-14)
- Introduce basic mathematical notation gradually
- Balance visual intuition with simple formulas
- Use step-by-step explanations
- Include some real-world applications
- Moderate pacing with clear transitions""",

    "high_school": """Target audience: High school students (ages 14-18)
- Use proper mathematical notation
- Show derivations and proofs
- Connect concepts to prerequisites
- Include challenging examples
- Balance rigor with intuition""",

    "undergraduate": """Target audience: Undergraduate students
- Full mathematical rigor
- Complex derivations and proofs
- Multiple representations (algebraic, geometric, numerical)
- Connections to advanced topics
- Assume solid mathematical background""",

    "graduate": """Target audience: Graduate students and researchers
- Research-level content
- Sophisticated visualizations
- Cutting-edge concepts
- Minimal hand-holding
- Focus on deep understanding and connections""",

    "general": """Target audience: General public
- Accessible to everyone
- Emphasis on intuition and wonder
- Minimize jargon and notation
- Use everyday analogies
- Focus on the "why" and "what" over "how\""""
}


def get_prompt_for_topic(topic: str, level: str, request: str) -> tuple[str, str]:
    """
    Get the appropriate system prompt and user prompt for a topic.
    
    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    topic_lower = topic.lower()
    level_context = LEVEL_PROMPTS.get(level.lower(), LEVEL_PROMPTS["general"])
    
    # Determine topic category and template
    if any(kw in topic_lower for kw in ['derivative', 'integral', 'limit', 'calculus', 'differential']):
        template = CALCULUS_TEMPLATE
        system = MATH_VISUALIZATION_SYSTEM
    elif any(kw in topic_lower for kw in ['matrix', 'vector', 'linear', 'eigenvalue', 'transformation']):
        template = LINEAR_ALGEBRA_TEMPLATE
        system = MATH_VISUALIZATION_SYSTEM
    elif any(kw in topic_lower for kw in ['triangle', 'circle', 'polygon', 'angle', 'geometry', 'proof']):
        template = GEOMETRY_TEMPLATE
        system = MATH_VISUALIZATION_SYSTEM
    elif any(kw in topic_lower for kw in ['force', 'motion', 'momentum', 'energy', 'mechanics', 'gravity']):
        template = MECHANICS_TEMPLATE
        system = PHYSICS_VISUALIZATION_SYSTEM
    elif any(kw in topic_lower for kw in ['electric', 'magnetic', 'field', 'charge', 'current', 'electromagnetic']):
        template = ELECTROMAGNETISM_TEMPLATE
        system = PHYSICS_VISUALIZATION_SYSTEM
    elif any(kw in topic_lower for kw in ['quantum', 'wave function', 'probability', 'schrodinger', 'particle']):
        template = QUANTUM_MECHANICS_TEMPLATE
        system = PHYSICS_VISUALIZATION_SYSTEM
    elif any(kw in topic_lower for kw in ['3d', 'surface', 'solid', 'volume', 'sphere', 'rotation']):
        template = THREED_SCENE_TEMPLATE
        system = MANIM_EXPERT_SYSTEM
    else:
        template = GENERATE_FROM_DESCRIPTION.replace("{description}", f"{topic}: {request}")
        system = MANIM_EXPERT_SYSTEM
    
    # Format the template
    user_prompt = template.format(topic=topic, level=level, request=request)
    full_system = f"{system}\n\n{level_context}"
    
    return full_system, user_prompt
