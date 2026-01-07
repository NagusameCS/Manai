"""
Agent System - Thinking, tools, and reasoning for intelligent video generation
"""

import json
import re
from typing import Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

from manai.ui import ui, console
from manai.math_engine import math_engine, MathResult
from manai.ollama_client import OllamaClient


# =============================================================================
# TOOL SYSTEM
# =============================================================================

@dataclass
class ToolResult:
    """Result from tool execution."""
    success: bool
    output: Any
    error: Optional[str] = None


class Tool(ABC):
    """Base class for agent tools."""
    
    name: str
    description: str
    parameters: dict
    
    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters."""
        pass
    
    def to_schema(self) -> dict:
        """Convert to schema for LLM."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }


class CalculateTool(Tool):
    """Tool for mathematical calculations."""
    
    name = "calculate"
    description = "Perform mathematical calculations, simplify expressions, or evaluate formulas"
    parameters = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Mathematical expression to evaluate (e.g., 'sin(pi/4)', 'x^2 + 2x + 1')"
            },
            "operation": {
                "type": "string",
                "enum": ["evaluate", "simplify", "expand", "factor"],
                "description": "Operation to perform"
            },
            "values": {
                "type": "object",
                "description": "Variable values for evaluation (e.g., {'x': 2})"
            }
        },
        "required": ["expression"]
    }
    
    def execute(self, expression: str, operation: str = "simplify", values: dict = None) -> ToolResult:
        try:
            if operation == "evaluate" and values:
                result = math_engine.evaluate(expression, values)
            else:
                result = math_engine.simplify_expr(expression)
            
            if result.success:
                return ToolResult(True, {
                    "result": str(result.result),
                    "latex": result.latex,
                })
            else:
                return ToolResult(False, None, result.error)
        except Exception as e:
            return ToolResult(False, None, str(e))


class DerivativeTool(Tool):
    """Tool for computing derivatives."""
    
    name = "derivative"
    description = "Compute the derivative of a function"
    parameters = {
        "type": "object",
        "properties": {
            "function": {
                "type": "string",
                "description": "Function to differentiate (e.g., 'x^3 + sin(x)')"
            },
            "variable": {
                "type": "string",
                "description": "Variable to differentiate with respect to",
                "default": "x"
            },
            "order": {
                "type": "integer",
                "description": "Order of derivative (1 for first, 2 for second, etc.)",
                "default": 1
            }
        },
        "required": ["function"]
    }
    
    def execute(self, function: str, variable: str = "x", order: int = 1) -> ToolResult:
        result = math_engine.derivative(function, variable, order)
        
        if result.success:
            return ToolResult(True, {
                "derivative": str(result.result),
                "latex": result.latex,
                "steps": result.steps
            })
        return ToolResult(False, None, result.error)


class IntegralTool(Tool):
    """Tool for computing integrals."""
    
    name = "integral"
    description = "Compute the integral of a function"
    parameters = {
        "type": "object",
        "properties": {
            "function": {
                "type": "string",
                "description": "Function to integrate"
            },
            "variable": {
                "type": "string",
                "description": "Variable to integrate with respect to",
                "default": "x"
            },
            "lower": {
                "type": "string",
                "description": "Lower limit for definite integral"
            },
            "upper": {
                "type": "string",
                "description": "Upper limit for definite integral"
            }
        },
        "required": ["function"]
    }
    
    def execute(
        self, 
        function: str, 
        variable: str = "x", 
        lower: str = None, 
        upper: str = None
    ) -> ToolResult:
        limits = (lower, upper) if lower and upper else None
        result = math_engine.integral(function, variable, limits)
        
        if result.success:
            return ToolResult(True, {
                "integral": str(result.result),
                "latex": result.latex,
                "steps": result.steps
            })
        return ToolResult(False, None, result.error)


class VerifyMathTool(Tool):
    """Tool for verifying mathematical claims."""
    
    name = "verify_math"
    description = "Verify if a mathematical statement or equation is correct"
    parameters = {
        "type": "object",
        "properties": {
            "expression1": {
                "type": "string",
                "description": "First expression or the left side of equation"
            },
            "expression2": {
                "type": "string",
                "description": "Second expression or the right side of equation"
            },
            "claim_type": {
                "type": "string",
                "enum": ["equality", "derivative", "integral"],
                "description": "Type of claim to verify"
            }
        },
        "required": ["expression1", "expression2"]
    }
    
    def execute(
        self,
        expression1: str,
        expression2: str,
        claim_type: str = "equality"
    ) -> ToolResult:
        if claim_type == "derivative":
            result = math_engine.verify_derivative(expression1, expression2)
        elif claim_type == "integral":
            result = math_engine.verify_integral(expression1, expression2)
        else:
            result = math_engine.verify_equality(expression1, expression2)
        
        if result.success:
            return ToolResult(True, {
                "verified": result.result,
                "verification": result.verification.value,
                "details": result.latex
            })
        return ToolResult(False, None, result.error)


class SolveEquationTool(Tool):
    """Tool for solving equations."""
    
    name = "solve_equation"
    description = "Solve algebraic equations or systems of equations"
    parameters = {
        "type": "object",
        "properties": {
            "equations": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of equations to solve (e.g., ['x^2 - 4 = 0', '2x + y = 5'])"
            },
            "variables": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Variables to solve for (e.g., ['x', 'y'])"
            }
        },
        "required": ["equations", "variables"]
    }
    
    def execute(self, equations: list, variables: list) -> ToolResult:
        result = math_engine.solve_system(equations, variables)
        
        if result.success:
            return ToolResult(True, {
                "solutions": str(result.result),
                "latex": result.latex
            })
        return ToolResult(False, None, result.error)


class MatrixTool(Tool):
    """Tool for matrix operations."""
    
    name = "matrix_operation"
    description = "Perform matrix operations like determinant, inverse, eigenvalues"
    parameters = {
        "type": "object",
        "properties": {
            "matrix": {
                "type": "array",
                "items": {"type": "array", "items": {"type": "number"}},
                "description": "Matrix as nested array (e.g., [[1, 2], [3, 4]])"
            },
            "operation": {
                "type": "string",
                "enum": ["determinant", "inverse", "transpose", "eigenvalues", "eigenvectors", "rref", "rank"],
                "description": "Matrix operation to perform"
            }
        },
        "required": ["matrix", "operation"]
    }
    
    def execute(self, matrix: list, operation: str) -> ToolResult:
        result = math_engine.matrix_ops(matrix, operation)
        
        if result.success:
            return ToolResult(True, {
                "result": str(result.result),
                "latex": result.latex
            })
        return ToolResult(False, None, result.error)


class LookupDocsTool(Tool):
    """Tool for looking up Manim documentation."""
    
    name = "lookup_docs"
    description = "Look up Manim documentation for specific classes, methods, or concepts"
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to look up (e.g., 'ThreeDScene', 'Axes', 'FadeIn')"
            },
            "category": {
                "type": "string",
                "enum": ["mobjects", "animations", "cameras", "scenes", "utilities"],
                "description": "Category to search in"
            }
        },
        "required": ["query"]
    }
    
    def execute(self, query: str, category: str = None) -> ToolResult:
        # Import docs database
        from manai.knowledge import manim_docs
        
        results = manim_docs.search(query, category)
        
        if results:
            return ToolResult(True, {
                "results": results,
                "count": len(results)
            })
        return ToolResult(True, {"results": [], "count": 0})


class PhysicsFormulaTool(Tool):
    """Tool for physics calculations."""
    
    name = "physics_formula"
    description = "Calculate physics quantities using standard formulas"
    parameters = {
        "type": "object",
        "properties": {
            "formula_type": {
                "type": "string",
                "enum": ["kinematics", "force", "energy", "momentum", "waves", "electric_field"],
                "description": "Type of physics calculation"
            },
            "known_values": {
                "type": "object",
                "description": "Known values (e.g., {'v0': 10, 'a': 9.8, 't': 2})"
            },
            "solve_for": {
                "type": "string",
                "description": "Variable to solve for"
            }
        },
        "required": ["formula_type", "known_values", "solve_for"]
    }
    
    def execute(
        self,
        formula_type: str,
        known_values: dict,
        solve_for: str
    ) -> ToolResult:
        if formula_type == "kinematics":
            result = math_engine.kinematics(known_values, solve_for)
            if result.success:
                return ToolResult(True, {
                    "result": str(result.result),
                    "latex": result.latex
                })
            return ToolResult(False, None, result.error)
        
        return ToolResult(False, None, f"Formula type '{formula_type}' not yet implemented")


# =============================================================================
# AGENT THINKING
# =============================================================================

class ThinkingMode(Enum):
    """Agent thinking modes."""
    QUICK = "quick"          # Minimal thinking
    STANDARD = "standard"    # Normal reasoning
    DEEP = "deep"            # Extended analysis
    VERIFY = "verify"        # Math verification focus


@dataclass
class Thought:
    """Represents a single thought/reasoning step."""
    step: int
    content: str
    type: str = "reasoning"  # reasoning, observation, plan, conclusion


@dataclass
class ThinkingResult:
    """Result of agent thinking process."""
    thoughts: list[Thought]
    conclusion: str
    plan: list[str]
    tool_calls: list[dict]
    confidence: float = 0.0


class AgentThinking:
    """
    Agent thinking and reasoning system.
    
    Provides structured reasoning capabilities for the agent.
    """
    
    def __init__(self, llm: OllamaClient, mode: ThinkingMode = ThinkingMode.STANDARD):
        self.llm = llm
        self.mode = mode
        self.thoughts: list[Thought] = []
    
    def think(
        self,
        query: str,
        context: str = "",
        available_tools: list[Tool] = None
    ) -> ThinkingResult:
        """
        Process a query with structured thinking.
        
        Args:
            query: The user's request
            context: Additional context
            available_tools: Tools the agent can use
            
        Returns:
            ThinkingResult with reasoning and plan
        """
        self.thoughts = []
        
        # Build thinking prompt based on mode
        thinking_prompt = self._build_thinking_prompt(query, context, available_tools)
        
        # Get structured response from LLM
        if self.mode == ThinkingMode.QUICK:
            return self._quick_think(query, thinking_prompt)
        elif self.mode == ThinkingMode.DEEP:
            return self._deep_think(query, thinking_prompt)
        elif self.mode == ThinkingMode.VERIFY:
            return self._verify_think(query, thinking_prompt)
        else:
            return self._standard_think(query, thinking_prompt)
    
    def _build_thinking_prompt(
        self,
        query: str,
        context: str,
        tools: list[Tool] = None
    ) -> str:
        """Build the thinking prompt."""
        tool_descriptions = ""
        if tools:
            tool_list = "\n".join([
                f"  - {t.name}: {t.description}"
                for t in tools
            ])
            tool_descriptions = f"\nAvailable tools:\n{tool_list}\n"
        
        return f"""You are an expert educational video creator specializing in mathematics and physics visualization.

Think through this request step by step before generating any code.

REQUEST: {query}

{f"CONTEXT: {context}" if context else ""}
{tool_descriptions}

Analyze the request and provide your thinking in this JSON format:
{{
    "understanding": "What the user wants",
    "mathematical_concepts": ["List of math/physics concepts involved"],
    "visualization_approach": "How to visualize this",
    "scene_type": "2D or 3D",
    "required_elements": ["List of Manim elements needed"],
    "animation_sequence": ["Step 1", "Step 2", ...],
    "tool_calls": [
        {{"tool": "tool_name", "reason": "why needed", "args": {{}}}}
    ],
    "potential_issues": ["Things that could go wrong"],
    "confidence": 0.0 to 1.0
}}

Think carefully and be thorough."""
    
    def _standard_think(self, query: str, prompt: str) -> ThinkingResult:
        """Standard thinking process."""
        ui.thinking_start("Analyzing request...")
        
        response = self.llm.generate(prompt)
        
        # Parse structured response
        try:
            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                thinking = json.loads(json_match.group())
            else:
                thinking = {"understanding": response, "confidence": 0.5}
        except json.JSONDecodeError:
            thinking = {"understanding": response, "confidence": 0.5}
        
        # Build thoughts
        self.thoughts = []
        step = 1
        
        if "understanding" in thinking:
            self._add_thought(step, thinking["understanding"], "understanding")
            ui.thinking_step(f"Understanding: {thinking['understanding'][:80]}...")
            step += 1
        
        if "mathematical_concepts" in thinking:
            concepts = ", ".join(thinking["mathematical_concepts"][:3])
            self._add_thought(step, f"Concepts: {concepts}", "analysis")
            ui.thinking_step(f"Concepts: {concepts}")
            step += 1
        
        if "visualization_approach" in thinking:
            self._add_thought(step, thinking["visualization_approach"], "plan")
            ui.thinking_step(f"Approach: {thinking['visualization_approach'][:60]}...")
            step += 1
        
        ui.thinking_end()
        
        # Build plan
        plan = thinking.get("animation_sequence", [])
        if not plan and "required_elements" in thinking:
            plan = [f"Create {elem}" for elem in thinking["required_elements"]]
        
        # Parse tool calls
        tool_calls = []
        for tc in thinking.get("tool_calls", []):
            if isinstance(tc, dict) and "tool" in tc:
                tool_calls.append(tc)
        
        return ThinkingResult(
            thoughts=self.thoughts,
            conclusion=thinking.get("visualization_approach", ""),
            plan=plan,
            tool_calls=tool_calls,
            confidence=thinking.get("confidence", 0.7)
        )
    
    def _quick_think(self, query: str, prompt: str) -> ThinkingResult:
        """Quick thinking for simple requests."""
        ui.thinking_start("Quick analysis...")
        
        quick_prompt = f"""Briefly analyze: {query}
        
Return JSON: {{"scene_type": "2D/3D", "main_element": "...", "approach": "..."}}"""
        
        response = self.llm.generate(quick_prompt)
        
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                thinking = json.loads(json_match.group())
            else:
                thinking = {"approach": response}
        except:
            thinking = {"approach": response}
        
        ui.thinking_end()
        
        return ThinkingResult(
            thoughts=[Thought(1, thinking.get("approach", ""), "quick")],
            conclusion=thinking.get("approach", ""),
            plan=[],
            tool_calls=[],
            confidence=0.6
        )
    
    def _deep_think(self, query: str, prompt: str) -> ThinkingResult:
        """Deep thinking with multiple passes."""
        ui.thinking_start("Deep analysis...")
        
        # First pass: understand
        ui.thinking_step("Phase 1: Understanding the problem...")
        understanding = self.llm.generate(
            f"Deeply analyze this request. What exactly is being asked? What are the implicit requirements?\n\nRequest: {query}"
        )
        self._add_thought(1, understanding, "understanding")
        
        # Second pass: math analysis
        ui.thinking_step("Phase 2: Mathematical analysis...")
        math_analysis = self.llm.generate(
            f"What mathematical concepts and formulas are relevant to: {query}\n\nBe specific about equations and relationships."
        )
        self._add_thought(2, math_analysis, "math")
        
        # Third pass: visualization plan
        ui.thinking_step("Phase 3: Visualization planning...")
        viz_plan = self.llm.generate(
            f"How should we visualize this concept? What Manim elements, colors, and animation sequence?\n\nConcept: {query}\nMath: {math_analysis[:500]}"
        )
        self._add_thought(3, viz_plan, "plan")
        
        # Fourth pass: structured output
        ui.thinking_step("Phase 4: Finalizing approach...")
        response = self.llm.generate(prompt)
        
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                thinking = json.loads(json_match.group())
            else:
                thinking = {}
        except:
            thinking = {}
        
        ui.thinking_end()
        
        return ThinkingResult(
            thoughts=self.thoughts,
            conclusion=thinking.get("visualization_approach", viz_plan[:200]),
            plan=thinking.get("animation_sequence", []),
            tool_calls=thinking.get("tool_calls", []),
            confidence=thinking.get("confidence", 0.85)
        )
    
    def _verify_think(self, query: str, prompt: str) -> ThinkingResult:
        """Thinking focused on mathematical verification."""
        ui.thinking_start("Verifying mathematics...")
        
        # Extract mathematical claims
        extract_prompt = f"""Extract all mathematical claims, equations, or formulas from this request:

{query}

Return JSON: {{"claims": [{{"expression": "...", "claim_type": "equation/derivative/integral/identity"}}]}}"""
        
        response = self.llm.generate(extract_prompt)
        
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            claims = json.loads(json_match.group()).get("claims", []) if json_match else []
        except:
            claims = []
        
        tool_calls = []
        for claim in claims:
            tool_calls.append({
                "tool": "verify_math",
                "reason": f"Verify: {claim.get('expression', '')}",
                "args": {
                    "expression1": claim.get("expression", ""),
                    "claim_type": claim.get("claim_type", "equality")
                }
            })
        
        self._add_thought(1, f"Found {len(claims)} mathematical claims to verify", "observation")
        
        ui.thinking_end()
        
        return ThinkingResult(
            thoughts=self.thoughts,
            conclusion="Mathematical verification required",
            plan=["Verify each mathematical claim", "Generate animation with verified math"],
            tool_calls=tool_calls,
            confidence=0.9
        )
    
    def _add_thought(self, step: int, content: str, thought_type: str):
        """Add a thought to the list."""
        self.thoughts.append(Thought(step, content, thought_type))


# =============================================================================
# AGENT
# =============================================================================

class Agent:
    """
    Intelligent agent for video generation with tools and reasoning.
    
    Features:
    - Structured thinking and reasoning
    - Tool use for math verification
    - Documentation lookup
    - Multi-step planning
    """
    
    def __init__(
        self,
        llm: OllamaClient,
        thinking_mode: ThinkingMode = ThinkingMode.STANDARD
    ):
        self.llm = llm
        self.thinking = AgentThinking(llm, thinking_mode)
        
        # Initialize tools
        self.tools: dict[str, Tool] = {
            "calculate": CalculateTool(),
            "derivative": DerivativeTool(),
            "integral": IntegralTool(),
            "verify_math": VerifyMathTool(),
            "solve_equation": SolveEquationTool(),
            "matrix_operation": MatrixTool(),
            "physics_formula": PhysicsFormulaTool(),
            "lookup_docs": LookupDocsTool(),
        }
    
    def set_thinking_mode(self, mode: ThinkingMode):
        """Change thinking mode."""
        self.thinking.mode = mode
    
    def process(
        self,
        query: str,
        context: str = "",
        verify: bool = True
    ) -> dict:
        """
        Process a user query with thinking and tools.
        
        Args:
            query: User's request
            context: Additional context
            verify: Whether to verify mathematical content
            
        Returns:
            Dict with thinking results and tool outputs
        """
        # Think through the request
        thinking_result = self.thinking.think(
            query,
            context,
            list(self.tools.values())
        )
        
        # Show reasoning if we have it
        if thinking_result.thoughts:
            ui.reasoning_panel(
                [t.content[:100] for t in thinking_result.thoughts],
                title="Agent Reasoning"
            )
        
        # Execute tool calls
        tool_outputs = []
        for tool_call in thinking_result.tool_calls:
            tool_name = tool_call.get("tool")
            tool_args = tool_call.get("args", {})
            
            if tool_name in self.tools:
                ui.tool_call(tool_name, tool_args)
                
                result = self.tools[tool_name].execute(**tool_args)
                tool_outputs.append({
                    "tool": tool_name,
                    "success": result.success,
                    "output": result.output,
                    "error": result.error
                })
                
                ui.tool_result(
                    str(result.output)[:60] if result.success else result.error,
                    result.success
                )
        
        return {
            "thinking": thinking_result,
            "tool_outputs": tool_outputs,
            "plan": thinking_result.plan,
            "confidence": thinking_result.confidence
        }
    
    def execute_tool(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute a specific tool."""
        if tool_name not in self.tools:
            return ToolResult(False, None, f"Unknown tool: {tool_name}")
        
        ui.tool_call(tool_name, kwargs)
        result = self.tools[tool_name].execute(**kwargs)
        ui.tool_result(
            str(result.output)[:60] if result.success else result.error,
            result.success
        )
        
        return result
    
    def verify_animation_math(self, description: str, code: str) -> dict:
        """
        Verify the mathematical correctness of an animation.
        
        Extracts mathematical claims from code and verifies them.
        """
        # Extract math from code comments and MathTex
        math_pattern = r'MathTex\(["\']([^"\']+)["\']'
        tex_pattern = r'Tex\(["\']([^"\']+)["\']'
        
        math_expressions = re.findall(math_pattern, code)
        math_expressions.extend(re.findall(tex_pattern, code))
        
        verifications = []
        for expr in math_expressions:
            # Try to verify if it's an equation
            if '=' in expr and not any(op in expr for op in ['\\leq', '\\geq', '\\neq']):
                parts = expr.split('=')
                if len(parts) == 2:
                    result = self.execute_tool(
                        "verify_math",
                        expression1=parts[0].strip(),
                        expression2=parts[1].strip()
                    )
                    verifications.append({
                        "expression": expr,
                        "verified": result.success and result.output.get("verified", False),
                        "details": result.output if result.success else result.error
                    })
        
        return {
            "expressions_found": len(math_expressions),
            "verifications": verifications,
            "all_verified": all(v["verified"] for v in verifications) if verifications else True
        }
