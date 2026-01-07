"""
Math Engine - Symbolic mathematics verification and computation
Uses SymPy for backend verification of mathematical expressions
"""

import re
from typing import Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

import sympy as sp
from sympy import (
    symbols, Symbol, Function, Derivative, Integral, Limit,
    sin, cos, tan, cot, sec, csc,
    sinh, cosh, tanh, coth, sech, csch,
    exp, log, ln, sqrt, Abs, sign,
    pi, E, I, oo,
    simplify, expand, factor, collect, cancel,
    trigsimp, powsimp, radsimp, ratsimp,
    diff, integrate, limit, series, summation,
    solve, solveset, linsolve, nonlinsolve,
    Matrix, det, trace, eye, zeros, ones,
    latex, pretty, pprint,
    Eq, Ne, Lt, Le, Gt, Ge,
    And, Or, Not, Implies,
    sympify, parse_expr,
)
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)
from sympy.physics.vector import ReferenceFrame, Vector
from sympy.physics.mechanics import dynamicsymbols


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class VerificationResult(Enum):
    """Result of mathematical verification."""
    VALID = "valid"
    INVALID = "invalid"
    EQUIVALENT = "equivalent"
    NOT_EQUIVALENT = "not_equivalent"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass
class MathResult:
    """Result of a mathematical operation."""
    success: bool
    result: Any
    latex: str = ""
    simplified: str = ""
    steps: list[str] = None
    verification: VerificationResult = VerificationResult.UNKNOWN
    error: str = ""
    
    def __post_init__(self):
        if self.steps is None:
            self.steps = []


# =============================================================================
# MATH ENGINE
# =============================================================================

class MathEngine:
    """
    Mathematical computation and verification engine.
    
    Provides symbolic math capabilities for verifying
    animations and computing mathematical expressions.
    """
    
    def __init__(self):
        # Common symbols
        self.x, self.y, self.z = symbols('x y z', real=True)
        self.t = symbols('t', real=True, positive=True)
        self.n, self.m, self.k = symbols('n m k', integer=True)
        self.a, self.b, self.c = symbols('a b c', real=True)
        
        # Parser transformations
        self.transformations = (
            standard_transformations +
            (implicit_multiplication_application, convert_xor)
        )
        
        # Symbol cache
        self._symbol_cache = {
            'x': self.x, 'y': self.y, 'z': self.z, 't': self.t,
            'n': self.n, 'm': self.m, 'k': self.k,
            'a': self.a, 'b': self.b, 'c': self.c,
            'pi': pi, 'e': E, 'i': I,
        }
    
    # =========================================================================
    # PARSING
    # =========================================================================
    
    def parse(self, expr_str: str) -> Optional[sp.Expr]:
        """
        Parse a string expression into SymPy.
        
        Handles various input formats including LaTeX-like notation.
        """
        if not expr_str:
            return None
        
        # Clean the expression
        expr_str = self._clean_expression(expr_str)
        
        try:
            return parse_expr(
                expr_str,
                local_dict=self._symbol_cache,
                transformations=self.transformations,
                evaluate=True
            )
        except Exception as e:
            # Try alternative parsing
            try:
                return sympify(expr_str, locals=self._symbol_cache)
            except:
                return None
    
    def _clean_expression(self, expr: str) -> str:
        """Clean and normalize expression string."""
        # Remove LaTeX delimiters
        expr = re.sub(r'\$+', '', expr)
        expr = re.sub(r'\\left|\\right', '', expr)
        
        # Convert LaTeX to Python-like syntax
        replacements = [
            (r'\\frac\{([^}]+)\}\{([^}]+)\}', r'((\1)/(\2))'),
            (r'\\sqrt\{([^}]+)\}', r'sqrt(\1)'),
            (r'\\sqrt\[(\d+)\]\{([^}]+)\}', r'(\2)**(1/\1)'),
            (r'\\sin', 'sin'),
            (r'\\cos', 'cos'),
            (r'\\tan', 'tan'),
            (r'\\cot', 'cot'),
            (r'\\sec', 'sec'),
            (r'\\csc', 'csc'),
            (r'\\sinh', 'sinh'),
            (r'\\cosh', 'cosh'),
            (r'\\tanh', 'tanh'),
            (r'\\ln', 'log'),
            (r'\\log', 'log'),
            (r'\\exp', 'exp'),
            (r'\\pi', 'pi'),
            (r'\\infty', 'oo'),
            (r'\^', '**'),
            (r'\\cdot', '*'),
            (r'\\times', '*'),
            (r'\\div', '/'),
            (r'\{', '('),
            (r'\}', ')'),
        ]
        
        for pattern, replacement in replacements:
            expr = re.sub(pattern, replacement, expr)
        
        return expr.strip()
    
    def to_latex(self, expr: sp.Expr) -> str:
        """Convert SymPy expression to LaTeX."""
        try:
            return latex(expr)
        except:
            return str(expr)
    
    # =========================================================================
    # CALCULUS
    # =========================================================================
    
    def derivative(
        self,
        expr: Union[str, sp.Expr],
        var: str = 'x',
        order: int = 1
    ) -> MathResult:
        """Compute derivative with steps."""
        if isinstance(expr, str):
            expr = self.parse(expr)
        
        if expr is None:
            return MathResult(False, None, error="Could not parse expression")
        
        try:
            symbol = self._get_symbol(var)
            result = diff(expr, symbol, order)
            simplified = simplify(result)
            
            steps = [
                f"Original: {self.to_latex(expr)}",
                f"d/d{var}: {self.to_latex(result)}",
            ]
            
            if result != simplified:
                steps.append(f"Simplified: {self.to_latex(simplified)}")
            
            return MathResult(
                success=True,
                result=simplified,
                latex=self.to_latex(simplified),
                simplified=str(simplified),
                steps=steps,
                verification=VerificationResult.VALID
            )
        except Exception as e:
            return MathResult(False, None, error=str(e))
    
    def integral(
        self,
        expr: Union[str, sp.Expr],
        var: str = 'x',
        limits: Optional[tuple] = None
    ) -> MathResult:
        """Compute integral (definite or indefinite)."""
        if isinstance(expr, str):
            expr = self.parse(expr)
        
        if expr is None:
            return MathResult(False, None, error="Could not parse expression")
        
        try:
            symbol = self._get_symbol(var)
            
            if limits:
                lower = self.parse(str(limits[0])) if isinstance(limits[0], str) else limits[0]
                upper = self.parse(str(limits[1])) if isinstance(limits[1], str) else limits[1]
                result = integrate(expr, (symbol, lower, upper))
                integral_type = "Definite"
            else:
                result = integrate(expr, symbol)
                integral_type = "Indefinite"
            
            simplified = simplify(result)
            
            steps = [
                f"Integrand: {self.to_latex(expr)}",
                f"{integral_type} integral: {self.to_latex(result)}",
            ]
            
            if result != simplified:
                steps.append(f"Simplified: {self.to_latex(simplified)}")
            
            return MathResult(
                success=True,
                result=simplified,
                latex=self.to_latex(simplified),
                simplified=str(simplified),
                steps=steps,
                verification=VerificationResult.VALID
            )
        except Exception as e:
            return MathResult(False, None, error=str(e))
    
    def limit_compute(
        self,
        expr: Union[str, sp.Expr],
        var: str = 'x',
        point: Any = 0,
        direction: str = '+-'
    ) -> MathResult:
        """Compute limit of expression."""
        if isinstance(expr, str):
            expr = self.parse(expr)
        
        if expr is None:
            return MathResult(False, None, error="Could not parse expression")
        
        try:
            symbol = self._get_symbol(var)
            point_val = self.parse(str(point)) if isinstance(point, str) else point
            
            result = limit(expr, symbol, point_val, direction)
            
            steps = [
                f"Expression: {self.to_latex(expr)}",
                f"As {var} → {point}: {self.to_latex(result)}",
            ]
            
            return MathResult(
                success=True,
                result=result,
                latex=self.to_latex(result),
                simplified=str(result),
                steps=steps,
                verification=VerificationResult.VALID
            )
        except Exception as e:
            return MathResult(False, None, error=str(e))
    
    def taylor_series(
        self,
        expr: Union[str, sp.Expr],
        var: str = 'x',
        point: Any = 0,
        order: int = 5
    ) -> MathResult:
        """Compute Taylor series expansion."""
        if isinstance(expr, str):
            expr = self.parse(expr)
        
        if expr is None:
            return MathResult(False, None, error="Could not parse expression")
        
        try:
            symbol = self._get_symbol(var)
            point_val = self.parse(str(point)) if isinstance(point, str) else point
            
            result = series(expr, symbol, point_val, order + 1).removeO()
            
            steps = [
                f"Function: {self.to_latex(expr)}",
                f"Expansion point: {point}",
                f"Order: {order}",
                f"Series: {self.to_latex(result)}",
            ]
            
            return MathResult(
                success=True,
                result=result,
                latex=self.to_latex(result),
                simplified=str(result),
                steps=steps,
                verification=VerificationResult.VALID
            )
        except Exception as e:
            return MathResult(False, None, error=str(e))
    
    # =========================================================================
    # LINEAR ALGEBRA
    # =========================================================================
    
    def matrix_create(self, data: list[list]) -> sp.Matrix:
        """Create SymPy matrix from nested list."""
        return Matrix(data)
    
    def matrix_ops(
        self,
        matrix: Union[list[list], sp.Matrix],
        operation: str
    ) -> MathResult:
        """Perform matrix operation."""
        if isinstance(matrix, list):
            matrix = Matrix(matrix)
        
        try:
            if operation == "determinant":
                result = det(matrix)
            elif operation == "inverse":
                result = matrix.inv()
            elif operation == "transpose":
                result = matrix.T
            elif operation == "trace":
                result = trace(matrix)
            elif operation == "eigenvalues":
                result = matrix.eigenvals()
            elif operation == "eigenvectors":
                result = matrix.eigenvects()
            elif operation == "rref":
                result = matrix.rref()[0]
            elif operation == "rank":
                result = matrix.rank()
            elif operation == "nullspace":
                result = matrix.nullspace()
            else:
                return MathResult(False, None, error=f"Unknown operation: {operation}")
            
            return MathResult(
                success=True,
                result=result,
                latex=self.to_latex(result) if hasattr(result, '__iter__') else str(result),
                simplified=str(result),
                verification=VerificationResult.VALID
            )
        except Exception as e:
            return MathResult(False, None, error=str(e))
    
    def solve_system(
        self,
        equations: list[str],
        variables: list[str]
    ) -> MathResult:
        """Solve system of equations."""
        try:
            eqs = []
            for eq in equations:
                if '=' in eq:
                    left, right = eq.split('=')
                    eqs.append(Eq(self.parse(left), self.parse(right)))
                else:
                    eqs.append(self.parse(eq))
            
            syms = [self._get_symbol(v) for v in variables]
            result = solve(eqs, syms)
            
            return MathResult(
                success=True,
                result=result,
                latex=str(result),
                simplified=str(result),
                verification=VerificationResult.VALID
            )
        except Exception as e:
            return MathResult(False, None, error=str(e))
    
    # =========================================================================
    # VERIFICATION
    # =========================================================================
    
    def verify_equality(
        self,
        expr1: Union[str, sp.Expr],
        expr2: Union[str, sp.Expr]
    ) -> MathResult:
        """Verify if two expressions are mathematically equal."""
        if isinstance(expr1, str):
            expr1 = self.parse(expr1)
        if isinstance(expr2, str):
            expr2 = self.parse(expr2)
        
        if expr1 is None or expr2 is None:
            return MathResult(
                False, None,
                error="Could not parse expressions",
                verification=VerificationResult.ERROR
            )
        
        try:
            # Try simplification comparison
            diff_expr = simplify(expr1 - expr2)
            
            if diff_expr == 0:
                return MathResult(
                    success=True,
                    result=True,
                    latex=f"{self.to_latex(expr1)} = {self.to_latex(expr2)}",
                    verification=VerificationResult.EQUIVALENT
                )
            
            # Try expansion
            expanded1 = expand(expr1)
            expanded2 = expand(expr2)
            
            if simplify(expanded1 - expanded2) == 0:
                return MathResult(
                    success=True,
                    result=True,
                    latex=f"{self.to_latex(expr1)} = {self.to_latex(expr2)}",
                    verification=VerificationResult.EQUIVALENT
                )
            
            # Try trigsimp
            trig1 = trigsimp(expr1)
            trig2 = trigsimp(expr2)
            
            if simplify(trig1 - trig2) == 0:
                return MathResult(
                    success=True,
                    result=True,
                    latex=f"{self.to_latex(expr1)} = {self.to_latex(expr2)}",
                    verification=VerificationResult.EQUIVALENT
                )
            
            return MathResult(
                success=True,
                result=False,
                latex=f"{self.to_latex(expr1)} ≠ {self.to_latex(expr2)}",
                verification=VerificationResult.NOT_EQUIVALENT
            )
            
        except Exception as e:
            return MathResult(
                False, None,
                error=str(e),
                verification=VerificationResult.ERROR
            )
    
    def verify_derivative(
        self,
        function: str,
        claimed_derivative: str,
        var: str = 'x'
    ) -> MathResult:
        """Verify if a derivative claim is correct."""
        actual = self.derivative(function, var)
        
        if not actual.success:
            return actual
        
        return self.verify_equality(actual.result, claimed_derivative)
    
    def verify_integral(
        self,
        function: str,
        claimed_integral: str,
        var: str = 'x'
    ) -> MathResult:
        """Verify if an integral claim is correct (up to constant)."""
        # For indefinite integrals, the derivative of the claimed integral
        # should equal the original function
        claimed = self.parse(claimed_integral)
        if claimed is None:
            return MathResult(False, None, error="Could not parse claimed integral")
        
        derivative_of_claimed = self.derivative(claimed, var)
        if not derivative_of_claimed.success:
            return derivative_of_claimed
        
        return self.verify_equality(derivative_of_claimed.result, function)
    
    def verify_limit(
        self,
        function: str,
        claimed_limit: str,
        var: str = 'x',
        point: Any = 0
    ) -> MathResult:
        """Verify if a limit claim is correct."""
        actual = self.limit_compute(function, var, point)
        
        if not actual.success:
            return actual
        
        return self.verify_equality(actual.result, claimed_limit)
    
    # =========================================================================
    # PHYSICS COMPUTATIONS
    # =========================================================================
    
    def kinematics(
        self,
        known: dict,
        solve_for: str
    ) -> MathResult:
        """Solve kinematics problem."""
        v0, v, a, t, s = symbols('v_0 v a t s', real=True)
        
        # Kinematic equations
        equations = [
            Eq(v, v0 + a * t),
            Eq(s, v0 * t + sp.Rational(1, 2) * a * t**2),
            Eq(v**2, v0**2 + 2 * a * s),
            Eq(s, (v0 + v) / 2 * t),
        ]
        
        # Substitute known values
        subs = {}
        symbol_map = {'v0': v0, 'v': v, 'a': a, 't': t, 's': s}
        
        for key, value in known.items():
            if key in symbol_map:
                subs[symbol_map[key]] = value
        
        target = symbol_map.get(solve_for)
        if target is None:
            return MathResult(False, None, error=f"Unknown variable: {solve_for}")
        
        try:
            # Try each equation
            for eq in equations:
                if target in eq.free_symbols:
                    substituted = eq.subs(subs)
                    solution = solve(substituted, target)
                    if solution:
                        result = solution[0] if isinstance(solution, list) else solution
                        return MathResult(
                            success=True,
                            result=result,
                            latex=self.to_latex(result),
                            simplified=str(result),
                            verification=VerificationResult.VALID
                        )
            
            return MathResult(False, None, error="Could not solve with given information")
            
        except Exception as e:
            return MathResult(False, None, error=str(e))
    
    def vector_ops(
        self,
        v1: tuple,
        v2: tuple,
        operation: str
    ) -> MathResult:
        """Perform vector operations."""
        try:
            v1_matrix = Matrix(list(v1))
            v2_matrix = Matrix(list(v2))
            
            if operation == "dot":
                result = v1_matrix.dot(v2_matrix)
            elif operation == "cross":
                if len(v1) != 3 or len(v2) != 3:
                    return MathResult(False, None, error="Cross product requires 3D vectors")
                result = v1_matrix.cross(v2_matrix)
            elif operation == "add":
                result = v1_matrix + v2_matrix
            elif operation == "subtract":
                result = v1_matrix - v2_matrix
            elif operation == "angle":
                dot = v1_matrix.dot(v2_matrix)
                mag1 = sqrt(v1_matrix.dot(v1_matrix))
                mag2 = sqrt(v2_matrix.dot(v2_matrix))
                result = sp.acos(dot / (mag1 * mag2))
            else:
                return MathResult(False, None, error=f"Unknown operation: {operation}")
            
            return MathResult(
                success=True,
                result=result,
                latex=self.to_latex(result),
                simplified=str(simplify(result)),
                verification=VerificationResult.VALID
            )
        except Exception as e:
            return MathResult(False, None, error=str(e))
    
    # =========================================================================
    # HELPERS
    # =========================================================================
    
    def _get_symbol(self, name: str) -> Symbol:
        """Get or create a symbol."""
        if name in self._symbol_cache:
            return self._symbol_cache[name]
        
        sym = symbols(name, real=True)
        self._symbol_cache[name] = sym
        return sym
    
    def simplify_expr(self, expr: Union[str, sp.Expr]) -> MathResult:
        """Simplify expression."""
        if isinstance(expr, str):
            expr = self.parse(expr)
        
        if expr is None:
            return MathResult(False, None, error="Could not parse expression")
        
        try:
            result = simplify(expr)
            
            return MathResult(
                success=True,
                result=result,
                latex=self.to_latex(result),
                simplified=str(result),
                verification=VerificationResult.VALID
            )
        except Exception as e:
            return MathResult(False, None, error=str(e))
    
    def evaluate(
        self,
        expr: Union[str, sp.Expr],
        values: dict
    ) -> MathResult:
        """Evaluate expression with given values."""
        if isinstance(expr, str):
            expr = self.parse(expr)
        
        if expr is None:
            return MathResult(False, None, error="Could not parse expression")
        
        try:
            subs = {}
            for var, val in values.items():
                subs[self._get_symbol(var)] = val
            
            result = expr.subs(subs)
            
            # Try to get numerical value
            try:
                numerical = float(result.evalf())
                result = numerical
            except:
                pass
            
            return MathResult(
                success=True,
                result=result,
                latex=str(result),
                simplified=str(result),
                verification=VerificationResult.VALID
            )
        except Exception as e:
            return MathResult(False, None, error=str(e))


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

math_engine = MathEngine()
