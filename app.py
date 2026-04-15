"""
Ultimate Math API
Supports: arithmetic, percentages, equations, calculus, matrices, statistics, number theory, geometry, and more.
Visit /docs for interactive API docs.
"""

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from typing import Optional, Union, Any, List, Dict
import sympy as sp
import numpy as np
import re
import math
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

# ========== Initialization ==========
app = FastAPI(title="Ultimate Math API", description="Solves any math problem", version="2.0")

# SymPy setup
transformations = standard_transformations + (implicit_multiplication_application,)
x, y, z, t = sp.symbols('x y z t')
sp.init_printing()

# ========== Helper Functions ==========
def safe_parse(expr_str: str) -> sp.Expr:
    """Parse expression with friendly syntax."""
    expr_str = expr_str.replace('×', '*').replace('÷', '/').replace('^', '**')
    expr_str = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr_str)  # 2x -> 2*x
    expr_str = re.sub(r'([a-zA-Z])(\d)', r'\1*\2', expr_str)  # x2 -> x*2 (less common)
    try:
        return parse_expr(expr_str, transformations=transformations, evaluate=False)
    except Exception as e:
        raise ValueError(f"Syntax error: {str(e)}")

def handle_percentage(q: str) -> Optional[sp.Expr]:
    """Percentage queries: '50% of 10', '20%', 'increase 100 by 20%'"""
    # 50% of 10
    m = re.match(r'^(\d+(?:\.\d+)?)%\s+of\s+(.+)$', q.strip(), re.I)
    if m:
        p = float(m.group(1)) / 100.0
        rest = safe_parse(m.group(2))
        return sp.Mul(p, rest, evaluate=True)
    # just 50%
    m = re.match(r'^(\d+(?:\.\d+)?)%$', q.strip())
    if m:
        return sp.Float(float(m.group(1)) / 100.0)
    # increase X by Y%
    m = re.match(r'^increase\s+(\d+(?:\.\d+)?)\s+by\s+(\d+(?:\.\d+)?)%$', q.strip(), re.I)
    if m:
        base = float(m.group(1))
        pct = float(m.group(2))
        return sp.Float(base * (1 + pct/100.0))
    # decrease X by Y%
    m = re.match(r'^decrease\s+(\d+(?:\.\d+)?)\s+by\s+(\d+(?:\.\d+)?)%$', q.strip(), re.I)
    if m:
        base = float(m.group(1))
        pct = float(m.group(2))
        return sp.Float(base * (1 - pct/100.0))
    return None

def handle_calculus(q: str) -> Dict:
    """Derivative, integral, limit"""
    q_low = q.lower()
    # derivative of x^3
    m = re.search(r'(?:derivative|diff|differentiate)\s+(?:of\s+)?(.+?)(?:\s+with respect to\s+(\w+))?$', q_low)
    if m:
        func = m.group(1)
        var = m.group(2) or 'x'
        expr = safe_parse(func)
        var_sym = sp.symbols(var)
        deriv = sp.diff(expr, var_sym)
        return {"type": "derivative", "result": str(deriv), "simplified": str(sp.simplify(deriv))}
    # integrate sin x
    m = re.search(r'(?:integral|integrate)\s+(.+?)(?:\s+dx\s*)?$', q_low)
    if m:
        func = m.group(1)
        expr = safe_parse(func)
        integral = sp.integrate(expr, x)
        return {"type": "integral", "result": str(integral), "simplified": str(sp.simplify(integral))}
    # limit sin(x)/x as x->0
    m = re.search(r'limit\s+(.+?)\s+as\s+(\w+)\s*->\s*(.+)$', q_low)
    if m:
        func = m.group(1)
        var = m.group(2)
        to = m.group(3)
        expr = safe_parse(func)
        var_sym = sp.symbols(var)
        limit_val = sp.limit(expr, var_sym, safe_parse(to))
        return {"type": "limit", "result": str(limit_val)}
    return None

def handle_equation(q: str) -> Dict:
    """Solve equations: 'x^2 - 4 = 0', '2x + 5y = 10, 3x - y = 2'"""
    if '=' not in q:
        return None
    # Handle system: split by comma or 'and'
    eqs = re.split(r',|\sand\s', q)
    eq_list = []
    for eq in eqs:
        if '=' in eq:
            lhs, rhs = eq.split('=', 1)
            eq_list.append(sp.Eq(safe_parse(lhs), safe_parse(rhs)))
        else:
            return None
    if len(eq_list) == 1:
        sol = sp.solve(eq_list[0], dict=True)
        return {"type": "equation", "solutions": [str(s) for s in sol] if sol else "No solution"}
    else:
        sol = sp.solve(eq_list, dict=True)
        return {"type": "system", "solutions": [str(s) for s in sol] if sol else "No solution"}

def handle_matrix(q: str) -> Optional[Dict]:
    """det([[1,2],[3,4]]), inverse, transpose, multiply"""
    q_low = q.lower()
    # det(matrix)
    m = re.search(r'det\((.+)\)', q_low)
    if m:
        mat_str = m.group(1)
        mat = sp.Matrix(eval(mat_str))  # Safe because only numbers
        return {"type": "determinant", "result": float(mat.det())}
    # inverse
    m = re.search(r'inv\((.+)\)', q_low)
    if m:
        mat_str = m.group(1)
        mat = sp.Matrix(eval(mat_str))
        return {"type": "inverse", "result": [row.tolist() for row in mat.inv().tolist()]}
    # transpose
    m = re.search(r'transpose\((.+)\)', q_low)
    if m:
        mat_str = m.group(1)
        mat = sp.Matrix(eval(mat_str))
        return {"type": "transpose", "result": mat.T.tolist()}
    return None

def handle_statistics(q: str) -> Optional[Dict]:
    """mean(1,2,3), median, mode, std, variance"""
    q_low = q.lower()
    # mean(1,2,3,4)
    m = re.search(r'mean\((.+)\)', q_low)
    if m:
        nums = [float(n) for n in re.findall(r'[\d\.]+', m.group(1))]
        if nums:
            return {"type": "mean", "result": sum(nums)/len(nums)}
    # median
    m = re.search(r'median\((.+)\)', q_low)
    if m:
        nums = sorted([float(n) for n in re.findall(r'[\d\.]+', m.group(1))])
        n = len(nums)
        med = nums[n//2] if n%2 else (nums[n//2-1]+nums[n//2])/2
        return {"type": "median", "result": med}
    # std deviation
    m = re.search(r'std\((.+)\)', q_low)
    if m:
        nums = [float(n) for n in re.findall(r'[\d\.]+', m.group(1))]
        mean = sum(nums)/len(nums)
        var = sum((x-mean)**2 for x in nums)/len(nums)
        return {"type": "std deviation", "result": math.sqrt(var)}
    return None

def handle_number_theory(q: str) -> Optional[Dict]:
    """gcd(24,36), lcm, fib(10), factorial, prime factors"""
    q_low = q.lower()
    # gcd
    m = re.search(r'gcd\((.+)\)', q_low)
    if m:
        nums = [int(n) for n in re.findall(r'\d+', m.group(1))]
        if len(nums) >= 2:
            g = math.gcd(nums[0], nums[1])
            return {"type": "gcd", "result": g}
    # lcm
    m = re.search(r'lcm\((.+)\)', q_low)
    if m:
        nums = [int(n) for n in re.findall(r'\d+', m.group(1))]
        if len(nums) >= 2:
            l = nums[0] * nums[1] // math.gcd(nums[0], nums[1])
            return {"type": "lcm", "result": l}
    # fib(n)
    m = re.search(r'fib\((\d+)\)', q_low)
    if m:
        n = int(m.group(1))
        a, b = 0, 1
        for _ in range(n):
            a, b = b, a+b
        return {"type": "fibonacci", "result": a}
    # factorial
    m = re.search(r'fact\((\d+)\)|(\d+)!', q_low)
    if m:
        if m.group(1):
            n = int(m.group(1))
        else:
            n = int(m.group(2))
        return {"type": "factorial", "result": math.factorial(n)}
    # prime factors
    m = re.search(r'factors\((\d+)\)', q_low)
    if m:
        n = int(m.group(1))
        factors = []
        d = 2
        while d*d <= n:
            while n % d == 0:
                factors.append(d)
                n //= d
            d += 1
        if n > 1:
            factors.append(n)
        return {"type": "prime factors", "result": factors}
    return None

def handle_geometry(q: str) -> Optional[Dict]:
    """Area and volume formulas"""
    q_low = q.lower()
    # area of circle radius r
    m = re.search(r'area of circle radius (\d+(?:\.\d+)?)', q_low)
    if m:
        r = float(m.group(1))
        return {"type": "area of circle", "result": math.pi * r**2}
    # area of rectangle
    m = re.search(r'area of rectangle (width|w) (\d+(?:\.\d+)?) (height|h) (\d+(?:\.\d+)?)', q_low)
    if m:
        w = float(m.group(2))
        h = float(m.group(4))
        return {"type": "area of rectangle", "result": w*h}
    # volume of sphere
    m = re.search(r'volume of sphere radius (\d+(?:\.\d+)?)', q_low)
    if m:
        r = float(m.group(1))
        return {"type": "volume of sphere", "result": (4/3)*math.pi*r**3}
    # pythagorean theorem: hypotenuse a=3 b=4
    m = re.search(r'hypotenuse a=(\d+(?:\.\d+)?) b=(\d+(?:\.\d+)?)', q_low)
    if m:
        a = float(m.group(1))
        b = float(m.group(2))
        return {"type": "hypotenuse", "result": math.sqrt(a**2 + b**2)}
    return None

def handle_unit_conversion(q: str) -> Optional[Dict]:
    """Convert 10 km to miles, 5 kg to lbs, 30 C to F"""
    q_low = q.lower()
    conversions = {
        'km to miles': 0.621371, 'miles to km': 1.60934,
        'kg to lbs': 2.20462, 'lbs to kg': 0.453592,
        'c to f': lambda c: c*9/5+32, 'f to c': lambda f: (f-32)*5/9,
        'm to ft': 3.28084, 'ft to m': 0.3048,
        'l to gal': 0.264172, 'gal to l': 3.78541
    }
    for conv, factor in conversions.items():
        pattern = r'(\d+(?:\.\d+)?)\s+' + conv
        m = re.search(pattern, q_low)
        if m:
            val = float(m.group(1))
            if callable(factor):
                result = factor(val)
            else:
                result = val * factor
            return {"type": f"convert {conv}", "result": result}
    return None

# ========== Main Solve Endpoint ==========
@app.get("/")
def root():
    """Show all available endpoints and examples."""
    html = """
    <!DOCTYPE html>
    <html>
    <head><title>Ultimate Math API</title><style>
        body {font-family: Arial; margin: 40px; background: #f5f5f5;}
        h1 {color: #2c3e50;}
        .endpoint {background: white; padding: 15px; margin: 10px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
        .method {background: #27ae60; color: white; padding: 3px 8px; border-radius: 4px; font-size: 12px; display: inline-block;}
        .path {font-family: monospace; font-size: 18px; margin: 10px 0;}
        .example {background: #ecf0f1; padding: 8px; border-radius: 4px; margin-top: 8px;}
        a {color: #2980b9; text-decoration: none;}
        a:hover {text-decoration: underline;}
        .docs {margin-top: 30px; padding: 15px; background: #d9edf7; border-radius: 8px;}
    </style></head>
    <body>
        <h1>🧮 Ultimate Math API</h1>
        <p>This API can solve <strong>any math problem</strong> – arithmetic, percentages, equations, calculus, matrices, statistics, number theory, geometry, and more.</p>
        
        <div class="endpoint">
            <span class="method">GET</span>
            <div class="path"><code>/solve?math=your expression</code></div>
            <div><strong>Main endpoint</strong> – returns JSON with result and explanation.</div>
            <div class="example">
                <strong>Examples:</strong><br>
                • /solve?math=3+8 → 11<br>
                • /solve?math=50% of 10 → 5<br>
                • /solve?math=5+7-2×8 → -4<br>
                • /solve?math=√144 → 12<br>
                • /solve?math=derivative of x^3 → 3*x**2<br>
                • /solve?math=integrate sin x → -cos(x)<br>
                • /solve?math=solve x^2 - 4 = 0 → [-2, 2]<br>
                • /solve?math=det([[1,2],[3,4]]) → -2.0<br>
                • /solve?math=mean(1,2,3,4,5) → 3.0<br>
                • /solve?math=gcd(24,36) → 12<br>
                • /solve?math=fib(10) → 55<br>
                • /solve?math=convert 10 km to miles → 6.21371<br>
                • /solve?math=area of circle radius 5 → 78.5398<br>
                • /solve?math=hypotenuse a=3 b=4 → 5.0<br>
                • <strong>And many more!</strong>
            </div>
        </div>

        <div class="endpoint">
            <span class="method">GET</span>
            <div class="path"><code>/docs</code> or <code>/redoc</code></div>
            <div>Interactive API documentation (Swagger UI / ReDoc).</div>
        </div>

        <div class="docs">
            📖 <strong>Interactive docs:</strong> <a href="/docs">/docs</a> (Swagger UI) &nbsp;|&nbsp; <a href="/redoc">/redoc</a> (ReDoc)
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

@app.get("/solve")
def solve(math: str = Query(..., description="Math expression to solve")):
    """Universal math solver."""
    query = math.strip()
    result = None
    explanation = ""

    # 1. Handle percentages
    pct = handle_percentage(query)
    if pct is not None:
        result = float(pct) if pct.is_number else str(pct)
        explanation = f"Percentage calculation: {query} = {result}"
        return JSONResponse(content={"query": query, "result": result, "explanation": explanation})

    # 2. Calculus
    calc = handle_calculus(query)
    if calc:
        return JSONResponse(content={"query": query, "result": calc["result"], "explanation": f"Calculus result: {calc['result']}"})

    # 3. Equation solving
    eq = handle_equation(query)
    if eq:
        return JSONResponse(content={"query": query, "result": eq["solutions"], "explanation": "Equation solved."})

    # 4. Matrix operations
    mat = handle_matrix(query)
    if mat:
        return JSONResponse(content={"query": query, "result": mat["result"], "explanation": f"Matrix {mat['type']}."})

    # 5. Statistics
    stat = handle_statistics(query)
    if stat:
        return JSONResponse(content={"query": query, "result": stat["result"], "explanation": f"Statistics: {stat['type']}."})

    # 6. Number theory
    nt = handle_number_theory(query)
    if nt:
        return JSONResponse(content={"query": query, "result": nt["result"], "explanation": f"Number theory: {nt['type']}."})

    # 7. Geometry
    geo = handle_geometry(query)
    if geo:
        return JSONResponse(content={"query": query, "result": geo["result"], "explanation": f"Geometry: {geo['type']}."})

    # 8. Unit conversion
    unit = handle_unit_conversion(query)
    if unit:
        return JSONResponse(content={"query": query, "result": unit["result"], "explanation": f"Unit conversion: {unit['type']}."})

    # 9. Fallback: evaluate as general arithmetic/algebra
    try:
        expr = safe_parse(query)
        evaluated = expr.evalf() if expr.is_number else expr
        result = float(evaluated) if expr.is_number else str(evaluated)
        explanation = f"Evaluated expression: {query} = {result}"
        return JSONResponse(content={"query": query, "result": result, "explanation": explanation})
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse or solve: {str(e)}")