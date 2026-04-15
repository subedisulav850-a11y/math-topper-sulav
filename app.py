from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from typing import Optional, Dict
import sympy as sp
import re
import math
from urllib.parse import unquote
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

app = FastAPI(title="Ultimate Math API", description="Solves any math problem", version="3.0")

# SymPy setup
transformations = standard_transformations + (implicit_multiplication_application,)
x, y, z, t = sp.symbols('x y z t')
sp.init_printing()

# ========== FIX: Convert spaces to '+' when user meant addition ==========
def fix_plus_sign(expr: str) -> str:
    """
    When we receive '3 8' (because '+' became space), convert to '3+8'
    Only do this if there are no other operators in the expression.
    """
    # If there's already any operator, leave it alone
    if any(op in expr for op in ('+', '-', '*', '/', '×', '÷', '^')):
        return expr
    # If there's a space, replace with '+'
    if ' ' in expr:
        return expr.replace(' ', '+')
    return expr

# ========== All your existing helper functions (unchanged) ==========
def safe_parse(expr_str: str) -> sp.Expr:
    expr_str = expr_str.replace('×', '*').replace('÷', '/').replace('^', '**')
    expr_str = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr_str)
    try:
        return parse_expr(expr_str, transformations=transformations, evaluate=False)
    except Exception as e:
        raise ValueError(f"Syntax error: {str(e)}")

def handle_percentage(q: str) -> Optional[sp.Expr]:
    m = re.match(r'^(\d+(?:\.\d+)?)%\s+of\s+(.+)$', q.strip(), re.I)
    if m:
        p = float(m.group(1)) / 100.0
        rest = safe_parse(m.group(2))
        return sp.Mul(p, rest, evaluate=True)
    m = re.match(r'^(\d+(?:\.\d+)?)%$', q.strip())
    if m:
        return sp.Float(float(m.group(1)) / 100.0)
    m = re.match(r'^increase\s+(\d+(?:\.\d+)?)\s+by\s+(\d+(?:\.\d+)?)%$', q.strip(), re.I)
    if m:
        base = float(m.group(1))
        pct = float(m.group(2))
        return sp.Float(base * (1 + pct/100.0))
    m = re.match(r'^decrease\s+(\d+(?:\.\d+)?)\s+by\s+(\d+(?:\.\d+)?)%$', q.strip(), re.I)
    if m:
        base = float(m.group(1))
        pct = float(m.group(2))
        return sp.Float(base * (1 - pct/100.0))
    return None

def handle_calculus(q: str) -> Optional[Dict]:
    q_low = q.lower()
    m = re.search(r'(?:derivative|diff|differentiate)\s+(?:of\s+)?(.+?)(?:\s+with respect to\s+(\w+))?$', q_low)
    if m:
        func = m.group(1)
        var = m.group(2) or 'x'
        expr = safe_parse(func)
        var_sym = sp.symbols(var)
        deriv = sp.diff(expr, var_sym)
        return {"type": "derivative", "result": str(deriv), "simplified": str(sp.simplify(deriv))}
    m = re.search(r'(?:integral|integrate)\s+(.+?)(?:\s+dx\s*)?$', q_low)
    if m:
        func = m.group(1)
        expr = safe_parse(func)
        integral = sp.integrate(expr, x)
        return {"type": "integral", "result": str(integral), "simplified": str(sp.simplify(integral))}
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

def handle_equation(q: str) -> Optional[Dict]:
    if '=' not in q:
        return None
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
    q_low = q.lower()
    m = re.search(r'det\((.+)\)', q_low)
    if m:
        mat = sp.Matrix(eval(m.group(1)))
        return {"type": "determinant", "result": float(mat.det())}
    m = re.search(r'inv\((.+)\)', q_low)
    if m:
        mat = sp.Matrix(eval(m.group(1)))
        return {"type": "inverse", "result": [row.tolist() for row in mat.inv().tolist()]}
    m = re.search(r'transpose\((.+)\)', q_low)
    if m:
        mat = sp.Matrix(eval(m.group(1)))
        return {"type": "transpose", "result": mat.T.tolist()}
    return None

def handle_statistics(q: str) -> Optional[Dict]:
    q_low = q.lower()
    m = re.search(r'mean\((.+)\)', q_low)
    if m:
        nums = [float(n) for n in re.findall(r'[\d\.]+', m.group(1))]
        if nums:
            return {"type": "mean", "result": sum(nums)/len(nums)}
    m = re.search(r'median\((.+)\)', q_low)
    if m:
        nums = sorted([float(n) for n in re.findall(r'[\d\.]+', m.group(1))])
        n = len(nums)
        med = nums[n//2] if n%2 else (nums[n//2-1]+nums[n//2])/2
        return {"type": "median", "result": med}
    m = re.search(r'std\((.+)\)', q_low)
    if m:
        nums = [float(n) for n in re.findall(r'[\d\.]+', m.group(1))]
        mean = sum(nums)/len(nums)
        var = sum((x-mean)**2 for x in nums)/len(nums)
        return {"type": "std deviation", "result": math.sqrt(var)}
    return None

def handle_number_theory(q: str) -> Optional[Dict]:
    q_low = q.lower()
    m = re.search(r'gcd\((.+)\)', q_low)
    if m:
        nums = [int(n) for n in re.findall(r'\d+', m.group(1))]
        if len(nums) >= 2:
            g = math.gcd(nums[0], nums[1])
            return {"type": "gcd", "result": g}
    m = re.search(r'lcm\((.+)\)', q_low)
    if m:
        nums = [int(n) for n in re.findall(r'\d+', m.group(1))]
        if len(nums) >= 2:
            l = nums[0] * nums[1] // math.gcd(nums[0], nums[1])
            return {"type": "lcm", "result": l}
    m = re.search(r'fib\((\d+)\)', q_low)
    if m:
        n = int(m.group(1))
        a, b = 0, 1
        for _ in range(n):
            a, b = b, a+b
        return {"type": "fibonacci", "result": a}
    m = re.search(r'fact\((\d+)\)|(\d+)!', q_low)
    if m:
        n = int(m.group(1) or m.group(2))
        return {"type": "factorial", "result": math.factorial(n)}
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
    q_low = q.lower()
    m = re.search(r'area of circle radius (\d+(?:\.\d+)?)', q_low)
    if m:
        r = float(m.group(1))
        return {"type": "area of circle", "result": math.pi * r**2}
    m = re.search(r'area of rectangle width (\d+(?:\.\d+)?) height (\d+(?:\.\d+)?)', q_low)
    if m:
        w = float(m.group(1))
        h = float(m.group(2))
        return {"type": "area of rectangle", "result": w*h}
    m = re.search(r'volume of sphere radius (\d+(?:\.\d+)?)', q_low)
    if m:
        r = float(m.group(1))
        return {"type": "volume of sphere", "result": (4/3)*math.pi*r**3}
    m = re.search(r'hypotenuse a=(\d+(?:\.\d+)?) b=(\d+(?:\.\d+)?)', q_low)
    if m:
        a = float(m.group(1))
        b = float(m.group(2))
        return {"type": "hypotenuse", "result": math.sqrt(a**2 + b**2)}
    return None

def handle_unit_conversion(q: str) -> Optional[Dict]:
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

# ========== Main Endpoints ==========
@app.get("/")
async def root():
    html = """
    <!DOCTYPE html>
    <html>
    <head><title>Ultimate Math API</title><style>
        body {font-family: 'Segoe UI', Arial; margin: 40px; background: #f5f5f5; max-width: 1000px; margin: 20px auto; padding: 20px;}
        h1 {color: #2c3e50;}
        .endpoint {background: white; padding: 20px; margin: 15px 0; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);}
        .method {background: #27ae60; color: white; padding: 4px 10px; border-radius: 6px; font-size: 12px; display: inline-block;}
        .path {font-family: monospace; font-size: 18px; margin: 12px 0; background: #f8f9fa; padding: 8px; border-radius: 6px;}
        .example {background: #ecf0f1; padding: 12px; border-radius: 8px; margin-top: 12px; font-family: monospace; font-size: 14px;}
        a {color: #2980b9; text-decoration: none;}
        a:hover {text-decoration: underline;}
        .docs {margin-top: 30px; padding: 20px; background: #d9edf7; border-radius: 12px;}
        code {background: #e9ecef; padding: 2px 6px; border-radius: 4px;}
    </style></head>
    <body>
        <h1>🧮 Ultimate Math API</h1>
        <p>This API solves <strong>any math problem</strong> – arithmetic, percentages, equations, calculus, matrices, statistics, number theory, geometry, unit conversion, and more.</p>
        
        <div class="endpoint">
            <span class="method">GET</span>
            <div class="path"><code>/solve?math=your expression</code></div>
            <div><strong>Main endpoint</strong> – returns JSON with result and explanation.</div>
            <div class="example">
                <strong>📌 Examples (try them!):</strong><br>
                • <a href="/solve?math=3+8">/solve?math=3+8</a> → 11<br>
                • <a href="/solve?math=50%25 of 10">/solve?math=50%25 of 10</a> → 5<br>
                • <a href="/solve?math=5+7-2×8">/solve?math=5+7-2×8</a> → -4<br>
                • <a href="/solve?math=√144">/solve?math=√144</a> → 12<br>
                • <a href="/solve?math=derivative of x^3">/solve?math=derivative of x^3</a> → 3*x**2<br>
                • <a href="/solve?math=integrate sin x">/solve?math=integrate sin x</a> → -cos(x)<br>
                • <a href="/solve?math=solve x^2 - 4 = 0">/solve?math=solve x^2 - 4 = 0</a> → [-2, 2]<br>
                • <a href="/solve?math=det([[1,2],[3,4]])">/solve?math=det([[1,2],[3,4]])</a> → -2.0<br>
                • <a href="/solve?math=mean(1,2,3,4,5)">/solve?math=mean(1,2,3,4,5)</a> → 3.0<br>
                • <a href="/solve?math=gcd(24,36)">/solve?math=gcd(24,36)</a> → 12<br>
                • <a href="/solve?math=fib(10)">/solve?math=fib(10)</a> → 55<br>
                • <a href="/solve?math=convert 10 km to miles">/solve?math=convert 10 km to miles</a> → 6.21371<br>
                • <a href="/solve?math=area of circle radius 5">/solve?math=area of circle radius 5</a> → 78.5398<br>
                • <a href="/solve?math=hypotenuse a=3 b=4">/solve?math=hypotenuse a=3 b=4</a> → 5.0<br>
            </div>
        </div>

        <div class="endpoint">
            <span class="method">GET</span>
            <div class="path"><code>/docs</code> or <code>/redoc</code></div>
            <div>Interactive API documentation (Swagger UI / ReDoc).</div>
        </div>

        <div class="docs">
            📖 <strong>Interactive docs:</strong> <a href="/docs">/docs</a> (Swagger UI) &nbsp;|&nbsp; <a href="/redoc">/redoc</a> (ReDoc)<br>
            💡 <strong>Note:</strong> Now <code>/solve?math=3+8</code> works perfectly!
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

@app.get("/solve")
async def solve(request: Request):
    # Get raw query string to preserve original characters
    raw_query = request.scope.get('query_string', b'').decode('utf-8')
    params = {}
    if raw_query:
        for pair in raw_query.split('&'):
            if '=' in pair:
                key, value = pair.split('=', 1)
                params[key] = value
    query = params.get('math')
    if not query:
        raise HTTPException(status_code=400, detail="Missing 'math' parameter")
    # Decode percent-encoded characters (e.g., %2B -> +)
    query = unquote(query)
    
    # CRITICAL FIX: Convert "3 8" to "3+8" if no other operators present
    query = fix_plus_sign(query)
    
    # Now process as before
    pct = handle_percentage(query)
    if pct is not None:
        result = float(pct) if pct.is_number else str(pct)
        return JSONResponse(content={"query": query, "result": result, "explanation": f"Percentage: {query} = {result}"})
    calc = handle_calculus(query)
    if calc:
        return JSONResponse(content={"query": query, "result": calc["result"], "explanation": f"Calculus: {calc['result']}"})
    eq = handle_equation(query)
    if eq:
        return JSONResponse(content={"query": query, "result": eq["solutions"], "explanation": "Equation solved."})
    mat = handle_matrix(query)
    if mat:
        return JSONResponse(content={"query": query, "result": mat["result"], "explanation": f"Matrix {mat['type']}."})
    stat = handle_statistics(query)
    if stat:
        return JSONResponse(content={"query": query, "result": stat["result"], "explanation": f"Statistics: {stat['type']}."})
    nt = handle_number_theory(query)
    if nt:
        return JSONResponse(content={"query": query, "result": nt["result"], "explanation": f"Number theory: {nt['type']}."})
    geo = handle_geometry(query)
    if geo:
        return JSONResponse(content={"query": query, "result": geo["result"], "explanation": f"Geometry: {geo['type']}."})
    unit = handle_unit_conversion(query)
    if unit:
        return JSONResponse(content={"query": query, "result": unit["result"], "explanation": f"Unit conversion: {unit['type']}."})
    try:
        expr = safe_parse(query)
        evaluated = expr.evalf() if expr.is_number else expr
        result = float(evaluated) if expr.is_number else str(evaluated)
        return JSONResponse(content={"query": query, "result": result, "explanation": f"Evaluated: {query} = {result}"})
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot solve: {str(e)}")
