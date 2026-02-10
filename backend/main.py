from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List, Any
import json
import os
import re
import time
import signal
import threading
import multiprocessing
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ---------- CONFIG ----------
load_dotenv()
API_KEY = os.environ.get("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not set!")

client = genai.Client(api_key=API_KEY)
PROBLEM_POOL_FILE = Path("problem_pool.json")
POOL_SIZE = 20
TARGET_PER_DIFFICULTY = POOL_SIZE // 4
MAX_WORKERS = 8
MAX_GENERATION_RETRIES = 3
CODE_EXECUTION_TIMEOUT = 5  # seconds

# ---------- FastAPI App ----------
app = FastAPI(title="SmartTalk API")

ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Thread Lock ----------
file_lock = threading.Lock()

# ---------- Generator State ----------
generator_running = False

# ---------- Models ----------
class CodeSubmission(BaseModel):
    code: str
    problem_index: int
    problem: str
    func_signature: str

class TestCase(BaseModel):
    input: str
    expected: str

class RunTestsRequest(BaseModel):
    code: str
    func_signature: str
    test_cases: List[TestCase]

# ---------- Pool Management (Atomic Read-Modify-Write) ----------
def load_problem_pool() -> Dict:
    """Load pool — caller MUST hold file_lock."""
    if PROBLEM_POOL_FILE.exists():
        try:
            return json.loads(PROBLEM_POOL_FILE.read_text())
        except json.JSONDecodeError:
            pass
    return {"Easy": [], "Medium": [], "Hard": [], "Expert": []}

def save_problem_pool(pool: Dict):
    """Save pool — caller MUST hold file_lock."""
    PROBLEM_POOL_FILE.write_text(json.dumps(pool, indent=2))

def get_pool_status() -> Dict[str, int]:
    with file_lock:
        pool = load_problem_pool()
    return {diff: len(pool.get(diff, [])) for diff in ["Easy", "Medium", "Hard", "Expert"]}

def atomic_add_problem(diff: str, problem: Dict) -> bool:
    """Atomically add a problem to the pool. Returns True if added."""
    with file_lock:
        pool = load_problem_pool()
        if diff not in pool:
            pool[diff] = []
        if len(pool[diff]) >= TARGET_PER_DIFFICULTY:
            return False
        pool[diff].append(problem)
        save_problem_pool(pool)
        return True

def atomic_pop_problems() -> List[Dict]:
    """Atomically pop one problem per difficulty for a quiz."""
    difficulties = ["Easy", "Medium", "Hard", "Expert"]
    with file_lock:
        pool = load_problem_pool()
        problems = []
        for diff in difficulties:
            if not pool.get(diff):
                raise HTTPException(
                    status_code=400,
                    detail=f"No {diff} problems available. Pool status: "
                           + ", ".join(f"{d}: {len(pool.get(d, []))}" for d in difficulties)
                )
            problem = pool[diff].pop(0)
            problems.append({
                "difficulty": diff,
                "problem": problem["problem"],
                "func_signature": problem.get("func_signature", "def solve() -> None:"),
                "class_definitions": problem.get("class_definitions", "")
            })
        save_problem_pool(pool)
    return problems

# ---------- Problem Generation ----------
def enforce_good_signature(sig: str, fallback: str) -> str:
    sig = sig.strip()
    if not sig or "(" not in sig or ")" not in sig:
        return fallback
    if re.match(r"def\s+calculate\s*\(", sig):
        return fallback
    return sig

def fix_markdown_formatting(text: str) -> str:
    """Fix common markdown formatting issues from LLM responses."""

    # Normalize 4+ backticks down to 3
    text = re.sub(r'`{4,}', '```', text)

    # Fix inline code immediately followed by stray triple backticks:
    #   `segment` ``` containing  ->  `segment` containing
    text = re.sub(r'(`[^`\n]+`)\s*```\s*', r'\1 ', text)

    # Fix stray ``` on a line by itself that isn't part of a code block pair
    # First, identify proper code block fences (``` optionally followed by a language)
    # and remove orphan ``` lines
    lines = text.split('\n')
    cleaned = []
    in_code_block = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('```'):
            if in_code_block:
                # Closing fence
                in_code_block = False
                cleaned.append(line)
            elif stripped == '```' and not in_code_block:
                # Could be orphan or opening — peek: skip orphan ``` on its own
                # Keep it only if there's a matching close later
                remaining = '\n'.join(lines[len(cleaned) + 1:])
                if '```' in remaining:
                    in_code_block = True
                    cleaned.append(line)
                # else: orphan, skip it
            else:
                # Opening fence with language tag like ```python
                in_code_block = True
                cleaned.append(line)
        else:
            # Remove stray triple backticks mid-line that aren't inline code
            # e.g. "backtrack(start)``` :" -> "backtrack(start):"
            line = re.sub(r'```\s*', '', line) if not in_code_block and '```' in line and not stripped.startswith('```') else line
            cleaned.append(line)
    text = '\n'.join(cleaned)

    # Ensure code block openers have language hint for Python code
    text = re.sub(r'```\s*\n(def |class |import |from )', r'```python\n\1', text)

    # Ensure code block opener is on its own line
    text = re.sub(r'([^\n])```python', r'\1\n```python', text)

    # If we have an unclosed code block, close it
    fence_count = len(re.findall(r'^```', text, re.MULTILINE))
    if fence_count % 2 != 0:
        text += '\n```'

    return text

def generate_one_problem(difficulty: str) -> Optional[Dict]:
    """Generate a problem using Gemini's JSON mode with retries."""

    prompt = f"""Generate a {difficulty} coding interview problem.

Include:
1. Problem description (clear and concise)
2. Input format
3. Output format
4. 2-3 examples with input/output and explanation
5. Constraints
6. Python function signature with type hints

IMPORTANT FORMATTING RULES for the "problem" field:
- Use standard markdown formatting
- Use code blocks with backticks for code: `code here`
- Use **bold** for emphasis
- ALL math expressions and constraints MUST be wrapped in dollar signs for LaTeX rendering
- Use $\\leq$ for less-than-or-equal, $\\geq$ for greater-than-or-equal, $\\times$ for multiplication
- Example constraints: $0 \\leq n \\leq 10^5$, $1 \\leq \\text{{s.length}} \\leq 10^4$
- Example complexity: $O(n \\log n)$
- NEVER write raw LaTeX without dollar signs - always wrap in $...$
- Add new lines after each section

Rules for func_signature:
- Must be a valid Python function signature like: def function_name(params: Type) -> ReturnType:
- No imports needed (typing symbols like List, Dict, Optional exist)
- Function name should be descriptive (not "calculate" or "solve")

Keep description under 300 words. Make examples clear and varied.

Return a JSON object with these exact keys:
- "problem": full problem description with examples (string)
- "func_signature": Python function signature (string)
- "class_definitions": any helper class definitions needed, or empty string (string)"""

    for attempt in range(MAX_GENERATION_RETRIES):
        try:
            start_time = time.time()
            response = client.models.generate_content(
                model="gemini-3-pro-preview",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema={
                        "type": "OBJECT",
                        "properties": {
                            "problem": {"type": "STRING"},
                            "func_signature": {"type": "STRING"},
                            "class_definitions": {"type": "STRING"},
                        },
                        "required": ["problem", "func_signature"],
                    },
                ),
            )
            elapsed = time.time() - start_time

            data = json.loads(response.text)

            # Validate required fields
            if not data.get("problem") or not data.get("func_signature"):
                print(f"Attempt {attempt+1}: Empty required fields for {difficulty}")
                continue

            data["func_signature"] = enforce_good_signature(
                data.get("func_signature", ""),
                fallback="def solve(nums: List[int]) -> int:"
            )
            data["class_definitions"] = data.get("class_definitions", "")
            data["difficulty"] = difficulty
            data["generation_time"] = elapsed

            return data

        except json.JSONDecodeError as e:
            print(f"Attempt {attempt+1}: JSON parse error for {difficulty}: {e}")
            continue
        except Exception as e:
            print(f"Attempt {attempt+1}: Error generating {difficulty}: {e}")
            if attempt < MAX_GENERATION_RETRIES - 1:
                time.sleep(1 * (attempt + 1))  # backoff
            continue

    print(f"All {MAX_GENERATION_RETRIES} attempts failed for {difficulty}")
    return None

def fill_pool_parallel():
    with file_lock:
        pool = load_problem_pool()
    difficulties = ["Easy", "Medium", "Hard", "Expert"]

    tasks = []
    for diff in difficulties:
        current = len(pool.get(diff, []))
        needed = TARGET_PER_DIFFICULTY - current
        if needed > 0:
            tasks.extend([diff] * needed)

    if not tasks:
        return 0

    print(f"Generating {len(tasks)} problems with {MAX_WORKERS} workers...")
    start_time = time.time()
    generated = 0

    for i, diff in enumerate(tasks):
        try:
            if i > 0:
                time.sleep(5)  # 2s delay between API calls to avoid rate limiting
            problem = generate_one_problem(diff)

            if problem:
                problem["generated_at"] = datetime.now().isoformat()
                if atomic_add_problem(diff, problem):
                    generated += 1
                    with file_lock:
                        count = len(load_problem_pool().get(diff, []))
                    print(f"{diff}: {count}/{TARGET_PER_DIFFICULTY}")

        except Exception as e:
            print(f"{diff} failed: {e}")

    elapsed = time.time() - start_time
    print(f"Batch complete: {generated} problems in {elapsed:.1f}s")

    return generated

def background_generator():
    global generator_running
    difficulties = ["Easy", "Medium", "Hard", "Expert"]

    while generator_running:
        with file_lock:
            pool = load_problem_pool()

        tasks = []
        for diff in difficulties:
            needed = TARGET_PER_DIFFICULTY - len(pool.get(diff, []))
            tasks.extend([diff] * max(0, needed))

        if not tasks:
            time.sleep(60)
            continue

        batch = tasks[:8]

        for i, diff in enumerate(batch):
            if not generator_running:
                break
            try:
                if i > 0:
                    time.sleep(2)  # 2s delay between API calls to avoid rate limiting
                problem = generate_one_problem(diff)
                if problem:
                    problem["generated_at"] = datetime.now().isoformat()
                    atomic_add_problem(diff, problem)
            except Exception as e:
                print(f"Background gen {diff} failed: {e}")

        time.sleep(3)

# ---------- Sandboxed Code Execution ----------
def _run_code_in_process(code: str, func_name: str, test_input_str: str, expected_str: str, conn):
    """Run user code in an isolated process with restricted globals."""
    try:
        exec_globals = {"__builtins__": {
            # Allow safe builtins only
            "len": len, "range": range, "int": int, "float": float, "str": str,
            "bool": bool, "list": list, "dict": dict, "set": set, "tuple": tuple,
            "min": min, "max": max, "sum": sum, "abs": abs, "round": round,
            "sorted": sorted, "reversed": reversed, "enumerate": enumerate,
            "zip": zip, "map": map, "filter": filter, "any": any, "all": all,
            "isinstance": isinstance, "issubclass": issubclass, "type": type,
            "repr": repr, "print": print, "hash": hash, "id": id,
            "chr": chr, "ord": ord, "hex": hex, "bin": bin, "oct": oct,
            "pow": pow, "divmod": divmod,
            "ValueError": ValueError, "TypeError": TypeError, "KeyError": KeyError,
            "IndexError": IndexError, "StopIteration": StopIteration,
            "RuntimeError": RuntimeError, "ZeroDivisionError": ZeroDivisionError,
            "Exception": Exception, "True": True, "False": False, "None": None,
        }}

        # Import typing symbols
        from typing import List, Dict, Set, Tuple, Optional, Union
        exec_globals.update({
            "List": List, "Dict": Dict, "Set": Set, "Tuple": Tuple,
            "Optional": Optional, "Union": Union,
        })

        exec(code, exec_globals)

        if func_name not in exec_globals:
            conn.send({"status": "error", "message": f"Function '{func_name}' not found"})
            return

        user_function = exec_globals[func_name]

        # Use a safe eval namespace too
        eval_globals = dict(exec_globals)
        test_input = eval(test_input_str, eval_globals)

        if isinstance(test_input, tuple):
            output = user_function(*test_input)
        else:
            output = user_function(test_input)

        result = {"status": "executed", "output": repr(output)}

        if expected_str:
            expected = eval(expected_str, eval_globals)
            result["status"] = "passed" if output == expected else "failed"

        conn.send(result)

    except Exception as e:
        conn.send({"status": "error", "message": f"{type(e).__name__}: {str(e)}"})


def run_code_sandboxed(code: str, func_name: str, test_input: str, expected: str) -> Dict:
    """Execute user code in a separate process with a timeout."""
    parent_conn, child_conn = multiprocessing.Pipe()

    proc = multiprocessing.Process(
        target=_run_code_in_process,
        args=(code, func_name, test_input, expected, child_conn),
        daemon=True,
    )
    proc.start()
    proc.join(timeout=CODE_EXECUTION_TIMEOUT)

    if proc.is_alive():
        proc.kill()
        proc.join(timeout=1)
        return {"status": "error", "message": f"Execution timed out ({CODE_EXECUTION_TIMEOUT}s limit)"}

    if parent_conn.poll():
        return parent_conn.recv()
    else:
        return {"status": "error", "message": "No result from execution (process may have crashed)"}


# ---------- API Endpoints ----------
@app.get("/")
def root():
    return {"message": "SmartTalk API", "status": "running"}

@app.get("/pool/status")
def pool_status():
    status = get_pool_status()
    total = sum(status.values())
    return {
        "status": status,
        "total": total,
        "target": POOL_SIZE,
        "ready": total >= 4,
        "generator_running": generator_running
    }

@app.post("/pool/generate")
def generate_pool(background_tasks: BackgroundTasks):
    """Generate problems to fill the pool"""
    background_tasks.add_task(fill_pool_parallel)
    return {"message": "Generation started"}

@app.post("/generator/start")
def start_generator():
    global generator_running
    if not generator_running:
        generator_running = True
        thread = threading.Thread(target=background_generator, daemon=True)
        thread.start()
        return {"message": "Generator started"}
    return {"message": "Generator already running"}

@app.post("/generator/stop")
def stop_generator():
    global generator_running
    generator_running = False
    return {"message": "Generator stopped"}

@app.get("/quiz/start")
def start_quiz():
    """Get 4 problems (one of each difficulty) to start a quiz"""
    problems = atomic_pop_problems()
    return {"problems": problems}

@app.post("/quiz/submit")
def submit_solution(submission: CodeSubmission):
    """Analyze submitted code and return score + feedback"""

    ai_prompt = f"""You are a coding interviewer. Analyze this solution:

Problem:
{submission.problem}

Function signature: {submission.func_signature}

User's code:
{submission.code}

Provide:
1. Score (0-10)
2. Brief feedback on correctness, style, and edge cases

IMPORTANT: Format your response EXACTLY as shown below.
Start your response with "SCORE:" followed by a number, then "FEEDBACK:" followed by your feedback.
Use markdown formatting for feedback (use `code` for inline code, **bold** for emphasis).

SCORE: [number]
FEEDBACK: [your feedback with markdown formatting]"""

    try:
        response = client.models.generate_content(
            model="gemini-3-pro-preview",
            contents=ai_prompt
        )

        response_text = fix_markdown_formatting(response.text)

        score_match = re.search(r'SCORE:\s*(\d+)', response_text)
        score = int(score_match.group(1)) if score_match else 5
        score = max(0, min(10, score))  # clamp to 0-10

        return {
            "score": score,
            "feedback": response_text,
            "problem_index": submission.problem_index
        }

    except Exception as e:
        print(f"ERROR in submit_code: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to evaluate submission: {str(e)}")

@app.post("/quiz/give-up")
def give_up_solution(submission: CodeSubmission):
    """Generate solution when user gives up and return 0 score"""

    # Log the incoming data for debugging
    print(f"Give up request received:")
    print(f"  - Problem index: {submission.problem_index}")
    print(f"  - Func signature: {submission.func_signature}")
    print(f"  - Problem length: {len(submission.problem) if submission.problem else 0}")
    print(f"  - Code length: {len(submission.code) if submission.code else 0}")

    ai_prompt = f"""You are a coding interviewer. The user gave up on this problem. Generate a complete, correct solution.

Problem:
{submission.problem}

Function signature: {submission.func_signature}

Provide:
1. A complete, working Python solution with clear comments
2. Brief explanation of the approach
3. Time and space complexity analysis

CRITICAL FORMATTING RULES:
- You MUST use proper markdown code blocks with triple backticks
- Format code blocks as: ```python (on its own line) then code then ``` (on its own line)
- Do NOT use single backticks for multi-line code
- Use simple text for Big-O notation like O(n), O(1), etc.

Format your response EXACTLY as:

**SOLUTION:**
```python
[complete code here with comments]
```

**EXPLANATION:**
[explanation of the approach here]

**COMPLEXITY:**
- Time: O(n)
- Space: O(1)"""

    try:
        response = client.models.generate_content(
            model="gemini-3-pro-preview",
            contents=ai_prompt
        )

        response_text = fix_markdown_formatting(response.text)

        return {
            "score": 0,
            "feedback": response_text,
            "problem_index": submission.problem_index,
            "gave_up": True
        }

    except Exception as e:
        print(f"ERROR in give_up_solution: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to generate solution: {str(e)}")

@app.post("/quiz/run-tests")
def run_tests(request: RunTestsRequest):
    """Run test cases against user code in a sandboxed process with timeout."""

    func_sig = request.func_signature
    match = re.search(r'def\s+(\w+)\s*\(', func_sig)
    if not match:
        raise HTTPException(status_code=400, detail="Could not extract function name")

    func_name = match.group(1)

    # Basic code safety check — reject obvious dangerous patterns
    dangerous_patterns = [
        r'\bimport\s+os\b', r'\bimport\s+sys\b', r'\bimport\s+subprocess\b',
        r'\bimport\s+shutil\b', r'\b__import__\b', r'\beval\b', r'\bexec\b',
        r'\bopen\s*\(', r'\bos\.', r'\bsys\.', r'\bsubprocess\.',
        r'\bglobals\s*\(', r'\blocals\s*\(', r'\bgetattr\b', r'\bsetattr\b',
        r'\bdelattr\b', r'\bcompile\b',
    ]
    for pattern in dangerous_patterns:
        if re.search(pattern, request.code):
            return {
                "results": [{"test_num": 1, "status": "error",
                             "message": f"Disallowed construct detected: {pattern}"}],
                "passed": 0, "total": len(request.test_cases)
            }

    results = []

    for i, test_case in enumerate(request.test_cases):
        result = {"test_num": i + 1, "input": test_case.input, "expected": test_case.expected}

        exec_result = run_code_sandboxed(
            request.code, func_name, test_case.input, test_case.expected
        )

        result["status"] = exec_result.get("status", "error")
        if "output" in exec_result:
            result["output"] = exec_result["output"]
        if "message" in exec_result:
            result["message"] = exec_result["message"]

        results.append(result)

    passed = sum(1 for r in results if r["status"] == "passed")

    return {
        "results": results,
        "passed": passed,
        "total": len(results)
    }

@app.post("/pool/clear")
def clear_pool():
    """Clear the problem pool"""
    with file_lock:
        if PROBLEM_POOL_FILE.exists():
            PROBLEM_POOL_FILE.unlink()
    return {"message": "Pool cleared"}

# Start generator on startup
@app.on_event("startup")
async def startup_event():
    global generator_running
    generator_running = True
    thread = threading.Thread(target=background_generator, daemon=True)
    thread.start()
    print("Background generator started")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)