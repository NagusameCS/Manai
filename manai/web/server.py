"""
Manai Web Server - FastAPI backend for web UI
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Manai imports
from manai.client import ManaiClient, ManaiConfig
from manai.ollama_client import OllamaClient, OllamaConfig
from manai.generator import ManimConfig
from manai.agent import Agent, ThinkingMode
from manai.math_engine import math_engine
from manai.templates_2d import list_2d_templates, get_2d_template
from manai.templates_3d import list_3d_templates, get_3d_template
from manai.transitions import list_transitions
from manai.voiceover import voiceover_engine, VoiceConfig, VOICES


# =============================================================================
# APP SETUP
# =============================================================================

app = FastAPI(
    title="Manai",
    description="AI-powered Math & Physics Video Generator",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# State
class AppState:
    client: Optional[ManaiClient] = None
    ollama: Optional[OllamaClient] = None
    agent: Optional[Agent] = None
    current_model: str = None  # Auto-detect on startup
    last_code: str = ""
    last_video: str = ""

state = AppState()


def auto_detect_model() -> str:
    """Auto-detect the first available Ollama model."""
    try:
        ollama = OllamaClient()
        models = ollama.list_models()
        if models:
            return models[0].get("name", "llama3.2")
    except:
        pass
    return "llama3.2"


# =============================================================================
# MODELS
# =============================================================================

class GenerateRequest(BaseModel):
    description: str
    topic: Optional[str] = None
    level: str = "general"
    thinking_mode: str = "standard"
    voice: Optional[str] = None  # Voice for narration
    add_captions: bool = True


class VoiceoverRequest(BaseModel):
    text: str
    voice: str = "aria"


class EquationRequest(BaseModel):
    equation: str
    title: Optional[str] = None
    derivation: bool = True

class TemplateRequest(BaseModel):
    name: str
    
class VerifyRequest(BaseModel):
    expression: str
    
class ModelRequest(BaseModel):
    model: str


# =============================================================================
# INITIALIZATION
# =============================================================================

def get_client(model: str = None) -> ManaiClient:
    """Get or create the Manai client."""
    # Auto-detect model if not set
    if state.current_model is None:
        state.current_model = auto_detect_model()
    
    model = model or state.current_model
    
    if state.client is None or state.current_model != model:
        config = ManaiConfig(
            ollama=OllamaConfig(model=model),
            manim=ManimConfig(quality="high", output_dir="./output")
        )
        state.client = ManaiClient(config=config)
        state.ollama = OllamaClient(OllamaConfig(model=model))
        state.agent = Agent(state.ollama, ThinkingMode.STANDARD)
        state.current_model = model
    
    return state.client


# =============================================================================
# STREAMING AGENT THINKING
# =============================================================================

async def stream_agent_thinking(websocket: WebSocket, description: str, thinking_mode: str) -> dict:
    """
    Stream agent thinking process over WebSocket for real-time UI updates.
    
    This replaces the blocking agent.process() call with streaming output.
    """
    import re
    step = 0
    
    async def send_step(label: str, content: str, step_type: str = "thinking"):
        nonlocal step
        step += 1
        await websocket.send_json({
            "type": "thinking_step",
            "step": step,
            "content": f"**{label}**: {content}",
            "step_type": step_type
        })
        await asyncio.sleep(0.05)  # Small delay for UI update
    
    # Get thinking prompt based on mode
    mode_prompts = {
        "quick": "Briefly analyze this request and identify key elements:",
        "standard": "Analyze this request step by step for a COMPREHENSIVE educational video:",
        "deep": "Deeply analyze this request with mathematical rigor for a DETAILED long-form video:",
        "verify": "Analyze and identify all mathematical claims to verify:"
    }
    
    base_prompt = mode_prompts.get(thinking_mode, mode_prompts["standard"])
    
    # Build the thinking prompt
    thinking_prompt = f"""You are an expert educational video creator for math and physics topics.
{base_prompt}

REQUEST: {description}

IMPORTANT: Plan for a COMPREHENSIVE video (10-30 minutes) that includes:
- Multiple sections and subtopics
- Several worked examples with step-by-step solutions
- Visual demonstrations and intuitive explanations
- Mathematical derivations where appropriate
- Real-world applications

Respond with your analysis in JSON format:
{{
    "understanding": "What the user wants - be specific about depth and scope",
    "concepts": ["List of 5-10 math/physics concepts to cover"],
    "approach": "Detailed visualization approach with multiple sections",
    "scene_type": "2D or 3D",
    "elements": ["All Manim elements needed"],
    "animation_steps": ["Step 1: Introduction", "Step 2: Background", "Step 3: Main concept", "Step 4: Example 1", "Step 5: Example 2", "Step 6: Derivation", "Step 7: Applications", "Step 8: Summary"],
    "tools_needed": ["calculate", "lookup_docs", etc.],
    "estimated_duration": "15-25 minutes",
    "confidence": 0.0 to 1.0
}}"""

    try:
        # Stream the thinking process
        await send_step("Analyzing", "Processing your request...", "init")
        
        thinking_chunks = []
        found_json = False
        
        # Stream from LLM with real-time output
        for chunk in state.ollama.generate(thinking_prompt, stream=True):
            thinking_chunks.append(chunk)
            current_text = "".join(thinking_chunks)
            
            # Send periodic thinking updates (every ~50 chars)
            if len(thinking_chunks) % 10 == 0:
                # Try to parse partial JSON for early extraction
                if not found_json:
                    try:
                        json_match = re.search(r'\{[\s\S]*?\}', current_text)
                        if json_match:
                            partial = json.loads(json_match.group())
                            
                            if "understanding" in partial and step < 2:
                                await send_step("Understanding", partial["understanding"][:150], "understanding")
                            
                            if "concepts" in partial and step < 3:
                                concepts = ", ".join(partial["concepts"][:4])
                                await send_step("Concepts", concepts, "concepts")
                                
                            if "approach" in partial and step < 4:
                                await send_step("Approach", partial["approach"][:120], "approach")
                                found_json = True
                    except json.JSONDecodeError:
                        pass
        
        # Parse final result
        full_response = "".join(thinking_chunks)
        
        try:
            json_match = re.search(r'\{[\s\S]*\}', full_response)
            if json_match:
                thinking = json.loads(json_match.group())
            else:
                thinking = {"understanding": full_response[:200], "confidence": 0.5}
        except json.JSONDecodeError:
            thinking = {"understanding": full_response[:200], "confidence": 0.5}
        
        # Send parsed thinking steps if not already sent
        if step < 2 and "understanding" in thinking:
            await send_step("Understanding", thinking["understanding"][:150], "understanding")
        
        if step < 3 and "concepts" in thinking:
            concepts = ", ".join(thinking["concepts"][:5])
            await send_step("Key Concepts", concepts, "concepts")
        
        if step < 4 and "approach" in thinking:
            await send_step("Visualization Approach", thinking["approach"][:150], "approach")
        
        if "scene_type" in thinking:
            await send_step("Scene Type", thinking["scene_type"], "scene")
        
        # Execute any tool calls
        tools_needed = thinking.get("tools_needed", [])
        tool_outputs = []
        
        for tool_name in tools_needed[:3]:  # Limit to 3 tools
            if tool_name in state.agent.tools:
                await send_step(f"Using Tool", f"{tool_name}...", "tool")
                
                try:
                    # Execute tool with default args
                    if tool_name == "lookup_docs":
                        result = state.agent.tools[tool_name].execute(
                            query=description[:50]
                        )
                    elif tool_name == "calculate":
                        # Extract first equation-like thing
                        eq_match = re.search(r'[a-z]\s*=\s*[^,]+', description.lower())
                        if eq_match:
                            result = state.agent.tools[tool_name].execute(
                                expression=eq_match.group()
                            )
                        else:
                            continue
                    else:
                        continue
                    
                    if result.success:
                        await send_step(f"Tool Result", f"{tool_name}: {str(result.output)[:80]}", "tool_result")
                        tool_outputs.append({
                            "tool": tool_name,
                            "success": True,
                            "output": result.output
                        })
                except Exception as e:
                    await send_step("Tool Warning", f"{tool_name} failed: {str(e)[:50]}", "warning")
        
        # Final planning step
        if "animation_steps" in thinking:
            steps_preview = " → ".join(thinking["animation_steps"][:4])
            await send_step("Animation Plan", steps_preview, "plan")
        
        confidence = thinking.get("confidence", 0.7)
        await send_step("Confidence", f"{int(confidence * 100)}%", "confidence")
        
        return {
            "thinking": thinking,
            "tool_outputs": tool_outputs,
            "plan": thinking.get("animation_steps", []),
            "confidence": confidence
        }
        
    except Exception as e:
        await send_step("Error", f"Thinking process failed: {str(e)}", "error")
        return {
            "thinking": {},
            "tool_outputs": [],
            "plan": [],
            "confidence": 0.3
        }


# =============================================================================
# API ROUTES
# =============================================================================

@app.get("/")
async def root():
    """Serve the main UI."""
    return HTMLResponse(content=get_html(), status_code=200)


@app.get("/api/status")
async def status():
    """Get system status."""
    try:
        ollama = OllamaClient()
        models = ollama.list_models()
        return {
            "status": "ok",
            "current_model": state.current_model,
            "models": [m.get("name", "") for m in models],
            "last_video": state.last_video,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/api/models")
async def list_models():
    """List available Ollama models."""
    try:
        ollama = OllamaClient()
        models = ollama.list_models()
        return {
            "models": models,
            "current": state.current_model
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/models/switch")
async def switch_model(request: ModelRequest):
    """Switch the active model."""
    state.current_model = request.model
    state.client = None  # Force recreation
    get_client(request.model)
    return {"success": True, "model": request.model}


@app.get("/api/templates")
async def get_templates():
    """Get all available templates."""
    return {
        "2d": list_2d_templates(),
        "3d": list_3d_templates(),
        "transitions": list_transitions()
    }


@app.post("/api/templates/get")
async def get_template(request: TemplateRequest):
    """Get template code by name."""
    code = get_2d_template(request.name) or get_3d_template(request.name)
    if code:
        return {"success": True, "code": code.strip()}
    return {"success": False, "error": "Template not found"}


@app.post("/api/generate")
async def generate(request: GenerateRequest):
    """Generate Manim code from description."""
    try:
        client = get_client()
        
        # Use agent for thinking
        thinking_mode = {
            "quick": ThinkingMode.QUICK,
            "standard": ThinkingMode.STANDARD,
            "deep": ThinkingMode.DEEP,
            "verify": ThinkingMode.VERIFY,
        }.get(request.thinking_mode, ThinkingMode.STANDARD)
        
        state.agent.set_thinking_mode(thinking_mode)
        
        # Process with agent first
        agent_result = state.agent.process(request.description)
        
        # Generate code
        result = client.create(
            description=request.description,
            topic=request.topic,
            level=request.level,
            render=request.render,
            preview=False
        )
        
        state.last_code = result.code or ""
        if result.output_path:
            state.last_video = str(result.output_path)
        
        return {
            "success": result.success,
            "code": result.code,
            "output_path": str(result.output_path) if result.output_path else None,
            "error": result.error,
            "thinking": {
                "plan": agent_result.get("plan", []),
                "confidence": agent_result.get("confidence", 0),
                "tool_outputs": agent_result.get("tool_outputs", [])
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/verify")
async def verify(request: VerifyRequest):
    """Verify a mathematical expression."""
    try:
        result = math_engine.parse(request.expression)
        
        response = {
            "success": result.success,
            "latex": result.latex,
            "error": result.error
        }
        
        if result.success:
            simplified = math_engine.simplify_expr(request.expression)
            if simplified.success:
                response["simplified"] = simplified.latex
        
        return response
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/video/{filename}")
async def get_video(filename: str):
    """Serve a generated video file."""
    output_dir = Path("./output")
    
    # First check direct path
    video_path = output_dir / filename
    if video_path.exists():
        return FileResponse(video_path, media_type="video/mp4")
    
    # Search recursively in output/videos subdirectories
    for f in output_dir.rglob(filename):
        if f.is_file():
            return FileResponse(f, media_type="video/mp4")
    
    # Also try in the videos subdirectory patterns
    for f in output_dir.rglob(f"**/1080p60/{filename}"):
        if f.is_file():
            return FileResponse(f, media_type="video/mp4")
    
    raise HTTPException(status_code=404, detail=f"Video not found: {filename}")


@app.get("/api/gallery")
async def get_gallery():
    """Get list of all generated videos."""
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)
    
    videos = []
    # Search recursively for all mp4 files
    for f in sorted(output_dir.rglob("*.mp4"), key=lambda x: x.stat().st_mtime, reverse=True):
        videos.append({
            "name": f.name,
            "path": f"/api/video/{f.name}",
            "size": f.stat().st_size,
            "created": f.stat().st_mtime
        })
    
    return {"videos": videos[:50]}  # Last 50 videos


@app.get("/api/voices")
async def get_voices():
    """Get available TTS voices."""
    return {
        "voices": list(VOICES.keys()),
        "default": "aria"
    }


@app.post("/api/voiceover")
async def create_voiceover(request: VoiceoverRequest):
    """Generate voiceover audio and subtitles."""
    try:
        voiceover_engine.set_voice(request.voice)
        audio_path, subtitles = voiceover_engine.generate_audio(request.text)
        
        # Generate SRT
        srt_path = audio_path.with_suffix('.srt')
        voiceover_engine.generate_srt(subtitles, srt_path)
        
        return {
            "success": True,
            "audio": f"/api/audio/{audio_path.name}",
            "srt": f"/api/audio/{srt_path.name}"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/audio/{filename}")
async def get_audio(filename: str):
    """Serve audio files."""
    audio_path = Path("./output/audio") / filename
    if audio_path.exists():
        media_type = "audio/mpeg" if filename.endswith(".mp3") else "text/plain"
        return FileResponse(audio_path, media_type=media_type)
    raise HTTPException(status_code=404, detail="Audio not found")


# =============================================================================
# WEBSOCKET FOR STREAMING
# =============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time streaming during generation."""
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "generate":
                description = message.get("description", "")
                thinking_mode = message.get("thinking_mode", "standard")
                
                # Phase 1: Initialize
                await websocket.send_json({"type": "phase", "phase": "init", "content": "Initializing..."})
                client = get_client()
                
                # Phase 2: Agent Thinking with streaming
                await websocket.send_json({"type": "phase", "phase": "thinking", "content": "Agent analyzing request..."})
                
                mode_map = {
                    "quick": ThinkingMode.QUICK,
                    "standard": ThinkingMode.STANDARD,
                    "deep": ThinkingMode.DEEP,
                    "verify": ThinkingMode.VERIFY,
                }
                state.agent.set_thinking_mode(mode_map.get(thinking_mode, ThinkingMode.STANDARD))
                
                # Stream agent thinking process
                agent_result = await stream_agent_thinking(websocket, description, thinking_mode)
                
                # Phase 3: Code Generation with streaming
                await websocket.send_json({"type": "phase", "phase": "generating", "content": "Generating intuitive visual explanation..."})
                
                # Build prompt for COMPREHENSIVE, TOPIC-SPECIFIC content
                generation_prompt = f"""Create a COMPREHENSIVE Manim animation (2-3 MINUTES LONG) that fully explains: {description}

THIS VIDEO MUST:
1. Be directly about "{description}" - every visual must relate to this specific topic
2. Be educational and thorough - assume the viewer knows nothing
3. Include multiple animated demonstrations, not just static equations
4. Run for at LEAST 90 seconds (self.wait() calls should total 60+ seconds)

===== REQUIRED STRUCTURE (FOLLOW EXACTLY) =====

PART 1 - INTRODUCTION (15-20 seconds):
- Title showing "{description}"
- Brief text explaining WHY this matters
- Fade to visual setup
Code pattern:
```
title = Text("{description}").scale(0.6)
self.play(Write(title))
self.wait(2)
subtitle = Text("Why this matters...").scale(0.4).next_to(title, DOWN)
self.play(Write(subtitle))
self.wait(3)
self.play(FadeOut(title), FadeOut(subtitle))
```

PART 2 - CORE CONCEPT VISUALIZATION (40-60 seconds):
- Show the physical/geometric setup (Axes, shapes, objects)
- Use ValueTracker + always_redraw for ANIMATED demonstrations
- Multiple self.wait(2-3) pauses for viewer to absorb
- Add labels and annotations as things move
Code pattern:
```
axes = Axes(x_range=[-4, 4], y_range=[-3, 3], x_length=7, y_length=5)
self.play(Create(axes))
self.wait(1)

tracker = ValueTracker(start)
dynamic = always_redraw(lambda: SomeObject(...).move_to(axes.c2p(tracker.get_value(), ...)))
self.play(Create(dynamic))
self.play(tracker.animate.set_value(end), run_time=4)  # Slow animation!
self.wait(2)
```

PART 3 - DERIVE THE EQUATION (30-40 seconds):
- Build the equation piece by piece
- Show WHERE each term comes from visually
- Use TransformMatchingShapes or ReplacementTransform
- Highlight with color coding
Code pattern:
```
step1 = MathTex(r"\\text{{term1}}").scale(0.5)
self.play(Write(step1))
self.wait(2)
step2 = MathTex(r"\\text{{term1}} = \\text{{term2}}").scale(0.5)
self.play(TransformMatchingShapes(step1, step2))
self.wait(2)
```

PART 4 - EXAMPLE APPLICATION (20-30 seconds):
- Show a specific example with actual numbers
- Verify the concept works
- Final summary
Code pattern:
```
example = Text("Example: ...").scale(0.4)
self.play(Write(example))
self.wait(2)
# Show calculation with actual numbers
```

===== CRITICAL RULES =====
1. Screen bounds: x ∈ [-6.5, 6.5], y ∈ [-3.5, 3.5]
2. ALWAYS scale text: .scale(0.4) to .scale(0.6)
3. ALWAYS scale MathTex: .scale(0.5) to .scale(0.6)
4. Use positioning: .to_edge(UP), .to_corner(UL), .shift(), .next_to()
5. Clear screen between major parts with FadeOut()
6. Include self.wait(1) to self.wait(3) between ALL steps
7. Total video should be 90-180 seconds

===== FORBIDDEN =====
- Text larger than scale(0.6)
- MathTex larger than scale(0.6)
- Elements outside screen bounds
- Videos shorter than 60 seconds
- Content unrelated to "{description}"
- Static slideshows (must have ValueTracker animations)

Generate the complete Python code now. The class should be called `{description.replace(' ', '').replace('-', '').replace("'", '')}Explanation`."""
                
                # Stream the code as it's generated
                code_chunks = []
                chunk_count = 0
                try:
                    # Generate with streaming - send every chunk for real-time preview
                    for chunk in state.ollama.generate(
                        prompt=generation_prompt,
                        system_prompt=client.generator.get_system_prompt(),
                        stream=True
                    ):
                        code_chunks.append(chunk)
                        chunk_count += 1
                        
                        # Send code updates every 2 chunks for smooth streaming
                        if chunk_count % 2 == 0:
                            await websocket.send_json({
                                "type": "code_stream",
                                "content": "".join(code_chunks)
                            })
                            await asyncio.sleep(0.01)  # Small delay for UI
                    
                    full_code = "".join(code_chunks)
                    # Extract code from response
                    extracted = client.generator.extract_code(full_code)
                    if extracted:
                        full_code = extracted
                        
                except Exception as e:
                    # Fallback to non-streaming
                    result = client.create(description, render=False, preview=False)
                    full_code = result.code or ""
                
                await websocket.send_json({
                    "type": "code_complete",
                    "content": full_code
                })
                
                # Phase 4: Code Validation - fix common errors
                invalid_classes = {
                    'ThreeDMathTex': 'MathTex',
                    'ThreeDText': 'Text',
                    'ThreeDTex': 'Tex',
                    '3DAxes': 'ThreeDAxes',
                    'ThreeDFrameBox': 'SurroundingRectangle',
                    'ThreeDTitle': 'Title',
                    'ThreeDLabel': 'Text',
                    'ThreeDDot': 'Dot3D',
                    'FrameBox': 'SurroundingRectangle',
                }
                
                # Also fix common syntax issues
                code_fixes = [
                    # Fix bullet point lists that cause issues
                    (r'\\begin{itemize}', ''),
                    (r'\\end{itemize}', ''),
                    (r'\\item ', '• '),
                ]
                
                code_fixed = False
                for invalid, valid in invalid_classes.items():
                    if invalid in full_code:
                        full_code = full_code.replace(invalid, valid)
                        code_fixed = True
                        await websocket.send_json({
                            "type": "thinking_step",
                            "step": 99,
                            "content": f"**Auto-fix**: Replaced invalid `{invalid}` with `{valid}`",
                            "step_type": "warning"
                        })
                
                # Apply regex-based fixes
                import re as regex_module
                for pattern, replacement in code_fixes:
                    if regex_module.search(pattern, full_code):
                        full_code = regex_module.sub(pattern, replacement, full_code)
                        code_fixed = True
                
                if code_fixed:
                    await websocket.send_json({
                        "type": "code_complete",
                        "content": full_code
                    })
                
                # Phase 5: Auto-Verification
                await websocket.send_json({"type": "phase", "phase": "verifying", "content": "Verifying math expressions..."})
                
                # Extract and verify any math expressions
                import re
                math_patterns = [
                    r'MathTex\(["\'](.+?)["\']\)',
                    r'Tex\(["\'](.+?)["\']\)',
                    r'equation\s*=\s*["\'](.+?)["\']',
                ]
                
                verified_expressions = []
                for pattern in math_patterns:
                    for match in re.finditer(pattern, full_code):
                        expr = match.group(1)
                        try:
                            result = math_engine.parse(expr.replace('\\\\', '\\'))
                            verified_expressions.append({
                                "expression": expr[:50],
                                "valid": result.success,
                                "latex": result.latex if result.success else None
                            })
                        except:
                            pass
                
                if verified_expressions:
                    await websocket.send_json({
                        "type": "verification",
                        "expressions": verified_expressions
                    })
                
                # Phase 5: Rendering
                await websocket.send_json({"type": "phase", "phase": "rendering", "content": "Rendering video..."})
                
                # Render the code
                render_result = client.generator.render(full_code)
                
                state.last_code = full_code
                
                video_path = None
                voiced_video = None
                srt_path = None
                
                if render_result.success and render_result.output_path:
                    state.last_video = str(render_result.output_path)
                    video_path = render_result.output_path
                    filename = Path(render_result.output_path).name
                    
                    # Phase 6: Voiceover (if requested)
                    voice = message.get("voice")
                    add_captions = message.get("add_captions", True)
                    
                    if voice:
                        try:
                            await websocket.send_json({"type": "phase", "phase": "voiceover", "content": "Generating voiceover..."})
                            
                            # Generate narration script
                            narration = voiceover_engine.generate_narration_script(full_code, description)
                            
                            # Generate audio
                            voiceover_engine.set_voice(voice)
                            audio_path, subtitles = await voiceover_engine.generate_audio_async(narration)
                            
                            # Generate SRT
                            srt_file = audio_path.with_suffix('.srt')
                            voiceover_engine.generate_srt(subtitles, srt_file)
                            srt_path = str(srt_file)
                            
                            # Merge audio with video
                            await websocket.send_json({"type": "phase", "phase": "merging", "content": "Adding voiceover to video..."})
                            
                            voiced_path = voiceover_engine.merge_audio_video(
                                video_path,
                                audio_path,
                                subtitle_path=srt_file if add_captions else None
                            )
                            voiced_video = f"/api/video/{voiced_path.name}"
                            
                        except Exception as vo_error:
                            await websocket.send_json({
                                "type": "warning",
                                "content": f"Voiceover failed: {str(vo_error)}"
                            })
                    
                    await websocket.send_json({
                        "type": "complete",
                        "code": full_code,
                        "video": voiced_video or f"/api/video/{filename}",
                        "original_video": f"/api/video/{filename}",
                        "srt": f"/api/audio/{Path(srt_path).name}" if srt_path else None,
                        "verified": len(verified_expressions),
                        "confidence": agent_result.get("confidence", 0)
                    })
                else:
                    await websocket.send_json({
                        "type": "error",
                        "content": render_result.error or "Rendering failed",
                        "code": full_code
                    })
                    
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "content": str(e)})
        except:
            pass


# =============================================================================
# HTML UI
# =============================================================================

def get_html() -> str:
    """Return the main HTML page."""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Manai</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        :root {
            --bg-primary: #0a0a0a;
            --bg-secondary: #141414;
            --bg-tertiary: #1e1e1e;
            --bg-hover: #282828;
            --text-primary: #e0e0e0;
            --text-secondary: #a0a0a0;
            --text-muted: #606060;
            --border: #2a2a2a;
            --accent: #888;
            --success: #4a4a4a;
            --error: #8a4a4a;
        }
        
        body {
            font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
        }
        
        /* Header */
        header {
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border);
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .logo {
            font-size: 1.5rem;
            font-weight: 600;
            letter-spacing: -0.02em;
        }
        
        .logo span {
            color: var(--text-muted);
            font-weight: 400;
        }
        
        .header-controls {
            display: flex;
            gap: 1rem;
            align-items: center;
        }
        
        /* Model Selector */
        .model-selector {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .model-selector label {
            color: var(--text-muted);
            font-size: 0.8rem;
        }
        
        select {
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            color: var(--text-primary);
            padding: 0.5rem 1rem;
            border-radius: 0;
            font-family: inherit;
            cursor: pointer;
        }
        
        select:focus {
            outline: none;
            border-color: var(--accent);
        }
        
        /* Main Layout */
        .container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            height: calc(100vh - 90px);
            overflow: hidden;
        }
        
        /* Left Panel - Input */
        .input-panel {
            border-right: 1px solid var(--border);
            display: flex;
            flex-direction: column;
        }
        
        .panel-header {
            padding: 1rem;
            border-bottom: 1px solid var(--border);
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: var(--text-muted);
        }
        
        .prompt-area {
            flex: 1;
            padding: 1rem;
            display: flex;
            flex-direction: column;
        }
        
        textarea {
            flex: 1;
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            border-radius: 0;
            color: var(--text-primary);
            padding: 1rem;
            font-family: inherit;
            font-size: 0.9rem;
            resize: none;
            line-height: 1.6;
            min-height: 150px;
        }
        
        textarea:focus {
            outline: none;
            border-color: var(--accent);
        }
        
        textarea::placeholder {
            color: var(--text-muted);
        }
        
        /* Controls */
        .controls {
            padding: 1rem;
            border-top: 1px solid var(--border);
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
        }
        
        button {
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            color: var(--text-primary);
            padding: 0.6rem 1.2rem;
            border-radius: 0;
            font-family: inherit;
            font-size: 0.85rem;
            cursor: pointer;
            transition: all 0.15s;
        }
        
        button:hover {
            background: var(--bg-hover);
            border-color: var(--accent);
        }
        
        button.primary {
            background: var(--text-primary);
            color: var(--bg-primary);
            border-color: var(--text-primary);
        }
        
        button.primary:hover {
            background: var(--text-secondary);
        }
        
        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        /* Right Panel - Output */
        .output-panel {
            display: flex;
            flex-direction: column;
            background: var(--bg-secondary);
        }
        
        /* Tabs */
        .tabs {
            display: flex;
            border-bottom: 1px solid var(--border);
        }
        
        .tab {
            padding: 0.8rem 1.2rem;
            font-size: 0.8rem;
            color: var(--text-muted);
            cursor: pointer;
            border-bottom: 2px solid transparent;
            transition: all 0.15s;
        }
        
        .tab:hover {
            color: var(--text-secondary);
        }
        
        .tab.active {
            color: var(--text-primary);
            border-bottom-color: var(--text-primary);
        }
        
        /* Tab Content */
        .tab-content {
            flex: 1;
            overflow: hidden;
            display: none;
        }
        
        .tab-content.active {
            display: flex;
            flex-direction: column;
        }
        
        /* Code Area */
        .code-area {
            flex: 1;
            overflow: auto;
            padding: 1rem;
        }
        
        pre {
            background: var(--bg-primary);
            border-radius: 0;
            border: 1px solid var(--border);
            padding: 1rem;
            font-size: 0.85rem;
            line-height: 1.5;
            overflow-x: auto;
        }
        
        code {
            color: var(--text-primary);
        }
        
        /* Video Preview */
        .video-area {
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            background: var(--bg-primary);
        }
        
        video {
            max-width: 100%;
            max-height: 100%;
        }
        
        .no-video {
            color: var(--text-muted);
            font-size: 0.9rem;
        }
        
        /* Thinking Panel */
        .thinking-area {
            flex: 1;
            overflow: auto;
            padding: 1rem;
        }
        
        .thinking-step {
            padding: 0.8rem;
            margin-bottom: 0.5rem;
            background: var(--bg-primary);
            border-radius: 0;
            border-left: 3px solid var(--border);
            animation: fadeIn 0.2s ease-out;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-5px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .thinking-step.active {
            border-left-color: var(--accent);
        }
        
        .thinking-step.understanding { border-left-color: #6b8afd; }
        .thinking-step.concepts { border-left-color: #a78bfa; }
        .thinking-step.approach { border-left-color: #22c55e; }
        .thinking-step.tool { border-left-color: #fbbf24; }
        .thinking-step.tool_result { border-left-color: #4ade80; }
        .thinking-step.plan { border-left-color: #38bdf8; }
        .thinking-step.warning { border-left-color: #f87171; }
        .thinking-step.confidence { border-left-color: #e879f9; }
        
        .step-number {
            color: var(--text-muted);
            font-size: 0.75rem;
            margin-bottom: 0.3rem;
            display: flex;
            justify-content: space-between;
        }
        
        .step-type {
            font-size: 0.65rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            opacity: 0.7;
        }
        
        .step-content {
            line-height: 1.5;
        }
        
        .step-content strong {
            color: var(--text-primary);
        }
        
        /* Code Header */
        .code-header {
            display: flex;
            justify-content: space-between;
            padding: 0.5rem 1rem;
            background: var(--bg-tertiary);
            border-bottom: 1px solid var(--border);
            font-size: 0.75rem;
        }
        
        #code-status {
            color: var(--text-muted);
        }
        
        #verify-status {
            color: var(--success);
        }
        
        #verify-status.warning {
            color: var(--error);
        }
        
        /* Progress Bar */
        .progress-container {
            padding: 0 1rem;
            margin-bottom: 0.5rem;
        }
        
        .progress-bar {
            height: 4px;
            background: var(--bg-tertiary);
            border-radius: 0;
            overflow: hidden;
        }
        
        .progress-fill {
            height: 100%;
            background: var(--text-secondary);
            width: 0%;
            transition: width 0.3s;
        }
        
        .progress-text {
            font-size: 0.7rem;
            color: var(--text-muted);
            margin-top: 0.3rem;
        }
        
        /* Gallery */
        .gallery-area {
            padding: 1rem;
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 1rem;
            overflow-y: auto;
            max-height: 100%;
        }
        
        .gallery-item {
            background: var(--bg-primary);
            border-radius: 0;
            border: 1px solid var(--border);
            overflow: hidden;
            cursor: pointer;
            transition: transform 0.15s, border-color 0.15s;
        }
        
        .gallery-item:hover {
            transform: none;
            border-color: var(--accent);
        }
        

        
        .gallery-thumb {
            width: 100%;
            aspect-ratio: 16/9;
            background: var(--bg-tertiary);
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--text-muted);
            font-size: 2rem;
        }
        
        .gallery-info {
            padding: 0.5rem;
            font-size: 0.75rem;
        }
        
        .gallery-name {
            color: var(--text-primary);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        
        .gallery-date {
            color: var(--text-muted);
            margin-top: 0.2rem;
        }
        
        .gallery-loading {
            grid-column: 1 / -1;
            text-align: center;
            color: var(--text-muted);
            padding: 2rem;
        }
        
        /* Streaming indicator */
        .streaming-indicator {
            display: inline-block;
            width: 8px;
            height: 8px;
            background: var(--accent);
            border-radius: 50%;
            margin-left: 0.5rem;
            animation: pulse 0.8s infinite;
        }
        
        /* Phase indicator */
        .phase-indicator {
            padding: 0.5rem 1rem;
            background: var(--bg-tertiary);
            border-radius: 0;
            border-left: 3px solid var(--accent);
            margin-bottom: 0.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.85rem;
        }
        
        .phase-dot {
            width: 8px;
            height: 8px;
            border-radius: 0;
            background: var(--text-muted);
        }
        
        .phase-dot.active {
            background: var(--text-primary);
            animation: square-blink 1s infinite;
        }
        
        .phase-dot.done {
            background: var(--success);
        }

        /* Templates Sidebar */
        .templates-panel {
            position: fixed;
            top: 60px;
            right: -300px;
            width: 300px;
            height: calc(100vh - 60px);
            background: var(--bg-secondary);
            border-left: 1px solid var(--border);
            transition: right 0.3s;
            overflow-y: auto;
            z-index: 100;
        }
        
        .templates-panel.open {
            right: 0;
        }
        
        .template-group {
            padding: 1rem;
            border-bottom: 1px solid var(--border);
        }
        
        .template-group h3 {
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: var(--text-muted);
            margin-bottom: 0.8rem;
        }
        
        .template-item {
            padding: 0.5rem 0.8rem;
            margin-bottom: 0.3rem;
            border-radius: 0;
            cursor: pointer;
            font-size: 0.85rem;
            transition: background 0.15s;
            border-left: 2px solid transparent;
        }
        
        .template-item:hover {
            background: var(--bg-hover);
            border-left-color: var(--accent);
        }
        

        
        /* Status Bar */
        .status-bar {
            padding: 0.5rem 1rem;
            background: var(--bg-tertiary);
            border-top: 1px solid var(--border);
            font-size: 0.75rem;
            color: var(--text-muted);
            display: flex;
            justify-content: space-between;
        }
        
        .status-indicator {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 0;
            background: var(--success);
        }
        
        .status-dot.error {
            background: var(--error);
        }
        
        .status-dot.loading {
            animation: square-blink 1s infinite;
        }
        
        @keyframes square-blink {
            0%, 100% { opacity: 0.4; }
            50% { opacity: 1; }
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 0.4; }
            50% { opacity: 1; }
        }
        
        /* Streaming Indicator */
        .streaming-indicator {
            display: inline-block;
            width: 8px;
            height: 8px;
            background: var(--text-secondary);
            border-radius: 0;
            margin-left: 6px;
            animation: square-blink 0.8s ease-in-out infinite;
        }
        
        /* Thinking Mode Select */
        .thinking-select {
            display: flex;
            gap: 0.3rem;
        }
        
        .thinking-option {
            padding: 0.4rem 0.8rem;
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            border-radius: 0;
            cursor: pointer;
            font-size: 0.75rem;
            color: var(--text-muted);
        }
        
        .thinking-option.active {
            background: var(--bg-hover);
            border-color: var(--accent);
            color: var(--text-primary);
        }
        
        /* Caption Toggle */
        .caption-toggle {
            display: flex;
            align-items: center;
            gap: 0.3rem;
            cursor: pointer;
            padding: 0.4rem 0.8rem;
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            border-radius: 0;
            font-size: 0.75rem;
            color: var(--text-muted);
        }
        
        .caption-toggle input {
            display: none;
        }
        
        .caption-toggle:has(input:checked) {
            background: var(--bg-hover);
            border-color: var(--accent);
            color: var(--text-primary);
        }

        /* Loading Overlay */
        .loading-overlay {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0,0,0,0.8);
            display: none;
            align-items: center;
            justify-content: center;
            z-index: 1000;
        }
        
        .loading-overlay.active {
            display: flex;
        }
        
        .loader {
            text-align: center;
        }
        
        .loader-spinner {
            width: 48px;
            height: 48px;
            border: 2px solid var(--border);
            position: relative;
            margin: 0 auto 1rem;
        }
        
        .loader-spinner::before {
            content: '';
            position: absolute;
            inset: 4px;
            border: 2px solid var(--text-primary);
            animation: square-pulse 1.2s ease-in-out infinite;
        }
        
        .loader-spinner::after {
            content: '';
            position: absolute;
            inset: 12px;
            background: var(--text-primary);
            animation: square-pulse 1.2s ease-in-out infinite 0.3s;
        }
        
        @keyframes square-pulse {
            0%, 100% { opacity: 0.3; transform: scale(0.8); }
            50% { opacity: 1; transform: scale(1); }
        }
        
        .loader-text {
            color: var(--text-secondary);
            font-size: 0.9rem;
        }
    </style>
</head>
<body>
    <header>
        <div class="logo">manai <span>v2.0</span></div>
        <div class="header-controls">
            <div class="model-selector">
                <label>Model:</label>
                <select id="model-select">
                    <option value="llama3.2">llama3.2</option>
                </select>
            </div>
            <div class="model-selector">
                <label>Voice:</label>
                <select id="voice-select">
                    <option value="">None</option>
                </select>
            </div>
            <label class="caption-toggle">
                <input type="checkbox" id="captions-toggle" checked>
                <span>CC</span>
            </label>
            <button onclick="toggleTemplates()">Templates</button>
        </div>
    </header>
    
    <div class="container">
        <div class="input-panel">
            <div class="panel-header">Describe your visualization</div>
            <div class="prompt-area">
                <textarea id="prompt" placeholder="Example: Show the derivative of sin(x) with a moving tangent line..."></textarea>
            </div>
            <div class="controls">
                <div class="thinking-select">
                    <div class="thinking-option" data-mode="quick">Quick</div>
                    <div class="thinking-option active" data-mode="standard">Standard</div>
                    <div class="thinking-option" data-mode="deep">Deep</div>
                </div>
                <button class="primary" onclick="generate()" id="generate-btn">Generate</button>
            </div>
        </div>
        
        <div class="output-panel">
            <div class="tabs">
                <div class="tab active" data-tab="code">Code</div>
                <div class="tab" data-tab="video">Video</div>
                <div class="tab" data-tab="thinking">Thinking</div>
                <div class="tab" data-tab="gallery">Gallery</div>
            </div>
            
            <div class="tab-content active" id="tab-code">
                <div class="code-area">
                    <div class="code-header">
                        <span id="code-status">Ready</span>
                        <span id="verify-status"></span>
                    </div>
                    <pre><code id="code-output"># Generated code will appear here...</code></pre>
                </div>
            </div>
            
            <div class="tab-content" id="tab-video">
                <div class="video-area">
                    <div class="no-video" id="video-placeholder">No video generated yet</div>
                    <video id="video-player" controls style="display:none;"></video>
                </div>
            </div>
            
            <div class="tab-content" id="tab-thinking">
                <div class="thinking-area" id="thinking-output">
                    <div class="thinking-step">
                        <div class="step-number">Waiting...</div>
                        <div>Agent reasoning will appear here during generation.</div>
                    </div>
                </div>
            </div>
            
            <div class="tab-content" id="tab-gallery">
                <div class="gallery-area" id="gallery-output">
                    <div class="gallery-loading">Loading gallery...</div>
                </div>
            </div>
        </div>
    </div>
    
    <div class="status-bar">
        <div class="status-indicator">
            <div class="status-dot" id="status-dot"></div>
            <span id="status-text">Ready</span>
        </div>
        <div id="model-info">Model: <span id="current-model-name">loading...</span></div>
    </div>
    
    <!-- Templates Sidebar -->
    <div class="templates-panel" id="templates-panel">
        <div class="template-group">
            <h3>2D Templates</h3>
            <div id="templates-2d"></div>
        </div>
        <div class="template-group">
            <h3>3D Templates</h3>
            <div id="templates-3d"></div>
        </div>
        <div class="template-group">
            <h3>Transitions</h3>
            <div id="templates-transitions"></div>
        </div>
    </div>
    
    <!-- Loading Overlay -->
    <div class="loading-overlay" id="loading">
        <div class="loader">
            <div class="loader-spinner"></div>
            <div class="loader-text" id="loader-text">Generating...</div>
            <div class="progress-container">
                <div class="progress-bar">
                    <div class="progress-fill" id="progress-fill"></div>
                </div>
                <div class="progress-text" id="progress-phase">Initializing...</div>
            </div>
        </div>
    </div>
    
    <script>
        // State
        let currentThinkingMode = 'standard';
        let ws = null;
        let isGenerating = false;
        let reconnectAttempts = 0;
        const maxReconnectAttempts = 5;
        
        // Phase progress mapping
        const phases = {
            init: { progress: 5, label: 'Initializing...' },
            thinking: { progress: 15, label: 'Agent analyzing...' },
            generating: { progress: 35, label: 'Generating code...' },
            verifying: { progress: 55, label: 'Verifying math...' },
            rendering: { progress: 70, label: 'Rendering video...' },
            voiceover: { progress: 85, label: 'Generating voiceover...' },
            merging: { progress: 95, label: 'Adding audio & captions...' }
        };
        
        // Initialize
        document.addEventListener('DOMContentLoaded', async () => {
            try {
                await Promise.all([
                    loadModels(),
                    loadVoices(),
                    loadTemplates(),
                    loadGallery()
                ]);
                setupTabs();
                setupThinkingOptions();
                connectWebSocket();
            } catch (e) {
                console.error('Initialization error:', e);
            }
        });
        
        // Load models
        async function loadModels() {
            try {
                const res = await fetch('/api/models');
                const data = await res.json();
                const select = document.getElementById('model-select');
                select.innerHTML = data.models.map(m => 
                    `<option value="${m.name}" ${m.name === data.current ? 'selected' : ''}>${m.name}</option>`
                ).join('');
                
                // Update status bar with current model
                document.getElementById('current-model-name').textContent = data.current || data.models[0]?.name || 'none';
                
                select.addEventListener('change', async (e) => {
                    await fetch('/api/models/switch', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({model: e.target.value})
                    });
                    document.getElementById('current-model-name').textContent = e.target.value;
                });
            } catch (e) {
                console.error('Failed to load models:', e);
            }
        }
        
        // Load voices
        async function loadVoices() {
            try {
                const res = await fetch('/api/voices');
                const data = await res.json();
                const select = document.getElementById('voice-select');
                select.innerHTML = '<option value="">None (no voiceover)</option>' +
                    data.voices.map(v => 
                        `<option value="${v}">${v}</option>`
                    ).join('');
            } catch (e) {
                console.error('Failed to load voices:', e);
            }
        }
        
        // Load templates
        async function loadTemplates() {
            try {
                const res = await fetch('/api/templates');
                const data = await res.json();
                
                document.getElementById('templates-2d').innerHTML = 
                    data['2d'].map(t => `<div class="template-item" onclick="useTemplate('${t}')">${t}</div>`).join('');
                document.getElementById('templates-3d').innerHTML = 
                    data['3d'].map(t => `<div class="template-item" onclick="useTemplate('${t}')">${t}</div>`).join('');
                document.getElementById('templates-transitions').innerHTML = 
                    data.transitions.map(t => `<div class="template-item" onclick="useTemplate('${t}')">${t}</div>`).join('');
            } catch (e) {
                console.error('Failed to load templates:', e);
            }
        }
        
        // Load gallery
        async function loadGallery() {
            try {
                const res = await fetch('/api/gallery');
                const data = await res.json();
                const gallery = document.getElementById('gallery-output');
                
                if (data.videos.length === 0) {
                    gallery.innerHTML = '<div class="gallery-loading">No videos yet. Generate your first one!</div>';
                    return;
                }
                
                gallery.innerHTML = data.videos.map(v => `
                    <div class="gallery-item" onclick="playGalleryVideo('${v.path}')">
                        <div class="gallery-thumb">▶</div>
                        <div class="gallery-info">
                            <div class="gallery-name">${v.name}</div>
                            <div class="gallery-date">${new Date(v.created * 1000).toLocaleDateString()}</div>
                        </div>
                    </div>
                `).join('');
            } catch (e) {
                console.error('Failed to load gallery:', e);
            }
        }
        
        // Play gallery video
        function playGalleryVideo(path) {
            showVideo(path);
            document.querySelector('.tab[data-tab="video"]').click();
        }
        
        // Setup tabs
        function setupTabs() {
            document.querySelectorAll('.tab').forEach(tab => {
                tab.addEventListener('click', () => {
                    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                    tab.classList.add('active');
                    document.getElementById(`tab-${tab.dataset.tab}`).classList.add('active');
                    
                    // Refresh gallery when clicked
                    if (tab.dataset.tab === 'gallery') {
                        loadGallery();
                    }
                });
            });
        }
        
        // Setup thinking options
        function setupThinkingOptions() {
            document.querySelectorAll('.thinking-option').forEach(opt => {
                opt.addEventListener('click', () => {
                    document.querySelectorAll('.thinking-option').forEach(o => o.classList.remove('active'));
                    opt.classList.add('active');
                    currentThinkingMode = opt.dataset.mode;
                });
            });
        }
        
        // Connect WebSocket with better error handling
        function connectWebSocket() {
            if (reconnectAttempts >= maxReconnectAttempts) {
                updateStatus('Connection failed - using HTTP fallback', true);
                return;
            }
            
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            
            try {
                ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
                
                ws.onopen = () => {
                    reconnectAttempts = 0;
                    updateStatus('Connected', false);
                };
                
                ws.onmessage = (event) => {
                    try {
                        const data = JSON.parse(event.data);
                        handleWSMessage(data);
                    } catch (e) {
                        console.error('Failed to parse WS message:', e);
                    }
                };
                
                ws.onclose = (event) => {
                    if (!event.wasClean) {
                        reconnectAttempts++;
                        updateStatus('Reconnecting...', false);
                        setTimeout(connectWebSocket, 2000 * reconnectAttempts);
                    }
                };
                
                ws.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    // Don't show error to user, onclose will handle reconnect
                };
            } catch (e) {
                console.error('WebSocket connection failed:', e);
                updateStatus('Using HTTP mode', false);
            }
        }
        
        // Handle WebSocket messages
        function handleWSMessage(data) {
            const thinkingOutput = document.getElementById('thinking-output');
            const codeOutput = document.getElementById('code-output');
            
            switch(data.type) {
                case 'phase':
                    const phase = phases[data.phase];
                    if (phase) {
                        updateProgress(phase.progress, phase.label);
                    }
                    document.getElementById('code-status').innerHTML = 
                        data.content + '<span class="streaming-indicator"></span>';
                    
                    // Auto-switch to thinking tab during thinking phase
                    if (data.phase === 'thinking') {
                        document.querySelector('.tab[data-tab="thinking"]').click();
                    }
                    // Auto-switch to code tab during generation
                    if (data.phase === 'generating') {
                        document.querySelector('.tab[data-tab="code"]').click();
                    }
                    break;
                    
                case 'thinking_step':
                    const stepType = data.step_type || 'thinking';
                    const stepTypeLabels = {
                        'init': '[INIT]',
                        'understanding': '[ANALYZE]',
                        'concepts': '[CONCEPTS]',
                        'approach': '[APPROACH]',
                        'scene': '[SCENE]',
                        'tool': '[TOOL]',
                        'tool_result': '[RESULT]',
                        'plan': '[PLAN]',
                        'warning': '[WARN]',
                        'confidence': '[CONF]',
                        'error': '[ERROR]'
                    };
                    const icon = stepTypeLabels[stepType] || '[THINK]';
                    
                    thinkingOutput.innerHTML += `
                        <div class="thinking-step ${stepType}">
                            <div class="step-number">
                                <span>${icon} Step ${data.step}</span>
                                <span class="step-type">${stepType}</span>
                            </div>
                            <div class="step-content">${formatThinkingContent(data.content)}</div>
                        </div>`;
                    thinkingOutput.scrollTop = thinkingOutput.scrollHeight;
                    break;
                    
                case 'code_stream':
                    // Live code preview during generation
                    codeOutput.textContent = data.content;
                    document.getElementById('code-status').innerHTML = 
                        'Generating... <span class="streaming-indicator"></span>';
                    break;
                    
                case 'code_complete':
                    codeOutput.textContent = data.content;
                    document.getElementById('code-status').textContent = 'Code ready';
                    break;
                    
                case 'verification':
                    const verified = data.expressions.filter(e => e.valid).length;
                    const total = data.expressions.length;
                    const verifyStatus = document.getElementById('verify-status');
                    
                    if (verified === total) {
                        verifyStatus.textContent = `✓ ${verified}/${total} expressions verified`;
                        verifyStatus.className = '';
                    } else {
                        verifyStatus.textContent = `⚠ ${verified}/${total} expressions verified`;
                        verifyStatus.className = 'warning';
                    }
                    break;
                
                case 'warning':
                    // Show warning but continue
                    console.warn('Generation warning:', data.content);
                    const thinkingArea = document.getElementById('thinking-output');
                    thinkingArea.innerHTML += `
                        <div class="thinking-step" style="border-left-color: var(--error);">
                            <div class="step-number">Warning</div>
                            <div>${data.content}</div>
                        </div>`;
                    break;
                    
                case 'complete':
                    hideLoading();
                    isGenerating = false;
                    
                    if (data.code) {
                        codeOutput.textContent = data.code;
                    }
                    
                    document.getElementById('code-status').textContent = 'Complete';
                    
                    if (data.verified) {
                        document.getElementById('verify-status').textContent = 
                            `✓ ${data.verified} expressions verified`;
                    }
                    
                    if (data.video) {
                        showVideo(data.video);
                        loadGallery(); // Refresh gallery
                    }
                    
                    updateStatus('Complete', false);
                    updateProgress(100, 'Done!');
                    break;
                    
                case 'error':
                    hideLoading();
                    isGenerating = false;
                    
                    if (data.code) {
                        codeOutput.textContent = data.code;
                    }
                    
                    document.getElementById('code-status').textContent = 'Error';
                    updateStatus(`Error: ${data.content}`, true);
                    break;
            }
        }
        
        // Generate using WebSocket
        async function generate() {
            const prompt = document.getElementById('prompt').value;
            if (!prompt.trim() || isGenerating) return;
            
            isGenerating = true;
            showLoading('Generating...');
            clearOutputs();
            updateProgress(0, 'Starting...');
            
            // Get voice and caption settings
            const voice = document.getElementById('voice-select').value;
            const addCaptions = document.getElementById('captions-toggle').checked;
            
            // Use WebSocket for streaming
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    type: 'generate',
                    description: prompt,
                    thinking_mode: currentThinkingMode,
                    voice: voice || null,
                    add_captions: addCaptions
                }));
            } else {
                // Fallback to HTTP
                try {
                    const res = await fetch('/api/generate', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            description: prompt,
                            thinking_mode: currentThinkingMode,
                            voice: voice || null,
                            add_captions: addCaptions
                        })
                    });
                    
                    const data = await res.json();
                    hideLoading();
                    isGenerating = false;
                    
                    if (data.success) {
                        document.getElementById('code-output').textContent = data.code || '';
                        if (data.output_path) {
                            const filename = data.output_path.split('/').pop();
                            showVideo(`/api/video/${filename}`);
                        }
                        updateStatus('Complete', false);
                    } else {
                        updateStatus(`Error: ${data.error}`, true);
                    }
                } catch (e) {
                    hideLoading();
                    isGenerating = false;
                    updateStatus(`Error: ${e.message}`, true);
                }
            }
        }
        
        // Update progress
        function updateProgress(percent, text) {
            document.getElementById('progress-fill').style.width = `${percent}%`;
            document.getElementById('progress-phase').textContent = text;
        }
        
        // Format thinking content (handle markdown-like **bold**)
        function formatThinkingContent(content) {
            return content
                .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
                .replace(/`(.+?)`/g, '<code>$1</code>')
                .replace(/→/g, '<span style="color: var(--text-muted)">→</span>');
        }
        
        // Use template
        async function useTemplate(name) {
            try {
                const res = await fetch('/api/templates/get', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({name})
                });
                
                const data = await res.json();
                if (data.success) {
                    document.getElementById('code-output').textContent = data.code;
                    document.querySelector('.tab[data-tab="code"]').click();
                    toggleTemplates();
                }
            } catch (e) {
                console.error('Failed to load template:', e);
            }
        }
        
        // Toggle templates panel
        function toggleTemplates() {
            document.getElementById('templates-panel').classList.toggle('open');
        }
        
        // Show video
        function showVideo(src) {
            const placeholder = document.getElementById('video-placeholder');
            const player = document.getElementById('video-player');
            placeholder.style.display = 'none';
            player.style.display = 'block';
            player.src = src;
            document.querySelector('.tab[data-tab="video"]').click();
        }
        
        // Clear outputs
        function clearOutputs() {
            document.getElementById('code-output').textContent = '# Generating...';
            document.getElementById('thinking-output').innerHTML = '';
            document.getElementById('code-status').textContent = 'Generating...';
            document.getElementById('verify-status').textContent = '';
        }
        
        // Loading
        function showLoading(text) {
            document.getElementById('loading').classList.add('active');
            document.getElementById('loader-text').textContent = text;
            document.getElementById('generate-btn').disabled = true;
        }
        
        function hideLoading() {
            document.getElementById('loading').classList.remove('active');
            document.getElementById('generate-btn').disabled = false;
        }
        
        // Status
        function updateStatus(text, isError = false) {
            document.getElementById('status-text').textContent = text;
            const dot = document.getElementById('status-dot');
            dot.classList.toggle('error', isError);
            dot.classList.remove('loading');
        }
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
                generate();
            }
        });
    </script>
</body>
</html>
'''


# =============================================================================
# SERVER RUNNER
# =============================================================================

def run_server(host: str = "127.0.0.1", port: int = 8000, reload: bool = False):
    """Run the web server."""
    import uvicorn
    
    print(f"\n  Manai Web UI")
    print(f"  ────────────────────────────")
    print(f"  Local:   http://{host}:{port}")
    print(f"  ────────────────────────────\n")
    
    uvicorn.run(
        "manai.web.server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="warning"
    )


if __name__ == "__main__":
    run_server()
