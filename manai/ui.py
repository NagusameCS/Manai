"""
Monotone UI System - Clean, minimal terminal interface for Manai
"""

from typing import Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.layout import Layout
from rich.live import Live
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.prompt import Prompt, Confirm
from rich.rule import Rule
from rich.align import Align
from rich.box import MINIMAL, SIMPLE, ROUNDED, HEAVY
from rich.style import Style
from rich.theme import Theme


# =============================================================================
# MONOTONE THEME
# =============================================================================

class MonotoneTheme:
    """Monotone color palette for consistent UI."""
    
    # Base colors - grayscale monotone
    BLACK = "#0a0a0a"
    DARK = "#1a1a1a"
    GRAY_DARK = "#2a2a2a"
    GRAY = "#4a4a4a"
    GRAY_LIGHT = "#6a6a6a"
    SILVER = "#8a8a8a"
    LIGHT = "#aaaaaa"
    WHITE = "#dadada"
    BRIGHT = "#ffffff"
    
    # Accent colors (subtle, desaturated)
    ACCENT_PRIMARY = "#7a9cc6"      # Muted blue
    ACCENT_SECONDARY = "#8fb8a8"    # Muted green
    ACCENT_WARNING = "#c9a87c"      # Muted gold
    ACCENT_ERROR = "#b87a7a"        # Muted red
    ACCENT_SUCCESS = "#8fb88f"      # Muted green
    ACCENT_INFO = "#9a8fb8"         # Muted purple
    
    # UI Elements
    BORDER = GRAY_DARK
    TEXT = LIGHT
    TEXT_DIM = GRAY_LIGHT
    TEXT_BRIGHT = WHITE
    BACKGROUND = BLACK
    
    @classmethod
    def get_theme(cls) -> Theme:
        """Create Rich theme from palette."""
        return Theme({
            "info": f"{cls.ACCENT_INFO}",
            "warning": f"{cls.ACCENT_WARNING}",
            "error": f"{cls.ACCENT_ERROR}",
            "success": f"{cls.ACCENT_SUCCESS}",
            "primary": f"{cls.ACCENT_PRIMARY}",
            "secondary": f"{cls.ACCENT_SECONDARY}",
            "dim": f"{cls.TEXT_DIM}",
            "bright": f"{cls.TEXT_BRIGHT}",
            "border": f"{cls.BORDER}",
            "title": f"bold {cls.WHITE}",
            "subtitle": f"{cls.SILVER}",
            "code": f"{cls.ACCENT_PRIMARY}",
            "thinking": f"italic {cls.ACCENT_INFO}",
            "tool": f"{cls.ACCENT_SECONDARY}",
            "result": f"{cls.ACCENT_SUCCESS}",
        })


# Create themed console
console = Console(theme=MonotoneTheme.get_theme())


# =============================================================================
# UI COMPONENTS
# =============================================================================

class UIStyle(Enum):
    """UI style variants."""
    MINIMAL = "minimal"
    COMPACT = "compact"
    FULL = "full"


@dataclass
class UIConfig:
    """UI configuration."""
    style: UIStyle = UIStyle.MINIMAL
    show_thinking: bool = True
    show_tools: bool = True
    animate: bool = True
    box_style: Any = MINIMAL


class UI:
    """
    Monotone UI system for Manai.
    
    Provides consistent, clean interface components.
    """
    
    def __init__(self, config: Optional[UIConfig] = None):
        self.config = config or UIConfig()
        self.console = console
    
    # =========================================================================
    # HEADERS & TITLES
    # =========================================================================
    
    def header(self, title: str = "MANAI", subtitle: str = "") -> None:
        """Display main header."""
        header_text = Text()
        header_text.append("█▀▄▀█ ", style=MonotoneTheme.GRAY)
        header_text.append(title, style=f"bold {MonotoneTheme.WHITE}")
        header_text.append(" ▀█▀█▀", style=MonotoneTheme.GRAY)
        
        self.console.print()
        self.console.print(Align.center(header_text))
        
        if subtitle:
            self.console.print(Align.center(
                Text(subtitle, style=MonotoneTheme.SILVER)
            ))
        
        self.console.print()
    
    def title(self, text: str, style: str = "title") -> None:
        """Display section title."""
        self.console.print()
        self.console.print(Rule(text, style=style, characters="─"))
    
    def subtitle(self, text: str) -> None:
        """Display subtitle."""
        self.console.print(Text(f"  {text}", style="subtitle"))
    
    # =========================================================================
    # PANELS & BOXES
    # =========================================================================
    
    def panel(
        self,
        content: str | Text,
        title: str = "",
        style: str = "border",
        padding: tuple = (0, 1)
    ) -> Panel:
        """Create styled panel."""
        return Panel(
            content,
            title=title if title else None,
            title_align="left",
            border_style=style,
            box=self.config.box_style,
            padding=padding,
        )
    
    def code_panel(self, code: str, language: str = "python", title: str = "Code") -> None:
        """Display code in styled panel."""
        syntax = Syntax(
            code,
            language,
            theme="nord",
            line_numbers=True,
            background_color=MonotoneTheme.DARK,
        )
        self.console.print(self.panel(syntax, title=title))
    
    def info_panel(self, message: str, title: str = "Info") -> None:
        """Display info panel."""
        self.console.print(Panel(
            Text(message, style="info"),
            title=f"[info]● {title}[/info]",
            border_style="info",
            box=MINIMAL,
        ))
    
    def error_panel(self, message: str, title: str = "Error") -> None:
        """Display error panel."""
        self.console.print(Panel(
            Text(message, style="error"),
            title=f"[error]✕ {title}[/error]",
            border_style="error",
            box=MINIMAL,
        ))
    
    def success_panel(self, message: str, title: str = "Success") -> None:
        """Display success panel."""
        self.console.print(Panel(
            Text(message, style="success"),
            title=f"[success]✓ {title}[/success]",
            border_style="success",
            box=MINIMAL,
        ))
    
    def warning_panel(self, message: str, title: str = "Warning") -> None:
        """Display warning panel."""
        self.console.print(Panel(
            Text(message, style="warning"),
            title=f"[warning]! {title}[/warning]",
            border_style="warning",
            box=MINIMAL,
        ))
    
    # =========================================================================
    # THINKING & REASONING
    # =========================================================================
    
    def thinking_start(self, message: str = "Thinking...") -> None:
        """Display thinking indicator."""
        if self.config.show_thinking:
            self.console.print()
            self.console.print(Text(f"  ◐ {message}", style="thinking"))
    
    def thinking_step(self, step: str) -> None:
        """Display a thinking step."""
        if self.config.show_thinking:
            self.console.print(Text(f"    → {step}", style="dim"))
    
    def thinking_end(self) -> None:
        """End thinking indicator."""
        pass
    
    def reasoning_panel(self, thoughts: list[str], title: str = "Reasoning") -> None:
        """Display reasoning/thinking process."""
        if not self.config.show_thinking:
            return
        
        content = Text()
        for i, thought in enumerate(thoughts, 1):
            content.append(f"  {i}. ", style="dim")
            content.append(f"{thought}\n", style="thinking")
        
        self.console.print(Panel(
            content,
            title=f"[thinking]◈ {title}[/thinking]",
            border_style="dim",
            box=MINIMAL,
        ))
    
    # =========================================================================
    # TOOL DISPLAY
    # =========================================================================
    
    def tool_call(self, tool_name: str, args: dict = None) -> None:
        """Display tool being called."""
        if not self.config.show_tools:
            return
        
        self.console.print()
        self.console.print(Text(f"  ⚙ Using: {tool_name}", style="tool"))
        
        if args:
            for key, value in args.items():
                display_value = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
                self.console.print(Text(f"      {key}: {display_value}", style="dim"))
    
    def tool_result(self, result: str, success: bool = True) -> None:
        """Display tool result."""
        if not self.config.show_tools:
            return
        
        style = "success" if success else "error"
        icon = "✓" if success else "✕"
        self.console.print(Text(f"    {icon} {result}", style=style))
    
    # =========================================================================
    # TABLES
    # =========================================================================
    
    def table(
        self,
        title: str,
        columns: list[str],
        rows: list[list[str]],
        show_header: bool = True
    ) -> None:
        """Display styled table."""
        table = Table(
            title=title,
            box=MINIMAL,
            border_style="border",
            header_style="bright",
            show_header=show_header,
        )
        
        for col in columns:
            table.add_column(col, style="dim")
        
        for row in rows:
            table.add_row(*row)
        
        self.console.print(table)
    
    def model_table(self, models: list[dict], current: str = "") -> None:
        """Display model selection table."""
        table = Table(
            title="Available Models",
            box=MINIMAL,
            border_style="border",
        )
        
        table.add_column("", width=3)
        table.add_column("Model", style="primary")
        table.add_column("Size", style="dim")
        table.add_column("Description", style="dim")
        
        for model in models:
            name = model.get("name", "")
            is_current = name.startswith(current)
            marker = "●" if is_current else "○"
            marker_style = "success" if is_current else "dim"
            
            size = model.get("size", 0)
            size_str = f"{size / 1e9:.1f}GB" if size else "—"
            
            table.add_row(
                Text(marker, style=marker_style),
                name,
                size_str,
                model.get("description", "")[:40]
            )
        
        self.console.print(table)
    
    # =========================================================================
    # PROGRESS & STATUS
    # =========================================================================
    
    def progress(self, description: str = "Processing") -> Progress:
        """Create progress indicator."""
        return Progress(
            SpinnerColumn(spinner_name="dots", style="primary"),
            TextColumn("[dim]{task.description}[/dim]"),
            BarColumn(bar_width=30, style="dim", complete_style="primary"),
            console=self.console,
        )
    
    def status(self, message: str, style: str = "dim") -> None:
        """Display status message."""
        self.console.print(Text(f"  {message}", style=style))
    
    def spinner(self, message: str) -> Live:
        """Create spinner context."""
        return Live(
            Text(f"  ◌ {message}", style="dim"),
            console=self.console,
            refresh_per_second=10,
        )
    
    # =========================================================================
    # INPUT
    # =========================================================================
    
    def prompt(self, message: str = "manai", default: str = "") -> str:
        """Get user input with styled prompt."""
        prompt_text = f"[{MonotoneTheme.ACCENT_PRIMARY}]{message}[/] [dim]>[/dim]"
        return Prompt.ask(prompt_text, default=default, console=self.console)
    
    def confirm(self, message: str, default: bool = True) -> bool:
        """Get confirmation."""
        return Confirm.ask(
            f"[dim]{message}[/dim]",
            default=default,
            console=self.console
        )
    
    def select(self, message: str, choices: list[str]) -> str:
        """Display selection menu."""
        self.console.print(f"\n[dim]{message}[/dim]\n")
        
        for i, choice in enumerate(choices, 1):
            self.console.print(f"  [{MonotoneTheme.GRAY}]{i}.[/] {choice}")
        
        self.console.print()
        
        while True:
            selection = self.prompt("Select")
            
            # Try by number
            try:
                idx = int(selection) - 1
                if 0 <= idx < len(choices):
                    return choices[idx]
            except ValueError:
                pass
            
            # Try by name
            if selection in choices:
                return selection
            
            self.console.print("[error]Invalid selection[/error]")
    
    # =========================================================================
    # SPECIAL DISPLAYS
    # =========================================================================
    
    def equation(self, latex: str, title: str = "") -> None:
        """Display mathematical equation."""
        eq_panel = Panel(
            Align.center(Text(latex, style="bright")),
            title=title if title else None,
            border_style="primary",
            box=MINIMAL,
        )
        self.console.print(eq_panel)
    
    def verification(self, original: str, verified: bool, details: str = "") -> None:
        """Display math verification result."""
        status = "[success]✓ Verified[/success]" if verified else "[error]✕ Invalid[/error]"
        
        content = Text()
        content.append(f"Expression: {original}\n", style="dim")
        content.append(f"Status: ")
        
        self.console.print(Panel(
            Group(
                Text(f"Expression: {original}", style="dim"),
                Text(f"Status: {'Verified ✓' if verified else 'Invalid ✕'}", 
                     style="success" if verified else "error"),
                Text(f"Details: {details}", style="dim") if details else Text(""),
            ),
            title="[info]Math Verification[/info]",
            border_style="info",
            box=MINIMAL,
        ))
    
    def generation_result(
        self,
        success: bool,
        code: Optional[str] = None,
        output_path: Optional[str] = None,
        error: Optional[str] = None
    ) -> None:
        """Display generation result."""
        if success:
            content = Text()
            content.append("Generation successful\n\n", style="success")
            
            if output_path:
                content.append("Output: ", style="dim")
                content.append(output_path, style="primary")
            
            self.console.print(Panel(
                content,
                title="[success]✓ Complete[/success]",
                border_style="success",
                box=MINIMAL,
            ))
            
            if code:
                self.code_panel(code, title="Generated Code")
        else:
            self.error_panel(error or "Unknown error", title="Generation Failed")
    
    def help_display(self, commands: dict[str, str]) -> None:
        """Display help information."""
        table = Table(
            title="Commands",
            box=MINIMAL,
            border_style="border",
            show_header=False,
        )
        
        table.add_column("Command", style="primary", width=20)
        table.add_column("Description", style="dim")
        
        for cmd, desc in commands.items():
            table.add_row(cmd, desc)
        
        self.console.print(table)
    
    # =========================================================================
    # DIVIDERS & SPACING
    # =========================================================================
    
    def divider(self, style: str = "dim") -> None:
        """Display divider line."""
        self.console.print(Rule(style=style, characters="─"))
    
    def space(self, lines: int = 1) -> None:
        """Add vertical space."""
        for _ in range(lines):
            self.console.print()
    
    def clear(self) -> None:
        """Clear console."""
        self.console.clear()


# =============================================================================
# GLOBAL UI INSTANCE
# =============================================================================

ui = UI()
