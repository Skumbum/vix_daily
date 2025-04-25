from prompt_toolkit import Application
from prompt_toolkit.layout import Layout, VSplit, HSplit, Window
from prompt_toolkit.widgets import Frame, TextArea
from prompt_toolkit.layout.containers import WindowAlign
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.styles import Style
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.layout.controls import BufferControl
from prompt_toolkit.document import Document


class TUI:
    def __init__(self, ticker, stats):
        self.ticker = ticker
        self.stats = stats
        self.sections = list(stats.keys())
        self.selected_section = 0
        self.expanded = {}  # Track expanded/collapsed state for nested sections
        self.focus_path = []  # Track the current focus path for nested navigation

        # Initialize the layout
        self.menu_area = TextArea(
            text=self._render_menu(),
            focusable=True,
            width=30,
            style="class:menu",
            wrap_lines=False
        )
        self.content_area = TextArea(
            text=self._render_content(),
            focusable=False,
            wrap_lines=True,
            style="class:content"
        )
        # Use BufferControl for the header with initial text
        header_document = Document(
            text=f"Financial Statistics Viewer - {ticker}",
            cursor_position=0
        )
        header_buffer = Buffer(
            document=header_document,
            read_only=True,
            multiline=False
        )
        self.header_area = Window(
            content=BufferControl(buffer=header_buffer),
            height=1,
            align=WindowAlign.CENTER,
            style="class:header"
        )
        # Use BufferControl for the footer with initial text
        footer_document = Document(
            text="↑↓: Navigate | Enter: Expand/Collapse | Ctrl+C: Exit",
            cursor_position=0
        )
        footer_buffer = Buffer(
            document=footer_document,
            read_only=True,
            multiline=False
        )
        self.footer_area = Window(
            content=BufferControl(buffer=footer_buffer),
            height=1,
            align=WindowAlign.CENTER,
            style="class:footer"
        )

        # Define the layout
        body = VSplit([
            Frame(self.menu_area, title="Sections", style="class:menu-frame"),
            Frame(self.content_area, title="Details", style="class:content-frame")
        ])
        root_container = HSplit([
            Frame(self.header_area, style="class:header-frame"),
            body,
            Frame(self.footer_area, style="class:footer-frame")
        ])
        self.layout = Layout(root_container)

        # Define key bindings
        bindings = KeyBindings()
        @bindings.add('up')
        def _(event):
            self.selected_section = max(0, self.selected_section - 1)
            self.focus_path = []  # Reset focus path when changing sections
            self._update_display()

        @bindings.add('down')
        def _(event):
            self.selected_section = min(len(self.sections) - 1, self.selected_section + 1)
            self.focus_path = []  # Reset focus path when changing sections
            self._update_display()

        @bindings.add('enter')
        def _(event):
            # Toggle expand/collapse for the current focus path
            selected_key = self.sections[self.selected_section]
            current_path = selected_key
            if self.focus_path:
                current_path = f"{selected_key}:{':'.join(self.focus_path)}"
            if current_path in self.expanded:
                self.expanded[current_path] = not self.expanded[current_path]
            else:
                # Check if the current path points to a dictionary
                current_data = self.stats[selected_key]
                for key in self.focus_path:
                    if isinstance(current_data, dict):
                        current_data = current_data.get(key)
                    else:
                        return  # Not a dictionary, can't expand
                if isinstance(current_data, dict):
                    self.expanded[current_path] = True
            self._update_display()

        # Define the style
        self.style = Style([
            ('menu', 'bg:#444444 fg:#ffffff'),
            ('menu-frame', 'bg:#444444 fg:#ffffff'),
            ('content', 'bg:#222222 fg:#dddddd'),
            ('content-frame', 'bg:#222222 fg:#dddddd'),
            ('header', 'bg:#333333 fg:#ffffff bold'),
            ('header-frame', 'bg:#333333 fg:#ffffff'),
            ('footer', 'bg:#333333 fg:#aaaaaa'),
            ('footer-frame', 'bg:#333333 fg:#aaaaaa'),
            ('selected', 'bg:#666666 fg:#ffffff bold')
        ])

        # Create the application without specifying output
        self.app = Application(
            layout=self.layout,
            key_bindings=bindings,
            style=self.style,
            full_screen=True
        )

    def _render_menu(self):
        """Render the menu with the selected section highlighted."""
        lines = []
        for i, section in enumerate(self.sections):
            if i == self.selected_section:
                lines.append(HTML(f"<selected>{section}</selected>"))
            else:
                lines.append(section)
        return "\n".join(str(line) for line in lines)

    def _render_content(self, indent=0):
        """Render the content of the selected section."""
        selected_key = self.sections[self.selected_section]
        value = self.stats[selected_key]
        lines = []
        self.focus_path = []  # Reset focus path for rendering

        if isinstance(value, dict):
            for key, val in value.items():
                if isinstance(val, dict):
                    path = f"{selected_key}:{key}"
                    is_expanded = self.expanded.get(path, False)
                    prefix = "▶ " if not is_expanded else "▼ "
                    lines.append("  " * indent + prefix + str(key))
                    if is_expanded:
                        nested_lines, _ = self._render_content_dict(val, indent + 1, [key])
                        lines.extend(nested_lines)
                else:
                    lines.append("  " * indent + f"{key}: {val}")
        else:
            lines.append(value)  # For notes/conclusions
        return "\n".join(str(line) for line in lines)

    def _render_content_dict(self, data, indent=0, path=None):
        """Helper to render nested dictionaries."""
        if path is None:
            path = []
        lines = []
        focus_index = len(self.focus_path) - len(path)
        is_focused = focus_index == 0

        for key, val in data.items():
            current_path = path + [key]
            path_key = f"{self.sections[self.selected_section]}:{':'.join(current_path)}"
            if isinstance(val, dict):
                is_expanded = self.expanded.get(path_key, False)
                prefix = "▶ " if not is_expanded else "▼ "
                line = "  " * indent + prefix + str(key)
                if is_focused and current_path[:len(self.focus_path)] == self.focus_path:
                    line = HTML(f"<selected>{line}</selected>")
                lines.append(line)
                if is_expanded:
                    nested_lines, _ = self._render_content_dict(val, indent + 1, current_path)
                    lines.extend(nested_lines)
            else:
                line = "  " * indent + f"{key}: {val}"
                if is_focused and current_path[:len(self.focus_path)] == self.focus_path:
                    line = HTML(f"<selected>{line}</selected>")
                lines.append(line)
        return lines, len(lines)

    def _update_display(self):
        """Update the menu and content areas."""
        self.menu_area.text = self._render_menu()
        self.content_area.text = self._render_content()

    def run(self):
        """Run the TUI application."""
        self.app.run()


def display_stats(ticker, stats):
    """Entry point for displaying stats in the TUI."""
    tui = TUI(ticker, stats)
    tui.run()