import tkinter as tk
from tkinter import ttk
import subprocess
import os
import re
import sys

class AutoScrollbar(tk.Scrollbar):
    """A scrollbar that hides itself if it's not needed."""
    def set(self, lo, hi):
        if float(lo) <= 0.0 and float(hi) >= 1.0:
            self.grid_remove()
        else:
            self.grid()
        super().set(lo, hi)

    def pack(self, **kw):
        raise tk.TclError("cannot use pack with this widget, use grid instead")

    def place(self, **kw):
        raise tk.TclError("cannot use place with this widget, use grid instead")


class SegObC_IDE:
    def __init__(self, root):
        self.root = root
        self.root.title("SegObC IDE")
        
        # --- Center Window Logic ---
        window_width = 1200
        window_height = 700
        
        # Get the screen width and height
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Calculate the x and y coordinates to center the window
        center_x = int((screen_width / 2) - (window_width / 2))
        center_y = int((screen_height / 2) - (window_height / 2))
        
        # Apply the geometry with the calculated offsets
        self.root.geometry(f"{window_width}x{window_height}+{center_x}+{center_y}")
        # ---------------------------
        try:
            icon_file = self.resource_path("segobClogo.ico")
            self.root.iconbitmap(icon_file)
        except Exception:
            pass
            
        # Modern "Nord" Inspired Color Palette
        self.colors = {
            'bg_primary': "#222834",      # Deep slate (main background)
            'bg_secondary': "#14171D",    # Lighter slate (toolbars, tabs)
            'bg_tertiary': "#222834",     # Highlight slate (consoles)
            'bg_editor': '#222834',       # Editor background
            'fg_primary': "#FFFFFF",      # Soft white text
            'fg_secondary': '#4C566A',    # Muted text (line numbers, comments)
            'accent_green': "#18E729",    # Soft green (Run button, strings)
            'accent_blue': '#81A1C1',     # Soft blue (Keywords, Machine Code)
            'accent_red': "#F2273B",      # Soft red (Clear button, errors)
            'accent_yellow': "#FFB700",   # Soft yellow (Undo, Assembly)
            'line_numbers': "#151820",    # Line number gutter
            'selection': "#1E2431",       # Text selection
            'scrollbar': '#4C566A' ,       # Scrollbars
            'selection': "#434C5E",       
            'scrollbar': '#4C566A'        # Scrollbars
        }
        
        self.fonts = {
            'button': ('Segoe UI', 10, 'bold'),
            'code': ('Consolas', 13),
            'ui': ('Segoe UI', 10)
        }
        
        self.setup_styles()
        self.create_widgets()
        self.load_default_code()
        
    def resource_path(self, relative_path):
        """ Get absolute path to resource, works for dev and for PyInstaller """
        try:
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")
        return os.path.join(base_path, relative_path)
      
    def setup_styles(self):
        """Configure ttk styles"""
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure('Custom.TNotebook', background=self.colors['bg_primary'], borderwidth=0)
        style.configure('Custom.TNotebook.Tab', 
                        padding=[16, 12], 
                        font=self.fonts['ui'],
                        background=self.colors['bg_secondary'],
                        foreground=self.colors['fg_primary'],
                        width=15, 
                        anchor='center',
                        focuscolor=self.colors['bg_tertiary'],
                        borderwidth=0)
        
        style.map('Custom.TNotebook.Tab', 
                  background=[("selected", self.colors['bg_tertiary'])],
                  expand=[("selected", [0, 0, 0, 0])],
                  padding=[("selected", [16, 12])])

    def create_widgets(self):
        """Create main UI"""
        main_frame = tk.Frame(self.root, bg=self.colors['bg_primary'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        self.create_toolbar(main_frame)
        
        self.main_paned = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        self.main_paned.pack(fill=tk.BOTH, expand=True)
        
        self.editor_section = self.create_editor_section()
        self.sidebar_section = self.create_sidebar_section()
        
        self.main_paned.add(self.editor_section, weight=6)
        self.main_paned.add(self.sidebar_section, weight=4) 
        
    def create_toolbar(self, parent):
        """Modern toolbar"""
        toolbar = tk.Frame(parent, bg=self.colors['bg_secondary'], height=60)
        toolbar.pack(fill=tk.X, pady=(0, 8))
        toolbar.pack_propagate(False)
        
        left_frame = tk.Frame(toolbar, bg=self.colors['bg_secondary'])
        left_frame.pack(side=tk.LEFT, padx=16, pady=12)
        
        self.run_btn = self.create_modern_button(left_frame, "▶ Run", self.colors['accent_green'], self.run_code, "#FFFFFF")
        self.clear_btn = self.create_modern_button(left_frame, "🗑 Clear", self.colors['accent_red'], self.clear_all, "#FFFFFF")
        
        right_frame = tk.Frame(toolbar, bg=self.colors['bg_secondary'])
        right_frame.pack(side=tk.RIGHT, padx=16, pady=12)
        
        self.undo_btn = self.create_modern_button(right_frame, "↶ Undo", self.colors['accent_yellow'], self.undo_code, "#FFFFFF")
        
    def create_modern_button(self, parent, text, bg_color, command, fg_color='white', state='normal'):
        """Modern button with hover"""
        btn = tk.Button(parent, text=text, bg=bg_color, fg=fg_color,
                       font=self.fonts['button'], relief='flat', bd=0,
                       padx=20, pady=8, cursor='hand2', command=command)
        btn.configure(state=state)
        
        original_bg = bg_color
        def on_enter(e): btn.configure(bg=self.lighten_color(bg_color, 0.15))
        def on_leave(e): btn.configure(bg=original_bg)
        
        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)
        btn.pack(side=tk.LEFT, padx=4)
        return btn
    
    def create_editor_section(self):
        """Code editor"""
        editor_frame = tk.Frame(self.main_paned, bg=self.colors['bg_primary'])
        editor_container = tk.Frame(editor_frame, bg=self.colors['bg_editor'], bd=0)
        editor_container.pack(fill=tk.BOTH, expand=True, padx=(0, 4))
        
        # Grid Configuration for dynamic scrollbars
        editor_container.grid_rowconfigure(0, weight=1)
        editor_container.grid_columnconfigure(1, weight=1)
        
        # Line numbers
        self.line_numbers = tk.Text(editor_container, width=4, padx=8, pady=8, takefocus=0, 
                                   bg=self.colors['line_numbers'], fg=self.colors['fg_secondary'],
                                   font=self.fonts['code'], state='disabled', border=0)
        self.line_numbers.grid(row=0, column=0, sticky='ns')
        
        # AutoScrollbars
        v_scrollbar = AutoScrollbar(editor_container, bg=self.colors['scrollbar'], troughcolor=self.colors['bg_primary'], bd=0, relief='flat')
        h_scrollbar = AutoScrollbar(editor_container, orient=tk.HORIZONTAL, bg=self.colors['scrollbar'], troughcolor=self.colors['bg_primary'], bd=0, relief='flat')
        
        self.code_editor = tk.Text(editor_container, font=self.fonts['code'],
                                  bg=self.colors['bg_editor'], fg=self.colors['fg_primary'],
                                  insertbackground=self.colors['accent_blue'],
                                  selectbackground=self.colors['selection'],
                                  undo=True, wrap=tk.NONE, border=0, padx=8, pady=8,
                                  yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set,
                                  highlightthickness=0, relief='flat')
        
        v_scrollbar.config(command=self.code_editor.yview)
        h_scrollbar.config(command=self.code_editor.xview)
        
        self.code_editor.grid(row=0, column=1, sticky='nsew')
        v_scrollbar.grid(row=0, column=2, sticky='ns')
        h_scrollbar.grid(row=1, column=1, sticky='ew')
        
        self.setup_syntax_highlighting()
        self.bind_editor_events()
        return editor_frame
    
    def create_sidebar_section(self):
        """Output tabs"""
        sidebar_frame = tk.Frame(self.main_paned, bg=self.colors['bg_primary'])
        self.notebook = ttk.Notebook(sidebar_frame, style='Custom.TNotebook')
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=(4, 0))
        
        # --- Terminal Tab ---
        self.output_frame = tk.Frame(self.notebook, bg=self.colors['bg_tertiary'])
        self.output_frame.grid_rowconfigure(0, weight=1)
        self.output_frame.grid_columnconfigure(0, weight=1)
        
        output_scroll = AutoScrollbar(self.output_frame)
        self.output_console = tk.Text(self.output_frame, font=self.fonts['code'],
                                     bg=self.colors['bg_tertiary'], fg=self.colors['fg_primary'], state=tk.DISABLED,
                                     wrap=tk.WORD, border=0, padx=8, pady=8,
                                     yscrollcommand=output_scroll.set)
        
        output_scroll.config(command=self.output_console.yview)
        
        self.output_console.grid(row=0, column=0, sticky='nsew')
        output_scroll.grid(row=0, column=1, sticky='ns')
        self.notebook.add(self.output_frame, text="Terminal")
        
        # --- Assembly Tab ---
        self.assembly_frame = tk.Frame(self.notebook, bg=self.colors['bg_tertiary'])
        self.assembly_frame.grid_rowconfigure(0, weight=1)
        self.assembly_frame.grid_columnconfigure(0, weight=1)
        
        asm_scroll_v = AutoScrollbar(self.assembly_frame)
        asm_scroll_h = AutoScrollbar(self.assembly_frame, orient=tk.HORIZONTAL)
        
        self.assembly_console = tk.Text(self.assembly_frame, font=self.fonts['code'],
                                       bg=self.colors['bg_tertiary'], fg=self.colors['accent_yellow'], state=tk.DISABLED,
                                       wrap=tk.NONE, border=0, padx=8, pady=8,
                                       yscrollcommand=asm_scroll_v.set, xscrollcommand=asm_scroll_h.set)
        
        asm_scroll_v.config(command=self.assembly_console.yview)
        asm_scroll_h.config(command=self.assembly_console.xview)
        
        self.assembly_console.grid(row=0, column=0, sticky='nsew')
        asm_scroll_v.grid(row=0, column=1, sticky='ns')
        asm_scroll_h.grid(row=1, column=0, sticky='ew')
        self.notebook.add(self.assembly_frame, text="Assembly Code")
        
        # --- Machine Code Tab ---
        self.machine_frame = tk.Frame(self.notebook, bg=self.colors['bg_tertiary'])
        self.machine_frame.grid_rowconfigure(0, weight=1)
        self.machine_frame.grid_columnconfigure(0, weight=1)
        
        mc_scroll_v = AutoScrollbar(self.machine_frame)
        mc_scroll_h = AutoScrollbar(self.machine_frame, orient=tk.HORIZONTAL)
        
        self.machine_console = tk.Text(self.machine_frame, font=self.fonts['code'],
                                       bg=self.colors['bg_tertiary'], fg=self.colors['accent_blue'], state=tk.DISABLED,
                                       wrap=tk.NONE, border=0, padx=8, pady=8,
                                       yscrollcommand=mc_scroll_v.set, xscrollcommand=mc_scroll_h.set)
        
        mc_scroll_v.config(command=self.machine_console.yview)
        mc_scroll_h.config(command=self.machine_console.xview)
        
        self.machine_console.grid(row=0, column=0, sticky='nsew')
        mc_scroll_v.grid(row=0, column=1, sticky='ns')
        mc_scroll_h.grid(row=1, column=0, sticky='ew')
        self.notebook.add(self.machine_frame, text="Machine Code")
        
        return sidebar_frame
    
    def setup_syntax_highlighting(self):
        self.code_editor.tag_configure("keyword", foreground=self.colors['accent_blue'], font=(self.fonts['code'][0], 13, 'bold'))
        self.code_editor.tag_configure("string", foreground=self.colors['accent_green'])
        self.code_editor.tag_configure("comment", foreground=self.colors['fg_secondary'], font=(self.fonts['code'][0], 13, 'italic'))
    
    def bind_editor_events(self):
        self._typing_timer = None 
        self.code_editor.bind("<KeyRelease>", self.on_text_change)
        self.code_editor.bind("<MouseWheel>", self.on_text_change)
        self.code_editor.bind("<Configure>", self.on_text_change)

    def on_text_change(self, event=None):
        if self._typing_timer:
            self.root.after_cancel(self._typing_timer)
        self._typing_timer = self.root.after(200, self.perform_updates)

    def perform_updates(self):
        self.update_line_numbers()
        self.highlight_syntax()

    def highlight_syntax(self):
        try:
            content = self.code_editor.get("1.0", tk.END)
            self.code_editor.tag_remove("keyword", "1.0", tk.END)
            self.code_editor.tag_remove("string", "1.0", tk.END)
            self.code_editor.tag_remove("comment", "1.0", tk.END)
            
            keywords = ["_start", "_done", "show>>", "input<<"]
            for kw in keywords:
                for match in re.finditer(re.escape(kw), content):
                    start_pos = f"1.0+{match.start()}c"
                    end_pos = f"1.0+{match.end()}c"
                    self.code_editor.tag_add("keyword", start_pos, end_pos)

            for match in re.finditer(r'".*?"', content):
                start_pos = f"1.0+{match.start()}c"
                end_pos = f"1.0+{match.end()}c"
                self.code_editor.tag_add("string", start_pos, end_pos)
                
            for match in re.finditer(r'//.*', content):
                start_pos = f"1.0+{match.start()}c"
                end_pos = f"1.0+{match.end()}c"
                self.code_editor.tag_add("comment", start_pos, end_pos)
        except Exception:
            pass  
    
    def load_default_code(self):
        default_code = '''_start{
    show>>( "Hello! World. This is our Custom PL." );
_done}'''
        self.code_editor.insert(tk.END, default_code)
        self.root.after_idle(self.highlight_syntax)
    
    def lighten_color(self, color, factor=0.1):
        try:
            color = color.lstrip('#')
            rgb = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
            lightened = tuple(min(255, int(c + (255 - c) * factor)) for c in rgb)
            return f"#{lightened[0]:02x}{lightened[1]:02x}{lightened[2]:02x}"
        except:
            return color
    
    def update_line_numbers(self, event=None):
        try:
            line_count = int(self.code_editor.index('end-1c').split('.')[0])
            lines = [str(i) for i in range(1, line_count + 1)]
            line_numbers_string = "\n".join(lines) + "\n"
            
            self.line_numbers.config(state='normal')
            self.line_numbers.delete('1.0', tk.END)
            self.line_numbers.insert('1.0', line_numbers_string)
            self.line_numbers.config(state='disabled')
            self.line_numbers.yview_moveto(self.code_editor.yview()[0])
        except:
            pass
    
    def clear_all(self):
        self.code_editor.delete(1.0, tk.END)
        self.clear_output()
    
    def clear_output(self):
        self.output_console.config(state=tk.NORMAL)
        self.output_console.delete(1.0, tk.END)
        self.output_console.config(state=tk.DISABLED)
        
        self.assembly_console.config(state=tk.NORMAL)
        self.assembly_console.delete(1.0, tk.END)
        self.assembly_console.config(state=tk.DISABLED)
        
        self.machine_console.config(state=tk.NORMAL)
        self.machine_console.delete(1.0, tk.END)
        self.machine_console.config(state=tk.DISABLED)
    
    def undo_code(self):
        try:
            self.code_editor.edit_undo()
            self.on_text_change()
        except:
            pass

    def generate_machine_code(self, assembly_text):
        """Translates MIPS64 assembly directly into Hex and Binary"""
        mc_lines = []
        
        # --- NEW: PASS 1 - Build a Symbol Table for Variables ---
        symbol_table = {}
        current_offset = 0
        
        for line in assembly_text.split('\n'):
            line = line.strip()
            # Find data labels (like "a: .byte 0") to record their memory offsets
            if ':' in line:
                label = line.split(':')[0].strip()
                symbol_table[label] = current_offset
                current_offset += 8  # Add 8 bytes for the next variable, matching your architecture
                
        # --- PASS 2 - Generate the Machine Code ---
        for line in assembly_text.split('\n'):
            line = line.strip()
            
            if not line or line.startswith('.') or ':' in line:
                continue

            binary = "00000000 00000 00000 00000 00000 000000"
            hex_code = "0x00000000"
            
            try:
                if line.startswith("daddiu "):
                    match = re.match(r"daddiu\s+r(\d+),\s*r(\d+),\s*(-?\d+)", line)
                    if match:
                        rt, rs, imm = map(int, match.groups())
                        imm &= 0xFFFF
                        code = (0x19 << 26) | (rs << 21) | (rt << 16) | imm
                        hex_code = f"0x{code:08X}"
                        binary = f"{0x19:06b} {rs:05b} {rt:05b} {imm:016b}"
                        
                elif line.startswith("daddi "):
                    match = re.match(r"daddi\s+r(\d+),\s*r(\d+),\s*(\w+)", line)
                    if match:
                        rt, rs, label = match.groups()
                        rs, rt = int(rs), int(rt)
                        imm = 0 
                        code = (0x18 << 26) | (rs << 21) | (rt << 16) | imm
                        hex_code = f"0x{code:08X}"
                        binary = f"{0x18:06b} {rs:05b} {rt:05b} {imm:016b}"

                elif line.startswith("daddu "):
                    match = re.match(r"daddu\s+r(\d+),\s*r(\d+),\s*r(\d+)", line)
                    if match:
                        rd, rs, rt = map(int, match.groups())
                        code = (0 << 26) | (rs << 21) | (rt << 16) | (rd << 11) | 0x2D
                        hex_code = f"0x{code:08X}"
                        binary = f"000000 {rs:05b} {rt:05b} {rd:05b} 00000 101101"

                elif line.startswith("dsubu "):
                    match = re.match(r"dsubu\s+r(\d+),\s*r(\d+),\s*r(\d+)", line)
                    if match:
                        rd, rs, rt = map(int, match.groups())
                        code = (0 << 26) | (rs << 21) | (rt << 16) | (rd << 11) | 0x2F
                        hex_code = f"0x{code:08X}"
                        binary = f"000000 {rs:05b} {rt:05b} {rd:05b} 00000 101111"

                # FIXED: dmult now accepts two registers
                elif line.startswith("dmult "):
                    match = re.match(r"dmult\s+r(\d+),\s*r(\d+)", line)
                    if match:
                        rs, rt = map(int, match.groups())
                        code = (0 << 26) | (rs << 21) | (rt << 16) | 0x1C
                        hex_code = f"0x{code:08X}"
                        binary = f"000000 {rs:05b} {rt:05b} 00000 00000 011100"

                elif line.startswith("ddiv "):
                    match = re.match(r"ddiv\s+r(\d+),\s*r(\d+)", line)
                    if match:
                        rs, rt = map(int, match.groups())
                        code = (0 << 26) | (rs << 21) | (rt << 16) | 0x1E
                        hex_code = f"0x{code:08X}"
                        binary = f"000000 {rs:05b} {rt:05b} 00000 00000 011110"

                elif line.startswith("mflo "):
                    match = re.match(r"mflo\s+r(\d+)", line)
                    if match:
                        rd = int(match.group(1))
                        code = (0 << 26) | (rd << 11) | 0x12
                        hex_code = f"0x{code:08X}"
                        binary = f"000000 00000 00000 {rd:05b} 00000 010010"
                        
                elif line.startswith("mfhi "):
                    match = re.match(r"mfhi\s+r(\d+)", line)
                    if match:
                        rd = int(match.group(1))
                        code = (0 << 26) | (rd << 11) | 0x10
                        hex_code = f"0x{code:08X}"
                        binary = f"000000 00000 00000 {rd:05b} 00000 010000"

                # FIXED: sb now accepts words (\w+) and uses the symbol table
                elif line.startswith("sb "):
                    match = re.match(r"sb\s+r(\d+),\s*(-?\d+|\w+)\(r(\d+)\)", line)
                    if match:
                        rt_str, offset_str, rs_str = match.groups()
                        rt, rs = int(rt_str), int(rs_str)
                        
                        # Look up the offset if it's a variable word, otherwise use the integer
                        imm = symbol_table.get(offset_str, 0) if not offset_str.lstrip('-').isdigit() else int(offset_str)
                        imm &= 0xFFFF
                        
                        code = (0x28 << 26) | (rs << 21) | (rt << 16) | imm
                        hex_code = f"0x{code:08X}"
                        binary = f"{0x28:06b} {rs:05b} {rt:05b} {imm:016b}"

                # FIXED: lb now accepts words (\w+) and uses the symbol table
                elif line.startswith("lb "):
                    match = re.match(r"lb\s+r(\d+),\s*(-?\d+|\w+)\(r(\d+)\)", line)
                    if match:
                        rt_str, offset_str, rs_str = match.groups()
                        rt, rs = int(rt_str), int(rs_str)
                        
                        imm = symbol_table.get(offset_str, 0) if not offset_str.lstrip('-').isdigit() else int(offset_str)
                        imm &= 0xFFFF
                        
                        code = (0x20 << 26) | (rs << 21) | (rt << 16) | imm
                        hex_code = f"0x{code:08X}"
                        binary = f"{0x20:06b} {rs:05b} {rt:05b} {imm:016b}"
                        
            except Exception:
                pass 
            
            mc_lines.append(f"{line:<25} {hex_code:<12} {binary}")

        return '\n'.join(mc_lines)
    
    def run_code(self):
        self.run_btn.config(state='disabled')
        
        code = self.code_editor.get(1.0, tk.END).strip()
        if not code:
            self.run_btn.config(state='normal')
            return
        
        temp_file = "temp_program.txt"
        try:
            with open(temp_file, "w") as f:
                f.write(code)

            if not os.path.exists("parser.exe"):
                self.show_error("❌ parser.exe not found!")
                return

            self.clear_output()
            
            with open(temp_file, "r") as input_file:
                result = subprocess.run(["parser.exe"],stdin=input_file, 
                                      capture_output=True, text=True, timeout=5,
                                      creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                
                self.output_console.config(state=tk.NORMAL)
                self.output_console.tag_configure("stdout", foreground=self.colors['accent_green'])
                self.output_console.tag_configure("error", foreground=self.colors['accent_red'])
                
            if result.stdout:
                self.output_console.insert(tk.END, result.stdout, "stdout")
            if result.stderr:
                self.output_console.insert(tk.END, f"\n🔴 ERRORS:\n{result.stderr}\n", "error")
            
            self.output_console.config(state=tk.DISABLED)
            self.output_console.see(tk.END)
            
            if os.path.exists("output.s"):
                with open("output.s", "r") as f:
                    asm_code = f.read()
                    
                    self.assembly_console.config(state=tk.NORMAL)
                    self.assembly_console.delete(1.0, tk.END)
                    self.assembly_console.insert(tk.END, asm_code)
                    self.assembly_console.config(state=tk.DISABLED)
                    
                    machine_code = self.generate_machine_code(asm_code)
                    self.machine_console.config(state=tk.NORMAL)
                    self.machine_console.delete(1.0, tk.END)
                    self.machine_console.insert(tk.END, machine_code)
                    self.machine_console.config(state=tk.DISABLED)
            
        except Exception as e:
            self.show_error(f"Error: {str(e)}")
        finally:
            self.run_btn.config(state='normal')
    
    def show_error(self, message):
        self.output_console.config(state=tk.NORMAL)
        self.output_console.insert(tk.END, f"{message}\n\n")
        self.output_console.config(state=tk.DISABLED)
        self.run_btn.config(state='normal')

if __name__ == "__main__":
    root = tk.Tk()
    app = SegObC_IDE(root)
    root.mainloop()
