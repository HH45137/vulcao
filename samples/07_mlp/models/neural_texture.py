"""Fit a single image with a coordinate MLP ("neural texture") using Tkinter.

The whole pipeline lives in one window:

* pick an image file,
* a small MLP maps normalized pixel coordinates (x, y) in [-1, 1] to RGB,
* the network is trained to reproduce the image with an MSE loss,
* the window shows the original image, the live reconstruction, the loss and
  the PSNR while training runs in a background thread,
* the trained weights can be saved and reloaded later for pure inference.

The GUI is English-only and relies solely on the standard library plus the
packages this repository already depends on (torch, numpy, Pillow).
"""

import math
import queue
import threading
import time
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import torch
from PIL import Image, ImageTk
from torch import nn

# The directory that contains this script is used as the root path.
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "neural_texture.pth"

DEFAULT_RESOLUTION = 128
DEFAULT_HIDDEN = 256
DEFAULT_LAYERS = 4
DEFAULT_FREQUENCIES = 8
DEFAULT_ACTIVATION = "relu"
DEFAULT_LR = 1e-3
DEFAULT_BATCH = 8192
DEFAULT_STEPS = 3000

IMAGE_FILETYPES = [
    ("Images", "*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff *.webp"),
    ("All files", "*.*"),
]


def resolve_device(name):
    """Turn a device name coming from the GUI into a torch.device."""
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def device_label(device):
    """Human readable description of a device, shown in the status bar and log."""
    if device.type == "cuda":
        return f"CUDA ({torch.cuda.get_device_name(0)})"
    return "CPU"


def make_coords(width, height, device):
    """Return the (H * W, 2) normalized pixel coordinate grid of an image."""
    xs = torch.linspace(-1.0, 1.0, width, device=device)
    ys = torch.linspace(-1.0, 1.0, height, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((grid_x, grid_y), dim=-1).reshape(-1, 2)


def psnr_from_mse(mse):
    """Peak signal-to-noise ratio for data in [0, 1]."""
    if mse <= 0.0:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def image_to_target(image, max_side):
    """Resize a PIL image so its longest side is max_side and flatten it.

    Returns the (N, 3) float target in [0, 1] together with the working width
    and height actually used for training.
    """
    source_width, source_height = image.size
    scale = max_side / max(source_width, source_height)
    width = max(1, round(source_width * scale))
    height = max(1, round(source_height * scale))
    resized = image.resize((width, height), Image.LANCZOS)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    target = torch.from_numpy(array).reshape(-1, 3).contiguous()
    return target, width, height




# 1. Model: a coordinate MLP, optionally wrapped in Fourier features
class SineLayer(nn.Module):
    """Linear layer followed by sin(omega_0 * x) with SIREN initialization."""

    def __init__(self, in_features, out_features, omega=30.0, is_first=False):
        super().__init__()
        self.omega = omega
        self.linear = nn.Linear(in_features, out_features)
        with torch.no_grad():
            if is_first:
                bound = 1.0 / in_features
            else:
                bound = math.sqrt(6.0 / in_features) / omega
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega * self.linear(x))


class CoordinateMLP(nn.Module):
    """Maps normalized (x, y) coordinates to RGB values in [0, 1].

    The optional positional encoding expands each coordinate into
    sin/cos bands at increasing frequencies, which lets a plain ReLU MLP
    reproduce fine image detail.
    """

    def __init__(
        self,
        hidden_dim=DEFAULT_HIDDEN,
        num_layers=DEFAULT_LAYERS,
        num_frequencies=DEFAULT_FREQUENCIES,
        activation=DEFAULT_ACTIVATION,
        out_dim=3,
    ):
        super().__init__()
        self.num_frequencies = int(num_frequencies)
        self.activation = activation
        input_dim = 2 + 4 * self.num_frequencies if self.num_frequencies > 0 else 2

        layers = []
        if activation == "sine":
            layers.append(SineLayer(input_dim, hidden_dim, omega=30.0, is_first=True))
            for _ in range(max(0, num_layers - 1)):
                layers.append(SineLayer(hidden_dim, hidden_dim, omega=30.0))
        else:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            for _ in range(max(0, num_layers - 1)):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def encode(self, coords):
        if self.num_frequencies <= 0:
            return coords
        base = torch.arange(
            self.num_frequencies, device=coords.device, dtype=coords.dtype
        )
        frequencies = torch.pow(2.0, base) * math.pi
        angles = coords.unsqueeze(-1) * frequencies
        return torch.cat(
            (coords, torch.sin(angles).flatten(1), torch.cos(angles).flatten(1)), dim=1
        )

    def forward(self, coords):
        return torch.sigmoid(self.net(self.encode(coords)))


def build_model(config):
    """Rebuild a model from the configuration stored next to its weights."""
    return CoordinateMLP(
        hidden_dim=int(config["hidden_dim"]),
        num_layers=int(config["num_layers"]),
        num_frequencies=int(config["num_frequencies"]),
        activation=config["activation"],
    )


@torch.no_grad()
def render(model, width, height):
    """Run the model over a whole grid and return an (H, W, 3) uint8 array."""
    model.eval()
    device = next(model.parameters()).device
    coords = make_coords(width, height, device)
    image = model(coords).view(height, width, 3).clamp(0.0, 1.0).cpu().numpy()
    return (image * 255.0).astype(np.uint8)


def snapshot_state(model):
    """Copy the weights to CPU so the GUI thread can use them safely."""
    return {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }


# 2. Tkinter GUI
class NeuralTextureApp:
    """Fit one image with a coordinate MLP, train, observe and infer in one window."""

    def __init__(self, root):
        self.root = root
        self.root.title("Neural Texture - MLP Image Fitter")
        self.root.minsize(1120, 720)

        # Background training plumbing.
        self.event_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker = None

        # Target image state.
        self.target = None
        self.width = 0
        self.height = 0
        self.original_pil = None
        self.recon_array = None
        # Full resolution source kept so "Image size" can be re-applied later.
        self.source_image = None
        self.source_name = ""
        self.target_resolution = 0

        # Trained model kept on CPU for inference.
        self.model = None
        self.model_config = None
        self.model_render_size = None

        # History backing the chart.
        self.steps = []
        self.losses = []
        self.psnrs = []

        # Tk image references, kept alive so they are not garbage collected.
        self._image_refs = {}

        self.path_var = tk.StringVar(value="No image loaded.")
        self.resolution_var = tk.StringVar(value=str(DEFAULT_RESOLUTION))
        self.hidden_var = tk.StringVar(value=str(DEFAULT_HIDDEN))
        self.layers_var = tk.StringVar(value=str(DEFAULT_LAYERS))
        self.freq_var = tk.StringVar(value=str(DEFAULT_FREQUENCIES))
        self.activation_var = tk.StringVar(value=DEFAULT_ACTIVATION)
        self.lr_var = tk.StringVar(value=str(DEFAULT_LR))
        self.batch_var = tk.StringVar(value=str(DEFAULT_BATCH))
        self.steps_var = tk.StringVar(value=str(DEFAULT_STEPS))
        self.device_var = tk.StringVar(value="auto")
        self.status_var = tk.StringVar(value="Ready.")

        self._build_layout()
        self._draw_chart()
        self._redraw_original()
        self._redraw_recon()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(100, self._poll_events)

    def _build_layout(self):
        outer = ttk.Frame(self.root, padding=10)
        outer.grid(row=0, column=0, sticky="nsew")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        outer.columnconfigure(0, weight=0)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(0, weight=1)

        left = ttk.Frame(outer)
        left.grid(row=0, column=0, sticky="nsw", padx=(0, 10))
        right = ttk.Frame(outer)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        self._build_image_panel(left)
        self._build_config_panel(left)
        self._build_log_panel(left)
        self._build_view_panel(right)
        self._build_chart_panel(right)

        status = ttk.Label(
            outer,
            textvariable=self.status_var,
            anchor="w",
            relief="sunken",
            padding=4,
        )
        status.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))

    def _build_image_panel(self, parent):
        frame = ttk.LabelFrame(parent, text="Image", padding=10)
        frame.grid(row=0, column=0, sticky="ew")
        frame.columnconfigure(0, weight=1)

        self.load_image_button = ttk.Button(
            frame, text="Load image...", command=self.load_image
        )
        self.load_image_button.grid(row=0, column=0, sticky="ew")

        ttk.Label(
            frame, textvariable=self.path_var, wraplength=260, justify="left"
        ).grid(row=1, column=0, sticky="w", pady=(6, 0))

    def _build_config_panel(self, parent):
        frame = ttk.LabelFrame(parent, text="Model and training", padding=10)
        frame.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        frame.columnconfigure(1, weight=1)

        def add_row(row, label, widget):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=2)
            widget.grid(row=row, column=1, sticky="ew", pady=2)
            return widget

        self.resolution_spin = add_row(
            0,
            "Image size",
            ttk.Spinbox(
                frame, from_=32, to=512, increment=16, textvariable=self.resolution_var
            ),
        )
        # Re-apply the training resolution as soon as the value is committed.
        for sequence in ("<Return>", "<KP_Enter>", "<FocusOut>"):
            self.resolution_spin.bind(
                sequence, lambda event: self._apply_resolution()
            )
        self.hidden_spin = add_row(
            1,
            "Hidden size",
            ttk.Spinbox(
                frame, from_=16, to=1024, increment=16, textvariable=self.hidden_var
            ),
        )
        self.layers_spin = add_row(
            2,
            "Hidden layers",
            ttk.Spinbox(frame, from_=1, to=12, textvariable=self.layers_var),
        )
        self.freq_spin = add_row(
            3,
            "Frequencies",
            ttk.Spinbox(frame, from_=0, to=16, textvariable=self.freq_var),
        )
        self.activation_combo = add_row(
            4,
            "Activation",
            ttk.Combobox(
                frame,
                textvariable=self.activation_var,
                values=("relu", "sine"),
                state="readonly",
            ),
        )
        self.lr_entry = add_row(
            5, "Learning rate", ttk.Entry(frame, textvariable=self.lr_var)
        )
        self.batch_spin = add_row(
            6,
            "Batch size",
            ttk.Spinbox(
                frame, from_=64, to=65536, increment=64, textvariable=self.batch_var
            ),
        )
        self.steps_spin = add_row(
            7,
            "Total steps",
            ttk.Spinbox(
                frame, from_=100, to=50000, increment=100, textvariable=self.steps_var
            ),
        )
        self.device_combo = add_row(
            8,
            "Device",
            ttk.Combobox(
                frame,
                textvariable=self.device_var,
                values=("auto", "cpu", "cuda"),
                state="readonly",
            ),
        )

        self.progress = ttk.Progressbar(frame, mode="determinate", maximum=100)
        self.progress.grid(row=9, column=0, columnspan=2, sticky="ew", pady=(8, 4))

        self.metrics = ttk.Label(
            frame,
            text="Step: -\nLoss: -\nPSNR: -\nElapsed: -",
            justify="left",
        )
        self.metrics.grid(row=10, column=0, columnspan=2, sticky="w", pady=(4, 8))

        buttons = ttk.Frame(frame)
        buttons.grid(row=11, column=0, columnspan=2, sticky="ew")
        buttons.columnconfigure(0, weight=1)
        buttons.columnconfigure(1, weight=1)

        self.start_button = ttk.Button(buttons, text="Start", command=self.start_fit)
        self.start_button.grid(row=0, column=0, sticky="ew", padx=2, pady=2)
        self.stop_button = ttk.Button(
            buttons, text="Stop", command=self.stop_fit, state="disabled"
        )
        self.stop_button.grid(row=0, column=1, sticky="ew", padx=2, pady=2)

        self.save_model_button = ttk.Button(
            buttons, text="Save model...", command=self.save_model, state="disabled"
        )
        self.save_model_button.grid(row=1, column=0, sticky="ew", padx=2, pady=2)
        self.save_render_button = ttk.Button(
            buttons, text="Save render...", command=self.save_render, state="disabled"
        )
        self.save_render_button.grid(row=1, column=1, sticky="ew", padx=2, pady=2)

        self.load_model_button = ttk.Button(
            buttons, text="Load model...", command=self.load_model
        )
        self.load_model_button.grid(
            row=2, column=0, columnspan=2, sticky="ew", padx=2, pady=2
        )

    def _build_log_panel(self, parent):
        frame = ttk.LabelFrame(parent, text="Log", padding=10)
        frame.grid(row=2, column=0, sticky="nsew", pady=(10, 0))
        parent.rowconfigure(2, weight=1)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        self.log_text = tk.Text(
            frame, height=10, width=40, state="disabled", wrap="word"
        )
        self.log_text.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(frame, command=self.log_text.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=scroll.set)

    def _build_view_panel(self, parent):
        frame = ttk.LabelFrame(parent, text="Observation", padding=10)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(1, weight=1)

        ttk.Label(frame, text="Original").grid(row=0, column=0)
        ttk.Label(frame, text="MLP fit").grid(row=0, column=1)

        self.original_canvas = tk.Canvas(
            frame,
            width=300,
            height=300,
            background="#f2f2f2",
            highlightthickness=1,
            highlightbackground="#c8c8c8",
        )
        self.original_canvas.grid(row=1, column=0, sticky="nsew", padx=(0, 5))
        self.original_canvas.bind("<Configure>", lambda event: self._redraw_original())

        self.recon_canvas = tk.Canvas(
            frame,
            width=300,
            height=300,
            background="#f2f2f2",
            highlightthickness=1,
            highlightbackground="#c8c8c8",
        )
        self.recon_canvas.grid(row=1, column=1, sticky="nsew", padx=(5, 0))
        self.recon_canvas.bind("<Configure>", lambda event: self._redraw_recon())

    def _build_chart_panel(self, parent):
        frame = ttk.LabelFrame(parent, text="Training progress", padding=10)
        frame.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        self.chart_canvas = tk.Canvas(
            frame,
            width=560,
            height=260,
            background="white",
            highlightthickness=1,
            highlightbackground="#c8c8c8",
        )
        self.chart_canvas.grid(row=0, column=0, sticky="nsew")
        self.chart_canvas.bind("<Configure>", lambda event: self._draw_chart())

    @staticmethod
    def _canvas_size(canvas):
        """Return the live canvas size, falling back to the configured size."""
        width = canvas.winfo_width()
        height = canvas.winfo_height()
        if width <= 1:
            width = int(canvas["width"])
        if height <= 1:
            height = int(canvas["height"])
        return width, height

    def _draw_placeholder(self, canvas, text):
        canvas.delete("all")
        width, height = self._canvas_size(canvas)
        canvas.create_text(width / 2, height / 2, text=text, fill="#aaaaaa")

    def _show_image(self, canvas, image, resample=Image.LANCZOS, key="image"):
        """Draw a PIL image centered inside a canvas, fitted to its size."""
        width, height = self._canvas_size(canvas)
        image_width, image_height = image.size
        scale = min(width / image_width, height / image_height)
        size = (max(1, int(image_width * scale)), max(1, int(image_height * scale)))
        photo = ImageTk.PhotoImage(image.resize(size, resample))
        canvas.delete("all")
        canvas.create_image(width / 2, height / 2, image=photo)
        self._image_refs[key] = photo

    def _redraw_original(self):
        if self.original_pil is None:
            self._draw_placeholder(self.original_canvas, "No image loaded")
        else:
            self._show_image(self.original_canvas, self.original_pil, key="original")

    def _redraw_recon(self):
        if self.recon_array is None:
            self._draw_placeholder(self.recon_canvas, "Press 'Start' to fit")
        else:
            self._show_image(
                self.recon_canvas,
                Image.fromarray(self.recon_array),
                resample=Image.NEAREST,
                key="recon",
            )

    def _draw_chart(self):
        canvas = self.chart_canvas
        canvas.delete("all")
        width, height = self._canvas_size(canvas)
        left, right, top, bottom = 56, width - 56, 28, height - 34
        if right <= left or bottom <= top:
            return

        canvas.create_text(
            (left + right) / 2,
            14,
            text="Loss (red, left axis) / PSNR in dB (blue, right axis)",
            fill="#444444",
        )

        for step in range(6):
            fraction = step / 5
            y = bottom - fraction * (bottom - top)
            canvas.create_line(left, y, right, y, fill="#eeeeee")
            canvas.create_text(
                left - 6, y, text=f"{fraction:.1f}", anchor="e", fill="#888888"
            )
            canvas.create_text(
                right + 6, y, text=f"{fraction:.1f}", anchor="w", fill="#888888"
            )

        canvas.create_line(left, top, left, bottom, fill="#888888")
        canvas.create_line(left, bottom, right, bottom, fill="#888888")

        if not self.losses:
            canvas.create_text(
                (left + right) / 2,
                (top + bottom) / 2,
                text="Load an image and press 'Start'",
                fill="#aaaaaa",
            )
            return

        count = len(self.losses)
        loss_max = max(self.losses) or 1.0
        psnr_max = max(max(self.psnrs), 1.0)

        def x_at(index):
            if count == 1:
                return (left + right) / 2
            return left + index * (right - left) / (count - 1)

        loss_points = []
        psnr_points = []
        for index, (loss, psnr) in enumerate(zip(self.losses, self.psnrs)):
            x = x_at(index)
            loss_points.extend((x, bottom - (loss / loss_max) * (bottom - top)))
            psnr_points.extend((x, bottom - (psnr / psnr_max) * (bottom - top)))

        if len(loss_points) >= 4:
            canvas.create_line(*loss_points, fill="#d64545", width=2, smooth=True)
        if len(psnr_points) >= 4:
            canvas.create_line(*psnr_points, fill="#3b6bd6", width=2, smooth=True)

        for index in range(count):
            x = x_at(index)
            loss_y = loss_points[index * 2 + 1]
            psnr_y = psnr_points[index * 2 + 1]
            canvas.create_oval(
                x - 3, loss_y - 3, x + 3, loss_y + 3, fill="#d64545", outline=""
            )
            canvas.create_oval(
                x - 3, psnr_y - 3, x + 3, psnr_y + 3, fill="#3b6bd6", outline=""
            )

        canvas.create_text(
            left, bottom + 16, text=f"step {self.steps[0]}", anchor="w", fill="#888888"
        )
        canvas.create_text(
            right,
            bottom + 16,
            text=f"step {self.steps[-1]}",
            anchor="e",
            fill="#888888",
        )
        canvas.create_text(
            (left + right) / 2,
            bottom + 16,
            text=f"updates: {count}",
            fill="#888888",
        )

    def load_image(self):
        path = filedialog.askopenfilename(
            title="Open image", initialdir=str(BASE_DIR), filetypes=IMAGE_FILETYPES
        )
        if not path:
            return
        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:
            messagebox.showerror("Image error", f"Could not load the image:\n{exc}")
            return

        self.source_image = image
        self.source_name = Path(path).name
        self.original_pil = image
        self.model = None
        self.model_config = None
        self.save_model_button.configure(state="disabled")
        self.save_render_button.configure(state="disabled")
        self._redraw_original()
        self.status_var.set("Image loaded.")
        self._log(f"Loaded '{self.source_name}' ({image.width} x {image.height}).")
        self._apply_resolution(force=True)

    def _apply_resolution(self, force=False):
        """(Re)build the training target at the size shown in 'Image size'."""
        if self.source_image is None:
            return
        if self.worker is not None and self.worker.is_alive():
            return
        try:
            resolution = int(self.resolution_var.get())
        except ValueError:
            resolution = DEFAULT_RESOLUTION
        resolution = max(8, min(resolution, 4096))
        self.resolution_var.set(str(resolution))
        if not force and resolution == self.target_resolution:
            return

        self.target, self.width, self.height = image_to_target(
            self.source_image, resolution
        )
        self.target_resolution = resolution
        self.recon_array = None
        self._redraw_recon()
        self._reset_history()
        self.metrics.configure(text="Step: -\nLoss: -\nPSNR: -\nElapsed: -")
        self.path_var.set(
            f"{self.source_name}  ({self.width} x {self.height} training pixels)"
        )
        self._log(f"Training resolution set to {self.width} x {self.height}.")


    def _reset_history(self):
        self.steps.clear()
        self.losses.clear()
        self.psnrs.clear()
        self._draw_chart()

    def _read_settings(self):
        if self.target is None:
            messagebox.showinfo("No image", "Load an image first.")
            return None
        try:
            hidden_dim = int(self.hidden_var.get())
            num_layers = int(self.layers_var.get())
            num_frequencies = int(self.freq_var.get())
            learning_rate = float(self.lr_var.get())
            batch_size = int(self.batch_var.get())
            total_steps = int(self.steps_var.get())
        except ValueError:
            messagebox.showerror(
                "Invalid input",
                "Hidden size, layers, frequencies, batch size and total steps must "
                "be integers and the learning rate must be a number.",
            )
            return None
        if (
            hidden_dim < 1
            or num_layers < 1
            or num_frequencies < 0
            or batch_size < 1
            or total_steps < 1
            or learning_rate <= 0
        ):
            messagebox.showerror(
                "Invalid input",
                "Hidden size, layers, batch size and total steps must be at least 1, "
                "frequencies cannot be negative and the learning rate must be positive.",
            )
            return None
        return {
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "num_frequencies": num_frequencies,
            "activation": self.activation_var.get(),
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "total_steps": total_steps,
        }

    def start_fit(self):
        if self.worker is not None and self.worker.is_alive():
            return
        # Pick up any "Image size" edit that was not committed with Enter yet.
        self._apply_resolution()
        config = self._read_settings()
        if config is None:
            return

        device = resolve_device(self.device_var.get())
        self._reset_history()
        self.recon_array = None
        self._redraw_recon()
        self.metrics.configure(text="Step: -\nLoss: -\nPSNR: -\nElapsed: -")
        self.progress.configure(value=0, maximum=config["total_steps"])
        self.stop_event.clear()
        self._set_running(True)
        self.status_var.set("Fitting...")
        self._log(
            f"Fitting a {config['activation']} MLP on {device_label(device)} "
            f"({self.width}x{self.height} pixels, hidden={config['hidden_dim']}, "
            f"layers={config['num_layers']}, freqs={config['num_frequencies']}, "
            f"lr={config['learning_rate']:g}, batch={config['batch_size']}, "
            f"steps={config['total_steps']})."
        )

        self.worker = threading.Thread(
            target=self._fit_worker,
            args=(device, config),
            daemon=True,
        )
        self.worker.start()

    def stop_fit(self):
        if self.worker is not None and self.worker.is_alive():
            self.stop_event.set()
            self.stop_button.configure(state="disabled")
            self._log("Stop requested; finishing the current step...")

    @staticmethod
    @torch.no_grad()
    def _snapshot(model, coords, target, criterion, width, height):
        """Full-grid render (H, W, 3) uint8 plus the MSE of the current weights."""
        model.eval()
        full = model(coords)
        mse = criterion(full, target).item()
        image = full.view(height, width, 3).clamp(0.0, 1.0).cpu().numpy()
        return (image * 255.0).astype(np.uint8), mse

    def _fit_worker(self, device, config):
        try:
            width, height = self.width, self.height
            target = self.target.to(device)
            coords = make_coords(width, height, device)
            count = coords.shape[0]
            batch = min(config["batch_size"], count)
            total_steps = config["total_steps"]
            report_every = max(1, total_steps // 150)

            model = CoordinateMLP(
                hidden_dim=config["hidden_dim"],
                num_layers=config["num_layers"],
                num_frequencies=config["num_frequencies"],
                activation=config["activation"],
            ).to(device)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

            started = time.time()
            stopped = False
            step = 0
            for step in range(1, total_steps + 1):
                if self.stop_event.is_set():
                    stopped = True
                    break

                index = torch.randint(0, count, (batch,), device=device)
                prediction = model(coords[index])
                loss = criterion(prediction, target[index])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if step % report_every == 0 or step == total_steps:
                    image, mse = self._snapshot(
                        model, coords, target, criterion, width, height
                    )
                    self.event_queue.put(
                        (
                            "progress",
                            {
                                "step": step,
                                "total": total_steps,
                                "loss": mse,
                                "psnr": psnr_from_mse(mse),
                                "image": image,
                                "elapsed": time.time() - started,
                            },
                        )
                    )

            image, mse = self._snapshot(model, coords, target, criterion, width, height)
            self.event_queue.put(
                (
                    "done",
                    {
                        "stopped": stopped,
                        "state": snapshot_state(model),
                        "config": config,
                        "step": step,
                        "total": total_steps,
                        "loss": mse,
                        "psnr": psnr_from_mse(mse),
                        "image": image,
                        "width": width,
                        "height": height,
                        "elapsed": time.time() - started,
                    },
                )
            )
        except Exception as exc:  # pragma: no cover - surfaced through the GUI
            self.event_queue.put(("error", str(exc)))

    def _poll_events(self):
        while True:
            try:
                kind, payload = self.event_queue.get_nowait()
            except queue.Empty:
                break
            self._handle_event(kind, payload)
        self.root.after(100, self._poll_events)

    def _handle_event(self, kind, payload):
        if kind == "progress":
            self.steps.append(payload["step"])
            self.losses.append(payload["loss"])
            self.psnrs.append(payload["psnr"])
            self.progress.configure(value=payload["step"])
            self._update_metrics(
                payload["step"],
                payload["total"],
                payload["loss"],
                payload["psnr"],
                payload["elapsed"],
            )
            self.recon_array = payload["image"]
            self._redraw_recon()
            self._draw_chart()
        elif kind == "done":
            if self.steps and self.steps[-1] == payload["step"]:
                self.losses[-1] = payload["loss"]
                self.psnrs[-1] = payload["psnr"]
            else:
                self.steps.append(payload["step"])
                self.losses.append(payload["loss"])
                self.psnrs.append(payload["psnr"])
            self.recon_array = payload["image"]
            self._redraw_recon()
            self._draw_chart()
            self._update_metrics(
                payload["step"],
                payload["total"],
                payload["loss"],
                payload["psnr"],
                payload["elapsed"],
            )

            self.model = build_model(payload["config"])
            self.model.load_state_dict(payload["state"])
            self.model.eval()
            self.model_config = payload["config"]
            self.model_render_size = (payload["width"], payload["height"])
            self.save_model_button.configure(state="normal")
            self.save_render_button.configure(state="normal")
            self._set_running(False)

            if payload["stopped"]:
                self.status_var.set("Stopped.")
                self._log(
                    f"Stopped after {payload['step']} steps: "
                    f"loss={payload['loss']:.5f}, psnr={payload['psnr']:.2f} dB."
                )
            else:
                self.status_var.set("Fitting finished.")
                self._log(
                    f"Finished in {payload['elapsed']:.1f}s: "
                    f"loss={payload['loss']:.5f}, psnr={payload['psnr']:.2f} dB."
                )
        elif kind == "log":
            self._log(payload)
        elif kind == "error":
            self._set_running(False)
            self.status_var.set("Fitting failed.")
            self._log(f"Error: {payload}")
            messagebox.showerror("Fitting error", payload)

    def _update_metrics(self, step, total, loss, psnr, elapsed):
        self.metrics.configure(
            text=(
                f"Step: {step} / {total}\n"
                f"Loss: {loss:.5f}\n"
                f"PSNR: {psnr:.2f} dB\n"
                f"Elapsed: {elapsed:.1f}s"
            )
        )

    def _set_running(self, running):
        state = "disabled" if running else "normal"
        self.start_button.configure(state=state)
        self.load_image_button.configure(state=state)
        self.load_model_button.configure(state=state)
        self.resolution_spin.configure(state=state)
        self.hidden_spin.configure(state=state)
        self.layers_spin.configure(state=state)
        self.freq_spin.configure(state=state)
        self.lr_entry.configure(state=state)
        self.batch_spin.configure(state=state)
        self.steps_spin.configure(state=state)
        self.device_combo.configure(state="disabled" if running else "readonly")
        self.activation_combo.configure(state="disabled" if running else "readonly")
        self.stop_button.configure(state="normal" if running else "disabled")
        if running:
            self.save_model_button.configure(state="disabled")
            self.save_render_button.configure(state="disabled")

    def _log(self, message):
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"[{timestamp}] {message}\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def save_model(self):
        if self.model is None:
            messagebox.showinfo("No model", "Fit or load a model first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save model",
            initialdir=str(BASE_DIR),
            initialfile=MODEL_PATH.name,
            defaultextension=".pth",
            filetypes=[("PyTorch model", "*.pth"), ("All files", "*.*")],
        )
        if not path:
            return
        save_width, save_height = self.model_render_size or (self.width, self.height)
        torch.save(
            {
                "config": self.model_config,
                "width": save_width,
                "height": save_height,
                "state_dict": self.model.state_dict(),
            },
            path,
        )
        self._log(f"Saved model to {path}")
        self.status_var.set(f"Saved model to {path}")

    def load_model(self):
        path = filedialog.askopenfilename(
            title="Load model",
            initialdir=str(BASE_DIR),
            filetypes=[("PyTorch model", "*.pth"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            payload = torch.load(path, map_location="cpu")
            model = build_model(payload["config"])
            model.load_state_dict(payload["state_dict"])
            model.eval()
            render_width = int(payload.get("width", DEFAULT_RESOLUTION))
            render_height = int(payload.get("height", DEFAULT_RESOLUTION))
        except Exception as exc:
            messagebox.showerror("Load error", f"Could not load the model:\n{exc}")
            return

        self.model = model
        self.model_config = payload["config"]
        self.model_render_size = (render_width, render_height)
        self.recon_array = render(model, render_width, render_height)
        self._redraw_recon()
        self.save_model_button.configure(state="normal")
        self.save_render_button.configure(state="normal")

        config = payload["config"]
        self.hidden_var.set(str(config["hidden_dim"]))
        self.layers_var.set(str(config["num_layers"]))
        self.freq_var.set(str(config["num_frequencies"]))
        self.activation_var.set(config["activation"])

        self.status_var.set(f"Loaded model from {path}")
        self._log(
            f"Loaded a model from {path} and rendered it at "
            f"{render_width} x {render_height}."
        )

    def save_render(self):
        if self.recon_array is None:
            messagebox.showinfo("Nothing to save", "Fit an image first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save render",
            initialdir=str(BASE_DIR),
            initialfile="neural_texture_render.png",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("All files", "*.*")],
        )
        if not path:
            return
        Image.fromarray(self.recon_array).save(path)
        self._log(f"Saved render to {path}")
        self.status_var.set(f"Saved render to {path}")

    def _on_close(self):
        self.stop_event.set()
        # Stop a pending <FocusOut> handler from touching the widgets while the
        # window is being destroyed.
        self.source_image = None
        self.root.destroy()


def main():
    root = tk.Tk()
    try:
        ttk.Style().theme_use("clam")
    except tk.TclError:
        pass
    NeuralTextureApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
