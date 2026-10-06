"""Train a small MLP on MNIST with a Tkinter GUI.

This is the interactive counterpart of the headless training script. It keeps
the exact same model, data pipeline and optimizer, but adds a desktop GUI that
can:

* configure the device, number of epochs, learning rate and batch size,
* run the training loop in a background thread and stop it at any time,
* plot the loss and accuracy curves while they improve,
* save and load the trained weights,
* run inference on random MNIST test samples and show all class probabilities.

The GUI is English-only and relies solely on the standard library plus the
packages this repository already depends on (torch, torchvision, numpy, Pillow).
"""

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
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# The directory that contains this script is used as the root path.
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "mlp_mnist.pth"

# MNIST normalization constants (mean and standard deviation).
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 128
DEFAULT_LR = 1e-3


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


# 1. Data preprocessing
def build_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
        ]
    )


def load_datasets(data_root=None):
    """Load the MNIST train/test splits, downloading them on first use."""
    root = Path(data_root) if data_root is not None else BASE_DIR / "data"
    transform = build_transform()
    train_ds = datasets.MNIST(
        root=str(root), train=True, download=True, transform=transform
    )
    test_ds = datasets.MNIST(
        root=str(root), train=False, download=True, transform=transform
    )
    return train_ds, test_ds


# 2. MLP model definition
class MLP(nn.Module):
    def __init__(self, in_dim=28 * 28, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# 3. Training and evaluation helpers (used by the background worker thread)
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    sample_count = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        sample_count += x.size(0)

    return total_loss / sample_count


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total


def snapshot_state(model):
    """Copy the weights to CPU so the GUI thread can use them safely."""
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


# 4. Tkinter GUI
class MlpApp:
    """Desktop front-end for the MLP training and inference script."""

    def __init__(self, root):
        self.root = root
        self.root.title("MNIST MLP Trainer")
        self.root.minsize(1000, 660)

        # Background training plumbing.
        self.event_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker = None

        # Datasets are loaded lazily the first time they are needed.
        self.train_ds = None
        self.test_ds = None

        # A CPU copy of the model used for interactive inference.
        self.model = MLP()
        self.current_sample = None
        self.probabilities = None
        self._image_ref = None

        # History backing the chart.
        self.losses = []
        self.accuracies = []

        self.device_var = tk.StringVar(value="auto")
        self.epochs_var = tk.StringVar(value=str(DEFAULT_EPOCHS))
        self.lr_var = tk.StringVar(value=str(DEFAULT_LR))
        self.batch_var = tk.StringVar(value=str(DEFAULT_BATCH_SIZE))
        self.status_var = tk.StringVar(value="Ready.")

        self._build_layout()
        self._draw_chart()
        self._draw_probabilities(None)
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

        self._build_training_panel(left)
        self._build_log_panel(left)
        self._build_chart_panel(right)
        self._build_inference_panel(right)

        status = ttk.Label(
            outer,
            textvariable=self.status_var,
            anchor="w",
            relief="sunken",
            padding=4,
        )
        status.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))


    def _build_training_panel(self, parent):
        frame = ttk.LabelFrame(parent, text="Training", padding=10)
        frame.grid(row=0, column=0, sticky="ew")
        frame.columnconfigure(1, weight=1)

        def add_row(row, label, widget):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=3)
            widget.grid(row=row, column=1, sticky="ew", pady=3)
            return widget

        self.device_combo = add_row(
            0,
            "Device",
            ttk.Combobox(
                frame,
                textvariable=self.device_var,
                values=("auto", "cpu", "cuda"),
                state="readonly",
                width=12,
            ),
        )
        self.epochs_spin = add_row(
            1,
            "Epochs",
            ttk.Spinbox(frame, from_=1, to=200, textvariable=self.epochs_var, width=12),
        )
        self.lr_entry = add_row(
            2, "Learning rate", ttk.Entry(frame, textvariable=self.lr_var, width=12)
        )
        self.batch_spin = add_row(
            3,
            "Batch size",
            ttk.Spinbox(frame, from_=1, to=1024, textvariable=self.batch_var, width=12),
        )

        self.progress = ttk.Progressbar(frame, mode="determinate", maximum=100)
        self.progress.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(8, 4))

        self.metrics = ttk.Label(
            frame,
            text="Epoch: -\nLoss: -\nAccuracy: -\nElapsed: -",
            justify="left",
        )
        self.metrics.grid(row=5, column=0, columnspan=2, sticky="w", pady=(4, 8))

        buttons = ttk.Frame(frame)
        buttons.grid(row=6, column=0, columnspan=2, sticky="ew")
        buttons.columnconfigure(0, weight=1)
        buttons.columnconfigure(1, weight=1)

        self.start_button = ttk.Button(
            buttons, text="Start", command=self.start_training
        )
        self.start_button.grid(row=0, column=0, sticky="ew", padx=2, pady=2)
        self.stop_button = ttk.Button(
            buttons, text="Stop", command=self.stop_training, state="disabled"
        )
        self.stop_button.grid(row=0, column=1, sticky="ew", padx=2, pady=2)

        self.save_button = ttk.Button(
            buttons, text="Save model...", command=self.save_model
        )
        self.save_button.grid(row=1, column=0, sticky="ew", padx=2, pady=2)
        self.load_button = ttk.Button(
            buttons, text="Load model...", command=self.load_model
        )
        self.load_button.grid(row=1, column=1, sticky="ew", padx=2, pady=2)

    def _build_log_panel(self, parent):
        frame = ttk.LabelFrame(parent, text="Log", padding=10)
        frame.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        parent.rowconfigure(1, weight=1)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        self.log_text = tk.Text(
            frame, height=12, width=36, state="disabled", wrap="word"
        )
        self.log_text.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(frame, command=self.log_text.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=scroll.set)


    def _build_chart_panel(self, parent):
        frame = ttk.LabelFrame(parent, text="Training progress", padding=10)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        self.chart_canvas = tk.Canvas(
            frame,
            width=560,
            height=300,
            background="white",
            highlightthickness=1,
            highlightbackground="#c8c8c8",
        )
        self.chart_canvas.grid(row=0, column=0, sticky="nsew")
        self.chart_canvas.bind("<Configure>", lambda event: self._draw_chart())

    def _build_inference_panel(self, parent):
        frame = ttk.LabelFrame(parent, text="Inference", padding=10)
        frame.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        frame.columnconfigure(1, weight=1)

        self.image_canvas = tk.Canvas(
            frame,
            width=168,
            height=168,
            background="#f2f2f2",
            highlightthickness=1,
            highlightbackground="#c8c8c8",
        )
        self.image_canvas.grid(row=0, column=0, rowspan=3, padx=(0, 12))

        self.prediction_var = tk.StringVar(value="No sample selected.")
        ttk.Label(
            frame, textvariable=self.prediction_var, font=("Segoe UI", 11, "bold")
        ).grid(row=0, column=1, sticky="w")

        buttons = ttk.Frame(frame)
        buttons.grid(row=1, column=1, sticky="ew", pady=6)
        self.sample_button = ttk.Button(
            buttons, text="Random test sample", command=self.show_random_sample
        )
        self.sample_button.grid(row=0, column=0, padx=(0, 6))
        self.predict_button = ttk.Button(
            buttons, text="Predict", command=self.predict_current
        )
        self.predict_button.grid(row=0, column=1)

        self.bars_canvas = tk.Canvas(
            frame,
            width=320,
            height=150,
            background="white",
            highlightthickness=1,
            highlightbackground="#c8c8c8",
        )
        self.bars_canvas.grid(row=2, column=1, sticky="ew")
        self.bars_canvas.bind(
            "<Configure>",
            lambda event: self._draw_probabilities(self.probabilities),
        )


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

    def _draw_chart(self):
        canvas = self.chart_canvas
        canvas.delete("all")
        width, height = self._canvas_size(canvas)
        left, right, top, bottom = 50, width - 50, 28, height - 34
        if right <= left or bottom <= top:
            return

        canvas.create_text(
            (left + right) / 2,
            14,
            text="Loss (red, left axis) / Accuracy (blue, right axis)",
            fill="#444444",
        )

        # Horizontal guide lines and their labels on both axes.
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

        # Axes.
        canvas.create_line(left, top, left, bottom, fill="#888888")
        canvas.create_line(left, bottom, right, bottom, fill="#888888")

        if not self.losses:
            canvas.create_text(
                (left + right) / 2,
                (top + bottom) / 2,
                text="Press 'Start' to begin training",
                fill="#aaaaaa",
            )
            return

        count = len(self.losses)
        loss_max = max(self.losses) or 1.0

        def x_at(index):
            if count == 1:
                return (left + right) / 2
            return left + index * (right - left) / (count - 1)

        loss_points = []
        acc_points = []
        for index, (loss, acc) in enumerate(zip(self.losses, self.accuracies)):
            x = x_at(index)
            loss_points.extend((x, bottom - (loss / loss_max) * (bottom - top)))
            acc_points.extend((x, bottom - acc * (bottom - top)))

        if len(loss_points) >= 4:
            canvas.create_line(*loss_points, fill="#d64545", width=2, smooth=True)
        if len(acc_points) >= 4:
            canvas.create_line(*acc_points, fill="#3b6bd6", width=2, smooth=True)

        for index in range(count):
            x = x_at(index)
            loss_y = loss_points[index * 2 + 1]
            acc_y = acc_points[index * 2 + 1]
            canvas.create_oval(
                x - 3, loss_y - 3, x + 3, loss_y + 3, fill="#d64545", outline=""
            )
            canvas.create_oval(
                x - 3, acc_y - 3, x + 3, acc_y + 3, fill="#3b6bd6", outline=""
            )
            canvas.create_text(x, bottom + 12, text=str(index + 1), fill="#888888")

        canvas.create_text(
            right, bottom + 24, text=f"epochs: {count}", anchor="e", fill="#888888"
        )


    def _draw_probabilities(self, probabilities):
        self.probabilities = probabilities
        canvas = self.bars_canvas
        canvas.delete("all")
        width, height = self._canvas_size(canvas)
        left, right, top, bottom = 24, width - 10, 18, height - 24
        if right <= left or bottom <= top:
            return

        canvas.create_text(
            left,
            8,
            text="Class probabilities",
            anchor="w",
            fill="#444444",
        )
        canvas.create_line(left, bottom, right, bottom, fill="#888888")

        slot = (right - left) / 10
        for digit in range(10):
            x0 = left + digit * slot + 4
            x1 = left + (digit + 1) * slot - 4
            value = 0.0 if probabilities is None else float(probabilities[digit])
            bar_height = value * (bottom - top)
            canvas.create_rectangle(
                x0, bottom - bar_height, x1, bottom, fill="#3b6bd6", outline=""
            )
            canvas.create_text(
                (x0 + x1) / 2, bottom + 12, text=str(digit), fill="#555555"
            )
            if probabilities is not None:
                canvas.create_text(
                    (x0 + x1) / 2,
                    bottom - bar_height - 8,
                    text=f"{value:.2f}",
                    fill="#555555",
                )


    def _read_settings(self):
        try:
            epochs = int(self.epochs_var.get())
            learning_rate = float(self.lr_var.get())
            batch_size = int(self.batch_var.get())
        except ValueError:
            messagebox.showerror(
                "Invalid input",
                "Epochs and batch size must be integers and the learning rate "
                "must be a number.",
            )
            return None
        if epochs < 1 or batch_size < 1 or learning_rate <= 0:
            messagebox.showerror(
                "Invalid input",
                "Epochs and batch size must be at least 1 and the learning rate "
                "must be positive.",
            )
            return None
        return epochs, learning_rate, batch_size

    def _ensure_datasets(self):
        """Load MNIST once; returns False if loading failed."""
        if self.train_ds is not None:
            return True
        self._log("Loading the MNIST dataset...")
        self.root.config(cursor="watch")
        self.root.update_idletasks()
        try:
            self.train_ds, self.test_ds = load_datasets()
        except Exception as exc:  # pragma: no cover - depends on the environment
            self.root.config(cursor="")
            messagebox.showerror("Dataset error", f"Could not load MNIST:\n{exc}")
            return False
        self.root.config(cursor="")
        self._log(
            f"Loaded {len(self.train_ds)} train and {len(self.test_ds)} test samples."
        )
        return True

    def start_training(self):
        if self.worker is not None and self.worker.is_alive():
            return
        settings = self._read_settings()
        if settings is None:
            return
        epochs, learning_rate, batch_size = settings

        if not self._ensure_datasets():
            return

        device = resolve_device(self.device_var.get())

        self.losses.clear()
        self.accuracies.clear()
        self.probabilities = None
        self.progress.configure(value=0, maximum=epochs)
        self.metrics.configure(text="Epoch: -\nLoss: -\nAccuracy: -\nElapsed: -")
        self._draw_chart()
        self._draw_probabilities(None)
        self.stop_event.clear()
        self._set_running(True)
        self.status_var.set("Training...")
        self._log(
            f"Training for {epochs} epoch(s) on {device_label(device)} "
            f"(lr={learning_rate:g}, batch={batch_size})."
        )

        self.worker = threading.Thread(
            target=self._training_worker,
            args=(device, epochs, learning_rate, batch_size),
            daemon=True,
        )
        self.worker.start()

    def stop_training(self):
        if self.worker is not None and self.worker.is_alive():
            self.stop_event.set()
            self.stop_button.configure(state="disabled")
            self._log("Stop requested; finishing the current epoch...")

    def _training_worker(self, device, epochs, learning_rate, batch_size):
        try:
            train_loader = DataLoader(self.train_ds, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(self.test_ds, batch_size=256, shuffle=False)
            model = MLP().to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            started = time.time()
            stopped = False
            for epoch in range(epochs):
                if self.stop_event.is_set():
                    stopped = True
                    break

                loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
                acc = evaluate(model, test_loader, device)
                self.event_queue.put(
                    (
                        "epoch",
                        {
                            "epoch": epoch + 1,
                            "total": epochs,
                            "loss": loss,
                            "acc": acc,
                            "elapsed": time.time() - started,
                            "state": snapshot_state(model),
                        },
                    )
                )

            final_state = snapshot_state(model)
            torch.save(model.state_dict(), str(MODEL_PATH))
            self.event_queue.put(
                ("done", {"stopped": stopped, "state": final_state})
            )
            self.event_queue.put(("log", f"Model saved to {MODEL_PATH}"))
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
        if kind == "epoch":
            self.losses.append(payload["loss"])
            self.accuracies.append(payload["acc"])
            self.progress.configure(value=payload["epoch"])
            self.metrics.configure(
                text=(
                    f"Epoch: {payload['epoch']} / {payload['total']}\n"
                    f"Loss: {payload['loss']:.4f}\n"
                    f"Accuracy: {payload['acc'] * 100:.2f}%\n"
                    f"Elapsed: {payload['elapsed']:.1f}s"
                )
            )
            self.model.load_state_dict(payload["state"])
            self._draw_chart()
            self._log(
                f"Epoch {payload['epoch']}/{payload['total']}: "
                f"loss={payload['loss']:.4f}, acc={payload['acc']:.4f}"
            )
        elif kind == "done":
            self.model.load_state_dict(payload["state"])
            self._set_running(False)
            if payload["stopped"]:
                self.status_var.set("Training stopped.")
                self._log("Training stopped by the user.")
            else:
                self.status_var.set("Training finished.")
                self._log("Training finished.")
        elif kind == "log":
            self._log(payload)
        elif kind == "error":
            self._set_running(False)
            self.status_var.set("Training failed.")
            self._log(f"Error: {payload}")
            messagebox.showerror("Training error", payload)

    def _set_running(self, running):
        state = "disabled" if running else "normal"
        self.start_button.configure(state=state)
        self.epochs_spin.configure(state=state)
        self.lr_entry.configure(state=state)
        self.batch_spin.configure(state=state)
        self.load_button.configure(state=state)
        self.device_combo.configure(state="disabled" if running else "readonly")
        self.stop_button.configure(state="normal" if running else "disabled")

    def _log(self, message):
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"[{timestamp}] {message}\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")


    def show_random_sample(self):
        if not self._ensure_datasets():
            return
        index = int(torch.randint(len(self.test_ds), (1,)).item())
        image, label = self.test_ds[index]
        label = int(label)
        self.current_sample = (image, label)
        self._image_ref = self._tensor_to_photoimage(image)
        self.image_canvas.delete("all")
        self.image_canvas.create_image(84, 84, image=self._image_ref)
        self.prediction_var.set(f"True label: {label}")
        self._draw_probabilities(None)
        self._log(f"Selected a random test sample with true label {label}.")

    def predict_current(self):
        if self.current_sample is None:
            messagebox.showinfo("No sample", "Pick a random test sample first.")
            return
        image, label = self.current_sample
        self.model.eval()
        with torch.no_grad():
            logits = self.model(image.unsqueeze(0))
            probabilities = torch.softmax(logits, dim=1).squeeze(0)
        predicted = int(probabilities.argmax().item())
        confidence = float(probabilities[predicted].item())
        self.prediction_var.set(
            f"True: {label}    Predicted: {predicted} ({confidence * 100:.1f}%)"
        )
        self._draw_probabilities(probabilities.tolist())
        self._log(
            f"Inference: true={label}, predicted={predicted}, "
            f"confidence={confidence:.4f}"
        )

    def _tensor_to_photoimage(self, tensor, size=168):
        """Denormalize a CHW tensor into a grayscale Tk image."""
        array = (tensor.squeeze() * MNIST_STD + MNIST_MEAN).clamp(0.0, 1.0).numpy()
        array = (array * 255.0).astype(np.uint8)
        image = Image.fromarray(array, mode="L").resize((size, size), Image.NEAREST)
        return ImageTk.PhotoImage(image)

    def save_model(self):
        path = filedialog.asksaveasfilename(
            title="Save model",
            initialdir=str(BASE_DIR),
            initialfile=MODEL_PATH.name,
            defaultextension=".pth",
            filetypes=[("PyTorch model", "*.pth"), ("All files", "*.*")],
        )
        if not path:
            return
        torch.save(self.model.state_dict(), path)
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
            state = torch.load(path, map_location="cpu")
        except Exception as exc:
            messagebox.showerror("Load error", f"Could not load the model:\n{exc}")
            return
        try:
            self.model.load_state_dict(state)
        except Exception as exc:
            messagebox.showerror(
                "Load error", f"The file does not match the MLP layout:\n{exc}"
            )
            return
        self._log(f"Loaded model from {path}")
        self.status_var.set(f"Loaded model from {path}")

    def _on_close(self):
        self.stop_event.set()
        self.root.destroy()


def main():
    root = tk.Tk()
    try:
        ttk.Style().theme_use("clam")
    except tk.TclError:
        pass
    MlpApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()









