"""
Visualize intermediate feature maps of each layer in the YOLO model.

Usage:
    python visualize_layers.py image.jpg
    python visualize_layers.py image.jpg --model runs/pose/dart_pose6/weights/best.pt

Controls:
    Left/Right arrows  - Navigate layers
    Up/Down arrows     - Navigate channels within a layer
    G                  - Toggle grid view (all channels) vs single channel
    O                  - Open new image
    Q / Esc            - Quit
"""

import os
import cv2
import torch
import numpy as np
from tkinter import Tk, filedialog
from ultralytics import YOLO

DEFAULT_MODEL = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "runs", "pose", "dart_pose6", "weights", "best.pt"
)

WINDOW = "Layer Visualizer"


class LayerVisualizer:
    def __init__(self, model_path):
        self.yolo = YOLO(model_path)
        self.model = self.yolo.model.model
        self.activations = {}
        self.layer_names = []
        self.current_layer = 0
        self.current_channel = 0
        self.grid_mode = True
        self.input_img = None
        self.display_img = None

        # Register hooks on every layer
        for i, layer in enumerate(self.model):
            name = f"{i:02d}_{layer.__class__.__name__}"
            self.layer_names.append(name)
            layer.register_forward_hook(self._make_hook(name))

    def _make_hook(self, name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                self.activations[name] = output.detach().cpu()
        return hook

    def load_image(self, path):
        self.input_img = cv2.imread(path)
        if self.input_img is None:
            return
        # Run inference to populate activations
        self.yolo(self.input_img, verbose=False)
        self.current_layer = 0
        self.current_channel = 0
        self.update_display()

    def open_file(self):
        root = Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp")]
        )
        root.destroy()
        if path:
            self.load_image(path)

    def _normalize_map(self, feat):
        """Normalize a 2D feature map to 0-255 for display."""
        feat = feat.numpy().astype(np.float32)
        fmin, fmax = feat.min(), feat.max()
        if fmax - fmin > 1e-6:
            feat = (feat - fmin) / (fmax - fmin)
        else:
            feat = np.zeros_like(feat)
        return (feat * 255).astype(np.uint8)

    def _colorize(self, gray):
        """Apply colormap to grayscale feature map."""
        return cv2.applyColorMap(gray, cv2.COLORMAP_VIRIDIS)

    def update_display(self):
        name = self.layer_names[self.current_layer]
        act = self.activations.get(name)

        if act is None or act.dim() < 3:
            # Non-tensor or low-dim output
            img = np.zeros((400, 700, 3), dtype=np.uint8)
            cv2.putText(img, f"Layer {name}: no spatial feature map",
                        (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            self.display_img = img
            return

        # act shape: [batch, channels, H, W]
        feat = act[0]  # first batch item
        num_channels = feat.shape[0]
        h, w = feat.shape[1], feat.shape[2]

        # Clamp channel index
        self.current_channel = max(0, min(self.current_channel, num_channels - 1))

        if self.grid_mode and num_channels > 1:
            img = self._make_grid(feat, num_channels)
        else:
            gray = self._normalize_map(feat[self.current_channel])
            img = self._colorize(gray)
            # Scale up for visibility
            scale = max(1, min(800 // max(h, 1), 800 // max(w, 1)))
            if scale > 1:
                img = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)

        # Draw HUD
        disp_h, disp_w = img.shape[:2]
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (disp_w, 40), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

        layer_text = f"Layer {self.current_layer}/{len(self.layer_names)-1}: {name}"
        shape_text = f"Shape: {num_channels}x{h}x{w}"
        mode_text = "GRID" if self.grid_mode else f"Ch {self.current_channel}/{num_channels-1}"
        help_text = "</>:Layer  ^/v:Channel  G:Grid  O:Open  Q:Quit"

        cv2.putText(img, layer_text, (8, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(img, f"{shape_text}  [{mode_text}]", (8, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.putText(img, help_text, (disp_w - 420, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

        self.display_img = img

    def _make_grid(self, feat, num_channels):
        """Create a grid of all channel feature maps."""
        # Limit to first 64 channels for readability
        show = min(num_channels, 64)
        cols = int(np.ceil(np.sqrt(show)))
        rows = int(np.ceil(show / cols))

        h, w = feat.shape[1], feat.shape[2]
        # Target cell size for visibility
        cell = max(48, min(128, 800 // cols))
        grid = np.zeros((rows * cell, cols * cell, 3), dtype=np.uint8)

        for idx in range(show):
            r, c = divmod(idx, cols)
            gray = self._normalize_map(feat[idx])
            colored = self._colorize(gray)
            resized = cv2.resize(colored, (cell, cell), interpolation=cv2.INTER_NEAREST)
            grid[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = resized

        if num_channels > 64:
            cv2.putText(grid, f"Showing 64/{num_channels} channels",
                        (5, grid.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        return grid

    def run(self, initial_image=None):
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)

        if initial_image:
            self.load_image(initial_image)
        else:
            self.display_img = np.zeros((400, 700, 3), dtype=np.uint8)
            cv2.putText(self.display_img, "Press 'O' to open an image",
                        (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

        cv2.imshow(WINDOW, self.display_img)

        while True:
            key = cv2.waitKey(50) & 0xFF

            if key == ord('q') or key == 27:
                break

            elif key == ord('o'):
                self.open_file()

            elif key == 83 or key == ord('d'):  # Right
                if self.layer_names:
                    self.current_layer = (self.current_layer + 1) % len(self.layer_names)
                    self.current_channel = 0
                    self.update_display()

            elif key == 81 or key == ord('a'):  # Left
                if self.layer_names:
                    self.current_layer = (self.current_layer - 1) % len(self.layer_names)
                    self.current_channel = 0
                    self.update_display()

            elif key == 82 or key == ord('w'):  # Up
                self.current_channel += 1
                self.update_display()

            elif key == 84 or key == ord('s'):  # Down
                self.current_channel = max(0, self.current_channel - 1)
                self.update_display()

            elif key == ord('g'):
                self.grid_mode = not self.grid_mode
                self.update_display()

            if self.display_img is not None:
                cv2.imshow(WINDOW, self.display_img)

        cv2.destroyAllWindows()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="YOLO layer feature map visualizer")
    parser.add_argument("image", nargs="?", default=None, help="Input image")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to model weights")
    args = parser.parse_args()

    viz = LayerVisualizer(args.model)
    viz.run(args.image)


if __name__ == "__main__":
    main()
