import os
import shutil
import gradio as gr
import numpy as np
import cv2
from ultralytics import SAM
from ultralytics.models.sam import SAM3SemanticPredictor

SAM3_PT = "sam3.pt"

# Auto-download sam3.pt from ModelScope if not present
if not os.path.exists(SAM3_PT):
    print("sam3.pt not found — downloading from ModelScope (facebook/sam3)...")
    from modelscope import snapshot_download
    model_dir = snapshot_download("facebook/sam3")
    src = os.path.join(model_dir, "sam3.pt")
    shutil.copy2(src, SAM3_PT)
    print(f"Copied sam3.pt to {os.path.abspath(SAM3_PT)}")

# Point/box model (SAM2-compatible interface)
model = SAM(SAM3_PT)

# Text-based concept segmentation predictor
semantic_predictor = SAM3SemanticPredictor(
    overrides=dict(conf=0.25, task="segment", mode="predict", model=SAM3_PT, half=True)
)

points_state = []
labels_state = []
original_image = None


def load_image(image):
    global points_state, labels_state, original_image
    points_state = []
    labels_state = []
    original_image = image.copy()
    return image, "Image loaded. Use text prompts or click to segment."


def draw_points(img):
    for (x, y), label in zip(points_state, labels_state):
        color = (0, 255, 0) if label == 1 else (255, 0, 0)
        cv2.circle(img, (x, y), 8, color, -1)
        cv2.circle(img, (x, y), 8, (255, 255, 255), 2)
    return img


def overlay_masks(base_img, results, multi_color=False):
    """Draw segmentation masks on the image."""
    overlay = base_img.copy()
    n_masks = 0
    if results and results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        n_masks = len(masks)
        if multi_color:
            np.random.seed(42)
            for mask in masks:
                bool_mask = mask.astype(bool)
                color = np.random.randint(50, 255, size=3).tolist()
                color_layer = np.zeros_like(overlay)
                color_layer[bool_mask] = color
                overlay = cv2.addWeighted(overlay, 1.0, color_layer, 0.3, 0)
                contour_mask = bool_mask.astype(np.uint8) * 255
                contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, color, 2)
        else:
            combined = np.zeros(overlay.shape[:2], dtype=bool)
            for mask in masks:
                combined |= mask.astype(bool)
            color_mask = np.zeros_like(overlay)
            color_mask[combined] = [0, 120, 255]
            overlay = cv2.addWeighted(overlay, 0.7, color_mask, 0.3, 0)
            contour_mask = combined.astype(np.uint8) * 255
            contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)
    return overlay, n_masks


def overlay_semantic_masks(base_img, masks_list, labels_list):
    """Draw semantic masks with per-concept colors and labels."""
    overlay = base_img.copy()
    n_masks = 0
    np.random.seed(42)
    for masks, label in zip(masks_list, labels_list):
        color = np.random.randint(50, 255, size=3).tolist()
        for mask in masks:
            n_masks += 1
            bool_mask = mask.astype(bool)
            color_layer = np.zeros_like(overlay)
            color_layer[bool_mask] = color
            overlay = cv2.addWeighted(overlay, 1.0, color_layer, 0.3, 0)
            contour_mask = bool_mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, color, 2)
            # Label the mask
            if contours:
                x, y, w, h = cv2.boundingRect(contours[0])
                cv2.putText(overlay, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return overlay, n_masks


# --- Text-based segmentation (SAM3 concept segmentation) ---

def run_text_segmentation(text_prompt):
    global original_image
    if original_image is None:
        return None, "Upload an image first."
    if not text_prompt.strip():
        return original_image.copy(), "Enter concept(s) separated by commas."

    concepts = [c.strip() for c in text_prompt.split(",") if c.strip()]

    # Save image to temp file for SAM3SemanticPredictor
    import tempfile, os
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp_path = tmp.name
    tmp.close()
    # Convert RGB (from gradio) to BGR for cv2
    cv2.imwrite(tmp_path, cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))

    try:
        semantic_predictor.set_image(tmp_path)
        results = semantic_predictor(text=concepts)

        if results and len(results) > 0 and results[0].masks is not None:
            overlay, n_masks = overlay_masks(original_image, results, multi_color=True)
            return overlay, f"Found {n_masks} mask(s) for: {', '.join(concepts)}"
        else:
            return original_image.copy(), f"No masks found for: {', '.join(concepts)}"
    finally:
        os.unlink(tmp_path)


# --- Point-based segmentation ---

def add_point(image, mode, evt: gr.SelectData):
    global points_state, labels_state
    if original_image is None:
        return image, "Upload an image first."

    x, y = evt.index
    label = 1 if mode == "Include (green)" else 0
    points_state.append([x, y])
    labels_state.append(label)

    annotated = draw_points(original_image.copy())
    label_name = "include" if label == 1 else "exclude"
    return annotated, f"Added {label_name} point ({x}, {y}). Total: {len(points_state)} points."


def run_point_segmentation():
    global original_image
    if original_image is None:
        return None, "Upload an image first."
    if len(points_state) == 0:
        return draw_points(original_image.copy()), "Add at least one point first."

    results = model(original_image, points=points_state, labels=labels_state)
    overlay, n_masks = overlay_masks(original_image, results)
    overlay = draw_points(overlay)
    return overlay, f"Done! Found {n_masks} mask(s)."


# --- Box segmentation ---

def run_box_segmentation(x1, y1, x2, y2):
    global original_image
    if original_image is None:
        return None, "Upload an image first."

    results = model(original_image, bboxes=[[int(x1), int(y1), int(x2), int(y2)]])
    overlay, n_masks = overlay_masks(original_image, results)
    cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)
    return overlay, f"Box segmentation done! Found {n_masks} mask(s)."


# --- Auto segmentation ---

def run_auto_segmentation():
    global original_image
    if original_image is None:
        return None, "Upload an image first."

    results = model(original_image)
    overlay, n_masks = overlay_masks(original_image, results, multi_color=True)
    return overlay, f"Auto-segmentation done! Found {n_masks} mask(s)."


def clear_points():
    global points_state, labels_state
    points_state = []
    labels_state = []
    if original_image is not None:
        return original_image.copy(), "Points cleared."
    return None, "Points cleared."


def undo_last_point():
    global points_state, labels_state
    if len(points_state) == 0:
        return original_image if original_image is not None else None, "No points to undo."
    points_state.pop()
    labels_state.pop()
    img = draw_points(original_image.copy()) if original_image is not None else None
    return img, f"Undone. {len(points_state)} points remaining."


# --- UI ---

with gr.Blocks(title="SAM 3 Segmentation", theme=gr.themes.Soft()) as app:
    gr.Markdown("# SAM 3 Segmentation Tool")
    gr.Markdown(
        "**Text prompt**: describe what you want segmented (e.g. `person, car, dog`). "
        "**Click**: point at specific objects. **Box**: enter coordinates."
    )

    with gr.Row():
        with gr.Column(scale=4):
            image_display = gr.Image(label="Image", type="numpy", interactive=True, height=600)
        with gr.Column(scale=1):
            gr.Markdown("### Text Prompt (SAM3)")
            text_input = gr.Textbox(
                label="Concepts",
                placeholder="person, car, glasses ...",
                lines=2,
            )
            text_btn = gr.Button("Segment by Text", variant="primary", size="lg")

            gr.Markdown("---")
            gr.Markdown("### Point Prompt")
            mode = gr.Radio(
                ["Include (green)", "Exclude (red)"],
                value="Include (green)",
                label="Click mode",
            )
            segment_btn = gr.Button("Segment Points", variant="primary")
            undo_btn = gr.Button("Undo Last Point")
            clear_btn = gr.Button("Clear Points", variant="stop")

            gr.Markdown("---")
            gr.Markdown("### Box Prompt")
            with gr.Row():
                bx1 = gr.Number(label="x1", precision=0)
                by1 = gr.Number(label="y1", precision=0)
            with gr.Row():
                bx2 = gr.Number(label="x2", precision=0)
                by2 = gr.Number(label="y2", precision=0)
            box_btn = gr.Button("Segment Box")

            gr.Markdown("---")
            auto_btn = gr.Button("Auto-Segment Everything")

    status = gr.Textbox(label="Status", interactive=False, value="Upload an image to start.")

    # Events
    image_display.upload(load_image, inputs=[image_display], outputs=[image_display, status])
    image_display.select(add_point, inputs=[image_display, mode], outputs=[image_display, status])
    text_btn.click(run_text_segmentation, inputs=[text_input], outputs=[image_display, status])
    segment_btn.click(run_point_segmentation, outputs=[image_display, status])
    undo_btn.click(undo_last_point, outputs=[image_display, status])
    clear_btn.click(clear_points, outputs=[image_display, status])
    box_btn.click(run_box_segmentation, inputs=[bx1, by1, bx2, by2], outputs=[image_display, status])
    auto_btn.click(run_auto_segmentation, outputs=[image_display, status])

if __name__ == "__main__":
    app.launch()
