import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import csv
from datetime import datetime
import os
import json
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import threading
import time
# import RPi.GPIO as GPIO  # Uncomment for real Raspberry Pi

class VisionHMI:
    def __init__(self, root):
        self.root = root
        self.root.title("VisionMaster HMI")
        self.root.geometry("1280x720")
        self.root.minsize(1024, 600)
        self.root.configure(bg="#e6e6e6")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Initialize camera
        self.cap = None
        self.use_static_image = False
        self.static_image = None
        self.init_camera()

        # Modes
        self.mode = tk.StringVar(value="Mode Réglage")
        self.mode.trace("w", self.update_mode)

        # ROI management
        self.rois = []  # (x, y, w, h, id, angle, shape, mask)
        self.selected_roi = None
        self.drawing = False
        self.resizing = False
        self.moving = False
        self.rotating = False
        self.drawing_mask = False
        self.ix, self.iy = -1, -1
        self.roi_id = 0
        self.hovered_roi = None
        self.snap_to_grid = tk.BooleanVar(value=False)
        self.roi_shape = tk.StringVar(value="rectangle")

        # Inspection parameters with min/max
        self.params = {
            "density_threshold_min": tk.DoubleVar(value=90.0),
            "density_threshold_max": tk.DoubleVar(value=110.0),
            "contrast_threshold_min": tk.DoubleVar(value=10.0),
            "contrast_threshold_max": tk.DoubleVar(value=30.0),
            "edge_threshold_min": tk.DoubleVar(value=50.0),
            "edge_threshold_max": tk.DoubleVar(value=150.0),
            "edge_canny_low": tk.DoubleVar(value=50.0),
            "edge_canny_high": tk.DoubleVar(value=150.0),
            "edge_sobel_kernel": tk.IntVar(value=3),
            "edge_median_blur": tk.IntVar(value=5),
            "template_threshold_min": tk.DoubleVar(value=0.7),
            "template_threshold_max": tk.DoubleVar(value=0.9),
            "contour_area_threshold_min": tk.DoubleVar(value=400.0),
            "contour_area_threshold_max": tk.DoubleVar(value=600.0),
            "contour_perimeter_min": tk.DoubleVar(value=50.0),
            "contour_perimeter_max": tk.DoubleVar(value=500.0),
            "contour_circularity_min": tk.DoubleVar(value=0.5),
            "contour_circularity_max": tk.DoubleVar(value=1.0),
            "contour_gaussian_blur": tk.IntVar(value=5),
            "contour_morph_kernel": tk.IntVar(value=3),
            "contour_hierarchy_mode": tk.StringVar(value="External"),
            # Blob Detection Parameters
            "blob_threshold_manual": tk.BooleanVar(value=True),
            "blob_threshold_value": tk.DoubleVar(value=128.0),
            "blob_area_min": tk.DoubleVar(value=50.0),
            "blob_area_max": tk.DoubleVar(value=200.0),
            "blob_width_min": tk.DoubleVar(value=10.0),
            "blob_width_max": tk.DoubleVar(value=100.0),
            "blob_height_min": tk.DoubleVar(value=10.0),
            "blob_height_max": tk.DoubleVar(value=100.0),
            "blob_circularity_min": tk.DoubleVar(value=0.8),
            "blob_circularity_max": tk.DoubleVar(value=1.0),
            "blob_aspect_ratio_min": tk.DoubleVar(value=0.5),
            "blob_aspect_ratio_max": tk.DoubleVar(value=2.0),
            "blob_solidity_min": tk.DoubleVar(value=0.8),
            "blob_solidity_max": tk.DoubleVar(value=1.0),
            "blob_bounding_shape": tk.StringVar(value="None"),  # Options: None, Rectangle, Circle
            "blob_color_mode": tk.StringVar(value="Grayscale"),  # Grayscale, RGB, HSV
            "blob_rgb_r_min": tk.DoubleVar(value=0.0),
            "blob_rgb_r_max": tk.DoubleVar(value=255.0),
            "blob_rgb_g_min": tk.DoubleVar(value=0.0),
            "blob_rgb_g_max": tk.DoubleVar(value=255.0),
            "blob_rgb_b_min": tk.DoubleVar(value=0.0),
            "blob_rgb_b_max": tk.DoubleVar(value=255.0),
            "blob_hsv_h_min": tk.DoubleVar(value=0.0),
            "blob_hsv_h_max": tk.DoubleVar(value=180.0),
            "blob_hsv_s_min": tk.DoubleVar(value=0.0),
            "blob_hsv_s_max": tk.DoubleVar(value=255.0),
            "blob_hsv_v_min": tk.DoubleVar(value=0.0),
            "blob_hsv_v_max": tk.DoubleVar(value=255.0),
            "blob_bilateral_sigma": tk.DoubleVar(value=10.0),
            "blob_count_min": tk.DoubleVar(value=1.0),
            "blob_count_max": tk.DoubleVar(value=6.0),
            "boundary_exclusion": tk.BooleanVar(value=True),
            # Measurement Parameters
            "measurement_tolerance_min": tk.DoubleVar(value=0.1),
            "measurement_tolerance_max": tk.DoubleVar(value=0.3),
            "focus_threshold_min": tk.DoubleVar(value=80.0),
            "focus_threshold_max": tk.DoubleVar(value=120.0),
            "gpio_trigger_pin": tk.IntVar(value=-1),
            "color_ratio_min": tk.DoubleVar(value=0.0),
            "color_ratio_max": tk.DoubleVar(value=100.0),
            "color_hue_min": tk.DoubleVar(value=0.0),
            "color_hue_max": tk.DoubleVar(value=180.0),
            "color_saturation_min": tk.DoubleVar(value=0.0),
            "color_saturation_max": tk.DoubleVar(value=255.0),
            "color_brightness_min": tk.DoubleVar(value=0.0),
            "color_brightness_max": tk.DoubleVar(value=255.0)
        }

        # Blob Output Parameters
        self.blob_outputs = {
            "count": tk.BooleanVar(value=True),
            "largest_area": tk.BooleanVar(value=False),
            "smallest_area": tk.BooleanVar(value=False),
            "center_of_gravity": tk.BooleanVar(value=False),
            "positions": tk.BooleanVar(value=False),
            "orientation": tk.BooleanVar(value=False),
            "total_area": tk.BooleanVar(value=False),
            "fill_percentage": tk.BooleanVar(value=False)
        }

        # Judgment Criteria
        self.judgment_criteria = {
            "blob_count_min": tk.DoubleVar(value=1.0),
            "blob_count_max": tk.DoubleVar(value=6.0),
            "blob_area_min": tk.DoubleVar(value=50.0),
            "blob_area_max": tk.DoubleVar(value=200.0),
            "criteria_type": tk.StringVar(value="At least one blob")  # Options: At least one blob, Blob count limit
        }

        self.template_image = None
        self.temp_selected_pin = tk.IntVar(value=-1)
        self.color_picking = False
        self.test_images = []
        self.test_image_index = tk.IntVar(value=-1)

        # Cycle logic
        self.cycle_state = "Idle"
        self.cycle_results = {}
        self.cycle_features = {
            "Density": tk.BooleanVar(value=True),
            "Contrast": tk.BooleanVar(value=False),
            "Edge": tk.BooleanVar(value=True),
            "Template Matching": tk.BooleanVar(value=False),
            "Contour Analysis": tk.BooleanVar(value=False),
            "Blob Detection": tk.BooleanVar(value=True),
            "Color Detection": tk.BooleanVar(value=False),
            "Measurement": tk.BooleanVar(value=False),
            "Focus Check": tk.BooleanVar(value=False)
        }

        # GPIO simulation
        self.gpio_trigger_active = False
        self.gpio_thread = None

        # Logging
        self.log_file = "inspection_log.csv"
        self.init_log()

        # Setup GUI
        self.setup_gui()
        self.load_settings()

        # Start video feed
        self.update_video()

        # Start GPIO simulation (if pin set)
        self.start_gpio_simulation()

    def init_camera(self):
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("Camera unavailable")
        except Exception:
            messagebox.showinfo("Info", "Camera unavailable. Using static image.")
            self.use_static_image = True
            self.static_image = cv2.imread("sample_image.jpg")
            if self.static_image is None:
                messagebox.showerror("Error", "No sample image found.")
                self.root.quit()

    def setup_gui(self):
        style = ttk.Style()
        style.theme_create("modern", parent="clam", settings={
            "TButton": {
                "configure": {
                    "padding": 8,
                    "font": ("Segoe UI", 11),
                    "background": "#0078d7",
                    "foreground": "#ffffff",
                    "bordercolor": "#005a9e",
                    "borderwidth": 1
                },
                "map": {
                    "background": [("active", "#005a9e"), ("disabled", "#b3d7ff")],
                    "foreground": [("active", "#ffffff")]
                }
            },
            "TLabel": {
                "configure": {
                    "background": "#e6e6e6",
                    "foreground": "#212121",
                    "font": ("Segoe UI", 11)
                }
            },
            "TFrame": {
                "configure": {"background": "#e6e6e6"}
            },
            "TNotebook": {
                "configure": {
                    "tabmargins": [5, 5, 5, 0],
                    "background": "#f5f5f5",
                    "foreground": "#212121"
                }
            },
            "TNotebook.Tab": {
                "configure": {
                    "padding": [15, 8],
                    "font": ("Segoe UI", 11, "bold"),
                    "background": "#f5f5f5",
                    "foreground": "#212121"
                },
                "map": {
                    "background": [("selected", "#ffffff"), ("active", "#e0e0e0")],
                    "foreground": [("selected", "#0078d7")]
                }
            },
            "TProgressbar": {
                "configure": {
                    "background": "#28a745",
                    "troughcolor": "#d4d4d4"
                }
            },
            "TEntry": {
                "configure": {
                    "fieldbackground": "#ffffff",
                    "foreground": "#212121",
                    "font": ("Segoe UI", 11)
                }
            },
            "TCombobox": {
                "configure": {
                    "fieldbackground": "#ffffff",
                    "foreground": "#212121",
                    "font": ("Segoe UI", 11)
                }
            }
        })
        style.theme_use("modern")

        # Custom title bar
        self.title_bar = tk.Frame(self.root, bg="#0078d7", relief="raised", bd=0)
        self.title_bar.pack(fill=tk.X)
        tk.Label(self.title_bar, text="VisionMaster HMI", bg="#0078d7", fg="#ffffff", font=("Segoe UI", 13, "bold")).pack(side=tk.LEFT, padx=10)
        self.mode_selector = ttk.Combobox(self.title_bar, textvariable=self.mode, values=["Mode Réglage", "Run Mode"], state="readonly", width=15)
        self.mode_selector.pack(side=tk.LEFT, padx=5)
        tk.Button(self.title_bar, text="🗕", bg="#0078d7", fg="#ffffff", bd=0, command=self.root.iconify).pack(side=tk.RIGHT)
        tk.Button(self.title_bar, text="🗖", bg="#0078d7", fg="#ffffff", bd=0, command=self.toggle_maximize).pack(side=tk.RIGHT)
        tk.Button(self.title_bar, text="🗙", bg="#0078d7", fg="#ffffff", bd=0, command=self.on_closing).pack(side=tk.RIGHT)
        self.title_bar.bind("<B1-Motion>", self.move_window)
        self.title_bar.bind("<Button-1>", self.get_pos)

        # Menu bar
        self.menubar = tk.Menu(self.root, bg="#f5f5f5", fg="#212121", font=("Segoe UI", 10))
        file_menu = tk.Menu(self.menubar, tearoff=0, bg="#f5f5f5", fg="#212121")
        file_menu.add_command(label="Save Config", command=self.save_cycle_config)
        file_menu.add_command(label="Load Config", command=self.load_cycle_config)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing)
        self.menubar.add_cascade(label="File", menu=file_menu)
        view_menu = tk.Menu(self.menubar, tearoff=0, bg="#f5f5f5", fg="#212121")
        view_menu.add_command(label="Toggle Results Pane", command=self.toggle_results)
        self.menubar.add_cascade(label="View", menu=view_menu)
        help_menu = tk.Menu(self.menubar, tearoff=0, bg="#f5f5f5", fg="#212121")
        help_menu.add_command(label="About", command=lambda: messagebox.showinfo("About", "VisionMaster HMI v2.0\n© 2025"))
        self.menubar.add_cascade(label="Help", menu=help_menu)
        self.root.config(menu=self.menubar)

        # Main layout
        self.main_frame = ttk.Frame(self.root, padding=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Status bar
        self.status_frame = ttk.Frame(self.main_frame, relief=tk.SUNKEN, borderwidth=1)
        self.status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_var = tk.StringVar(value="Ready | Mode Réglage")
        ttk.Label(self.status_frame, textvariable=self.status_var).pack(side=tk.LEFT, padx=5)
        self.time_var = tk.StringVar()
        ttk.Label(self.status_frame, textvariable=self.time_var).pack(side=tk.RIGHT, padx=5)
        self.led_ok = tk.Canvas(self.status_frame, width=16, height=16, bg="#e6e6e6", highlightthickness=0)
        self.led_ok.create_oval(2, 2, 14, 14, fill="#d4d4d4", tags="led_ok")
        self.led_ok.pack(side=tk.RIGHT, padx=5)
        self.led_ng = tk.Canvas(self.status_frame, width=16, height=16, bg="#e6e6e6", highlightthickness=0)
        self.led_ng.create_oval(2, 2, 14, 14, fill="#d4d4d4", tags="led_ng")
        self.led_ng.pack(side=tk.RIGHT, padx=5)
        self.update_time()

        # Paned window for resizable panels
        self.paned_window = ttk.PanedWindow(self.main_frame, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True)

        # Left panel: Video and results
        self.left_panel = ttk.Frame(self.paned_window)
        self.paned_window.add(self.left_panel, weight=1)

        self.canvas = tk.Canvas(self.left_panel, width=640, height=480, bg="#000000", highlightthickness=1, highlightbackground="#d4d4d4")
        self.canvas.pack(padx=5, pady=5)
        self.canvas.bind("<Button-1>", self.start_roi_or_pick_color)
        self.canvas.bind("<B1-Motion>", self.draw_roi_or_mask)
        self.canvas.bind("<ButtonRelease-1>", self.end_roi_or_mask)
        self.canvas.bind("<Button-3>", self.select_roi)
        self.canvas.bind("<B3-Motion>", self.move_resize_rotate_roi)
        self.canvas.bind("<ButtonRelease-3>", self.end_move_resize_rotate)
        self.canvas.bind("<Motion>", self.update_cursor)

        self.roi_info_frame = ttk.Frame(self.left_panel)
        self.roi_info_frame.pack(fill=tk.X, padx=5, pady=5)
        self.roi_info_var = tk.StringVar(value="ROI Info: None")
        ttk.Label(self.roi_info_frame, textvariable=self.roi_info_var).pack(side=tk.LEFT)
        ttk.Checkbutton(self.roi_info_frame, text="Snap to Grid", variable=self.snap_to_grid).pack(side=tk.RIGHT)

        self.result_frame = ttk.Frame(self.left_panel, relief=tk.SUNKEN, borderwidth=1)
        self.result_frame.pack(fill=tk.X, padx=5, pady=5)
        self.result_text = tk.Text(self.result_frame, height=6, width=50, font=("Segoe UI", 10), bg="#ffffff", fg="#212121")
        self.result_text.pack(pady=5)
        self.result_text.tag_config("green", foreground="#28a745")
        self.result_text.tag_config("red", foreground="#dc3545")
        self.result_button_frame = ttk.Frame(self.result_frame)
        self.result_button_frame.pack(fill=tk.X)
        ttk.Button(self.result_button_frame, text="📸 Save Image", command=self.save_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.result_button_frame, text="📄 Generate PDF", command=self.generate_pdf_report).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.result_button_frame, text="📜 View Log", command=self.view_log).pack(side=tk.LEFT, padx=5)

        # Right panel: Tabs
        self.right_panel = ttk.Frame(self.paned_window)
        self.paned_window.add(self.right_panel, weight=1)
        self.notebook = ttk.Notebook(self.right_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # ROI Tab
        self.roi_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.roi_frame, text="📍 ROI")
        ttk.Label(self.roi_frame, text="ROI Shape").pack(pady=5)
        ttk.Combobox(self.roi_frame, textvariable=self.roi_shape, values=["rectangle", "circle"], state="readonly", width=15).pack(pady=5)
        ttk.Button(self.roi_frame, text="➕ Add ROI", command=self.reset_roi).pack(pady=5)
        ttk.Button(self.roi_frame, text="🎨 Draw Mask", command=self.start_mask_drawing).pack(pady=5)
        ttk.Button(self.roi_frame, text="🗑️ Clear Mask", command=self.clear_mask).pack(pady=5)
        ttk.Button(self.roi_frame, text="🗑️ Clear All ROIs", command=self.clear_rois).pack(pady=5)
        self.roi_listbox = tk.Listbox(self.roi_frame, height=5, font=("Segoe UI", 10), bg="#ffffff", fg="#212121")
        self.roi_listbox.pack(pady=5, fill=tk.X)
        ttk.Button(self.roi_frame, text="❌ Delete Selected ROI", command=self.delete_selected_roi).pack(pady=5)

        # Inspection Tab
        self.inspection_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.inspection_frame, text="🔍 Inspections")
        ttk.Label(self.inspection_frame, text="ROI ID").pack(pady=5)
        self.roi_id_var = tk.StringVar()
        self.roi_id_menu = ttk.OptionMenu(self.inspection_frame, self.roi_id_var, "")
        self.roi_id_menu.pack(pady=5)
        inspection_canvas = tk.Canvas(self.inspection_frame, bg="#e6e6e6")
        inspection_scrollbar = ttk.Scrollbar(self.inspection_frame, orient=tk.VERTICAL, command=inspection_canvas.yview)
        inspection_inner = ttk.Frame(inspection_canvas)
        inspection_inner.bind(
            "<Configure>",
            lambda e: inspection_canvas.configure(scrollregion=inspection_canvas.bbox("all"))
        )
        inspection_canvas.create_window((0, 0), window=inspection_inner, anchor="nw")
        inspection_canvas.configure(yscrollcommand=inspection_scrollbar.set)
        inspection_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        inspection_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Collapsible sections for inspection controls
        def create_collapsible_section(parent, title):
            frame = ttk.Frame(parent)
            frame.pack(fill=tk.X, pady=2)
            var = tk.BooleanVar(value=True)
            header = ttk.Checkbutton(frame, text=title, variable=var, style="TButton")
            header.pack(fill=tk.X)
            content = ttk.Frame(frame)
            content.pack(fill=tk.X, padx=10)
            def toggle():
                if var.get():
                    content.pack(fill=tk.X, padx=10)
                else:
                    content.pack_forget()
            var.trace("w", lambda *args: toggle())
            return content

        # Density Inspection
        density_frame = create_collapsible_section(inspection_inner, "Density Inspection")
        ttk.Label(density_frame, text="Density Threshold", width=20).pack(side=tk.LEFT, padx=5)
        ttk.Label(density_frame, text="Min").pack(side=tk.LEFT)
        ttk.Scale(density_frame, from_=0, to=255, orient=tk.HORIZONTAL, variable=self.params["density_threshold_min"], length=80).pack(side=tk.LEFT)
        ttk.Entry(density_frame, textvariable=self.params["density_threshold_min"], width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(density_frame, text="Max").pack(side=tk.LEFT)
        ttk.Scale(density_frame, from_=0, to=255, orient=tk.HORIZONTAL, variable=self.params["density_threshold_max"], length=80).pack(side=tk.LEFT)
        ttk.Entry(density_frame, textvariable=self.params["density_threshold_max"], width=6).pack(side=tk.LEFT, padx=2)
        ttk.Button(density_frame, text="Run Density", command=self.run_density_inspection).pack(side=tk.LEFT, padx=5)
        ttk.Button(density_frame, text="Preview", command=lambda: self.run_density_inspection(preview=True)).pack(side=tk.LEFT, padx=5)

        # Contrast Inspection
        contrast_frame = create_collapsible_section(inspection_inner, "Contrast Inspection")
        ttk.Label(contrast_frame, text="Contrast Threshold", width=20).pack(side=tk.LEFT, padx=5)
        ttk.Label(contrast_frame, text="Min").pack(side=tk.LEFT)
        ttk.Scale(contrast_frame, from_=0, to=255, orient=tk.HORIZONTAL, variable=self.params["contrast_threshold_min"], length=80).pack(side=tk.LEFT)
        ttk.Entry(contrast_frame, textvariable=self.params["contrast_threshold_min"], width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(contrast_frame, text="Max").pack(side=tk.LEFT)
        ttk.Scale(contrast_frame, from_=0, to=255, orient=tk.HORIZONTAL, variable=self.params["contrast_threshold_max"], length=80).pack(side=tk.LEFT)
        ttk.Entry(contrast_frame, textvariable=self.params["contrast_threshold_max"], width=6).pack(side=tk.LEFT, padx=2)
        ttk.Button(contrast_frame, text="Run Contrast", command=self.run_contrast_inspection).pack(side=tk.LEFT, padx=5)
        ttk.Button(contrast_frame, text="Preview", command=lambda: self.run_contrast_inspection(preview=True)).pack(side=tk.LEFT, padx=5)

        # Edge Inspection
        edge_frame = create_collapsible_section(inspection_inner, "Edge Inspection")
        ttk.Label(edge_frame, text="Edge Threshold", width=20).pack(side=tk.LEFT, padx=5)
        ttk.Label(edge_frame, text="Min").pack(side=tk.LEFT)
        ttk.Scale(edge_frame, from_=0, to=10000, orient=tk.HORIZONTAL, variable=self.params["edge_threshold_min"], length=80).pack(side=tk.LEFT)
        ttk.Entry(edge_frame, textvariable=self.params["edge_threshold_min"], width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(edge_frame, text="Max").pack(side=tk.LEFT)
        ttk.Scale(edge_frame, from_=0, to=10000, orient=tk.HORIZONTAL, variable=self.params["edge_threshold_max"], length=80).pack(side=tk.LEFT)
        ttk.Entry(edge_frame, textvariable=self.params["edge_threshold_max"], width=6).pack(side=tk.LEFT, padx=2)
        ttk.Button(edge_frame, text="Run Edge", command=self.run_edge_inspection).pack(side=tk.LEFT, padx=5)
        ttk.Button(edge_frame, text="Preview", command=lambda: self.run_edge_inspection(preview=True)).pack(side=tk.LEFT, padx=5)
        edge_subframe = ttk.Frame(edge_frame)
        edge_subframe.pack(fill=tk.X, padx=10, pady=2)
        ttk.Label(edge_subframe, text="Canny Low", width=15).pack(side=tk.LEFT, padx=5)
        ttk.Entry(edge_subframe, textvariable=self.params["edge_canny_low"], width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(edge_subframe, text="Canny High", width=15).pack(side=tk.LEFT, padx=5)
        ttk.Entry(edge_subframe, textvariable=self.params["edge_canny_high"], width=6).pack(side=tk.LEFT, padx=2)
        # Blob Detection
        blob_frame = create_collapsible_section(inspection_inner, "Blob Detection")
        ttk.Label(blob_frame, text="Threshold", width=20).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(blob_frame, text="Manual Threshold", variable=self.params["blob_threshold_manual"]).pack(side=tk.LEFT, padx=5)
        ttk.Label(blob_frame, text="Value").pack(side=tk.LEFT)
        ttk.Scale(blob_frame, from_=0, to=255, orient=tk.HORIZONTAL, variable=self.params["blob_threshold_value"], length=80).pack(side=tk.LEFT)
        ttk.Entry(blob_frame, textvariable=self.params["blob_threshold_value"], width=6).pack(side=tk.LEFT, padx=2)
        # Size Filters
        size_frame = ttk.Frame(blob_frame)
        size_frame.pack(fill=tk.X, padx=10, pady=2)
        ttk.Label(size_frame, text="Area Min", width=15).pack(side=tk.LEFT, padx=5)
        ttk.Entry(size_frame, textvariable=self.params["blob_area_min"], width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(size_frame, text="Max", width=15).pack(side=tk.LEFT, padx=5)
        ttk.Entry(size_frame, textvariable=self.params["blob_area_max"], width=6).pack(side=tk.LEFT, padx=2)
        size_subframe = ttk.Frame(blob_frame)
        size_subframe.pack(fill=tk.X, padx=10, pady=2)
        ttk.Label(size_subframe, text="Width Min", width=15).pack(side=tk.LEFT, padx=5)
        ttk.Entry(size_subframe, textvariable=self.params["blob_width_min"], width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(size_subframe, text="Max", width=15).pack(side=tk.LEFT, padx=5)
        ttk.Entry(size_subframe, textvariable=self.params["blob_width_max"], width=6).pack(side=tk.LEFT, padx=2)
        size_subframe2 = ttk.Frame(blob_frame)
        size_subframe2.pack(fill=tk.X, padx=10, pady=2)
        ttk.Label(size_subframe2, text="Height Min", width=15).pack(side=tk.LEFT, padx=5)
        ttk.Entry(size_subframe2, textvariable=self.params["blob_height_min"], width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(size_subframe2, text="Max", width=15).pack(side=tk.LEFT, padx=5)
        ttk.Entry(size_subframe2, textvariable=self.params["blob_height_max"], width=6).pack(side=tk.LEFT, padx=2)

        # Shape Filters
        shape_frame = ttk.Frame(blob_frame)
        shape_frame.pack(fill=tk.X, padx=10, pady=2)
        ttk.Label(shape_frame, text="Circularity Min", width=15).pack(side=tk.LEFT, padx=5)
        ttk.Entry(shape_frame, textvariable=self.params["blob_circularity_min"], width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(shape_frame, text="Max", width=15).pack(side=tk.LEFT, padx=5)
        ttk.Entry(shape_frame, textvariable=self.params["blob_circularity_max"], width=6).pack(side=tk.LEFT, padx=2)
        shape_subframe = ttk.Frame(blob_frame)
        shape_subframe.pack(fill=tk.X, padx=10, pady=2)
        ttk.Label(shape_subframe, text="Aspect Ratio Min", width=15).pack(side=tk.LEFT, padx=5)
        ttk.Entry(shape_subframe, textvariable=self.params["blob_aspect_ratio_min"], width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(shape_subframe, text="Max", width=15).pack(side=tk.LEFT, padx=5)
        ttk.Entry(shape_subframe, textvariable=self.params["blob_aspect_ratio_max"], width=6).pack(side=tk.LEFT, padx=2)
        shape_subframe2 = ttk.Frame(blob_frame)
        shape_subframe2.pack(fill=tk.X, padx=10, pady=2)
        ttk.Label(shape_subframe2, text="Solidity Min", width=15).pack(side=tk.LEFT, padx=5)
        ttk.Entry(shape_subframe2, textvariable=self.params["blob_solidity_min"], width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(shape_subframe2, text="Max", width=15).pack(side=tk.LEFT, padx=5)
        ttk.Entry(shape_subframe2, textvariable=self.params["blob_solidity_max"], width=6).pack(side=tk.LEFT, padx=2)
        shape_subframe3 = ttk.Frame(blob_frame)
        shape_subframe3.pack(fill=tk.X, padx=10, pady=2)
        ttk.Label(shape_subframe3, text="Bounding Shape", width=15).pack(side=tk.LEFT, padx=5)
        ttk.Combobox(shape_subframe3, textvariable=self.params["blob_bounding_shape"], values=["None", "Rectangle", "Circle"], state="readonly", width=10).pack(side=tk.LEFT, padx=2)

        # Color Filters
        color_frame = ttk.Frame(blob_frame)
        color_frame.pack(fill=tk.X, padx=10, pady=2)
        ttk.Label(color_frame, text="Color Mode", width=15).pack(side=tk.LEFT, padx=5)
        ttk.Combobox(color_frame, textvariable=self.params["blob_color_mode"], values=["Grayscale", "RGB", "HSV"], state="readonly", width=10).pack(side=tk.LEFT, padx=2)
        color_subframe = ttk.Frame(blob_frame)
        color_subframe.pack(fill=tk.X, padx=10, pady=2)
        ttk.Label(color_subframe, text="RGB R Min", width=15).pack(side=tk.LEFT, padx=5)
        ttk.Entry(color_subframe, textvariable=self.params["blob_rgb_r_min"], width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(color_subframe, text="Max", width=15).pack(side=tk.LEFT, padx=5)
        ttk.Entry(color_subframe, textvariable=self.params["blob_rgb_r_max"], width=6).pack(side=tk.LEFT, padx=2)
        color_subframe2 = ttk.Frame(blob_frame)
        color_subframe2.pack(fill=tk.X, padx=10, pady=2)
        ttk.Label(color_subframe2, text="RGB G Min", width=15).pack(side=tk.LEFT, padx=5)
        ttk.Entry(color_subframe2, textvariable=self.params["blob_rgb_g_min"], width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(color_subframe2, text="Max", width=15).pack(side=tk.LEFT, padx=5)
        ttk.Entry(color_subframe2, textvariable=self.params["blob_rgb_g_max"], width=6).pack(side=tk.LEFT, padx=2)
        color_subframe3 = ttk.Frame(blob_frame)
        color_subframe3.pack(fill=tk.X, padx=10, pady=2)
        ttk.Label(color_subframe3, text="RGB B Min", width=15).pack(side=tk.LEFT, padx=5)
        ttk.Entry(color_subframe3, textvariable=self.params["blob_rgb_b_min"], width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(color_subframe3, text="Max", width=15).pack(side=tk.LEFT, padx=5)
        ttk.Entry(color_subframe3, textvariable=self.params["blob_rgb_b_max"], width=6).pack(side=tk.LEFT, padx=2)
        color_subframe4 = ttk.Frame(blob_frame)
        color_subframe4.pack(fill=tk.X, padx=10, pady=2)
        ttk.Label(color_subframe4, text="HSV H Min", width=15).pack(side=tk.LEFT, padx=5)
        ttk.Entry(color_subframe4, textvariable=self.params["blob_hsv_h_min"], width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(color_subframe4, text="Max", width=15).pack(side=tk.LEFT, padx=5)
        ttk.Entry(color_subframe4, textvariable=self.params["blob_hsv_h_max"], width=6).pack(side=tk.LEFT, padx=2)
        color_subframe5 = ttk.Frame(blob_frame)
        color_subframe5.pack(fill=tk.X, padx=10, pady=2)
        ttk.Label(color_subframe5, text="HSV S Min", width=15).pack(side=tk.LEFT, padx=5)
        ttk.Entry(color_subframe5, textvariable=self.params["blob_hsv_s_min"], width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(color_subframe5, text="Max", width=15).pack(side=tk.LEFT, padx=5)
        ttk.Entry(color_subframe5, textvariable=self.params["blob_hsv_s_max"], width=6).pack(side=tk.LEFT, padx=2)
        color_subframe6 = ttk.Frame(blob_frame)
        color_subframe6.pack(fill=tk.X, padx=10, pady=2)
        ttk.Label(color_subframe6, text="HSV V Min", width=15).pack(side=tk.LEFT, padx=5)
        ttk.Entry(color_subframe6, textvariable=self.params["blob_hsv_v_min"], width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(color_subframe6, text="Max", width=15).pack(side=tk.LEFT, padx=5)
        ttk.Entry(color_subframe6, textvariable=self.params["blob_hsv_v_max"], width=6).pack(side=tk.LEFT, padx=2)

        # Output Parameters
        output_frame = ttk.Frame(blob_frame)
        output_frame.pack(fill=tk.X, padx=10, pady=2)
        ttk.Label(output_frame, text="Output Parameters").pack(anchor=tk.W)
        for output, var in self.blob_outputs.items():
            ttk.Checkbutton(output_frame, text=output.replace("_", " ").title(), variable=var).pack(anchor=tk.W)

        # Judgment Criteria
        judgment_frame = ttk.Frame(blob_frame)
        judgment_frame.pack(fill=tk.X, padx=10, pady=2)
        ttk.Label(judgment_frame, text="Judgment Criteria").pack(anchor=tk.W)
        ttk.Label(judgment_frame, text="Blob Count Min").pack(side=tk.LEFT, padx=5)
        ttk.Entry(judgment_frame, textvariable=self.judgment_criteria["blob_count_min"], width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(judgment_frame, text="Max").pack(side=tk.LEFT, padx=5)
        ttk.Entry(judgment_frame, textvariable=self.judgment_criteria["blob_count_max"], width=6).pack(side=tk.LEFT, padx=2)
        judgment_subframe = ttk.Frame(blob_frame)
        judgment_subframe.pack(fill=tk.X, padx=10, pady=2)
        ttk.Label(judgment_subframe, text="Blob Area Min").pack(side=tk.LEFT, padx=5)
        ttk.Entry(judgment_subframe, textvariable=self.judgment_criteria["blob_area_min"], width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(judgment_subframe, text="Max").pack(side=tk.LEFT, padx=5)
        ttk.Entry(judgment_subframe, textvariable=self.judgment_criteria["blob_area_max"], width=6).pack(side=tk.LEFT, padx=2)
        judgment_subframe2 = ttk.Frame(blob_frame)
        judgment_subframe2.pack(fill=tk.X, padx=10, pady=2)
        ttk.Label(judgment_subframe2, text="Criteria Type").pack(side=tk.LEFT, padx=5)
        ttk.Combobox(judgment_subframe2, textvariable=self.judgment_criteria["criteria_type"], values=["At least one blob", "Blob count limit"], state="readonly", width=15).pack(side=tk.LEFT, padx=2)

        ttk.Button(blob_frame, text="Run Blob", command=self.run_blob_detection).pack(side=tk.LEFT, padx=5)
        ttk.Button(blob_frame, text="Preview", command=lambda: self.run_blob_detection(preview=True)).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(blob_frame, text="Boundary Exclusion", variable=self.params["boundary_exclusion"]).pack(side=tk.LEFT, padx=5)

        # Color Detection
        color_frame = create_collapsible_section(inspection_inner, "Color Detection")
        ttk.Label(color_frame, text="Color Ratio", width=20).pack(side=tk.LEFT, padx=5)
        ttk.Label(color_frame, text="Min").pack(side=tk.LEFT)
        ttk.Entry(color_frame, textvariable=self.params["color_ratio_min"], width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(color_frame, text="Max").pack(side=tk.LEFT)
        ttk.Entry(color_frame, textvariable=self.params["color_ratio_max"], width=6).pack(side=tk.LEFT, padx=2)
        ttk.Button(color_frame, text="Run Color", command=self.run_color_detection).pack(side=tk.LEFT, padx=5)
        ttk.Button(color_frame, text="Preview", command=lambda: self.run_color_detection(preview=True)).pack(side=tk.LEFT, padx=5)
        color_subframe = ttk.Frame(color_frame)
        color_subframe.pack(fill=tk.X, padx=10, pady=2)
        ttk.Label(color_subframe, text="Hue Min", width=15).pack(side=tk.LEFT, padx=5)
        ttk.Entry(color_subframe, textvariable=self.params["color_hue_min"], width=6).pack(side=tk.LEFT, padx=2)
        ttk.Button(color_subframe, text="+", command=lambda: self.params["color_hue_min"].set(min(self.params["color_hue_min"].get() + 5, 180))).pack(side=tk.LEFT, padx=1)
        ttk.Button(color_subframe, text="-", command=lambda: self.params["color_hue_min"].set(max(self.params["color_hue_min"].get() - 5, 0))).pack(side=tk.LEFT, padx=1)
        ttk.Label(color_subframe, text="Max", width=15).pack(side=tk.LEFT, padx=5)
        ttk.Entry(color_subframe, textvariable=self.params["color_hue_max"], width=6).pack(side=tk.LEFT, padx=2)
        ttk.Button(color_subframe, text="+", command=lambda: self.params["color_hue_max"].set(min(self.params["color_hue_max"].get() + 5, 180))).pack(side=tk.LEFT, padx=1)
        ttk.Button(color_subframe, text="-", command=lambda: self.params["color_hue_max"].set(max(self.params["color_hue_max"].get() - 5, 0))).pack(side=tk.LEFT, padx=1)
        color_subframe2 = ttk.Frame(color_frame)
        color_subframe2.pack(fill=tk.X, padx=10, pady=2)
        ttk.Label(color_subframe2, text="Saturation Min", width=15).pack(side=tk.LEFT, padx=5)
        ttk.Entry(color_subframe2, textvariable=self.params["color_saturation_min"], width=6).pack(side=tk.LEFT, padx=2)
        ttk.Button(color_subframe2, text="+", command=lambda: self.params["color_saturation_min"].set(min(self.params["color_saturation_min"].get() + 5, 255))).pack(side=tk.LEFT, padx=1)
        ttk.Button(color_subframe2, text="-", command=lambda: self.params["color_saturation_min"].set(max(self.params["color_saturation_min"].get() - 5, 0))).pack(side=tk.LEFT, padx=1)
        ttk.Label(color_subframe2, text="Saturation Max", width=15).pack(side=tk.LEFT, padx=5)
        ttk.Entry(color_subframe2, textvariable=self.params["color_saturation_max"], width=6).pack(side=tk.LEFT, padx=2)
        ttk.Button(color_subframe2, text="+", command=lambda: self.params["color_saturation_max"].set(min(self.params["color_saturation_max"].get() + 5, 255))).pack(side=tk.LEFT, padx=1)
        ttk.Button(color_subframe2, text="-", command=lambda: self.params["color_saturation_max"].set(max(self.params["color_saturation_max"].get() - 5, 0))).pack(side=tk.LEFT, padx=1)
        color_subframe3 = ttk.Frame(color_frame)
        color_subframe3.pack(fill=tk.X, padx=10, pady=2)
        ttk.Label(color_subframe3, text="Brightness Min", width=15).pack(side=tk.LEFT, padx=5)
        ttk.Entry(color_subframe3, textvariable=self.params["color_brightness_min"], width=6).pack(side=tk.LEFT, padx=2)
        ttk.Button(color_subframe3, text="+", command=lambda: self.params["color_brightness_min"].set(min(self.params["color_brightness_min"].get() + 5, 255))).pack(side=tk.LEFT, padx=1)
        ttk.Button(color_subframe3, text="-", command=lambda: self.params["color_brightness_min"].set(max(self.params["color_brightness_min"].get() - 5, 0))).pack(side=tk.LEFT, padx=1)
        ttk.Label(color_subframe3, text="Brightness Max", width=15).pack(side=tk.LEFT, padx=5)
        ttk.Entry(color_subframe3, textvariable=self.params["color_brightness_max"], width=6).pack(side=tk.LEFT, padx=2)
        ttk.Button(color_subframe3, text="+", command=lambda: self.params["color_brightness_max"].set(min(self.params["color_brightness_max"].get() + 5, 255))).pack(side=tk.LEFT, padx=1)
        ttk.Button(color_subframe3, text="-", command=lambda: self.params["color_brightness_max"].set(max(self.params["color_brightness_max"].get() - 5, 0))).pack(side=tk.LEFT, padx=1)
        ttk.Button(color_frame, text="🎯 Pick Color", command=self.start_color_picking).pack(side=tk.LEFT, pady=5)

        # Measurement
        measurement_frame = create_collapsible_section(inspection_inner, "Measurement")
        ttk.Label(measurement_frame, text="Tolerance", width=20).pack(side=tk.LEFT, padx=5)
        ttk.Label(measurement_frame, text="Min").pack(side=tk.LEFT)
        ttk.Entry(measurement_frame, textvariable=self.params["measurement_tolerance_min"], width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(measurement_frame, text="Max").pack(side=tk.LEFT)
        ttk.Entry(measurement_frame, textvariable=self.params["measurement_tolerance_max"], width=6).pack(side=tk.LEFT, padx=2)
        ttk.Button(measurement_frame, text="Run Measurement", command=self.run_measurement).pack(side=tk.LEFT, padx=5)
        ttk.Button(measurement_frame, text="Preview", command=lambda: self.run_measurement(preview=True)).pack(side=tk.LEFT, padx=5)

        # Focus Check
        focus_frame = create_collapsible_section(inspection_inner, "Focus Check")
        ttk.Label(focus_frame, text="Focus Threshold", width=20).pack(side=tk.LEFT, padx=5)
        ttk.Label(focus_frame, text="Min").pack(side=tk.LEFT)
        ttk.Scale(focus_frame, from_=0, to=500, orient=tk.HORIZONTAL, variable=self.params["focus_threshold_min"], length=80).pack(side=tk.LEFT)
        ttk.Entry(focus_frame, textvariable=self.params["focus_threshold_min"], width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(focus_frame, text="Max").pack(side=tk.LEFT)
        ttk.Scale(focus_frame, from_=0, to=500, orient=tk.HORIZONTAL, variable=self.params["focus_threshold_max"], length=80).pack(side=tk.LEFT)
        ttk.Entry(focus_frame, textvariable=self.params["focus_threshold_max"], width=6).pack(side=tk.LEFT, padx=2)
        ttk.Button(focus_frame, text="Run Focus", command=self.run_focus_check).pack(side=tk.LEFT, padx=5)
        ttk.Button(focus_frame, text="Preview", command=lambda: self.run_focus_check(preview=True)).pack(side=tk.LEFT, padx=5)

        # Cycle Logic Tab
        self.cycle_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.cycle_frame, text="🔄 Cycle Logic")
        ttk.Label(self.cycle_frame, text="Cycle Features").pack(pady=5)
        cycle_inner = ttk.Frame(self.cycle_frame)
        cycle_inner.pack(fill=tk.BOTH, padx=5)
        for feature, var in self.cycle_features.items():
            ttk.Checkbutton(cycle_inner, text=feature, variable=var).pack(anchor=tk.W)
        ttk.Button(self.cycle_frame, text="💾 Save Config", command=self.save_cycle_config).pack(pady=5)
        ttk.Button(self.cycle_frame, text="📜 Load Config", command=self.load_cycle_config).pack(pady=5)
        self.cycle_run_button = ttk.Button(self.cycle_frame, text="▶️ Run Cycle", command=self.run_cycle_logic)
        self.cycle_run_button.pack(pady=5)
        self.cycle_label = ttk.Label(self.cycle_frame, text="Cycle State: Idle")
        self.cycle_label.pack(pady=5)
        self.progress = ttk.Progressbar(self.cycle_frame, length=200, mode="determinate")
        self.progress.pack(pady=5)

        # Simulation Tab
        self.simulation_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.simulation_frame, text="🧪 Simulation")
        ttk.Button(self.simulation_frame, text="📂 Load Test Images", command=self.load_test_images).pack(pady=5)
        ttk.Button(self.simulation_frame, text="▶️ Next Image", command=self.next_test_image).pack(pady=5)
        ttk.Button(self.simulation_frame, text="🔄 Run Test Cycle", command=self.run_test_cycle).pack(pady=5)
        self.test_image_label = ttk.Label(self.simulation_frame, text="No test images loaded")
        self.test_image_label.pack(pady=5)

        # Settings Tab
        self.settings_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.settings_frame, text="⚙️ Settings")
        settings_canvas = tk.Canvas(self.settings_frame, bg="#e6e6e6")
        settings_scrollbar = ttk.Scrollbar(self.settings_frame, orient=tk.VERTICAL, command=settings_canvas.yview)
        settings_inner = ttk.Frame(settings_canvas)
        settings_inner.bind(
            "<Configure>",
            lambda e: settings_canvas.configure(scrollregion=settings_canvas.bbox("all"))
        )
        settings_canvas.create_window((0, 0), window=settings_inner, anchor="nw")
        settings_canvas.configure(yscrollcommand=settings_scrollbar.set)
        settings_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        settings_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        for param_name in sorted(set(k.rsplit("_", 1)[0] for k in self.params.keys() if k != "gpio_trigger_pin")):
            frame = ttk.Frame(settings_inner)
            frame.pack(fill=tk.X, pady=3)
            ttk.Label(frame, text=param_name.replace("_", " ").title(), width=20).pack(side=tk.LEFT, padx=5)
            if f"{param_name}_min" in self.params:
                ttk.Label(frame, text="Min").pack(side=tk.LEFT)
                ttk.Entry(frame, textvariable=self.params[f"{param_name}_min"], width=6).pack(side=tk.LEFT, padx=2)
            if f"{param_name}_max" in self.params:
                ttk.Label(frame, text="Max").pack(side=tk.LEFT)
                ttk.Entry(frame, textvariable=self.params[f"{param_name}_max"], width=6).pack(side=tk.LEFT, padx=2)
            if param_name in self.params and param_name not in [f"{k}_min"[:-4] for k in self.params.keys() if k.endswith("_min")]:
                if param_name == "boundary_exclusion" or param_name == "blob_threshold_manual":
                    ttk.Checkbutton(frame, text="Enable", variable=self.params[param_name]).pack(side=tk.LEFT, padx=2)
                elif param_name in ["contour_hierarchy_mode", "blob_color_mode", "blob_bounding_shape"]:
                    values = ["External", "All"] if param_name == "contour_hierarchy_mode" else ["Grayscale", "RGB", "HSV"] if param_name == "blob_color_mode" else ["None", "Rectangle", "Circle"]
                    ttk.Combobox(frame, textvariable=self.params[param_name], values=values, state="readonly", width=10).pack(side=tk.LEFT, padx=2)
                elif param_name in ["contour_gaussian_blur", "contour_morph_kernel", "edge_sobel_kernel", "edge_median_blur"]:
                    ttk.Combobox(frame, textvariable=self.params[param_name], values=[3, 5, 7], state="readonly", width=5).pack(side=tk.LEFT, padx=2)
                else:
                    ttk.Entry(frame, textvariable=self.params[param_name], width=10).pack(side=tk.LEFT, padx=2)
        gpio_frame = ttk.Frame(settings_inner)
        gpio_frame.pack(fill=tk.X, pady=5)
        ttk.Label(gpio_frame, text="GPIO Trigger Pin", width=20).pack(side=tk.LEFT, padx=5)
        self.gpio_label = ttk.Label(gpio_frame, text=f"Pin: {self.params['gpio_trigger_pin'].get() if self.params['gpio_trigger_pin'].get() != -1 else 'None'}")
        self.gpio_label.pack(side=tk.LEFT, padx=5)
        ttk.Button(gpio_frame, text="🔌 Setup GPIO", command=self.setup_gpio).pack(side=tk.LEFT, padx=5)
        ttk.Button(settings_inner, text="💾 Save Settings", command=self.save_settings).pack(pady=5)

    def init_log(self):
        try:
            if not os.path.exists(self.log_file):
                with open(self.log_file, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Timestamp", "ROI ID", "Inspection Type", "Result", "Details"])
        except Exception as e:
            self.show_toast(f"Log initialization failed: {e}")

    def update_time(self):
        self.time_var.set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.root.after(1000, self.update_time)

    def show_toast(self, message, duration=2000):
        toast = tk.Toplevel(self.root)
        toast.overrideredirect(True)
        toast.configure(bg="#333333")
        tk.Label(toast, text=message, bg="#333333", fg="#ffffff", font=("Segoe UI", 10), padx=10, pady=5).pack()
        x = self.root.winfo_x() + self.root.winfo_width() - 200
        y = self.root.winfo_y() + 50
        toast.geometry(f"200x40+{x}+{y}")
        toast.attributes("-alpha", 0.0)
        def fade_in(alpha=0.0):
            alpha += 0.1
            if alpha <= 0.9:
                toast.attributes("-alpha", alpha)
                toast.after(50, lambda: fade_in(alpha))
        def fade_out(alpha=0.9):
            alpha -= 0.1
            if alpha >= 0.0:
                toast.attributes("-alpha", alpha)
                toast.after(50, lambda: fade_out(alpha))
            else:
                toast.destroy()
        fade_in()
        toast.after(duration, lambda: fade_out())

    def validate_min_max(self, min_key, max_key):
        if min_key in self.params and max_key in self.params:
            min_val = self.params[min_key].get()
            max_val = self.params[max_key].get()
            if min_val > max_val:
                self.show_toast(f"{min_key} must be <= {max_key}")
                return False
        return True

    def update_mode(self, *args):
        mode = self.mode.get()
        self.status_var.set(f"Ready | {mode}")
        if mode == "Mode Réglage":
            self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            self.canvas.bind("<Button-1>", self.start_roi_or_pick_color)
            self.canvas.bind("<B1-Motion>", self.draw_roi_or_mask)
            self.canvas.bind("<ButtonRelease-1>", self.end_roi_or_mask)
            self.canvas.bind("<Button-3>", self.select_roi)
            self.canvas.bind("<B3-Motion>", self.move_resize_rotate_roi)
            self.canvas.bind("<ButtonRelease-3>", self.end_move_resize_rotate)
            self.roi_info_frame.pack(fill=tk.X, padx=5, pady=5)
            self.cycle_run_button.pack(pady=5)
            self.root.config(menu=self.menubar)
        else:  # Run Mode
            self.notebook.pack_forget()
            self.canvas.unbind("<Button-1>")
            self.canvas.unbind("<B1-Motion>")
            self.canvas.unbind("<ButtonRelease-1>")
            self.canvas.unbind("<Button-3>")
            self.canvas.unbind("<B3-Motion>")
            self.canvas.unbind("<ButtonRelease-3>")
            self.roi_info_frame.pack_forget()
            self.cycle_run_button.pack(pady=5)
            self.root.config(menu=tk.Menu(self.root))
            self.show_toast("Entered Run Mode: Only cycle execution allowed")

    def setup_gpio(self):
        if self.mode.get() != "Mode Réglage":
            self.show_toast("GPIO setup available only in Mode Réglage")
            return
        gpio_window = tk.Toplevel(self.root)
        gpio_window.title("GPIO Pin Selection")
        gpio_window.geometry("700x500")
        gpio_window.configure(bg="#e6e6e6")
        gpio_window.transient(self.root)
        gpio_window.grab_set()

        ttk.Label(gpio_window, text="Select GPIO Pin for Cycle Trigger (Raspberry Pi 4B)", font=("Segoe UI", 12, "bold")).pack(pady=10)
        ttk.Label(gpio_window, text="Click a green GPIO pin to set the trigger. Power (red) and ground (black) pins are not selectable.", font=("Segoe UI", 10)).pack()

        canvas = tk.Canvas(gpio_window, width=650, height=350, bg="#ffffff", highlightthickness=1, highlightbackground="#d4d4d4")
        canvas.pack(pady=10)

        pin_layout = [
            (None, "3.3V", "POWER", 1), (None, "5V", "POWER", 2),
            (2, "GPIO2 (SDA1)", "GPIO", 3), (None, "5V", "POWER", 4),
            (3, "GPIO3 (SCL1)", "GPIO", 5), (None, "GND", "GROUND", 6),
            (4, "GPIO4", "GPIO", 7), (14, "GPIO14 (TXD0)", "GPIO", 8),
            (None, "GND", "GROUND", 9), (15, "GPIO15 (RXD0)", "GPIO", 10),
            (17, "GPIO17", "GPIO", 11), (18, "GPIO18 (PCM_CLK)", "GPIO", 12),
            (27, "GPIO27", "GPIO", 13), (None, "GND", "GROUND", 14),
            (22, "GPIO22", "GPIO", 15), (23, "GPIO23", "GPIO", 16),
            (None, "3.3V", "POWER", 17), (24, "GPIO24", "GPIO", 18),
            (10, "GPIO10 (MOSI)", "GPIO", 19), (None, "GND", "GROUND", 20),
            (9, "GPIO9 (MISO)", "GPIO", 21), (25, "GPIO25", "GPIO", 22),
            (11, "GPIO11 (SCLK)", "GPIO", 23), (8, "GPIO8 (CE0)", "GPIO", 24),
            (None, "GND", "GROUND", 25), (7, "GPIO7 (CE1)", "GPIO", 26),
            (5, "GPIO5", "GPIO", 27), (None, "GND", "GROUND", 28),
            (6, "GPIO6", "GPIO", 29), (12, "GPIO12 (PCM_DIN)", "GPIO", 30),
            (13, "GPIO13 (PCM_DOUT)", "GPIO", 31), (None, "GND", "GROUND", 32),
            (19, "GPIO19 (MISO)", "GPIO", 33), (16, "GPIO16", "GPIO", 34),
            (26, "GPIO26", "GPIO", 35), (20, "GPIO20 (MOSI)", "GPIO", 36),
            (None, "GND", "GROUND", 37), (21, "GPIO21 (SCLK)", "GPIO", 38),
            (None, "GND", "GROUND", 39), (None, "GND", "GROUND", 40)
        ]

        pin_size = 20
        spacing_x = 30
        spacing_y = 25
        offset_x, offset_y = 50, 50
        self.gpio_pins = {}
        selected_pin = self.params["gpio_trigger_pin"].get()
        self.temp_selected_pin.set(selected_pin)

        self.selected_pin_label = ttk.Label(gpio_window, text=f"Selected Pin: GPIO{selected_pin}" if selected_pin != -1 else "Selected Pin: None")
        self.selected_pin_label.pack(pady=5)

        for i in range(20):
            for j in range(2):
                idx = i * 2 + j
                bcm, label, pin_type, phys_pin = pin_layout[idx]
                x = offset_x + j * (pin_size + spacing_x + 200)
                y = offset_y + i * (pin_size + spacing_y)
                color = "#28a745" if pin_type == "GPIO" else "#dc3545" if pin_type == "POWER" else "#212121"
                outline = "#0078d7" if bcm == selected_pin and pin_type == "GPIO" else "#000000"
                pin_id = canvas.create_oval(x, y, x + pin_size, y + pin_size, fill=color, outline=outline, width=2)
                label_text = f"Pin {phys_pin}: {'GPIO' + str(bcm) if bcm is not None else pin_type}"
                canvas.create_text(x + pin_size / 2, y + pin_size + 15, text=label_text, font=("Segoe UI", 8))
                if pin_type == "GPIO":
                    self.gpio_pins[pin_id] = (bcm, phys_pin)
                    canvas.tag_bind(pin_id, "<Button-1>", lambda e, p=bcm: self.select_gpio_pin(p, canvas))
                    canvas.tag_bind(pin_id, "<Enter>", lambda e, pid=pin_id: canvas.itemconfig(pid, fill="#90EE90"))
                    canvas.tag_bind(pin_id, "<Leave>", lambda e, pid=pin_id: canvas.itemconfig(pid, fill="#28a745"))
                    tooltip = tk.Toplevel(canvas, bg="#ffffe0")
                    tooltip.wm_overrideredirect(True)
                    tooltip_label = tk.Label(tooltip, text=label, bg="#ffffe0", font=("Segoe UI", 8))
                    tooltip_label.pack()
                    tooltip.withdraw()
                    def show_tooltip(event, x=x, y=y):
                        tooltip.geometry(f"+{event.x_root + 10}+{event.y_root + 10}")
                        tooltip.deiconify()
                    def hide_tooltip(event):
                        tooltip.withdraw()
                    canvas.tag_bind(pin_id, "<Enter>", show_tooltip)
                    canvas.tag_bind(pin_id, "<Leave>", hide_tooltip)
                else:
                    canvas.tag_bind(pin_id, "<Button-1>", lambda e: self.show_toast("Select a GPIO pin, not POWER or GROUND"))

        legend_frame = ttk.Frame(gpio_window)
        legend_frame.pack(pady=5)
        for color, text in [("#28a745", "GPIO"), ("#dc3545", "Power"), ("#212121", "Ground"), ("#0078d7", "Selected")]:
            canvas_widget = tk.Canvas(legend_frame, width=20, height=20, bg="#e6e6e6", highlightthickness=0)
            canvas_widget.create_oval(5, 5, 15, 15, fill=color)
            canvas_widget.pack(side=tk.LEFT, padx=2)
            ttk.Label(legend_frame, text=text).pack(side=tk.LEFT, padx=2)

        button_frame = ttk.Frame(gpio_window)
        button_frame.pack(pady=10)
        ttk.Button(button_frame, text="Save Selection", command=lambda: self.save_gpio_selection(gpio_window)).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=gpio_window.destroy).pack(side=tk.LEFT, padx=5)

    def select_gpio_pin(self, pin, canvas):
        self.temp_selected_pin.set(pin)
        self.selected_pin_label.config(text=f"Selected Pin: GPIO{pin}")
        self.show_toast(f"Selected GPIO{pin}")
        for pin_id, (bcm, _) in self.gpio_pins.items():
            canvas.itemconfig(pin_id, outline="#0078d7" if bcm == pin else "#000000")

    def save_gpio_selection(self, window):
        self.params["gpio_trigger_pin"].set(self.temp_selected_pin.get())
        self.gpio_label.config(text=f"Pin: GPIO{self.params['gpio_trigger_pin'].get()}" if self.params['gpio_trigger_pin'].get() != -1 else "Pin: None")
        self.save_settings()
        window.destroy()
        self.show_toast("GPIO selection saved")

    def start_gpio_simulation(self):
        if self.gpio_thread and self.gpio_thread.is_alive():
            self.params["gpio_trigger_pin"].set(-1)
        if self.params["gpio_trigger_pin"].get() != -1:
            def simulate_gpio():
                while self.params["gpio_trigger_pin"].get() != -1:
                    if self.mode.get() == "Run Mode":
                        self.gpio_trigger_active = np.random.choice([True, False], p=[0.1, 0.9])
                        if self.gpio_trigger_active:
                            self.root.after(0, self.run_cycle_logic)
                    time.sleep(0.5)
            self.gpio_thread = threading.Thread(target=simulate_gpio, daemon=True)
            self.gpio_thread.start()

    def update_video(self):
        try:
            if self.use_static_image:
                frame = self.static_image.copy()
            else:
                ret, frame = self.cap.read()
                if not ret:
                    self.show_toast("Failed to capture frame")
                    return

            # Draw ROIs
            for roi in self.rois:
                x, y, w, h, roi_id, angle, shape, mask = roi
                color = (255, 0, 0) if roi_id == self.selected_roi else (0, 255, 0) if roi_id == self.hovered_roi else (0, 200, 0)
                center = (x + w // 2, y + h // 2)
                if shape == "rectangle":
                    M = cv2.getRotationMatrix2D(center, angle, 1)
                    cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
                    new_w = int((h * sin) + (w * cos))
                    new_h = int((h * cos) + (w * sin))
                    M[0, 2] += (new_w - w) // 2
                    M[1, 2] += (new_h - h) // 2
                    points = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)
                    points = np.dot(points - center, M[:, :2].T) + center
                    points = points.astype(np.int32)
                    cv2.polylines(frame, [points], True, color, 2)
                else:  # circle
                    radius = min(w, h) // 2
                    cv2.ellipse(frame, (int(center[0]), int(center[1])), (radius, radius), angle, 0, 360, color, 2)
                if roi_id == self.selected_roi:
                    if shape == "rectangle":
                        cv2.polylines(frame, [points], True, (255, 255, 255), 1)
                        for px, py in points:
                            cv2.circle(frame, (px, py), 5, (255, 255, 0), -1)
                    else:
                        cv2.ellipse(frame, (int(center[0]), int(center[1])), (radius, radius), angle, 0, 360, (255, 255, 255), 1)
                        cv2.circle(frame, (int(center[0] + radius * np.cos(np.radians(angle))), int(center[1] + radius * np.sin(np.radians(angle)))), 5, (255, 255, 0), -1)
                    cv2.circle(frame, (int(center[0]), int(center[1])), 5, (0, 255, 255), -1)
                cv2.putText(frame, f"ROI {roi_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                # Draw mask if present
                if mask is not None and np.any(mask):
                    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
                    mask_rgb[mask > 0] = (255, 0, 255)  # Magenta for mask
                    frame[y:y+h, x:x+w] = cv2.addWeighted(frame[y:y+h, x:x+w], 0.7, mask_rgb, 0.3, 0)

            # Update ROI info
            if self.selected_roi is not None:
                for roi in self.rois:
                    if roi[4] == self.selected_roi:
                        x, y, w, h, _, angle, shape, _ = roi
                        self.roi_info_var.set(f"ROI {self.selected_roi}: X={x}, Y={y}, W={w}, H={h}, Angle={angle:.1f}°, Shape={shape}")
                        break
            else:
                self.roi_info_var.set("ROI Info: None")

            # Convert to ImageTk
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((640, 480), Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

            # Update ROI list and menu
            self.roi_listbox.delete(0, tk.END)
            roi_ids = [str(roi[4]) for roi in self.rois]
            for roi_id in roi_ids:
                self.roi_listbox.insert(tk.END, f"ROI {roi_id}")
            self.roi_id_menu["menu"].delete(0, tk.END)
            for roi_id in roi_ids:
                self.roi_id_menu["menu"].add_command(label=roi_id, command=lambda x=roi_id: self.roi_id_var.set(x))

            self.led_ok.itemconfig("led_ok", fill="#28a745" if self.cycle_state == "Completed" else "#d4d4d4")
            self.led_ng.itemconfig("led_ng", fill="#dc3545" if self.cycle_state.startswith("Failed") else "#d4d4d4")

            self.root.after(10, self.update_video)
        except Exception as e:
            self.show_toast(f"Video update error: {e}")

    def update_cursor(self, event):
        if self.mode.get() != "Mode Réglage":
            self.canvas.config(cursor="")
            return
        x, y = event.x, event.y
        self.hovered_roi = None
        for roi in self.rois:
            rx, ry, rw, rh, rid, angle, shape, _ = roi
            center_x, center_y = rx + rw / 2, ry + rh / 2
            if shape == "rectangle":
                if rx <= x <= rx + rw and ry <= y <= ry + rh:
                    self.hovered_roi = rid
                    if (abs(x - rx) < 10 or abs(x - (rx + rw)) < 10) and (abs(y - ry) < 10 or abs(y - (ry + rh)) < 10):
                        self.canvas.config(cursor="sizing")
                        self.status_var.set(f"ROI {rid}: Resize")
                    elif abs(x - center_x) < 10 and abs(y - center_y) < 10:
                        self.canvas.config(cursor="cross")
                        self.status_var.set(f"ROI {rid}: Rotate")
                    else:
                        self.canvas.config(cursor="fleur")
                        self.status_var.set(f"ROI {rid}: Move")
                    break
            else:  # circle
                radius = min(rw, rh) / 2
                dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                if dist <= radius:
                    self.hovered_roi = rid
                    if abs(dist - radius) < 10:
                        self.canvas.config(cursor="sizing")
                        self.status_var.set(f"ROI {rid}: Resize")
                    elif abs(x - center_x) < 10 and abs(y - center_y) < 10:
                        self.canvas.config(cursor="cross")
                        self.status_var.set(f"ROI {rid}: Rotate")
                    else:
                        self.canvas.config(cursor="fleur")
                        self.status_var.set(f"ROI {rid}: Move")
                    break
        else:
            self.canvas.config(cursor="")
            self.status_var.set(f"Ready | {self.mode.get()}")

    def start_roi_or_pick_color(self, event):
        if self.mode.get() != "Mode Réglage":
            return
        if self.color_picking:
            self.pick_color(event)
            return
        self.drawing = True
        self.ix, self.iy = event.x, event.y
        self.status_var.set("Drawing ROI")

    def draw_roi_or_mask(self, event):
        if self.mode.get() != "Mode Réglage":
            return
        if self.drawing:
            self.draw_roi(event)
        elif self.drawing_mask:
            self.draw_mask(event)

    def end_roi_or_mask(self, event):
        if self.mode.get() != "Mode Réglage":
            return
        if self.drawing:
            self.end_roi(event)
        elif self.drawing_mask:
            self.end_mask(event)

    def draw_roi(self, event):
        if not self.drawing:
            return
        self.canvas.delete("temp_roi")
        x, y = min(self.ix, event.x), min(self.iy, event.y)
        w, h = abs(event.x - self.ix), abs(event.y - self.iy)
        if self.snap_to_grid.get():
            grid_size = 10
            x, y = round(x / grid_size) * grid_size, round(y / grid_size) * grid_size
            w, h = round(w / grid_size) * grid_size, round(h / grid_size) * grid_size
        if self.roi_shape.get() == "rectangle":
            self.canvas.create_rectangle(x, y, x + w, y + h, outline="#0078d7", width=2, tags="temp_roi")
        else:
            radius = min(w, h) / 2
            self.canvas.create_oval(x, y, x + w, y + h, outline="#0078d7", width=2, tags="temp_roi")

    def end_roi(self, event):
        if not self.drawing:
            return
        self.drawing = False
        x, y = min(self.ix, event.x), min(self.iy, event.y)
        w, h = abs(event.x - self.ix), abs(event.y - self.iy)
        if self.snap_to_grid.get():
            grid_size = 10
            x, y = round(x / grid_size) * grid_size, round(y / grid_size) * grid_size
            w, h = round(w / grid_size) * grid_size, round(h / grid_size) * grid_size
        if w > 10 and h > 10:
            mask = np.zeros((h, w), dtype=np.uint8)  # Initialize empty mask
            self.rois.append((x, y, w, h, self.roi_id, 0.0, self.roi_shape.get(), mask))
            self.roi_id += 1
            self.show_toast(f"ROI {self.roi_id - 1} created")
        self.canvas.delete("temp_roi")
        self.status_var.set(f"Ready | {self.mode.get()}")

    def start_mask_drawing(self):
        if self.mode.get() != "Mode Réglage":
            self.show_toast("Mask drawing available only in Mode Réglage")
            return
        if self.selected_roi is None:
            self.show_toast("Select an ROI first")
            return
        self.drawing_mask = True
        self.status_var.set("Drawing Mask")
        self.show_toast("Draw mask with left mouse button")

    def draw_mask(self, event):
        if not self.drawing_mask or self.selected_roi is None:
            return
        for i, roi in enumerate(self.rois):
            if roi[4] == self.selected_roi:
                x, y, w, h, _, _, _, mask = roi
                rel_x, rel_y = event.x - x, event.y - y
                if 0 <= rel_x < w and 0 <= rel_y < h:
                    cv2.circle(mask, (int(rel_x), int(rel_y)), 5, 255, -1)
                self.rois[i] = (x, y, w, h, roi[4], roi[5], roi[6], mask)
                break

    def end_mask(self, event):
        if not self.drawing_mask:
            return
        self.drawing_mask = False
        self.status_var.set(f"Ready | {self.mode.get()}")
        self.show_toast("Mask drawing completed")

    def clear_mask(self):
        if self.mode.get() != "Mode Réglage":
            self.show_toast("Clear mask available only in Mode Réglage")
            return
        if self.selected_roi is None:
            self.show_toast("Select an ROI first")
            return
        for i, roi in enumerate(self.rois):
            if roi[4] == self.selected_roi:
                x, y, w, h, rid, angle, shape, _ = roi
                self.rois[i] = (x, y, w, h, rid, angle, shape, np.zeros((h, w), dtype=np.uint8))
                self.show_toast(f"Mask cleared for ROI {rid}")
                break

    def select_roi(self, event):
        if self.mode.get() != "Mode Réglage":
            return
        x, y = event.x, event.y
        for roi in self.rois:
            rx, ry, rw, rh, rid, angle, shape, _ = roi
            center_x, center_y = rx + rw / 2, ry + rh / 2
            if shape == "rectangle":
                if rx <= x <= rx + rw and ry <= y <= ry + rh:
                    self.selected_roi = rid
                    self.moving = True
                    self.ix, self.iy = x, y
                    self.status_var.set(f"ROI {rid}: Moving")
                    break
            else:  # circle
                radius = min(rw, rh) / 2
                dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                if dist <= radius:
                    self.selected_roi = rid
                    self.moving = True
                    self.ix, self.iy = x, y
                    self.status_var.set(f"ROI {rid}: Moving")
                    break
        else:
            self.selected_roi = None
            self.moving = False
            self.resizing = False
            self.rotating = False
            self.status_var.set(f"Ready | {self.mode.get()}")

    def move_resize_rotate_roi(self, event):
        if self.mode.get() != "Mode Réglage" or self.selected_roi is None:
            return
        x, y = event.x, event.y
        for i, roi in enumerate(self.rois):
            if roi[4] == self.selected_roi:
                rx, ry, rw, rh, rid, angle, shape, mask = roi
                center_x, center_y = rx + rw / 2, ry + rh / 2
                if self.moving:
                    dx, dy = x - self.ix, y - self.iy
                    new_x, new_y = rx + dx, ry + dy
                    if self.snap_to_grid.get():
                        grid_size = 10
                        new_x = round(new_x / grid_size) * grid_size
                        new_y = round(new_y / grid_size) * grid_size
                    self.rois[i] = (new_x, new_y, rw, rh, rid, angle, shape, mask)
                    self.ix, self.iy = x, y
                elif self.resizing:
                    if shape == "rectangle":
                        new_w = max(10, rw + (x - self.ix))
                        new_h = max(10, rh + (y - self.iy))
                    else:
                        radius = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                        new_w = new_h = max(10, int(radius * 2))
                    if self.snap_to_grid.get():
                        grid_size = 10
                        new_w = round(new_w / grid_size) * grid_size
                        new_h = round(new_h / grid_size) * grid_size
                    new_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                    self.rois[i] = (rx, ry, new_w, new_h, rid, angle, shape, new_mask)
                    self.ix, self.iy = x, y
                elif self.rotating:
                    dx, dy = x - center_x, y - center_y
                    new_angle = np.degrees(np.arctan2(dy, dx))
                    self.rois[i] = (rx, ry, rw, rh, rid, new_angle, shape, mask)
                break

    def end_move_resize_rotate(self, event):
        if self.mode.get() != "Mode Réglage":
            return
        self.moving = False
        self.resizing = False
        self.rotating = False
        self.status_var.set(f"Ready | {self.mode.get()}")

    def delete_selected_roi(self):
        if self.mode.get() != "Mode Réglage":
            self.show_toast("ROI deletion available only in Mode Réglage")
            return
        if self.selected_roi is not None:
            self.rois = [roi for roi in self.rois if roi[4] != self.selected_roi]
            self.show_toast(f"ROI {self.selected_roi} deleted")
            self.selected_roi = None

    def clear_rois(self):
        if self.mode.get() != "Mode Réglage":
            self.show_toast("Clear ROIs available only in Mode Réglage")
            return
        self.rois = []
        self.selected_roi = None
        self.roi_id = 0
        self.show_toast("All ROIs cleared")

    def reset_roi(self):
        if self.mode.get() != "Mode Réglage":
            self.show_toast("Reset ROI available only in Mode Réglage")
            return
        self.selected_roi = None
        self.drawing = False
        self.canvas.delete("temp_roi")
        self.status_var.set(f"Ready | {self.mode.get()}")

    def start_color_picking(self):
        if self.mode.get() != "Mode Réglage":
            self.show_toast("Color picking available only in Mode Réglage")
            return
        self.color_picking = True
        self.show_toast("Click on image to pick color")

    def pick_color(self, event):
        if not self.color_picking:
            return
        try:
            if self.use_static_image:
                frame = self.static_image.copy()
            else:
                ret, frame = self.cap.read()
                if not ret:
                    self.show_toast("Failed to capture frame")
                    return
            x, y = event.x, event.y
            h, w = frame.shape[:2]
            x_scaled = int(x * w / 640.0)
            y_scaled = int(y * h / 480.0 * h / w)
            b, g, r = frame[y_scaled, x_scaled]
            hsv = cv2.cvtColor(np.uint8([[b, g, r]]), cv2.COLOR_BGR2HSV)[0][0]
            self.params["color_hue_min"].set(max(0, h - 10))
            self.params["color_hue_max"].set(min(180, h + 10))
            self.params["color_saturation_min"].set(max(0, hsv[1] - 50))
            self.params["color_saturation_max"].set(min(255, hsv[1] + 50))
            self.params["color_brightness_min"].set(max(0, hsv[2] - 50))
            self.params["color_brightness_max"].set(min(255, hsv[2] + 50))
            self.show_toast(f"Picked color: RGB=({r}, {g}, {b}), HSV=({hsv[0]}, {hsv[1]}, {hsv[2]})")
            self.color_picking = False
            self.status_var.set(f"Ready | {self.mode.get()}")
        except Exception as e:
            self.show_toast(f"Color picking error: {e}")

    def run_blob_detection(self, preview=False):
        if not self.validate_selected_roi():
            return
        if not self.validate_parameters(["blob_threshold_value", "blob_area_min", "blob_area_max", "blob_width_min", "blob_width_max",
                                        "blob_height_min", "blob_height_max", "blob_circularity_min", "blob_circularity_max",
                                        "blob_aspect_ratio_min", "blob_aspect_ratio_max", "blob_soldity_min", "blob_soldity_max",
                                        "blob_rgb_r_min", "blob_rgb_r_max", "blob_rgb_g_min", "blob_rgb_g_max",
                                        "blob_rgb_b_min", "blob_rgb_b_max", "blob_hsv_h_min", "blob_hsv_h_max",
                                        "blob_hsv_s_min", "blob_hsv_s_max", "blob_hsv_v_min", "blob_hsv_v_max",
                                        "blob_count_min", "blob_count_max"]):
            return
        try:
            img = self.get_image()
            for roi in self.rois:
                if roi[4] == self.selected_roi:
                    x, y, w, h, _, _, _, mask = roi
                    roi_img = img[y:y+h, x:x+w]
                    if self.params["blob_color_mode"].get() == "Grayscale":
                        gray = cv2.cvtColor(roi_img, cv2.COLOR_RGB2GRAY)
                        if self.params["blob_threshold_manual"].get():
                            thresh = cv2.threshold(gray, self.params["blob_threshold_value"].get(), 255, cv2.THRESH_BINARY)[1]
                        else:
                            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                    elif self.params["blob_color_mode"].get() == "RGB":
                        lower_rgb = (self.params["blob_rgb_b_min"].get(), self.params["blob_rgb_g_min"].get(), self.params["blob_rgb_r_min"].get())
                        upper_rgb = (self.params["blob_rgb_b_max"].get(), self.params["blob_rgb_g_max"].get(), self.params["blob_rgb_r_max"].get())
                        thresh = cv2.inRange(roi_img, lower_rgb, upper_rgb)
                    else:  # HSV
                        hsv = cv2.cvtColor(roi_img, cv2.COLOR_RGB2HSV)
                        lower_hsv = (self.params["blob_hsv_h_min"].get(), self.params["blob_hsv_s_min"].get(), self.params["blob_hsv_v_min"].get())
                        upper_hsv = (self.params["get_hsv_h_max"].get(), self.params["blob_hsv_s_max"].get(), self.params["blob_hsv_v_max"].get())
                        thresh = cv2.inRange(hsv, lower_hsv, upper_hsv)
                    if np.any(mask):
                        thresh = thresh & mask
                    # Apply bilateral filter
                    thresh = cv2.bilateralFilter(thresh, 11, self.params["blob_bilateral_sigma"].get(), self.params["blob_bilateral_sigma"].get())
                    # Find contours
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    # Filter contours
                    filtered_blobs = []
                    blob_measurements = {
                        "count": 0,
                        "largest_area": 0,
                        "smallest_area": float("inf"),
                        "center_of_gravity": [],
                        "positions": [],
                        "orientation": [],
                        "total_area": 0,
                        "fill_percentage": 0
                    }
                    for contour in contours:
                        area = cv2.contourArea(contour)
                        if not (self.params["blob_area_min"].get() <= area <= self.params["blob_area_max"].get()):
                            continue
                        x_b, y_b, w_b, h_b = cv2.boundingRect(contour)
                        if not (self.params["blob_width_min"].get() <= w_b <= self.params["blob_width_max"].get() and
                                self.params["blob_height_min"].get() <= h_b <= self.params["blob_height_max"].get()):
                            continue
                        # Shape filters
                        perimeter = cv2.arcLength(contour, True)
                        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
                        if not (self.params["blob_circularity_min"].get() <= circularity <= self.params["blob_circularity_max"].get()):
                            continue
                        aspect_ratio = w_b / h_b if h_b > 0 else 0
                        if not (self.params["blob_aspect_ratio_min"].get() <= aspect_ratio <= self.params["blob_aspect_ratio_max"].get()):
                            continue
                        hull = cv2.convexHull(contour)
                        hull_area = cv2.contourArea(hull)
                        solidity = area / hull_area if hull_area > 0 else 0
                        if not (self.params["blob_solidity_min"].get() <= solidity <= self.params["blob_solidity_max"].get()):
                            continue
                        # Bounding shape match
                        if self.params["blob_bounding_shape"].get() == "Rectangle":
                            rect = cv2.minAreaRect(contour)
                            box = cv2.boxPoints(rect)
                            box_area = cv2.contourArea(np.int32([box]))
                            if box_area == 0 or area / box_area < 0.8:
                                continue
                        elif self.params["blob_bounding_shape"].get() == "Circle":
                            (cx, cy), radius = cv2.minEnclosingCircle(contour)
                            circle_area = np.pi * radius ** 2
                            if circle_area == 0 or area / circle_area < 0.8:
                                continue
                        # Boundary exclusion
                        if self.params["boundary_exclusion"].get():
                            if x_b <= 2 or y_b <= 2 or x_b + w_b >= w - 2 or y_b + h_b >= h - 2:
                                continue
                        filtered_blobs.append(contour)
                        # Update measurements
                        blob_measurements["count"] += 1
                        blob_measurements["largest_area"] = max(blob_measurements["largest_area"], area)
                        blob_measurements["smallest_area"] = min(blob_measurements["smallest_area"], area)
                        M = cv2.moments(contour)
                        cx = M["m10"] / M["m00"] if M["m00"] > 0 else 0
                        cy = M["m01"] / M["m00"] if M["m00"] > 0 else 0
                        blob_measurements["center_of_gravity"].append((cx + x, cy + y))
                        blob_measurements["positions"].append((x_b + x, y_b + y, w_b, h_b))
                        if len(contour) >= 5:
                            ellipse = cv2.fitEllipse(contour)
                            blob_measurements["orientation"].append(ellipse[2])
                        blob_measurements["total_area"] += area
                    if blob_measurements["count"] > 0:
                        blob_measurements["fill_percentage"] = blob_measurements["total_area"] / (w * h) * 100
                    else:
                        blob_measurements["smallest_area"] = 0
                    # Judgment criteria
                    result = "OK"
                    details = []
                    if self.judgment_criteria["criteria_type"].get() == "At least one blob":
                        if blob_measurements["count"] < 1:
                            result = "NG"
                            details.append("No blobs detected")
                    else:  # Blob count limit
                        if not (self.judgment_criteria["blob_count_min"].get() <= blob_measurements["count"] <= self.judgment_criteria["blob_count_max"].get()):
                            result = "NG"
                            details.append(f"Blob count {blob_measurements['count']} outside range [{self.judgment_criteria['blob_count_min'].get()}, {self.judgment_criteria['blob_count_max'].get()}]")
                    for contour in filtered_blobs:
                        area = cv2.contourArea(contour)
                        if not (self.judgment_criteria["blob_area_min"].get() <= area <= self.judgment_criteria["blob_area_max"].get()):
                            result = "NG"
                            details.append(f"Blob area {area:.1f} outside range [{self.judgment_criteria['blob_area_min'].get()}, {self.judgment_criteria['blob_area_max'].get()}]")
                    # Log results
                    self.log_result(roi[4], "Blob Detection", result, "; ".join(details) or "Passed")
                    # Display results
                    self.result_text.delete(1.0, tk.END)
                    self.result_text.insert(tk.END, f"Blob Detection ROI {roi[4]}: {result}\n")
                    for output, enabled in self.blob_outputs.items():
                        if enabled.get():
                            value = blob_measurements[output]
                            if output in ["center_of_gravity", "positions", "orientation"]:
                                value = "; ".join([f"({v[0]:.1f}, {v[1]:.1f})" if isinstance(v, tuple) else f"{v:.1f}" for v in value[:2]]) + ("..." if len(value) > 2 else "")
                            elif output == "fill_percentage":
                                value = f"{value:.2f}%"
                            elif output in ["largest_area", "smallest_area", "total_area"]:
                                value = f"{value:.1f} px²"
                            self.result_text.insert(tk.END, f"{output.replace('_', ' ').title()}: {value}\n")
                    self.result_text.tag_add("green" if result == "OK" else "red", "1.0", "1.end")
                    # Preview
                    if preview:
                        preview_img = roi_img.copy()
                        for contour in filtered_blobs:
                            cv2.drawContours(preview_img, [contour], -1, (0, 255, 0), 2)
                        cv2.imshow(f"Blob Detection Preview ROI {roi[4]}", cv2.cvtColor(preview_img, cv2.COLOR_RGB2BGR))
                        cv2.waitKey(1)
                    break
        except Exception as e:
            self.show_toast(f"Blob detection error: {e}")

    def run_density_inspection(self, preview=False):
        if not self.validate_selected_roi() or not self.validate_parameters(["density_threshold_min", "density_threshold_max"]):
            return
        try:
            img = self.get_image()
            for roi in self.rois:
                if roi[4] == self.selected_roi:
                    x, y, w, h, _, _, _, mask = roi
                    roi_img = img[y:y+h, x:x+w]
                    gray = cv2.cvtColor(roi_img, cv2.COLOR_RGB2GRAY)
                    if np.any(mask):
                        gray = cv2.bitwise_and(gray, gray, mask=mask)
                    mean_density = np.mean(gray[gray > 0]) if np.any(gray > 0) else 0
                    min_density, max_density = self.params["density_threshold_min"].get(), self.params["density_threshold_max"].get()
                    result = "OK" if min_density <= mean_density <= max_density else "NG"
                    details = f"Mean Density: {mean_density:.2f} (Range: [{min_density}, {max_density}])"
                    self.log_result(roi[4], "Density Inspection", result, details)
                    self.result_text.delete(1.0, tk.END)
                    self.result_text.insert(tk.END, f"Density Inspection ROI {roi[4]}: {result}\n{details}\n")
                    self.result_text.tag_add("green" if result == "OK" else "red", "1.0", "1.end")
                    if preview:
                        preview_img = roi_img.copy()
                        cv2.putText(preview_img, f"Density: {mean_density:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if result == "OK" else (0, 0, 255), 2)
                        cv2.imshow(f"Density Preview ROI {roi[4]}", cv2.cvtColor(preview_img, cv2.COLOR_RGB2BGR))
                        cv2.waitKey(1)
                    break
        except Exception as e:
            self.show_toast(f"Density inspection error: {e}")

    def run_contrast_inspection(self, preview=False):
        if not self.validate_selected_roi() or not self.validate_parameters(["contrast_threshold_min", "contrast_threshold_max"]):
            return
        try:
            img = self.get_image()
            for roi in self.rois:
                if roi[4] == self.selected_roi:
                    x, y, w, h, _, _, _, mask = roi
                    roi_img = img[y:y+h, x:x+w]
                    gray = cv2.cvtColor(roi_img, cv2.COLOR_RGB2GRAY)
                    if np.any(mask):
                        gray = cv2.bitwise_and(gray, gray, mask=mask)
                    contrast = np.std(gray[gray > 0]) if np.any(gray > 0) else 0
                    min_contrast, max_contrast = self.params["contrast_threshold_min"].get(), self.params["contrast_threshold_max"].get()
                    result = "OK" if min_contrast <= contrast <= max_contrast else "NG"
                    details = f"Contrast: {contrast:.2f} (Range: [{min_contrast}, {max_contrast}])"
                    self.log_result(roi[4], "Contrast Inspection", result, details)
                    self.result_text.delete(1.0, tk.END)
                    self.result_text.insert(tk.END, f"Contrast Inspection ROI {roi[4]}: {result}\n{details}\n")
                    self.result_text.tag_add("green" if result == "OK" else "red", "1.0", "1.end")
                    if preview:
                        preview_img = roi_img.copy()
                        cv2.putText(preview_img, f"Contrast: {contrast:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if result == "OK" else (0, 0, 255), 2)
                        cv2.imshow(f"Contrast Preview ROI {roi[4]}", cv2.cvtColor(preview_img, cv2.COLOR_RGB2BGR))
                        cv2.waitKey(1)
                    break
        except Exception as e:
            self.show_toast(f"Contrast inspection error: {e}")

    def run_edge_inspection(self, preview=False):
        if not self.validate_selected_roi() or not self.validate_parameters(["edge_threshold_min", "edge_threshold_max", "edge_canny_low", "edge_canny_high"]):
            return
        try:
            img = self.get_image()
            for roi in self.rois:
                if roi[4] == self.selected_roi:
                    x, y, w, h, _, _, _, mask = roi
                    roi_img = img[y:y+h, x:x+w]
                    gray = cv2.cvtColor(roi_img, cv2.COLOR_RGB2GRAY)
                    if np.any(mask):
                        gray = cv2.bitwise_and(gray, gray, mask=mask)
                    gray = cv2.medianBlur(gray, self.params["edge_median_blur"].get())
                    edges = cv2.Canny(gray, self.params["edge_canny_low"].get(), self.params["edge_canny_high"].get())
                    edge_sum = np.sum(edges) / 255
                    min_edge, max_edge = self.params["edge_threshold_min"].get(), self.params["edge_threshold_max"].get()
                    result = "OK" if min_edge <= edge_sum <= max_edge else "NG"
                    details = f"Edge Sum: {edge_sum:.2f} (Range: [{min_edge}, {max_edge}])"
                    self.log_result(roi[4], "Edge Inspection", result, details)
                    self.result_text.delete(1.0, tk.END)
                    self.result_text.insert(tk.END, f"Edge Inspection ROI {roi[4]}: {result}\n{details}\n")
                    self.result_text.tag_add("green" if result == "OK" else "red", "1.0", "1.end")
                    if preview:
                        preview_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
                        cv2.putText(preview_img, f"Edge Sum: {edge_sum:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if result == "OK" else (0, 0, 255), 2)
                        cv2.imshow(f"Edge Preview ROI {roi[4]}", cv2.cvtColor(preview_img, cv2.COLOR_RGB2BGR))
                        cv2.waitKey(1)
                    break
        except Exception as e:
            self.show_toast(f"Edge inspection error: {e}")

    def run_color_detection(self, preview=False):
        if not self.validate_selected_roi() or not self.validate_parameters(["color_hue_min", "color_hue_max", "color_saturation_min", "color_saturation_max", "color_brightness_min", "color_brightness_max", "color_ratio_min", "color_ratio_max"]):
            return
        try:
            img = self.get_image()
            for roi in self.rois:
                if roi[4] == self.selected_roi:
                    x, y, w, h, _, _, _, mask = roi
                    roi_img = img[y:y+h, x:x+w]
                    hsv = cv2.cvtColor(roi_img, cv2.COLOR_RGB2HSV)
                    lower = (self.params["color_hue_min"].get(), self.params["color_saturation_min"].get(), self.params["color_brightness_min"].get())
                    upper = (self.params["color_hue_max"].get(), self.params["color_saturation_max"].get(), self.params["color_brightness_max"].get())
                    color_mask = cv2.inRange(hsv, lower, upper)
                    if np.any(mask):
                        color_mask = cv2.bitwise_and(color_mask, mask)
                    ratio = np.sum(color_mask) / (w * h * 255) * 100
                    min_ratio, max_ratio = self.params["color_ratio_min"].get(), self.params["color_ratio_max"].get()
                    result = "OK" if min_ratio <= ratio <= max_ratio else "NG"
                    details = f"Color Ratio: {ratio:.2f}% (Range: [{min_ratio}, {max_ratio}])"
                    self.log_result(roi[4], "Color Detection", result, details)
                    self.result_text.delete(1.0, tk.END)
                    self.result_text.insert(tk.END, f"Color Detection ROI {roi[4]}: {result}\n{details}\n")
                    self.result_text.tag_add("green" if result == "OK" else "red", "1.0", "1.end")
                    if preview:
                        preview_img = roi_img.copy()
                        preview_img[color_mask > 0] = (0, 255, 0)
                        cv2.putText(preview_img, f"Ratio: {ratio:.2f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                        cv2.imshow(f"Color Preview ROI {roi[4]}", cv2.cvtColor(preview_img, cv2.COLOR_RGB2BGR))
                        cv2.waitKey(1)
                    break
        except Exception as e:
            self.show_toast(f"Color detection error: {e}")

    def run_measurement(self, preview=False):
        if not self.validate_selected_roi() or not self.validate_parameters(["measurement_tolerance_min", "measurement_tolerance_max"]):
            return
        try:
            img = self.get_image()
            for roi in self.rois:
                if roi[4] == self.selected_roi:
                    x, y, w, h, _, _, _, mask = roi
                    roi_img = img[y:y+h, x:x+w]
                    gray = cv2.cvtColor(roi_img, cv2.COLOR_RGB2GRAY)
                    if np.any(mask):
                        gray = cv2.bitwise_and(gray, gray, mask=mask)
                    edges = cv2.Canny(gray, 100, 200)
                    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        largest_contour = max(contours, key=cv2.contourArea)
                        area = cv2.contourArea(largest_contour)
                        min_area, max_area = w * h * self.params["measurement_tolerance_min"].get(), w * h * self.params["measurement_tolerance_max"].get()
                        result = "OK" if min_area <= area <= max_area else "NG"
                        details = f"Area: {area:.2f} px² (Range: [{min_area:.2f}, {max_area:.2f}])"
                    else:
                        result = "NG"
                        details = "No contours found"
                    self.log_result(roi[4], "Measurement", result, details)
                    self.result_text.delete(1.0, tk.END)
                    self.result_text.insert(tk.END, f"Measurement ROI {roi[4]}: {result}\n{details}\n")
                    self.result_text.tag_add("green" if result == "OK" else "red", "1.0", "1.end")
                    if preview:
                        preview_img = roi_img.copy()
                        if contours:
                            cv2.drawContours(preview_img, [largest_contour], -1, (0, 255, 0), 2)
                        cv2.putText(preview_img, f"Area: {area:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if result == "OK" else (0, 0, 255), 2)
                        cv2.imshow(f"Measurement Preview ROI {roi[4]}", cv2.cvtColor(preview_img, cv2.COLOR_RGB2BGR))
                        cv2.waitKey(1)
                    break
        except Exception as e:
            self.show_toast(f"Measurement error: {e}")

    def run_focus_check(self, preview=False):
        if not self.validate_selected_roi() or not self.validate_parameters(["focus_threshold_min", "focus_threshold_max"]):
            return
        try:
            img = self.get_image()
            for roi in self.rois:
                if roi[4] == self.selected_roi:
                    x, y, w, h, _, _, _, mask = roi
                    roi_img = img[y:y+h, x:x+w]
                    gray = cv2.cvtColor(roi_img, cv2.COLOR_RGB2GRAY)
                    if np.any(mask):
                        gray = cv2.bitwise_and(gray, gray, mask=mask)
                    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                    min_focus, max_focus = self.params["focus_threshold_min"].get(), self.params["focus_threshold_max"].get()
                    result = "OK" if min_focus <= laplacian_var <= max_focus else "NG"
                    details = f"Focus Variance: {laplacian_var:.2f} (Range: [{min_focus}, {max_focus}])"
                    self.log_result(roi[4], "Focus Check", result, details)
                    self.result_text.delete(1.0, tk.END)
                    self.result_text.insert(tk.END, f"Focus Check ROI {roi[4]}: {result}\n{details}\n")
                    self.result_text.tag_add("green" if result == "OK" else "red", "1.0", "1.end")
                    if preview:
                        preview_img = roi_img.copy()
                        cv2.putText(preview_img, f"Focus: {laplacian_var:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if result == "OK" else (0, 0, 255), 2)
                        cv2.imshow(f"Focus Preview ROI {roi[4]}", cv2.cvtColor(preview_img, cv2.COLOR_RGB2BGR))
                        cv2.waitKey(1)
                    break
        except Exception as e:
            self.show_toast(f"Focus check error: {e}")

    def run_cycle_logic(self):
        if self.cycle_state != "Idle":
            self.show_toast("Cycle already running")
            return
        if not self.rois:
            self.show_toast("No ROIs defined")
            return
        self.cycle_state = "Running"
        self.cycle_label.config(text="Cycle State: Running")
        self.progress["value"] = 0
        self.cycle_results = {}
        total_steps = sum(1 for _, enabled in self.cycle_features.items() if enabled.get()) * len(self.rois)
        step = 100 / total_steps if total_steps > 0 else 100
        current_step = 0

        try:
            img = self.get_image()
            for roi in self.rois:
                roi_id = roi[4]
                self.cycle_results[roi_id] = {}
                self.selected_roi = roi_id
                for feature, enabled in self.cycle_features.items():
                    if not enabled.get():
                        continue
                    if feature == "Density":
                        self.run_density_inspection()
                    elif feature == "Contrast":
                        self.run_contrast_inspection()
                    elif feature == "Edge":
                        self.run_edge_inspection()
                    elif feature == "Blob Detection":
                        self.run_blob_detection()
                    elif feature == "Color Detection":
                        self.run_color_detection()
                    elif feature == "Measurement":
                        self.run_measurement()
                    elif feature == "Focus Check":
                        self.run_focus_check()
                    current_step += step
                    self.progress["value"] = current_step
                    self.root.update()
            overall_result = "OK"
            for roi_id, results in self.cycle_results.items():
                for insp_type, res in results.items():
                    if res["result"] == "NG":
                        overall_result = "NG"
                        break
                if overall_result == "NG":
                    break
            self.cycle_state = "Completed" if overall_result == "OK" else "Failed"
            self.cycle_label.config(text=f"Cycle State: {self.cycle_state}")
            self.show_toast(f"Cycle completed: {overall_result}")
        except Exception as e:
            self.cycle_state = "Failed"
            self.cycle_label.config(text="Cycle State: Failed")
            self.show_toast(f"Cycle error: {e}")
        finally:
            self.progress["value"] = 100
            self.cycle_state = "Idle"
            self.cycle_label.config(text="Cycle State: Idle")

    def load_test_images(self):
        if self.mode.get() != "Mode Réglage":
            self.show_toast("Test images loading available only in Mode Réglage")
            return
        files = filedialog.askopenfilenames(filetypes=[("Image files", "*.jpg *.png *.bmp")])
        if files:
            self.test_images = [cv2.imread(f) for f in files]
            self.test_images = [img for img in self.test_images if img is not None]
            self.test_image_index.set(-1)
            self.test_image_label.config(text=f"Loaded {len(self.test_images)} test images")
            self.show_toast(f"Loaded {len(self.test_images)} test images")

    def next_test_image(self):
        if not self.test_images:
            self.show_toast("No test images loaded")
            return
        self.test_image_index.set((self.test_image_index.get() + 1) % len(self.test_images))
        self.static_image = self.test_images[self.test_image_index.get()]
        self.use_static_image = True
        self.test_image_label.config(text=f"Test Image {self.test_image_index.get() + 1}/{len(self.test_images)}")
        self.show_toast(f"Displaying test image {self.test_image_index.get() + 1}")

    def run_test_cycle(self):
        if self.mode.get() != "Mode Réglage":
            self.show_toast("Test cycle available only in Mode Réglage")
            return
        if not self.test_images:
            self.show_toast("No test images loaded")
            return
        original_use_static = self.use_static_image
        original_static_image = self.static_image
        self.use_static_image = True
        for i, img in enumerate(self.test_images):
            self.static_image = img
            self.test_image_index.set(i)
            self.test_image_label.config(text=f"Test Image {i + 1}/{len(self.test_images)}")
            self.run_cycle_logic()
            self.root.update()
            time.sleep(1)  # Simulate real-time processing
        self.use_static_image = original_use_static
        self.static_image = original_static_image
        self.show_toast("Test cycle completed")

    def get_image(self):
        if self.use_static_image:
            return self.static_image.copy()
        ret, frame = self.cap.read()
        if not ret:
            raise Exception("Failed to capture frame")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def validate_selected_roi(self):
        if self.selected_roi is None:
            self.show_toast("Select an ROI first")
            return False
        return True

    def validate_parameters(self, param_keys):
        for key in param_keys:
            if key.endswith("_min") and f"{key[:-4]}_max" in self.params:
                if not self.validate_min_max(key, f"{key[:-4]}_max"):
                    return False
        return True

    def log_result(self, roi_id, inspection_type, result, details):
        try:
            with open(self.log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), roi_id, inspection_type, result, details])
        except Exception as e:
            self.show_toast(f"Logging error: {e}")

    def view_log(self):
        try:
            with open(self.log_file, "r") as f:
                log_window = tk.Toplevel(self.root)
                log_window.title("Inspection Log")
                log_window.geometry("800x600")
                text = tk.Text(log_window, font=("Segoe UI", 10), wrap=tk.NONE)
                text.pack(fill=tk.BOTH, expand=True)
                text.insert(tk.END, f.read())
                text.config(state="disabled")
                scrollbar_y = ttk.Scrollbar(log_window, orient=tk.VERTICAL, command=text.yview)
                scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
                scrollbar_x = ttk.Scrollbar(log_window, orient=tk.HORIZONTAL, command=text.xview)
                scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
                text.config(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
        except Exception as e:
            self.show_toast(f"Error viewing log: {e}")

    def save_image(self):
        try:
            img = self.get_image()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"capture_{timestamp}.png"
            cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            self.show_toast(f"Image saved as {filename}")
        except Exception as e:
            self.show_toast(f"Save image error: {e}")

    def generate_pdf_report(self):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"report_{timestamp}.pdf"
            c = canvas.Canvas(filename, pagesize=letter)
            c.setFont("Helvetica", 12)
            c.drawString(50, 750, "VisionMaster HMI Inspection Report")
            c.drawString(50, 730, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            y = 700
            for roi_id, results in self.cycle_results.items():
                c.drawString(50, y, f"ROI {roi_id}:")
                y -= 20
                for insp_type, res in results.items():
                    c.drawString(70, y, f"{insp_type}: {res['result']} - {res['details']}")
                    y -= 20
                y -= 10
            c.save()
            self.show_toast(f"PDF report saved as {filename}")
        except Exception as e:
            self.show_toast(f"PDF generation error: {e}")

    def save_settings(self):
        try:
            settings = {k: v.get() for k, v in self.params.items()}
            with open("settings.json", "w") as f:
                json.dump(settings, f, indent=4)
            self.show_toast("Settings saved")
        except Exception as e:
            self.show_toast(f"Save settings error: {e}")

    def load_settings(self):
        try:
            if os.path.exists("settings.json"):
                with open("settings.json", "r") as f:
                    settings = json.load(f)
                for k, v in settings.items():
                    if k in self.params:
                        self.params[k].set(v)
                self.gpio_label.config(text=f"Pin: GPIO{self.params['gpio_trigger_pin'].get()}" if self.params['gpio_trigger_pin'].get() != -1 else "Pin: None")
                self.start_gpio_simulation()
                self.show_toast("Settings loaded")
        except Exception as e:
            self.show_toast(f"Load settings error: {e}")

    def save_cycle_config(self):
        try:
            config = {
                "params": {k: v.get() for k, v in self.params.items()},
                "cycle_features": {k: v.get() for k, v in self.cycle_features.items()},
                "rois": [(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7].tolist()) for r in self.rois],
                "blob_outputs": {k: v.get() for k, v in self.blob_outputs.items()},
                "judgment_criteria": {k: v.get() for k, v in self.judgment_criteria.items()}
            }
            filename = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
            if filename:
                with open(filename, "w") as f:
                    json.dump(config, f, indent=4)
                self.show_toast("Cycle configuration saved")
        except Exception as e:
            self.show_toast(f"Save config error: {e}")

    def load_cycle_config(self):
        try:
            filename = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
            if filename:
                with open(filename, "r") as f:
                    config = json.load(f)
                for k, v in config["params"].items():
                    if k in self.params:
                        self.params[k].set(v)
                for k, v in config["cycle_features"].items():
                    if k in self.cycle_features:
                        self.cycle_features[k].set(v)
                for k, v in config.get("blob_outputs", {}).items():
                    if k in self.blob_outputs:
                        self.blob_outputs[k].set(v)
                for k, v in config.get("judgment_criteria", {}).items():
                    if k in self.judgment_criteria:
                        self.judgment_criteria[k].set(v)
                self.rois = [(r[0], r[1], r[2], r[3], r[4], r[5], r[6], np.array(r[7], dtype=np.uint8)) for r in config["rois"]]
                self.roi_id = max([r[4] for r in self.rois] + [-1]) + 1
                self.gpio_label.config(text=f"Pin: GPIO{self.params['gpio_trigger_pin'].get()}" if self.params['gpio_trigger_pin'].get() != -1 else "Pin: None")
                self.start_gpio_simulation()
                self.show_toast("Cycle configuration loaded")
        except Exception as e:
            self.show_toast(f"Load config error: {e}")

    def toggle_results(self):
        if self.result_frame.winfo_ismapped():
            self.result_frame.pack_forget()
            self.show_toast("Results pane hidden")
        else:
            self.result_frame.pack(fill=tk.X, padx=5, pady=5)
            self.show_toast("Results pane shown")

    def toggle_maximize(self):
        if self.root.state() == "zoomed":
            self.root.state("normal")
        else:
            self.root.state("zoomed")

    def move_window(self, event):
        self.root.geometry(f"+{event.x_root - self.offset_x}+{event.y_root - self.offset_y}")

    def get_pos(self, event):
        self.offset_x = event.x
        self.offset_y = event.y

    def on_closing(self):
        try:
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            if self.gpio_thread and self.gpio_thread.is_alive():
                self.params["gpio_trigger_pin"].set(-1)
            self.save_settings()
            self.root.destroy()
        except Exception as e:
            print(f"Error during closing: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = VisionHMI(root)
    root.mainloop()