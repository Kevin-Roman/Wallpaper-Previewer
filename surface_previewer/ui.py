from tkinter import filedialog

import customtkinter as ctk
from PIL import Image as PILImage

from layout_estimation.common import (
    LayoutSegmentationLabels,
    LayoutSegmentationLabelsOnlyWalls,
)

from .processor import SurfacePreviewer


class SurfacePreviewerApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.surface_previewer = SurfacePreviewer()
        self.room_image: PILImage.Image | None = None
        self.wallpaper_image: PILImage.Image | None = None
        self.output_image: PILImage.Image | None = None

        # Setup.
        self.title("room Previewer")
        self.geometry("800x800")

        # Image upload buttons.
        self.upload_room_button = ctk.CTkButton(
            self, text="Upload Room Photo", command=self.upload_room
        )
        self.upload_room_button.grid(row=0, column=0, padx=10, pady=10)

        self.upload_wallpaper_button = ctk.CTkButton(
            self, text="Upload Wallpaper", command=self.upload_wallpaper
        )
        self.upload_wallpaper_button.grid(row=0, column=1, padx=10, pady=10)

        # Checkboxes for selecting the walls to apply the wallpaper to.
        self.selected_walls: set[LayoutSegmentationLabelsOnlyWalls] = set()
        self.checkbox_vars = {}
        for i, option in enumerate(LayoutSegmentationLabels.walls()):
            var = ctk.BooleanVar()
            checkbox = ctk.CTkCheckBox(
                self,
                text=option.name,
                variable=var,
                command=lambda opt=option, v=var: self.update_selection(opt, v),
            )
            checkbox.grid(row=1, column=i, padx=10, pady=10)
            self.checkbox_vars[option] = var

        # Apply wallpaper button. Initiates the surface previewing logic.
        self.preview_button = ctk.CTkButton(
            self, text="Apply Wallpaper", command=self.preview_wallpaper
        )
        self.preview_button.grid(row=2, column=0, pady=20)

        # Uploaded image labels.
        self.room_label = ctk.CTkLabel(self, text="Room Photo")
        self.room_label.grid(row=3, column=0, padx=10, pady=10)

        self.wallpaper_label = ctk.CTkLabel(self, text="Wallpaper")
        self.wallpaper_label.grid(row=3, column=1, padx=10, pady=10)

        self.result_label = ctk.CTkLabel(self, text="Result Preview")
        self.result_label.grid(row=4, column=0, pady=10)

        # Surface previewing processing progress bar.
        self.progress_bar = ctk.CTkProgressBar(self, orientation="horizontal")
        self.progress_bar.set(0)
        self.progress_bar.grid(row=5, column=0, pady=10)

        # Save output image button.
        self.save_button = ctk.CTkButton(
            self, text="Save Output", command=self.save_image, state="disabled"
        )
        self.save_button.grid(row=6, column=0, pady=10)

    def upload_room(self) -> None:
        if not (
            file_path := filedialog.askopenfilename(
                filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
            )
        ):
            return

        self.room_image = PILImage.open(file_path)
        image_display = ctk.CTkImage(
            self.room_image,
            size=(
                int(self.room_image.width * (200 / self.room_image.height)),
                200,
            ),
        )
        self.room_label.configure(image=image_display, text="")
        self.room_label.image = image_display

    def upload_wallpaper(self) -> None:
        if not (
            file_path := filedialog.askopenfilename(
                filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
            )
        ):
            return

        self.wallpaper_image = PILImage.open(file_path)
        image_display = ctk.CTkImage(
            self.wallpaper_image,
            size=(
                200,
                int(self.wallpaper_image.height * (200 / self.wallpaper_image.width)),
            ),
        )
        self.wallpaper_label.configure(image=image_display, text="")
        self.wallpaper_label.image = image_display

    def preview_wallpaper(self) -> None:

        if not self.room_image or not self.wallpaper_image:
            self.result_label.configure(text="Upload both images first!")
            return

        self.progress_bar.set(0)
        self.update_idletasks()

        self.output_image = self.surface_previewer.apply_wallpaper(
            self.room_image, self.wallpaper_image, self.selected_walls
        )
        self.progress_bar.set(1)

        result_display = ctk.CTkImage(
            self.output_image,
            size=(
                int(self.output_image.width * (200 / self.output_image.height)),
                200,
            ),
        )
        self.result_label.configure(image=result_display, text="")
        self.result_label.image = result_display
        self.save_button.configure(state="normal")

    def save_image(self) -> None:
        if not self.output_image or not (
            file_path := filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[
                    ("PNG files", "*.png"),
                    ("JPEG files", "*.jpg"),
                    ("All Files", "*.*"),
                ],
            )
        ):
            return

        self.output_image.save(file_path)

    def update_selection(
        self, option: LayoutSegmentationLabelsOnlyWalls, var: ctk.BooleanVar
    ) -> None:
        if not var.get():
            self.selected_walls.discard(option)
            return

        self.selected_walls.add(option)
