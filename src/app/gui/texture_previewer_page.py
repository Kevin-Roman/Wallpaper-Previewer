from tkinter import Frame, filedialog, messagebox

import customtkinter as ctk
from PIL import Image as PILImage

from src.app.surface_previewer import TexturePreviewer
from src.common import LayoutSegmentationLabels, LayoutSegmentationLabelsOnlyWalls

from .wallpaper_previewer_page import PreviewFrame, UploadFrame

DEFAULT_FONT_FAMILY = "montserrat"


class OptionsFrame(ctk.CTkFrame):
    def __init__(self, master: Frame, **kwargs):
        super().__init__(master, **kwargs)

        self.title_label = ctk.CTkLabel(
            self,
            text="Options",
            font=ctk.CTkFont(family=DEFAULT_FONT_FAMILY, size=20, weight="bold"),
        )
        self.title_label.grid(
            row=0, column=0, columnspan=2, padx=10, pady=(10, 5), sticky="nw"
        )

        # Walls Label and Checkboxes (on the same row)
        self.walls_label = ctk.CTkLabel(self, text="Walls to apply wallpaper to: ")
        self.walls_label.grid(row=1, column=0, sticky="w", padx=10)

        # Checkboxes for Walls (left, centre, right) on the same row as the label
        self.wall_left = ctk.CTkCheckBox(self, text="Left Wall")
        self.wall_left.grid(row=1, column=1, sticky="w")
        self.wall_left.select()

        self.wall_centre = ctk.CTkCheckBox(self, text="Centre Wall")
        self.wall_centre.grid(row=1, column=2, sticky="w")
        self.wall_centre.select()

        self.wall_right = ctk.CTkCheckBox(self, text="Right Wall")
        self.wall_right.grid(row=1, column=3, sticky="w")
        self.wall_right.select()


class TexturePreviewerPage(ctk.CTk):
    def __init__(self, texture_previewer: TexturePreviewer) -> None:
        super().__init__()

        self.texture_previewer = texture_previewer

        self.room_photo_pil: PILImage.Image | None = None
        self.selected_walls: set[LayoutSegmentationLabelsOnlyWalls] = set()
        self.output_image: PILImage.Image | None = None

        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        self.title("Texture Previewer")
        self.geometry("800x600")
        self.minsize(600, 400)

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.scrollable_frame = ctk.CTkScrollableFrame(
            self, corner_radius=0, fg_color="transparent"
        )
        self.scrollable_frame.grid(row=0, column=0, sticky="nsew")
        self.scrollable_frame.grid_columnconfigure(0, weight=1)

        self.__create_header()
        self.__create_upload_content_frame()
        self.__create_preview_content_frame()

    def __create_header(self) -> None:
        self.header = ctk.CTkLabel(
            self.scrollable_frame,
            text="Texture Previewer",
            font=ctk.CTkFont(family=DEFAULT_FONT_FAMILY, size=32, weight="bold"),
            anchor="center",
        )
        self.header.grid(row=0, column=0, pady=20, sticky="nsew")

    def __create_upload_content_frame(self) -> None:
        self.content_frame = ctk.CTkFrame(self.scrollable_frame)
        self.content_frame.grid(row=1, column=0, padx=20, pady=20, sticky="nsew")
        self.content_frame.grid_columnconfigure(0, weight=1)

        # Room Photo Upload.
        self.room_photo_upload = UploadFrame(
            self.content_frame,
            title="Room Photo",
            command=self.__upload_room_photo,
        )
        self.room_photo_upload.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Options.
        self.options_frame = OptionsFrame(self.content_frame)
        self.options_frame.grid(
            row=1, column=0, columnspan=2, padx=10, pady=10, sticky="nsew"
        )

        # Apply texture button.
        self.upload_button = ctk.CTkButton(
            self.content_frame,
            text="Apply Texture",
            command=self.__apply_texture,
            font=ctk.CTkFont(family=DEFAULT_FONT_FAMILY, size=14),
        )
        self.upload_button.grid(row=2, column=0, pady=10)

    def __create_preview_content_frame(self) -> None:
        self.preview_frame = PreviewFrame(
            self.scrollable_frame, command=self.__save_output_image
        )
        self.preview_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")

    def __upload_room_photo(self) -> None:
        if not (
            file_path := filedialog.askopenfilename(
                filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
            )
        ):
            return

        self.room_photo_pil = PILImage.open(file_path)
        self.room_photo_upload.display_image(PILImage.open(file_path))

    def __apply_texture(self) -> None:
        if not self.room_photo_pil:
            messagebox.showerror("Error", "Please upload a room photo.")
            return

        self.selected_walls.clear()
        if self.options_frame.wall_left.get():
            self.selected_walls.add(LayoutSegmentationLabels.WALL_LEFT)
        if self.options_frame.wall_centre.get():
            self.selected_walls.add(LayoutSegmentationLabels.WALL_CENTRE)
        if self.options_frame.wall_right.get():
            self.selected_walls.add(LayoutSegmentationLabels.WALL_RIGHT)

        self.output_image = self.texture_previewer(
            self.room_photo_pil,
            self.selected_walls,
        )

        if self.output_image is None:
            messagebox.showerror("Error", "Failed to apply texture.")
            return

        self.preview_frame.display_image(self.output_image)
        self.preview_frame.save_button.configure(state="normal")

    def __save_output_image(self) -> None:
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
