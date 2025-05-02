import threading
from tkinter import Frame, Toplevel, filedialog, messagebox
from typing import Callable

import customtkinter as ctk
from PIL import Image as PILImage

from src.app.surface_previewer import WallpaperPreviewer
from src.common import LayoutSegmentationLabels, LayoutSegmentationLabelsOnlyWalls

DEFAULT_FONT_FAMILY = "montserrat"


class UploadFrame(ctk.CTkFrame):
    def __init__(
        self,
        master: Frame,
        title: str,
        command: Callable,
        display_image: bool = True,
        **kwargs,
    ):
        super().__init__(master, **kwargs)

        self.columnconfigure(0, weight=1)

        self.label = ctk.CTkLabel(
            self,
            text=title,
            font=ctk.CTkFont(family=DEFAULT_FONT_FAMILY, size=20, weight="bold"),
        )
        self.label.grid(row=0, column=0, pady=(10, 5))

        self.upload_button = ctk.CTkButton(
            self,
            text="Upload",
            command=command,
            font=ctk.CTkFont(family=DEFAULT_FONT_FAMILY, size=14),
        )
        self.upload_button.grid(row=1, column=0, pady=10)

        if display_image:
            self.image_label = ctk.CTkLabel(
                self,
                text="No image uploaded",
                font=ctk.CTkFont(family=DEFAULT_FONT_FAMILY, size=14),
                width=300,
                height=300,
                corner_radius=10,
                text_color="#242424",
                fg_color="#cccccc",
                anchor="center",
            )
            self.image_label.grid(row=2, column=0, pady=10)

    def display_image(self, image: PILImage.Image, max_side_length: int = 300) -> None:
        image_thumbnail = image.copy()
        image_thumbnail.thumbnail(
            (max_side_length, max_side_length), PILImage.Resampling.LANCZOS
        )

        self.image = ctk.CTkImage(
            image_thumbnail, size=(image_thumbnail.width, image_thumbnail.height)
        )
        self.image_label.configure(image=self.image, text="", fg_color="transparent")


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

        # Wallpaper Height
        self.room_height_label = ctk.CTkLabel(self, text="Room height (metres):")
        self.room_height_label.grid(row=2, column=0, sticky="w", padx=10)

        self.room_height_entry = ctk.CTkEntry(self, width=40)
        self.room_height_entry.grid(row=2, column=1, pady=5, padx=10)
        self.room_height_entry.insert(0, "2.0")

        self.wallpaper_sample_height_label = ctk.CTkLabel(
            self, text="Wallpaper sample height (metres):"
        )
        self.wallpaper_sample_height_label.grid(row=3, column=0, sticky="w", padx=10)

        self.wallpaper_sample_height_entry = ctk.CTkEntry(self, width=40)
        self.wallpaper_sample_height_entry.grid(row=3, column=1, pady=5, padx=10)
        self.wallpaper_sample_height_entry.insert(0, "2.0")


class PreviewFrame(ctk.CTkFrame):
    def __init__(self, master: Frame, command: Callable, **kwargs):
        super().__init__(master, **kwargs)
        self.window_image = None
        self.window_label = None

        self.columnconfigure(0, weight=1)

        # Title Label
        self.label = ctk.CTkLabel(
            self,
            text="Preview",
            font=ctk.CTkFont(family=DEFAULT_FONT_FAMILY, size=20, weight="bold"),
        )
        self.label.grid(row=0, column=0, pady=(10, 5))

        self.image_label = ctk.CTkLabel(
            self,
            text="No preview image generated",
            font=ctk.CTkFont(family=DEFAULT_FONT_FAMILY, size=14),
            width=300,
            height=300,
            corner_radius=10,
            text_color="#242424",
            fg_color="#cccccc",
            anchor="center",
        )
        self.image_label.grid(row=1, column=0, pady=10)

        # Save Button
        self.save_button = ctk.CTkButton(
            self,
            text="Save",
            command=command,
            font=ctk.CTkFont(family=DEFAULT_FONT_FAMILY, size=14),
        )
        self.save_button.grid(row=2, column=0, pady=10)
        self.save_button.configure(state="disabled")

    def display_image(self, image: PILImage.Image, max_side_length: int = 300) -> None:
        image_thumbnail = image.copy()
        image_thumbnail.thumbnail(
            (max_side_length, max_side_length), PILImage.Resampling.LANCZOS
        )

        self.image = ctk.CTkImage(
            image_thumbnail, size=(image_thumbnail.width, image_thumbnail.height)
        )
        self.image_label.configure(image=self.image, text="", fg_color="transparent")
        self.image_label.bind(
            "<Button-1>", lambda event: self.create_image_window(image)
        )

    def create_image_window(self, image: PILImage.Image) -> None:
        if self.window_image is None or not self.window_image.winfo_exists():
            top = Toplevel()
            top.title("Enlarged Image")
            self.window_image = top

            label = ctk.CTkLabel(self.window_image, text="", fg_color="transparent")
            label.pack()
            self.window_label = label

        ctk_image = ctk.CTkImage(image, size=(image.width, image.height))
        self.window_label.configure(image=ctk_image)
        self.window_label.image = ctk_image
        self.window_image.lift()


class WallpaperPreviewerPage(ctk.CTk):
    def __init__(self, wallpaper_previewer: WallpaperPreviewer) -> None:
        super().__init__()

        self.wallpaper_previewer = wallpaper_previewer

        self.room_photo_pil: PILImage.Image | None = None
        self.wallpaper_pil: PILImage.Image | None = None
        self.selected_walls: set[LayoutSegmentationLabelsOnlyWalls] = set()
        self.room_height: float = 2.0
        self.wallpaper_sample_height: float = 2.0
        self.output_image: PILImage.Image | None = None

        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        self.title("Wallpaper Previewer")
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
            text="Wallpaper Previewer",
            font=ctk.CTkFont(family=DEFAULT_FONT_FAMILY, size=32, weight="bold"),
            anchor="center",
        )
        self.header.grid(row=0, column=0, pady=20, sticky="nsew")

    def __create_upload_content_frame(self) -> None:
        self.content_frame = ctk.CTkFrame(self.scrollable_frame)
        self.content_frame.grid(row=1, column=0, padx=20, pady=20, sticky="nsew")
        self.content_frame.grid_columnconfigure((0, 1), weight=1)

        # Room Photo Upload.
        self.room_photo_upload = UploadFrame(
            self.content_frame,
            title="Room Photo",
            command=self.__upload_room_photo,
        )
        self.room_photo_upload.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Wallpaper Upload.
        self.wallpaper_upload = UploadFrame(
            self.content_frame,
            title="Wallpaper",
            command=self.__upload_wallpaper,
        )
        self.wallpaper_upload.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        # Options.
        self.options_frame = OptionsFrame(self.content_frame)
        self.options_frame.grid(
            row=1, column=0, columnspan=2, padx=10, pady=10, sticky="nsew"
        )

        # Apply wallpaper button.
        self.upload_button = ctk.CTkButton(
            self.content_frame,
            text="Apply Wallpaper",
            command=self.__apply_wallpaper,
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

    def __upload_wallpaper(self) -> None:
        if not (
            file_path := filedialog.askopenfilename(
                filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
            )
        ):
            return

        self.wallpaper_pil = PILImage.open(file_path)
        self.wallpaper_upload.display_image(PILImage.open(file_path))

    def __apply_wallpaper(self) -> None:
        if not self.room_photo_pil or not self.wallpaper_pil:
            messagebox.showerror(
                "Error", "Please upload both a room photo and a wallpaper."
            )
            return

        self.selected_walls.clear()
        if self.options_frame.wall_left.get():
            self.selected_walls.add(LayoutSegmentationLabels.WALL_LEFT)
        if self.options_frame.wall_centre.get():
            self.selected_walls.add(LayoutSegmentationLabels.WALL_CENTRE)
        if self.options_frame.wall_right.get():
            self.selected_walls.add(LayoutSegmentationLabels.WALL_RIGHT)

        try:
            self.room_height = float(self.options_frame.room_height_entry.get())
            self.wallpaper_sample_height = float(
                self.options_frame.wallpaper_sample_height_entry.get()
            )
        except ValueError:
            messagebox.showerror(
                "Error",
                "Please enter valid numerical values for room height and wallpaper "
                "sample height.",
            )
            return

        thread = threading.Thread(target=self.__apply_wallpaper_thread)
        thread.start()

    def __apply_wallpaper_thread(self) -> None:
        assert self.room_photo_pil is not None
        assert self.wallpaper_pil is not None

        self.output_image = self.wallpaper_previewer(
            self.room_photo_pil,
            self.wallpaper_pil,
            self.selected_walls,
            self.room_height,
            self.wallpaper_sample_height,
        )

        if self.output_image is None:
            messagebox.showerror("Error", "Failed to apply wallpaper.")
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
