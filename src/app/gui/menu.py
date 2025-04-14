import customtkinter as ctk

from .texture_previewer_page import TexturePreviewerPage
from .wallpaper_previewer_page import WallpaperPreviewerPage
from src.models.room_layout_estimation import FCNAugmentedRoomLayoutEstimator
from src.models.wall_segmentation import EncoderDecoderPPMWallSegmenter
from src.app.surface_previewer import WallpaperPreviewer, TexturePreviewer

DEFAULT_FONT_FAMILY = "montserrat"


class MainMenu(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.room_layout_estimator = FCNAugmentedRoomLayoutEstimator()
        self.wall_segmenter = EncoderDecoderPPMWallSegmenter()

        self.title("Main Menu")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.create_header()

        open_wallpaper_previewer_button = ctk.CTkButton(
            self, text="Open Wallpaper Previewer", command=self.open_wallpaper_previewer
        )
        open_wallpaper_previewer_button.grid(row=1, column=0, pady=5)

        open_texture_previewer_button = ctk.CTkButton(
            self, text="Open Texture Previewer", command=self.open_rendering_previewer
        )
        open_texture_previewer_button.grid(row=2, column=0, pady=5)

    def create_header(self) -> None:
        self.header = ctk.CTkLabel(
            self,
            text="Wallpaper Previewer",
            font=ctk.CTkFont(family=DEFAULT_FONT_FAMILY, size=32, weight="bold"),
            anchor="center",
        )
        self.header.grid(row=0, column=0, pady=20, sticky="n")

    def open_wallpaper_previewer(self):
        self.destroy()
        app = WallpaperPreviewerPage(
            WallpaperPreviewer(self.room_layout_estimator, self.wall_segmenter)
        )
        app.mainloop()

    def open_rendering_previewer(self):
        self.destroy()
        app = TexturePreviewerPage(
            TexturePreviewer(self.room_layout_estimator, self.wall_segmenter)
        )
        app.mainloop()
