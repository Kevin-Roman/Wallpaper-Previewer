import customtkinter as ctk

from .rendering_mode import RenderingMode
from .wallpaper_mode import WallpaperMode
from src.models.room_layout_estimation import FCNAugmentedRoomLayoutEstimator
from src.models.wall_segmentation import EncoderDecoderPPMWallSegmenter
from src.app.surface_previewer import WallpaperPreviewer, TexturePreviewer


class MainMenu(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.room_layout_estimator = FCNAugmentedRoomLayoutEstimator()
        self.wall_segmenter = EncoderDecoderPPMWallSegmenter()

        self.title("Main Menu")
        self.geometry("400x300")

        ctk.CTkButton(
            self, text="Open Wallpaper Previewer", command=self.open_wallpaper_previewer
        ).pack(pady=20)
        ctk.CTkButton(
            self, text="Open Surface Previewer", command=self.open_rendering_previewer
        ).pack(pady=20)

    def open_wallpaper_previewer(self):
        self.destroy()
        app = WallpaperMode(
            WallpaperPreviewer(self.room_layout_estimator, self.wall_segmenter)
        )
        app.mainloop()

    def open_rendering_previewer(self):
        self.destroy()
        app = RenderingMode(
            TexturePreviewer(self.room_layout_estimator, self.wall_segmenter)
        )
        app.mainloop()
