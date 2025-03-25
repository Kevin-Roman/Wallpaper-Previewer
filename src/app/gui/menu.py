import customtkinter as ctk

from ..surface_previewer import SurfacePreviewer
from .rendering_previewer import RenderingMode
from .wallpaper_previewer import WallpaperMode


class MainMenu(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.surface_previewer = SurfacePreviewer()

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
        app = WallpaperMode(self.surface_previewer)
        app.mainloop()

    def open_rendering_previewer(self):
        self.destroy()
        app = RenderingMode(self.surface_previewer)
        app.mainloop()
