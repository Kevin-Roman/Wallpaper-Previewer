import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

from .gui import MainMenu

if __name__ == "__main__":
    app = MainMenu()
    app.mainloop()
