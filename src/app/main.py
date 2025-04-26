import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

from src.app.gui import MainMenu

if __name__ == "__main__":
    app = MainMenu()
    app.mainloop()
