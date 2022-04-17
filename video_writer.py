import logging
import os
import re
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
from inotify_simple import INotify, flags

logging.basicConfig(level=logging.INFO)

class VideoWriterMp4:

    def __init__(self, name: str, fps: int, source: str, resolution: Optional[Tuple[int, int]] = None):
        self.LOGGER = logging.getLogger(self.__class__.__name__)
        self.LOGGER.setLevel(logging.INFO)
        self.name = Path(name).with_suffix('.mp4')
        self.resolution = resolution
        self._fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self._fps = fps
        self.source = source
        if self.resolution is None:
            self.__out = None
        else:
            self.__out = cv2.VideoWriter(self.name, self._fourcc, self._fps, self.resolution)

        self.__patterns = [r".*\.jpg", r".*\.jpeg"]
        self.__inotify = INotify()
        watch_flags = flags.CREATE
        self.watch_descriptor = self.__inotify.add_watch(self.source, watch_flags)
        self.handle_existing_content()

    def handle_existing_content(self):
        images = os.listdir(self.source)
        if len(images) == 0:
            return
        self.LOGGER.info(f"Found {len(images)} existing images. Writing...")
        for image in images:
            self.write(os.path.join(self.source, image))
        self.LOGGER.info("Done ✅")

    def write(self, image: str):
        img = cv2.imread(image)
        if self.__out is None:
            self.resolution = img.shape[:2][::-1]
            self.__out = cv2.VideoWriter(str(self.name.absolute()), self._fourcc, self._fps, self.resolution)
        self.__out.write(img)

    def on_created(self, event):
        for pattern in self.__patterns:
            if re.match(pattern, event.name):
                image = os.path.join(self.source, event.name)
                self.LOGGER.info(f"add {image} to {self.name}")
                time.sleep(0.2)
                self.write(image)
                break

    def run(self):
        self.LOGGER.info("Starting interactive mode...")
        try:
            while True:
                for event in self.__inotify.read(read_delay=1000):
                    for flag in flags.from_mask(event.mask):
                        if flag == flags.CREATE:
                            self.on_created(event)
        except KeyboardInterrupt:
            cv2.destroyAllWindows()
            if self.__out:
                self.__out.release()


if __name__ == "__main__":
    video = VideoWriterMp4("test.mp4", 30, "data/test")
    video.run()
