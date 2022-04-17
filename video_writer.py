import logging
import os
from pathlib import Path

import time
from typing import Optional, Tuple

from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler

import cv2
import numpy as np


class VideoWriterMp4:

    LOGGER = logging.getLogger(__name__)
    LOGGER.setLevel(logging.INFO)

    def __init__(self, name: str, fps: int, source: str, resolution: Optional[Tuple[int, int]] = None):
        self.name = Path(name).with_suffix('.mp4')
        self.resolution = resolution
        self._fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self._fps = fps
        if self.resolution is None:
            self.__out = None
        else:
            self.__out = cv2.VideoWriter(self.name, self._fourcc, self._fps, self.resolution)

        self.__patterns = ["*.jpg", "*.jpeg"]
        self.__ignore_patterns = None
        self.__event_handler = PatternMatchingEventHandler(self.__patterns, self.__ignore_patterns,
                                                           ignore_directories=True, case_sensitive=True)
        self.__init_event_handler()

        self.__observer = Observer()
        self.__observer.schedule(self.__event_handler, source, recursive=False)

    def __init_event_handler(self):
        self.__event_handler.on_created = self.on_created
        self.__event_handler.on_deleted = self.on_deleted
        self.__event_handler.on_modified = self.on_modified
        self.__event_handler.on_moved = self.on_moved

    def on_created(self, event):
        self.LOGGER.info(f"add {event.src_path} to {self.name}")
        time.sleep(0.2)
        img = cv2.imread(event.src_path)
        if self.__out is None:
            self.resolution = img.shape[:2][::-1]
            self.__out = cv2.VideoWriter(str(self.name.absolute()), self._fourcc, self._fps, self.resolution)
        self.__out.write(img)

    def on_deleted(self, event):
        pass

    def on_modified(self, event):
        pass

    def on_moved(self, event):
        pass

    def run(self):
        self.__observer.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.__observer.stop()
            self.__observer.join()
            cv2.destroyAllWindows()
            self.__out.release()


if __name__ == "__main__":
    video = VideoWriterMp4("test.mp4", 30, "data/test")
    video.run()