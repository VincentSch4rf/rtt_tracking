from yolov5 import YOLOv5

from squeezedet.image_classifier import ImageClassifier


class YoloDetector(ImageClassifier):

    def __init__(self, **kwargs):
        super(YoloDetector, self).__init__(**kwargs)
        self.model = YOLOv5(self.checkpoint_path, device='cpu', agnostic=True)

    def classify(self, image):
        preds = self.model.predict(image).pred[0]
        bboxes = preds[:, 0:4]
        probs = preds[:, 4:5]
        labels = preds[:, 5:]
        return bboxes.numpy(), probs.numpy(), labels.numpy()