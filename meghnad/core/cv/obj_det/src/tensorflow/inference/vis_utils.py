import cv2
from utils.common_defs import class_header, method_header


@class_header(
    description='''
    Class for colors and boxes''')
class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()


@method_header(
    description='''
        Function to draw bounding boxes to the images''',
        arguments='''
        image: image : input image to the function
        bboxes: pass bboxes to the function to draw bboxes around the objects
        classes: number and type of classes to the function
        scores: show scores on the top of bboxes
        class_map: provides the class_labels and the keys
        ''',
        returns='''
        returns an image_clone as an image''')
def draw_bboxes(image, bboxes, classes, scores, class_map=None):
    image_clone = image.copy()
    for bbox, cls, score in zip(bboxes, classes, scores):
        x1, y1, x2, y2 = list(map(int, bbox))
        color = colors(cls)
        cv2.rectangle(image_clone, (x1, y1), (x2, y2), color, 2)
        if class_map:
            class_name = class_map[cls]
            text = f'{class_name}: {score:.2f}'
        else:
            text = f'{cls}: {score:.2f}'
        cv2.putText(image_clone, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    1., color, 2, cv2.LINE_AA)
    return image_clone
