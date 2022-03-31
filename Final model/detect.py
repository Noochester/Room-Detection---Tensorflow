import os
import cv2
import numpy as np
import argparse
import tensorflow as tf
import sys

colors_main = [
    [0, 200, 0],    # green
    [0, 0, 0],      # black
    [255, 129, 0],  # orange
    [98, 31, 220],  # pink
    [0, 0, 255],    # blue
    [42, 229, 240]  # yellow
]


def parse_args(args):
    parser = argparse.ArgumentParser("Object detection model")
    parser.add_argument('--model-dir')
    parser.add_argument('--class-names')
    parser.add_argument('--score-threshold', default=0.2, type=float)
    parser.add_argument('--pic-dir')
    return parser.parse_args(args)


def detect_batch_img(img, model):
    boxes, scores, classes, null = model(tf.convert_to_tensor(img, dtype=tf.uint8))
    return boxes, scores, classes, null


def plot_one_box(img, c1, c2, color=None, obj_type=None, label=None):
    """plot one box on image."""
    if obj_type == 'pictureOrTV':
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
    else:
        cv2.rectangle(img, c1, c2, color, thickness=6, lineType=cv2.LINE_AA)  # regular box
    tl = round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    cv2.putText(img, label, (c1[0], c1[1] - 3), 0, fontScale=tl / 5, color=[225, 255, 255],
                thickness=tl - 1, lineType=cv2.LINE_AA)


def plot_boxes(img, boxes_to_draw, boxes, scores, classes, class_names):
    """plot all boxes on image."""

    boxesList = boxes.tolist()
    print("\n"*3)
    print("New image, Resolution is {}/{}".format(img.shape[1], img.shape[0]))
    for i in range(boxes_to_draw):

        x1y1 = (int(boxesList[i][1]), int(boxesList[i][0]))
        x2y2 = (int(boxesList[i][3]), int(boxesList[i][2]))

        obj_type = class_names[int(classes[i])]
        # Gives coordinates, scores and class name
        print("Box dimensions are", x1y1, "and", x2y2, "Score is", int(scores[i]*100),"%,", obj_type)

        plot_one_box(img, x1y1, x2y2, colors_main[int(classes[i])], obj_type=obj_type,
                     label=str("%s:%0.2f" % (obj_type, scores[i])))

    return img


def main(args):
    counter = 0  # for image naming
    boxes_to_detect = 5  # how much boxes to draw
    # load model
    model = tf.saved_model.load(args.model_dir)
    # read class labels
    with open(args.class_names) as f:
        class_names = f.read().splitlines()
    img_list = os.listdir(args.pic_dir)

    for img_name in img_list:
        img = cv2.imread(os.path.join(args.pic_dir, img_name))
        img_copy = img.copy()
        expanded = [img_copy]
        # predict model
        boxes, scores, classes, null = detect_batch_img(expanded, model)
        boxes = boxes.numpy()
        scores = scores.numpy()
        classes = classes.numpy()

        # Visualize results

        plot_boxes(img_copy, boxes_to_detect, boxes[0], scores[0], classes[0], class_names)
        cv2.imwrite("images/evald/" + str(counter) + ".jpg", img_copy)
        counter = counter + 1
        # cv2.waitKey(0)


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    main(args)
