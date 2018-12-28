import cv2
from imutils import face_utils

def dlib_to_box(box, factor=1.0, faceBoxScale=0.0):
    # scales and moves the box back to the size of the image according
    # to factor, keeping the box's center while also allowing
    # additional scaling of the size according to faceBoxScale
    b = face_utils.rect_to_bb(box)
    [x1, y1, bW, bH] = b
    dW = bW * factor * faceBoxScale / 2
    dH = bH * factor * faceBoxScale / 2
    x2 = int((x1 + bW)*factor + dW)
    y2 = int((y1 + bH)*factor + dH)
    x1 = int(x1*factor - dW)
    y1 = int(y1*factor - dH)
    return [x1, y1, x2, y2]

def draw_box(image, box, color=(0,255,0)):
    [x1, y1, x2, y2] = box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)

def draw_boxes(image, boxes, box_color=(255, 255, 255)):
    for box in boxes:
        draw_box(image, box, box_color)

def move_box(box, offset):
    """Move the box to direction specified by vector offset"""
    left_x = box[0] + offset[0]
    top_y = box[1] + offset[1]
    right_x = box[2] + offset[0]
    bottom_y = box[3] + offset[1]
    return [left_x, top_y, right_x, bottom_y]

def get_square_box(box, imageShape):
    """Get a square box out of the given box, by expanding it."""
    left_x = box[0]
    top_y = box[1]
    right_x = box[2]
    bottom_y = box[3]

    box_width = right_x - left_x
    box_height = bottom_y - top_y

    # Check if box is already a square. If not, make it a square.
    diff = box_height - box_width
    delta = int(abs(diff) / 2)

    if diff == 0:                   # Already a square.
        return box
    elif diff > 0:                  # Height > width, a slim box.
        left_x -= delta
        right_x += delta
        if diff % 2 == 1:
            right_x += 1
    else:                           # Width > height, a short box.
        top_y -= delta
        bottom_y += delta
        if diff % 2 == 1:
            bottom_y += 1

    # Make sure box is always square.
    assert ((right_x - left_x) == (bottom_y - top_y)), 'Box is not square.'

    return [
        max(left_x, 0),
        max(top_y, 0),
        min(right_x, imageShape[1]),
        min(bottom_y, imageShape[0])
    ]

def box_in_image(box, image):
    """Check if the box is in image"""
    rows = image.shape[0]
    cols = image.shape[1]
    return box[0] >= 0 and box[1] >= 0 and box[2] <= cols and box[3] <= rows
