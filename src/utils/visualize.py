""" Visualization functions """
import io
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.pyplot import colormaps

from api.app.models import ResultModel


def show(image: np.ndarray, figsize: tuple[int, int] = (15, 7)) -> None:
    """Show the np.ndarray object """

    plt.figure(figsize=figsize)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()


def result2show(image, result: ResultModel, cmap=colormaps['summer']) -> None:
    """
    Plot the original image and an empty white image for overlaying boxes and text
    A dictionary with the following keys and value types:
        - 'rec_texts': List of strings
        - 'rec_scores': List of floats
        - 'det_polygons': List of tuples [x1 y1 x2 y2 x3 y3 x4 y4] with four floats each
        - 'det_scores': List of floats

    """
    # plot image
    _, axis = plt.subplots(1, 2, figsize=(16, 16))

    axis[0].imshow(image, cmap='gray')
    axis[0].axis('off')

    # plot white image nearby
    white_img = np.ones_like(image, dtype=np.uint8)
    white_img.fill(255)
    axis[1].imshow(white_img, cmap='gray', vmin=0, vmax=255)
    axis[1].axis('off')
    # Iterate through the result dictionary
    for i in range(len(result.rec_texts)):
        rec_texts = result.rec_texts[i]
        rec_scores = result.rec_scores[i]
        det_polygons = result.det_polygons[i]
        det_scores = result.det_scores[i]

        x_coord = [int(coord) for coord in det_polygons[0::2]]
        y_coord = [int(coord) for coord in det_polygons[1::2]]

        rect0 = patches.Polygon(
            xy=list(zip(x_coord, y_coord)),
            closed=True,
            fill=False,
            edgecolor=cmap(1 - det_scores),
            lw=1)

        rect1 = patches.Polygon(
            xy=list(zip(x_coord, y_coord)),
            closed=True,
            fill=True,
            facecolor=cmap(1 - rec_scores / 2.),
            edgecolor='green',
            lw=1)

        axis[0].add_patch(rect0)
        axis[1].add_patch(rect1)
        axis[1].text(x_coord[0],
                     np.mean(y_coord),
                     rec_texts.replace('$', r'\$'),
                     color='black',
                     fontsize=12,
                     va='center')

    plt.tight_layout()

def draw_bounding_boxes(image, annotations):
    """
    Draw bounding boxes and recognized text on the image.

    :param image: Input image in OpenCV format.
    :param annotations: A dictionary with the following keys:
        - 'rec_texts': List of strings with recognized texts.
        - 'rec_scores': List of floats with recognition scores.
        - 'det_polygons': List of tuples, each containing 8 floats [x1 y1 x2 y2 x3 y3 x4 y4] representing the coordinates of the detection polygon.
        - 'det_scores': List of floats with detection scores.
    :return: Annotated image.
    """
    rec_texts = annotations.rec_texts
    rec_scores = annotations.rec_scores
    det_polygons = annotations.det_polygons
    det_scores = annotations.det_scores

    for i, polygon in enumerate(det_polygons):
        # Convert the flat list of coordinates to a list of tuples
        pts = [(int(polygon[j]), int(polygon[j + 1])) for j in range(0, len(polygon), 2)]
        pts = np.array(pts, dtype=np.int32)
        
        # Draw the polygon (bounding box)
        cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        
        # Prepare text to display
        if i < len(rec_texts) and i < len(rec_scores):
            text = f'{rec_texts[i]} ({rec_scores[i]:.2f})'
        else:
            text = 'Unknown'
        
        # Put the text on the image
        # Get the position for the text
        text_x, text_y = pts[0][0], pts[0][1] - 10
        # Draw a filled rectangle as the background for the text for better visibility
        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 0.5, 1)
        cv2.rectangle(image, (text_x, text_y - text_h - baseline), (text_x + text_w, text_y), (0, 255, 0), -1)
        # Put the text over the rectangle
        cv2.putText(image, text, (text_x, text_y - baseline), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)
        
        # If you also want to display the detection score
        if i < len(det_scores):
            det_text = f'Det: {det_scores[i]:.2f}'
            det_text_x, det_text_y = pts[3][0], pts[3][1] + 20
            cv2.putText(image, det_text, (det_text_x, det_text_y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)

    return image
