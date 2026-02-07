import cv2
from tkinter import Tk
from tkinter.filedialog import askopenfilename


# Iteration 1
def grey_scale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# Iteration 3
def distortion_correction(img):
    return img


# Iteration 2
def normalise_contrast(img):
    return cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)


# Iteration 1
def noise_reduction(img):
    return cv2.fastNlMeansDenoising(img, None, h=10, templateWindowSize=13, searchWindowSize=21)


# Iteration 3
def deskew(img):
    return img


# Iteration 2
def binarize(img):
    return img


def resize_for_screen(img, max_width=800, max_height=800):
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    return cv2.resize(img, (int(w * scale), int(h * scale)))


def preproc_image(img):
    img = grey_scale(img)
    img = distortion_correction(img)
    img = normalise_contrast(img)
    img = noise_reduction(img)
    img = deskew(img)
    img = binarize(img)
    return img


if __name__ == '__main__':
    # Hide Tkinter root window
    Tk().withdraw()

    # Open file dialog
    path = askopenfilename(
        title="Select Image",
        filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")]
    )

    if not path:
        print("No image selected.")
        exit()

    # Read image
    original = cv2.imread(path)

    if original is None:
        print("Failed to load image.")
        exit()

    # Run preprocessing pipeline
    processed = resize_for_screen(preproc_image(original))
    normalised = grey_scale(original)

    # Display results
    cv2.imshow("Original", resize_for_screen(original))
    cv2.imshow("Intermediate", resize_for_screen(normalise_contrast(normalised)))
    cv2.imshow("Processed", processed)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
