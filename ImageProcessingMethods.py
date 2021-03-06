import cv2
import numpy as np

# ===========================================================
# ===================+++ Preprocessing +++===================
# ===========================================================

def grayscale(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_img


def invert(image):
    inverted_img = cv2.bitwise_not(image)
    return inverted_img

def binarize(image, threshold=127, maxValue=255):
    threshold, img_thresholded = cv2.threshold(image, threshold, maxValue, cv2.THRESH_BINARY) # + cv2.THRESH_OTSU
    return img_thresholded

def add_borders(image, borderSize=20):
    color = [0, 0, 0]
    top, bottom, left, right = [borderSize]*4
    image_with_border = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return image_with_border

# ==========================================================
# ===================+++ Segmentation +++===================
# ==========================================================

def segment_to_lines(image, verbose=0):
    lines = []
    h, w = image.shape
    kernel = np.ones((int(h/40),w-10), dtype='uint8')
    dilated_image = cv2.dilate(image, kernel, iterations=1)
    contours, hierarchies = cv2.findContours(dilated_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_contour_lines = sorted(contours, key=lambda contour: cv2.boundingRect(contour)[1]) #(x,y,w,h)
    if verbose == 1 or verbose >= 4:
        print(len(sorted_contour_lines), " Lines Recognized")
    image_with_boxes = image.copy()
    for contour in sorted_contour_lines:
        if cv2.contourArea(contour) < 400:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        # if h < 10 or w < 10:
        #     continue

        line_image=image[y:y+h, x:x+w]
        lines.append(line_image)
        cv2.rectangle(image_with_boxes, (x, y), (x+w, y+h), (100,255,100), thickness=2)
    return image_with_boxes, lines

def segment_to_words(linesAsImages, verbose=0):
    allWords = []
    for idx, lineImage in enumerate(linesAsImages):
        h, w = lineImage.shape
        kernel_height=int(h*(1/4))
        kernel_width=int(w*(1/35))
        kernel = np.ones((int(kernel_height),int(kernel_width)), dtype='uint8')
        dilated_image = cv2.dilate(lineImage.copy(), kernel, iterations=1)
        contours, hierarchies = cv2.findContours(dilated_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        sorted_contour_words = sorted(contours, key=lambda contour: cv2.boundingRect(contour)[0]) #(x,y,w,h)
        if verbose == 2 or verbose >= 4:
            print("{} Words in {}. line Recognized".format(len(sorted_contour_words), (idx+1)))
        for contour_word in sorted_contour_words:
            if cv2.contourArea(contour_word) < 80:
                continue

            x, y, w, h = cv2.boundingRect(contour_word)
            word_image=lineImage[y:y+h, x:x+w]
            allWords.append(word_image)
    return allWords

def segment_to_chars(wordsAsImages, verbose=0):
    def rescaleFrame(frame, scale=1.5):
        width = int(frame.shape[1] * scale)
        height = int(frame.shape[0] * scale)
        dimensions = (width, height)

        return cv2.resize(frame, dimensions, interpolation=cv2.INTER_CUBIC)
        # Inter_Area is useful if we are shrinking the image to dimensions that are smaller than the original dimensions.
        # If we enlarge the image we probably can use INTER_LINEAR or INTER_CUBIC

    allCharactersAsWords = []
    for idx, wordImage in enumerate(wordsAsImages):
        allCharacters = []
        image = rescaleFrame(wordImage, 3)
        height, width = image.shape
        img_canny = cv2.Canny(image, 125, 200)
        contours, hierarchies = cv2.findContours(img_canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sorted_contour_lines = sorted(contours, key=lambda contour: cv2.boundingRect(contour)[0]) #(x,y,w,h)
        sorted_contour_lines_areas = sorted(contours, key=lambda contour: cv2.contourArea(contour))
        max_area = cv2.contourArea(sorted_contour_lines_areas[-1])
        for contour in sorted_contour_lines:
            x, y, w, h = cv2.boundingRect(contour)
            # Conditions to get rid of points recognized as letters as possible as we can
            if (y+h) < (int(height/100)*38): #(e.g. E????T??M)
                continue
            elif ((y+h) < (int(height/100)*48)) and (cv2.contourArea(contour) < (max_area*0.2)): #(e.g. diyor)
                continue
            elif ((y+h) < (int(height/100)*80)) and (cv2.contourArea(contour) < (max_area*0.07)): #(e.g. din)
                continue
            # To get all components of Multi-Component-Letters (e.g. i, j, ??, ??, ??, etc.)
            char_image=image[0:y+h, x:x+w]
            img_canny_2 = cv2.Canny(char_image.copy(), 125, 200)
            contours_2, hierarchies_2 = cv2.findContours(img_canny_2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours_2) > 1 :
                sorted_contour_lines_2 = sorted(contours_2, key=lambda contour: cv2.boundingRect(contour)[1]) #(x,y,w,h)
                highest_y = np.inf
                for contour_2 in sorted_contour_lines_2:
                    x2, y2, w2, h2 = cv2.boundingRect(contour_2)
                    char_image_temp=char_image[y2:y2+h2, x2:x2+w2]
                    highest_y = min(highest_y, y2)
                char_image_2=image[highest_y:y+h, x:x+w]
                allCharacters.append(char_image_2)
            else :
                char_image=image[y:y+h, x:x+w]
                allCharacters.append(char_image)
        if verbose >= 3:
            print("{} Characters in {}. word Recognized".format(len(allCharacters), (idx+1)))
        allCharactersAsWords.append(allCharacters)
    return allCharactersAsWords

# ==================================================================
# ===================+++ Feeding To The Model +++===================
# ==================================================================

def rescaleForModel(image, img_dimension=28):
    height, width = image.shape
    if height > width:
        if height > img_dimension:
            new_height = img_dimension-2 # -4 as margin
            aspect_ratio = height / width
            new_width = int(new_height / aspect_ratio)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)    #Resize To Smaller Image
            return image
        else:
            new_height = img_dimension-2 # -4 as margin
            aspect_ratio = height / width
            new_width = int(new_height / aspect_ratio)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)   #Resize To Bigger Image
            return image
    else:
        if width > img_dimension:
            new_width = img_dimension-2 # -4 as margin
            aspect_ratio = width / height
            new_height = int(new_width / aspect_ratio)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)    #Resize To Smaller Image
            return image
        else:
            new_width = img_dimension-2 # -4 as margin
            aspect_ratio = width / height
            new_height = int(new_width / aspect_ratio)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)   #Resize To Bigger Image
            return image

def addPadding(image, img_dimension=28):
    height, width = image.shape
    if height % 2 != 0:
        top = int((img_dimension-height)/2)+1
        bottom = int((img_dimension-height)/2)
        if width % 2 != 0:
            left = int((img_dimension-width)/2)+1
            right = int((img_dimension-width)/2)
            image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT,value=0)
            return image
        else:
            left = int((img_dimension-width)/2)
            right = int((img_dimension-width)/2)
            image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT,value=0)
            return image
    else:
        top = int((img_dimension-height)/2)
        bottom = int((img_dimension-height)/2)
        if width % 2 != 0:
            left = int((img_dimension-width)/2)+1
            right = int((img_dimension-width)/2)
            image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT,value=0)
            return image
        else:
            left = int((img_dimension-width)/2)
            right = int((img_dimension-width)/2)
            image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT,value=0)
            return image

def repairShapeOfCharacter(image, img_dimension=28):
    rescaled_char = rescaleForModel(image, img_dimension)
    padded_char = addPadding(rescaled_char, img_dimension)
    return padded_char