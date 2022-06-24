# ===================================================
# ==================+++ Imports +++==================
# ===================================================

import numpy as np
from ImageProcessingMethods import grayscale, invert, binarize, segment_to_lines, segment_to_words, segment_to_chars, repairShapeOfCharacter, add_borders
from autocorrect import Speller

# TensorFlow ≥2.0 is required
import tensorflow as tf
assert tf.__version__ >= "2.0"

# to make this notebook's output stable across runs
np.random.seed(42)
tf.random.set_seed(42)

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# ==================================================
# =============+++ Helper Functions +++=============
# ==================================================

def displayImage(image): #Gray_Image with one Channel
    dpi = 80
    height, width = image.shape
    # What Size does the figure need to be in inches to fit the image
    figSize = width/float(dpi), height/float(dpi)

    fig = plt.figure(figsize=figSize)
    ax = fig.add_axes([0,0,1,1])
    ax.axis('off')

    ax.imshow(image, cmap="gray")
    plt.show()


# ===================================================
# =============+++ Decoding Function +++=============
# ===================================================

def decodeLabel(value):
    if value <= 9:
        return(chr(value+48))
    elif value > 9 and value < 36:
        return(chr(value+55))
    elif value > 35 and value < 62:
        return(chr(value+61))
    elif value > 61 and value < 72:
        turkishDecoder = {
            62:286 , #Ğ
            63:304 , #İ
            64:214 , #Ö
            65:220 , #Ü
            66:231 , #ç
            67:287 , #ğ
            68:305 , #ı
            69:246 , #ö
            70:351 , #ş
            71:252  #ü
        }
        return (chr(turkishDecoder.get(value)))
    # elif value > 71:
    #     # Labels for punctuation marks
    #     return 0


# ======================================================
# =============+++ Recognition Function +++=============
# ======================================================

'''
verbose:
    0 ==> Outputs nothing
    1 ==> Outputs only segmented Lines
    2 ==> Outputs only segmented Words
    3 ==> Outputs only segmented Characters
    4 ==> Outputs segmented Lines, Words and Characters
    5 ==> Outputs all preprocessing steps (step by step)
'''
def segmentTextToChars(image, image_dimension=28, verbose=0):
    image_grayscale = grayscale(image)
    image_invert = invert(image_grayscale)
    image_binary = binarize(image_invert)
    # image = noise_removal(image_binary)
    image = image_binary.copy()
    if(verbose > 4):
        print("Grayscaling ...")
        displayImage(image_grayscale)
        print("Inverting ...")
        displayImage(image_invert)
        print("Binarizing (Thresholding) ...")
        displayImage(image_binary)
        # print("Reducing Noise ...")
        # displayImage(image)
    if verbose == 1 or verbose >= 4:
        print("================LINES===========================")
    line_segmented_img, linesAsImages = segment_to_lines(image, verbose=verbose)
    if verbose == 1 or verbose >= 4:
        displayImage(line_segmented_img)
    borderedLines = []
    for lineImage in linesAsImages:
        borderedLine = add_borders(lineImage, 5)
        if verbose == 1 or verbose >= 4:
            displayImage(borderedLine)
        borderedLines.append(borderedLine)

    if verbose == 2 or verbose >= 4:
        print("==============WORDS=============================")

    wordsAsImages = segment_to_words(borderedLines.copy(), verbose=verbose)
    for wordImage in wordsAsImages:
        if verbose == 2 or verbose >= 4:
            displayImage(wordImage)

    if verbose >= 3:
        print("======================LETTERS=====================")
    charsAsImages = segment_to_chars(wordsAsImages.copy(), verbose=verbose)
    repairedWords = []
    for word in charsAsImages:
        repairedChars = []
        for charImage in word:
            borderedChar = add_borders(charImage, 10)
            repaired_char = repairShapeOfCharacter(borderedChar)
            if verbose >= 3:
                displayImage(repaired_char)
            repairedChars.append(repaired_char)
        if verbose >= 3:
            print("===================================")
        repairedWords.append(repairedChars)
    flat_chars = np.concatenate(repairedWords)
    if verbose >= 3:
        print("Number of recognized Characters is ", len(flat_chars))
        print("===========================================")
    return repairedWords


# =====================================================
# =============+++ Prediction Function +++=============
# =====================================================

def predictText(charactersSeparatedInWords, model, show_top_k=False, k=3):
    text = ""
    probabilities = []
    for word in charactersSeparatedInWords:
        for charImage in word:
            # print(charImage.shape) #  (28, 28)
            image = charImage[np.newaxis, :, :, np.newaxis]
            # print(image.shape) #  (1, 28, 28, 1)
            Y_probas = model.predict(image, verbose=0)
            y_pred = np.argmax(Y_probas, axis=-1)
            probabilities.append(Y_probas[0][y_pred.item(0)])
            y_pred_value = decodeLabel(y_pred.item(0))
            if show_top_k == True:
                print("=======================================")
                displayImage(charImage)
                top_k = tf.nn.top_k(Y_probas, k=k)
                for k_index in range(k):
                    class_name = decodeLabel(int(top_k.indices[0, k_index]))
                    proba = 100 * top_k.values[0, k_index]
                    print("  {}. {} {:.3f}%".format(k_index + 1, class_name, proba))
                print("The predicted value is ", y_pred_value)
                print("=======================================")
            text = text + y_pred_value
        text = text + " "
        overall_probability = sum(probabilities)
        overall_probability = overall_probability / len(probabilities)
    return text, overall_probability


# =======================================================
# ============+++ Postprocessing Function +++============
# =======================================================

def autoCorrectText(text):
    spell = Speller(only_replacements=True, lang="tr")
    return spell(text)