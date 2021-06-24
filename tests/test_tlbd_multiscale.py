import numpy as np
import cv2
import pytlbd


def draw_multiscale_matches(img_left, img_right, segs_left, segs_right, matches):
    assert img_left.ndim == 2
    h, w = img_left.shape

    # store the matching results of the first and second images into a single image
    left_color_img = cv2.cvtColor(img_left, cv2.COLOR_GRAY2BGR)
    right_color_img = cv2.cvtColor(img_right, cv2.COLOR_GRAY2BGR)
    r1, g1, b1 = [], [], []  # the line colors

    for pair in range(len(matches)):
        r1.append(int(255 * np.random.rand()))
        g1.append(int(255 * np.random.rand()))
        b1.append(255 - r1[-1])
        line_id_l, line_id_r = matches[pair]

        octave, l = segs_left[line_id_l][0]
        l = l * np.sqrt(2) ** octave
        cv2.line(left_color_img, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), (r1[pair], g1[pair], b1[pair]), 3)

        octave, l = segs_right[line_id_r][0]
        l = l * np.sqrt(2) ** octave
        cv2.line(right_color_img, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), (r1[pair], g1[pair], b1[pair]), 3)

    result_img = np.hstack([left_color_img, right_color_img])
    result_img_tmp = result_img.copy()
    for pair in range(27, len(matches)):
        line_id_l, line_id_r = matches[pair]
        octave_left, seg_left = segs_left[line_id_l][0]
        octave_right, seg_right = segs_right[line_id_r][0]
        seg_left = seg_left[:4] * (np.sqrt(2) ** octave_left)
        seg_right = seg_right[:4] * (np.sqrt(2) ** octave_right)

        start_ptn = (int(seg_left[0]), int(seg_left[1]))
        end_ptn = (int(seg_right[0] + w), int(seg_right[1]))
        cv2.line(result_img_tmp, start_ptn, end_ptn, (r1[pair], g1[pair], b1[pair]), 2, cv2.LINE_AA)

    result_img = cv2.addWeighted(result_img, 0.5, result_img_tmp, 0.5, 0.0)

    return result_img


# read both images
imgLeft = cv2.imread('resources/boat1.jpg', cv2.IMREAD_GRAYSCALE)
imgRight = cv2.imread('resources/boat3.jpg', cv2.IMREAD_GRAYSCALE)

# Detect segments
multiscaleL = pytlbd.edlines_multiscale(imgLeft)
multiscaleR = pytlbd.edlines_multiscale(imgRight)

# Compute multi-scale descriptors
descriptorsL = pytlbd.lbd_multiscale(imgLeft, multiscaleL, 9, 7)
descriptorsR = pytlbd.lbd_multiscale(imgRight, multiscaleR, 9, 7)

# Find matches using the heuristic approach defined in the paper
matching_result = pytlbd.lbd_matching_multiscale(multiscaleL, multiscaleR, descriptorsL, descriptorsR)
print(f'Found {len(matching_result)} matches.')

# Draw the resulting matches
matchesImage = draw_multiscale_matches(imgLeft, imgRight, multiscaleL, multiscaleR, matching_result)
cv2.imshow("Matches found", matchesImage)
cv2.waitKey()
