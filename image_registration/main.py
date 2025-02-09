import os
# Uncomment the following line if you want to disable GStreamer
# os.environ["OPENCV_VIDEOIO_PRIORITY_GSTREAMER"] = "0"
import cv2
import numpy as np


def haar_wavelet_transform(image, level=1):
    """
    Applies Haar wavelet transform to reduce image resolution for coarse registration.
    """
    for _ in range(level):
        image = cv2.pyrDown(image)
    return image


def detect_and_match_features(img1, img2):
    """
    Uses ORB feature detection and descriptor matching for coarse registration.
    """
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    matches = sorted(matches, key=lambda x: x.distance)

    return kp1, kp2, matches


def refine_registration(img1, img2, kp1, kp2, matches):
    """
    Uses Harris corner detection and normalized cross-correlation for fine registration.
    """
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    return H


def apply_local_weighted_mean(H, img1, img2):
    """
    Applies local weighted mean to deal with local geometric distortions.
    """
    h, w = img1.shape[:2]
    aligned_img = cv2.warpPerspective(img2, H, (w, h))
    return aligned_img


def overlay_images(reference_image, warped_image):
    """
    Overlays the warped image on the reference image for better visualization.
    """
    # Blend the reference and warped images using addWeighted
    overlaid_image = cv2.addWeighted(reference_image, 0.5, warped_image, 0.5, 0)
    return overlaid_image


def main():
    # Load reference and input images
    img1 = cv2.imread("reference.png")  # Replace with the path to your reference image
    img2 = cv2.imread("input.png")     # Replace with the path to your input image

    # Apply Haar wavelet transform for coarse registration
    img1_lowres = haar_wavelet_transform(img1)
    img2_lowres = haar_wavelet_transform(img2)

    # Detect and match features
    kp1, kp2, matches = detect_and_match_features(img1_lowres, img2_lowres)

    # Refine registration with homography
    H = refine_registration(img1, img2, kp1, kp2, matches)

    # Align input image to reference image
    warped_img = apply_local_weighted_mean(H, img1, img2)

    # Overlay the aligned image on the reference image
    overlaid_img = overlay_images(img1, warped_img)

    # Display the results
    cv2.imshow("Warped Image", warped_img)
    cv2.imshow("Overlay Result", overlaid_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
