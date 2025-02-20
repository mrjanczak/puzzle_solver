import cv2
import numpy as np

# Load the image
image = cv2.imread("image.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding
_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if contours:
    # Get the largest contour
    contour = max(contours, key=cv2.contourArea)

    # Approximate the contour
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx_contour = cv2.approxPolyDP(contour, epsilon, True)

    # Initialize counts
    line_points = 0
    non_line_points = 0
    total_points = len(approx_contour)

    # Define lambda (λ) for line classification
    λ = 1.2  # Adjusts classification sensitivity

    # Define gamma (γ) forgetting factor
    γ = 0.01  # Higher γ means faster decay of older points

    # Compute cumulative distances for weighting
    distances = np.zeros(total_points)
    for i in range(1, total_points):
        distances[i] = distances[i - 1] + np.linalg.norm(approx_contour[i][0] - approx_contour[i - 1][0])

    # Iterate over contour points
    weighted_sum = 0
    weight_total = 0

    for i in range(1, total_points - 1):  # Avoid first & last points
        p1 = approx_contour[i - 1][0]  # Previous point
        p2 = approx_contour[i][0]      # Current point
        p3 = approx_contour[i + 1][0]  # Next point

        # Compute vectors
        v1 = p2 - p1
        v2 = p3 - p2

        # Compute cosine of angle using dot product
        dot_product = np.dot(v1, v2)
        mag_v1 = np.linalg.norm(v1)
        mag_v2 = np.linalg.norm(v2)

        if mag_v1 > 0 and mag_v2 > 0:  # Avoid division by zero
            cos_angle = dot_product / (mag_v1 * mag_v2)
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))  # Convert to radians
            angle_degrees = np.degrees(angle)

            # Define "line points" as those with angles close to 180°, adjusted by λ
            threshold_angle = 150 * λ  # Lambda-modified threshold

            # Compute forgetting weight using gamma
            weight = np.exp(-γ * distances[i])  # Older points get lower weight

            if angle_degrees > threshold_angle:
                line_points += weight
            else:
                non_line_points += weight

            weighted_sum += weight
            weight_total += weight

    # Compute weighted ratio
    ratio = (line_points / weight_total) if weight_total > 0 else 0

    print(f"Lambda (λ): {λ}")
    print(f"Gamma (γ): {γ}")
    print(f"Weighted Line Points: {line_points:.2f}, Weighted Non-Line Points: {non_line_points:.2f}")
    print(f"Weighted Line-to-Total Ratio: {ratio:.2f}")

    # Draw contour for visualization
    cv2.drawContours(image, [approx_contour], -1, (0, 255, 0), 2)
    cv2.imshow("Contour", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
