from selenium import webdriver
import time
import os
import datetime
import cv2
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Configuration
# -----------------------------
Tb_min_k = 190
Tb_max_k = 290
Tb_threshold = 240  # IRBT threshold in Kelvin

# -----------------------------
# 1. Capture Image from MOSDAC
# -----------------------------
def capture_mosdac_image(output_dir="screenshots"):
    os.makedirs(output_dir, exist_ok=True)

    driver = webdriver.Chrome()  # Open normal browser (not headless for debugging)

    try:
        driver.get("https://mosdac.gov.in/gallery/index.html?&prod=3RIMG_%27*_L1C_SGP_3D_IR1_V%27*.jpg&date=2025-07-07&count=60")
        time.sleep(20)  # Wait for page to load

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f"mosdac_live_debug_{timestamp}.png")
        driver.save_screenshot(filename)
        print(f"[✅] Screenshot saved: {filename}")
        return filename

    finally:
        driver.quit()

# -----------------------------
# 2. Analyze Image for TCC
# -----------------------------
def analyze_tcc(image_path):
    print(f"\n[📊] Analyzing image: {image_path}")
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise FileNotFoundError("Could not read image or image is invalid.")

    # Map pixel to brightness temperature
    Tb = Tb_min_k + (img_gray / 255.0) * (Tb_max_k - Tb_min_k)
    mask = Tb < Tb_threshold
    tcc_pixels = Tb[mask]

    if tcc_pixels.size == 0:
        print("❌ No TCC pixels found below threshold.")
        return

    # TCC Metrics
    mean_tb = np.mean(tcc_pixels)
    min_tb = np.min(tcc_pixels)
    max_tb = np.max(tcc_pixels)
    median_tb = np.median(tcc_pixels)
    std_tb = np.std(tcc_pixels)
    pixel_count = np.sum(mask)

    # Contour detection
    _, binary_mask = cv2.threshold(img_gray, ((Tb_threshold - Tb_min_k)/(Tb_max_k - Tb_min_k)) * 255, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("❌ No contours found.")
        return

    largest_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest_contour)
    cx = int(M["m10"]/M["m00"])
    cy = int(M["m01"]/M["m00"])

    # Radius metrics
    distances = [np.linalg.norm(np.array([cx, cy]) - pt[0]) for pt in largest_contour]
    min_radius = np.min(distances)
    max_radius = np.max(distances)
    mean_radius = np.mean(distances)

    def tb_to_height_km(tb_k):
        return (290 - tb_k) * 0.15  # lapse rate approx. 6.5 K/km

    max_height = tb_to_height_km(min_tb)
    mean_height = tb_to_height_km(mean_tb)

    # Display results
    print("🌩️ TCC Metrics:")
    print(f"Pixel Count: {pixel_count}")
    print(f"Mean Tb: {mean_tb:.2f} K")
    print(f"Min Tb: {min_tb:.2f} K")
    print(f"Max Tb: {max_tb:.2f} K")
    print(f"Median Tb: {median_tb:.2f} K")
    print(f"Std Dev Tb: {std_tb:.2f} K")
    print(f"Convective Center: (x={cx}, y={cy})")
    print(f"Min Radius: {min_radius:.2f} px")
    print(f"Max Radius: {max_radius:.2f} px")
    print(f"Mean Radius: {mean_radius:.2f} px")
    print(f"Max Cloud-Top Height: {max_height:.2f} km")
    print(f"Mean Cloud-Top Height: {mean_height:.2f} km")

    # Visual
    plt.figure(figsize=(8, 6))
    plt.imshow(Tb, cmap="inferno")
    plt.colorbar(label="Brightness Temperature (K)")
    plt.scatter(cx, cy, c="cyan", s=50, label="Convective Center")
    plt.title("Tropical Cloud Cluster Detection")
    plt.legend()
    plt.tight_layout()
    plt.show()

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    screenshot = capture_mosdac_image()
    analyze_tcc(screenshot)
