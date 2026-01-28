from selenium import webdriver
import time
import os
import datetime
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for server
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
def capture_mosdac_image(output_dir="static/screenshots"):
    # Ensure absolute path for robustness
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(os.getcwd(), output_dir)
        
    os.makedirs(output_dir, exist_ok=True)

    options = webdriver.ChromeOptions()
    # options.add_argument("--headless") # Uncomment if you want it invisible
    
    driver = webdriver.Chrome(options=options)

    try:
        driver.get("https://mosdac.gov.in/gallery/index.html?&prod=3RIMG_%27*_L1C_SGP_3D_IR1_V%27*.jpg&date=2025-07-07&count=60")
        time.sleep(20)  # Wait for page to load

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mosdac_live_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        
        driver.save_screenshot(filepath)
        print(f"[✅] Screenshot saved: {filepath}")
        
        # Return relative path for web serving
        return filename, filepath

    finally:
        driver.quit()

# -----------------------------
# 2. Analyze Image for TCC
# -----------------------------
def analyze_tcc(image_filename, image_path, static_dir="static"):
    print(f"\n[📊] Analyzing image: {image_path}")
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise FileNotFoundError("Could not read image or image is invalid.")

    # Map pixel to brightness temperature
    Tb = Tb_min_k + (img_gray / 255.0) * (Tb_max_k - Tb_min_k)
    mask = Tb < Tb_threshold
    tcc_pixels = Tb[mask]

    results = {}

    if tcc_pixels.size == 0:
        results["error"] = "No TCC pixels found below threshold."
        return results

    # TCC Metrics
    mean_tb = np.mean(tcc_pixels)
    min_tb = np.min(tcc_pixels)
    max_tb = np.max(tcc_pixels)
    median_tb = np.median(tcc_pixels)
    std_tb = np.std(tcc_pixels)
    pixel_count = int(np.sum(mask))

    # Contour detection
    _, binary_mask = cv2.threshold(img_gray, ((Tb_threshold - Tb_min_k)/(Tb_max_k - Tb_min_k)) * 255, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        results["error"] = "No contours found."
        return results

    largest_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest_contour)
    if M["m00"] != 0:
        cx = int(M["m10"]/M["m00"])
        cy = int(M["m01"]/M["m00"])
    else:
        cx, cy = 0, 0

    # Radius metrics
    distances = [np.linalg.norm(np.array([cx, cy]) - pt[0]) for pt in largest_contour]
    min_radius = np.min(distances)
    max_radius = np.max(distances)
    mean_radius = np.mean(distances)

    def tb_to_height_km(tb_k):
        return (290 - tb_k) * 0.15  # lapse rate approx. 6.5 K/km

    max_height = tb_to_height_km(min_tb)
    mean_height = tb_to_height_km(mean_tb)

    # Prepare results dictionary
    results = {
        "pixel_count": pixel_count,
        "mean_tb": round(mean_tb, 2),
        "min_tb": round(min_tb, 2),
        "max_tb": round(max_tb, 2),
        "median_tb": round(median_tb, 2),
        "std_tb": round(std_tb, 2),
        "center_x": cx,
        "center_y": cy,
        "min_radius": round(min_radius, 2),
        "max_radius": round(max_radius, 2),
        "mean_radius": round(mean_radius, 2),
        "max_cloud_height": round(max_height, 2),
        "mean_cloud_height": round(mean_height, 2),
        "original_image": f"screenshots/{image_filename}"
    }

    # Generate Heatmap Image
    plt.figure(figsize=(8, 6))
    plt.imshow(Tb, cmap="inferno")
    plt.colorbar(label="Brightness Temperature (K)")
    plt.scatter(cx, cy, c="cyan", s=50, label="Convective Center")
    plt.title("Tropical Cloud Cluster Detection")
    plt.legend()
    plt.tight_layout()
    
    heatmap_filename = f"heatmap_{image_filename}"
    heatmap_path = os.path.join(static_dir, "heatmaps", heatmap_filename)
    os.makedirs(os.path.dirname(heatmap_path), exist_ok=True)
    
    plt.savefig(heatmap_path)
    plt.close() # Close to free memory
    
    results["heatmap_image"] = f"heatmaps/{heatmap_filename}"

    return results
