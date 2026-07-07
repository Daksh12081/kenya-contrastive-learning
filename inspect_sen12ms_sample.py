import rasterio

s2_path = "datasets/SEN12MS/ROIs1158_spring/s2_100/ROIs1158_spring_s2_100_p101.tif"
s1_path = "datasets/SEN12MS/ROIs1158_spring/s1_100/ROIs1158_spring_s1_100_p101.tif"

with rasterio.open(s2_path) as src:
    print("S2 count:", src.count)
    print("S2 shape:", src.read().shape)
    print("S2 descriptions:", src.descriptions)

with rasterio.open(s1_path) as src:
    print("S1 count:", src.count)
    print("S1 shape:", src.read().shape)
    print("S1 descriptions:", src.descriptions)