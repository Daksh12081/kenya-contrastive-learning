import rasterio

s1_path = "datasets/sen12ms-subset/ROIs2017_winter_s1/s1_21/ROIs2017_winter_s1_21_p30.tif"
s2_path = "datasets/sen12ms-subset/ROIs2017_winter_s2/s2_21/ROIs2017_winter_s2_21_p30.tif"

print("===== Sentinel-1 =====")
with rasterio.open(s1_path) as src:
    print("Bands:", src.count)
    print("Shape:", src.read().shape)
    print("Dtype:", src.dtypes)
    print("Descriptions:", src.descriptions)

print()

print("===== Sentinel-2 =====")
with rasterio.open(s2_path) as src:
    print("Bands:", src.count)
    print("Shape:", src.read().shape)
    print("Dtype:", src.dtypes)
    print("Descriptions:", src.descriptions)