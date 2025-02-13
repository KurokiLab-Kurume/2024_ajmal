import os
import numpy as np
from PIL import Image

def load(K, V, image):

    # Folder containing the images
    folder_path = image

    # Supported image extensions
    image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".gif")

    # Initialize the array to hold the images
    S = np.zeros((K, V, V), dtype=np.float32)  # Assuming grayscale images

    # Counter for images
    count = 0

    # Iterate through the folder and process images
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(image_extensions):
            file_path = os.path.join(folder_path, filename)
            print(f"Loading image: {file_path}")

            try:
                # Open the image and convert to grayscale
                img = Image.open(file_path).convert("L")
                orig_width, orig_height = img.size

                # Case 1: Image is smaller -> Zero-padding
                if orig_width < V or orig_height < V:
                    padded_img = Image.new("L", (V, V), 0)  # Create a black canvas
                    x_offset = (V - orig_width) // 2
                    y_offset = (V - orig_height) // 2
                    padded_img.paste(img, (x_offset, y_offset))
                    img_processed = padded_img  # Use padded image

                # Case 2: Image is larger -> Downscale while keeping aspect ratio
                elif orig_width > V or orig_height > V:
                    # Compute new size while keeping aspect ratio
                    aspect_ratio = orig_width / orig_height

                    if orig_width > orig_height:
                        new_width = V
                        new_height = int(V / aspect_ratio)
                    else:
                        new_height = V
                        new_width = int(V * aspect_ratio)

                    img_resized = img.resize((new_width, new_height), Image.LANCZOS)

                    # Place the resized image in the center of a VxV black canvas
                    padded_img = Image.new("L", (V, V), 0)
                    x_offset = (V - new_width) // 2
                    y_offset = (V - new_height) // 2
                    padded_img.paste(img_resized, (x_offset, y_offset))
                    img_processed = padded_img  # Use the resized and centered image

                # Case 3: Image is already V x V -> No change
                else:
                    img_processed = img

                # Convert to NumPy array
                img_array = np.array(img_processed, dtype=np.float32)

                # Normalize the image (optional)
                img_array /= 255.0

                # Store the processed image in the array
                S[count] = img_array

                count += 1
                if count >= K:
                    print(f"Loaded {K} images into array S.")
                    break
            except Exception as e:
                print(f"Error processing image {filename}: {e}")

    # Verify the shape of S
    print(f"Array S shape: {S.shape}")

    return S


# import os
# import numpy as np
# from PIL import Image

# def load(K, V, image):

#     # Folder containing the images
#     folder_path = image

#     # Supported image extensions
#     image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".gif")

#     # Initialize the array to hold the images
#     S = np.zeros((K, V, V), dtype=np.float32)  # Assuming grayscale images
#     # S = np.zeros((K, V, V, 3), dtype=np.float32)  
#     # # For RGB images
#     # # In the loop, skip the grayscale conversion step and load images with 3 channels

#     # Counter for images
#     count = 0

#     # Iterate through the folder and process images
#     for filename in sorted(os.listdir(folder_path)):
#         if filename.lower().endswith(image_extensions):
#             file_path = os.path.join(folder_path, filename)
#             print(f"Loading image: {file_path}")

#             # Open the image, convert to grayscale, and resize
#             try:
#                 img = Image.open(file_path).convert("L")  # Convert to grayscale
#                 img_resized = img.resize((V, V))  # Resize to (V, V)
#                 img_array = np.array(img_resized, dtype=np.float32)  # Convert to NumPy array

#                 # Normalize the image (optional, scale to [0, 1])
#                 img_array /= 255.0

#                 # Store the image in the array
#                 S[count] = img_array

#                 count += 1
#                 if count >= K:
#                     print(f"Loaded {K} images into array S.")
#                     break
#             except Exception as e:
#                 print(f"Error processing image {filename}: {e}")

#     # Verify the shape of S
#     print(f"Array S shape: {S.shape}")

#     return S