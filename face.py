from face_recognition_sdk import FaceRecognition

# Initialize the SDK
face_sdk = FaceRecognition()

# Process an image
image_path = "path/to/your/image.jpg"
face_info = face_sdk.GetImageInfo(image_path, faceMaxCount=10)

# Compare two faces
similarity = face_sdk.get_similarity(feature1, feature2)

# Compare two images
image1 = "tf9.jpg"
image2 = "tf10.jpg"

# Get face information from both images
faces1 = face_sdk.GetImageInfo(image1, faceMaxCount=1)
faces2 = face_sdk.GetImageInfo(image2, faceMaxCount=1)

if faces1 and faces2:
    # Compare the first face from each image
    similarity = face_sdk.get_similarity(faces1[0]['embedding'], faces2[0]['embedding'])
    print(f"Similarity: {similarity}%")
    
    # Check if it's the same person (threshold = 75)
    is_same_person = similarity >= 75
    print(f"Same person: {is_same_person}")