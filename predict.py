def predict_image(image_path, model, scaler, pca):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, img_size)
    img = img / 255.0
    img = img.reshape(1, -1)
    img = scaler.transform(img)
    img_pca = pca.transform(img)
    prediction = model.predict(img_pca)
    return prediction

# Example prediction
image_path = 'data/new_image.png'
prediction = predict_image(image_path, clf, scaler, pca)
print("Prediction:", "Defective" if prediction[0] == 1 else "Non-Defective")
