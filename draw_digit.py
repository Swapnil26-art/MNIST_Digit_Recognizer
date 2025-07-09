import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained CNN model
model = load_model("mnist_cnn_model.h5")

# Create a Tkinter window
window = tk.Tk()
window.title("MNIST Digit Recognizer")
window.resizable(False, False)

# Create a canvas to draw digits
canvas_width, canvas_height = 280, 280
canvas = tk.Canvas(window, width=canvas_width, height=canvas_height, bg='white')
canvas.pack()

# Create PIL image to draw on (in background)
image = Image.new("L", (canvas_width, canvas_height), color=255)
draw = ImageDraw.Draw(image)

# Draw on canvas and save into image
def draw_digit(event):
    x, y = event.x, event.y
    r = 8  # brush radius
    canvas.create_oval(x - r, y - r, x + r, y + r, fill='black')
    draw.ellipse([x - r, y - r, x + r, y + r], fill=0)

canvas.bind("<B1-Motion>", draw_digit)

# Predict the digit
def predict():
    # Resize to 28x28 (MNIST format)
    img_resized = image.resize((28, 28))
    img_inverted = ImageOps.invert(img_resized)

    # Normalize and reshape for model
    img_array = np.array(img_inverted) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # Predict using model
    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)

    result_label.config(text=f"Predicted Digit: {predicted_digit}")

# Clear the canvas
def clear():
    canvas.delete("all")
    draw.rectangle([0, 0, canvas_width, canvas_height], fill=255)
    result_label.config(text="Draw a digit above")

# Buttons
button_frame = tk.Frame(window)
button_frame.pack()

predict_btn = tk.Button(button_frame, text="Predict", command=predict, width=10)
predict_btn.grid(row=0, column=0, padx=5, pady=5)

clear_btn = tk.Button(button_frame, text="Clear", command=clear, width=10)
clear_btn.grid(row=0, column=1, padx=5, pady=5)

# Result label
result_label = tk.Label(window, text="Draw a digit above", font=("Arial", 16))
result_label.pack(pady=10)

# Start the GUI
window.mainloop()
