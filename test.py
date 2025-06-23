from PIL import Image
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Replace this with your own image path
img = Image.open("rcb.jpeg")
text = pytesseract.image_to_string(img)
print(text)
