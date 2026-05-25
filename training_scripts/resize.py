from PIL import Image
baseheight = 224
img = Image.open('image.jpeg')
width = 224
img = img.resize((width, baseheight), Image.ANTIALIAS)
img.save('resizedimage.jpg')