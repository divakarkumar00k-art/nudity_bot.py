import telegram
from telegram.ext import Updater, MessageHandler, Filters, CallbackContext
import cv2
import numpy as np
import tensorflow as tf

# Load your model and labels here
model = tf.keras.models.load_model('nudity_model.pb')
labels = open('nudity_labels.txt').read().splitlines()

# Initialize bot
BOT_TOKEN = "YOUR_REAL_BOT_TOKEN"

updater = Updater(BOT_TOKEN, use_context=True)
dispatcher = updater.dispatcher


def detect_nudity(update, context):
    # Get image from user
    photo_file = update.message.photo[-1].get_file()
    photo_file.download("user_image.jpg")

    # Process image
    img = cv2.imread("user_image.jpg")
    img = cv2.resize(img, (300, 300))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0

    # Run model inference
    prediction = model.predict(np.expand_dims(img, axis=0))

    if np.max(prediction) > 0.8:
        update.message.reply_text("Nudity detected! 🚨")
    else:
        update.message.reply_text("No nudity detected. ✅")


# Handlers
image_handler = MessageHandler(Filters.photo, detect_nudity)
dispatcher.add_handler(image_handler)

# Start bot
updater.start_polling()
updater.idle()
