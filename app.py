import os
import cv2
import numpy as np
import tensorflow as tf
from telegram.ext import Updater, MessageHandler, Filters

# ==============================
# Load Environment Variables
# ==============================

BOT_TOKEN = os.getenv("BOT_TOKEN")

if not BOT_TOKEN:
    raise ValueError("BOT_TOKEN not found in environment variables!")

# ==============================
# Load Model & Labels
# ==============================

try:
    model = tf.keras.models.load_model("nudity_model.pb")
    labels = open("nudity_labels.txt").read().splitlines()
except Exception as e:
    print("Model loading error:", e)
    model = None

# ==============================
# Initialize Bot
# ==============================

updater = Updater(BOT_TOKEN, use_context=True)
dispatcher = updater.dispatcher


# ==============================
# Image Handler Function
# ==============================

def detect_nudity(update, context):
    if model is None:
        update.message.reply_text("Model not loaded properly.")
        return

    try:
        # Download image
        photo_file = update.message.photo[-1].get_file()
        photo_file.download("user_image.jpg")

        # Preprocess image
        img = cv2.imread("user_image.jpg")
        img = cv2.resize(img, (300, 300))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0

        # Prediction
        prediction = model.predict(np.expand_dims(img, axis=0))

        if np.max(prediction) > 0.8:
            update.message.reply_text("Nudity detected 🚨")
        else:
            update.message.reply_text("No nudity detected ✅")

    except Exception as e:
        update.message.reply_text("Error processing image.")
        print("Processing error:", e)


# ==============================
# Add Handlers
# ==============================

image_handler = MessageHandler(Filters.photo, detect_nudity)
dispatcher.add_handler(image_handler)

# ==============================
# Start Bot
# ==============================

print("Bot is running...")
updater.start_polling()
updater.idle()
