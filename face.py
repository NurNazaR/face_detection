# Import kivy dependencies first
from kivy.app import App 
from kivy.uix.boxlayout import BoxLayout
from kivy. uix.image import Image 
from kivy.uix.button import Button 
from kivy.uix. label import Label 
from kivy.clock import Clock 
from kivy.graphics.texture import Texture 
from kivy.logger import Logger
from layers import L1Dist 

# Import other dependencies
import cv2
import tensorflow as tf 
 
import os
import numpy as np

from pathlib import Path
import imghdr
import sys


# Build app and layout
class CamApp(App):
    def build(self):
        # Main lavout components
        self.web_cam = Image(size_hint=(1, .8))
        self.button = Button (text="Verify", on_press=self.verify, size_hint=(1, .1))
        self.verification_label = Label(text ="Verification Uninitiated", size_hint= (1, .1))
        
        # Add items to layout
        layout = BoxLayout(orientation= 'vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)
        
        # Load tensorflow/keras model
        self.model = tf.keras.models.load_model('facetracker.keras')
        self.model2 = tf.keras.models.load_model('siamesemodel.keras', custom_objects={'L1Dist':L1Dist})
        
        # Setup video capture device
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/33.0) 
        
        return layout
    
    # Run continuously to get webcam feed
    def update(self, *args):
        # Read frame from openc
        ret, frame = self.capture.read()
        frame = cv2.flip(frame[100:100+900, 500:500+900, :], 1)
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = tf.image.resize(rgb, (120,120))
        
        yhat = self.model.predict(np.expand_dims(resized/255,0))
        sample_coords = yhat[1][0]
        
        if yhat[0] > 0.5: 
            # Controls the main rectangle
            cv2.rectangle(frame, 
                        tuple(np.multiply(sample_coords[:2], [900,900]).astype(int)),
                        tuple(np.multiply(sample_coords[2:], [900,900]).astype(int)), 
                                (255,0,0), 2)
            # Controls the label rectangle
            cv2.rectangle(frame, 
                        tuple(np.add(np.multiply(sample_coords[:2], [900,900]).astype(int), 
                                        [0,-30])),
                        tuple(np.add(np.multiply(sample_coords[:2], [900,900]).astype(int),
                                        [80,0])), 
                                (255,0,0), -1)
            
            # Controls the text rendered
            cv2.putText(frame, 'face', tuple(np.add(np.multiply(sample_coords[:2], [900,900]).astype(int),
                                                [0,-5])),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    
        # Flip horizontall and convert image to texture
        buf = cv2.flip (frame, 0).tobytes()
        img_texture = Texture.create(size= (frame. shape [1], frame. shape [0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture
    
    
    # Load image from file and convert to 100x100px| 
    def preprocess(self, file_path):
        # Read in image from file path
        byte_img = tf.io.read_file(file_path)
        # Load in the image
        img = tf.io.decode_jpeg(byte_img)
        # Preprocessing steps - resizing the image to be 100×100x3
        img = tf.image.resize(img, (100, 100))
        img = img / 255.0
        return img
    
    def verify(self, *args):
        # Specify thresholds
        detection_threshold = 0.9
        verification_threshold = 0.5
        
        # Capture input image from our webcam
        SAVE_PATH = os.path.join('application data', 'input_image', 'input_image.jpg')
        _, frame = self.capture.read()
        
        frame = cv2.flip(frame[100:100+900, 500:500+900, :], 1)
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = tf.image.resize(rgb, (120,120))
        
        yhat = self.model.predict(np.expand_dims(resized/255,0))
        sample_coords = yhat[1][0]
        
        if yhat[0] > 0.5: 
            face = frame[int(sample_coords[1]*900):int(sample_coords[3]*900),
                        int(sample_coords[0]*900):int(sample_coords[2]*900)]
            resized_face = cv2.resize(face, (250, 250))
            cv2.imwrite(SAVE_PATH, resized_face)
        
            # Build results array
            results = []
            
            for image in os.listdir(os.path.join('application data','verification_images')):
                if image.endswith(".jpg"):
                    validation_img = self.preprocess(os.path.join('application data','verification_images', image))

                    
                    input_img = self.preprocess(os.path.join('application data', 'input_image', 'input_image.jpg'))
                    
                    # Make Predictions
                    result = self.model2.predict(list(np.expand_dims([input_img, validation_img], axis=1)), verbose=0)
                    results.append(result)
                
            # Detection Threshold: Metric above which a prediciton is considered positive
            detection = np.sum(np.array(results) > detection_threshold)
            
            # Verification Threshold: Proportion of positive predictions / total positive samples
            verification = detection / len(os.listdir(os.path.join('application data', 'verification_images')))
            verified = verification > verification_threshold
            
            # Set verification text
            self.verification_label.text = 'Verified' if verified == True else 'Unverified'
            
            # Log out details
            Logger.info(results)
            Logger.info(detection)
            Logger.info(verification)
            Logger.info(verified)
        
            return results, verified
         
if __name__ == '__main__':
    CamApp().run()