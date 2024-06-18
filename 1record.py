import cv2
import pygame
import numpy as np
import os
from datetime import datetime
from time import time, sleep
from gpiozero import Robot

class CarBot:
    def __init__(self, left_pins, right_pins, save_path="training_data", interval=0.5):
        # Initialize GPIO Robot with left and right motor pins
        self.carbot = Robot(left=left_pins, right=right_pins)
        # Default speeds for left and right motors
        self.left_speed = 1.0
        self.right_speed = 1.0
        # Path to save captured frames
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
        # Timestamp for the last recorded frame
        self.last_record_time = time()
        # Interval between recording frames
        self.interval = interval
        
        # Initialize Pygame for keyboard control and display
        pygame.init()
        self.screen = pygame.display.set_mode((400, 300))
        pygame.display.set_caption("Robot Car Control")
        
        # Initialize OpenCV capture from webcam
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Flag to control the main loop
        self.running = True
        
        # Initialize frame counter for captured images
        self.frame_counter = 0  

    def set_speed(self, left_speed, right_speed):
        # Set speeds for left and right motors
        self.left_speed = left_speed
        self.right_speed = right_speed

    def forward(self, duration=None):
        # Move the carbot forward
        self.carbot.forward(speed=self.left_speed)
        # Optionally stop after specified duration
        if duration:
            sleep(duration)
            self.carbot.stop()

    def forward_left(self, duration=None):
        # Move the carbot forward-left
        self.carbot.left_motor.backward(speed=0.5)
        self.carbot.right_motor.forward(speed=1.0)
        # Optionally stop after specified duration
        if duration:
            sleep(duration)
            self.carbot.stop()

    def forward_right(self, duration=None):
        # Move the carbot forward-right
        self.carbot.left_motor.forward(speed=1.0)
        self.carbot.right_motor.backward(speed=0.5)
        # Optionally stop after specified duration
        if duration:
            sleep(duration)
            self.carbot.stop()

    def stop(self):
        # Stop the carbot
        self.carbot.stop()

    def process_events(self):
        # Check for Pygame events (keyboard presses)
        keys = pygame.key.get_pressed()

        if keys[pygame.K_a]:
            self.forward_left()
            self.record_frame("forward_left")
        elif keys[pygame.K_d]:
            self.forward_right()
            self.record_frame("forward_right")
        else:
            self.forward()  # Keep moving forward
            self.record_frame("forward")

        # Check for quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def record_frame(self, control_command):
        # Capture a frame from the webcam
        ret, frame = self.capture.read()
        if not ret:
            print("Failed to capture frame")
            return
        # Generate timestamp for unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        # Construct filename with command and frame counter
        frame_filename = os.path.join(self.save_path, 
                                      f"{timestamp}_{control_command}_{self.frame_counter}.jpg")
        # Save the captured frame as an image
        cv2.imwrite(frame_filename, frame)
        # Increment frame counter
        self.frame_counter += 1  

    def run(self):
        # Start moving forward
        self.forward()
        try:
            while self.running:
                # Process Pygame events (keyboard inputs)
                self.process_events()
                # Update Pygame display
                pygame.display.flip()
        except KeyboardInterrupt:
            print("Keyboard interrupt detected. Stopping the car.")
            # Stop the carbot and release resources
            self.stop()
            pygame.quit()
            self.capture.release()

def main():
    # Initialize the CarBot with GPIO pins for left and right motors
    car_bot = CarBot(left_pins=(13, 18), right_pins=(12, 19))
    # Set initial speed (optional)
    car_bot.set_speed(0.5, 0.5)
    # Run the main loop for keyboard control
    car_bot.run()

if __name__ == "__main__":
    main()
