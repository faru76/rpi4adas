import os
import cv2
import numpy as np
import tensorflow as tf
import pygame
from gpiozero import Robot
import serial
from time import sleep

class CarBot:
    def __init__(self, left_pins, right_pins, model_path="final_model.tflite", serial_port="/dev/ttyS0"):
        # Initialize GPIO for controlling motors
        self.carbot = Robot(left=left_pins, right=right_pins)
        self.left_speed = 1.0
        self.right_speed = 1.0

        # Load TFLite model and allocate tensors
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Define the label to index mapping
        self.label_to_index = {
            "forward": 0,
            "left": 1,
            "right": 2,
        }

        # Create an index to label mapping
        self.index_to_label = {v: k for k, v in self.label_to_index.items()}

        # Initialize Pygame for keyboard control and display
        pygame.init()
        self.screen = pygame.display.set_mode((400, 300))
        pygame.display.set_caption("Robot Car Control")
        self.running = True

        # Initialize camera capture
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Initialize serial communication with TFmini sensor
        self.ser = serial.Serial(serial_port, 115200, timeout=1)
        if not self.ser.is_open:
            self.ser.open()
        print("Serial port for TFmini opened")

    def __del__(self):
        # Destructor to ensure serial port is closed properly
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("Serial port for TFmini closed")

    def set_speed(self, left_speed, right_speed):
        # Method to set the speed of both motors
        self.left_speed = left_speed
        self.right_speed = right_speed

    def forward(self):
        # Method to move forward
        self.carbot.forward(speed=self.left_speed)

    def left(self):
        # Method to turn left
        self.carbot.left_motor.backward(speed=1.0)
        self.carbot.right_motor.forward(speed=1.0)

    def right(self):
        # Method to turn right
        self.carbot.left_motor.forward(speed=1.0)
        self.carbot.right_motor.backward(speed=1.0)

    def stop(self):
        # Method to stop the carbot
        self.carbot.stop()

    def process_events(self):
        # Method to handle Pygame events for keyboard control
        keys = pygame.key.get_pressed()
        if keys[pygame.K_a]:
            self.forward_left()
        elif keys[pygame.K_d]:
            self.forward_right()
        else:
            self.stop()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def get_distance(self):
        # Method to read distance from TFmini LiDAR sensor
        count = self.ser.in_waiting
        if count > 8:
            recv = self.ser.read(9)
            if recv[0] == 0x59 and recv[1] == 0x59:  # Check header
                distance = recv[2] + recv[3] * 256
                print(f"distance: {distance}cm")
                self.ser.reset_input_buffer()  # Clear the buffer
                return distance
            else:
                print("Header mismatch:", recv[:2])
                self.ser.reset_input_buffer()  # Clear the buffer
                return None
        return None

    def predict_and_control(self):
        # Method to perform inference with the TFLite model and control the carbot accordingly
        ret, frame = self.capture.read()
        if not ret:
            print("Failed to capture frame")
            self.stop()
            return

        # Preprocess the frame for the TFLite model
        frame_resized = cv2.resize(frame, (64, 64))
        frame_normalized = frame_resized / 255.0
        frame_input = np.expand_dims(frame_normalized, axis=0).astype(np.float32)

        # Set the input tensor for the TFLite interpreter and invoke
        self.interpreter.set_tensor(self.input_details[0]['index'], frame_input)
        self.interpreter.invoke()

        # Get the prediction from the output tensor of the TFLite interpreter
        prediction = self.interpreter.get_tensor(self.output_details[0]['index'])
        command_index = np.argmax(prediction)
        command = self.index_to_label[command_index]
        
        print(f"Prediction: {command}")

        # Check distance and adjust carbot behavior based on prediction and distance
        distance = self.get_distance()
        if distance is not None and distance < 10:
            self.stop()
            print("Obstacle detected! Stopping the car.")
            while True:
                distance = self.get_distance()
                if distance is not None and distance >= 10:
                    break
        elif distance is None:
            self.stop()
            print("Error reading distance. Stopping the car.")
            while True:
                distance = self.get_distance()
                if distance is not None and distance >= 10:
                    break
        else:
            if command == "left":
                self.left()
                sleep(0.2)
            elif command == "right":
                self.right()
                sleep(0.2)
            elif command == "forward":
                self.forward()
                sleep(0.2)
            else:
                self.stop()

    def run(self):
        # Main loop to run the carbot
        try:
            while self.running:
                self.process_events()
                self.predict_and_control()
                pygame.display.flip()
        except KeyboardInterrupt:
            print("Keyboard interrupt detected. Stopping the car.")
        finally:
            self.stop()
            pygame.quit()
            self.capture.release()

def main():
    # Entry point of the program
    car_bot = CarBot(left_pins=(13, 18), right_pins=(12, 19))
    car_bot.set_speed(0.7, 0.7)
    car_bot.run()

if __name__ == "__main__":
    main()
