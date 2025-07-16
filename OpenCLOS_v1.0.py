import os
import sys
import psutil
import cv2
import dlib
import time
import numpy as np
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from scipy.spatial import distance as dist
import queue
import signal
import platform
import argparse
from PIL import Image, ImageTk

class DrowsinessDetector:
    def __init__(self, cli_mode=False):
        self.cli_mode = cli_mode
        self.video_window_name = "Drowsiness Detection Feed"
        self.running = True
        self._stop_event = threading.Event()
        self._gui_running = True
        
        # Initialize pygame availability
        self.HAS_PYGAME = False
        try:
            import pygame
            pygame.mixer.init()
            self.HAS_PYGAME = True
            print("Pygame audio support initialized")
        except ImportError:
            print("Note: pygame module not installed. Using system beep for alerts")
        except Exception as e:
            print(f"Error initializing pygame: {e}")

        # Initialize system components
        self.process = psutil.Process(os.getpid())
        print(f"System Info - CPU: {self.process.cpu_percent()}% | Memory: {self.process.memory_info().rss / 1024 / 1024:.1f} MB")
        
        # Initialize flags and configurations
        self.gui_ready = False
        self.camera_ready = False
        self.detection_running = False
        self.models_ready = False
        self.manual_silence = False
        self.sound_playing = False
        
        # Detection parameters
        self.CLOSED_THRESHOLD = 0.25
        self.CLOSED_CONSEC_FRAMES = 20  # Number of frames for drowsiness detection
        self.NO_FACE_FRAMES = 30
        
        # Tracking variables
        self.closed_eyes_frame = 0
        self.no_face_frame = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Eye landmark indices
        self.LEFT_EYE_IDX = list(range(36, 42))
        self.RIGHT_EYE_IDX = list(range(42, 48))
        
        # Calibration
        self.ear_history = []
        self.calibrated_threshold = None
        self.silence_until = 0
        
        # Thread-safe communication
        self.status_queue = queue.Queue()
        self.command_queue = queue.Queue()
        self.video_queue = queue.Queue(maxsize=1)  # For video frames to GUI
        
        # Camera and display
        self.available_cameras = []
        self.selected_camera = 0
        self.cap = None
        self.video_thread = None
        self._video_thread_lock = threading.Lock()
        self.last_frame = None
        
        # Initialize audio
        self.alert_sound = None
        if self.HAS_PYGAME:
            try:
                if os.path.exists("alert.wav"):
                    import pygame
                    self.alert_sound = pygame.mixer.Sound("alert.wav")
                else:
                    print("Warning: alert.wav not found. Using system beep instead.")
            except Exception as e:
                print(f"Error loading alert sound: {e}")
                self.HAS_PYGAME = False
        
        # Initialize models
        self.init_models()
        
        if not self.cli_mode:
            self.setup_gui()
        else:
            self.setup_cli()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def setup_cli(self):
        """Initialize command-line interface mode"""
        print("\nInitializing CLI Mode:")
        print("-" * 40)
        
        # Initialize camera
        print("Initializing camera...")
        if not self.init_camera(self.selected_camera):
            print("Error: Failed to initialize camera")
            sys.exit(1)
        
        # Verify models
        if not self.models_ready:
            print("Error: Required models not loaded")
            sys.exit(1)
        
        # Print instructions
        print("\nCLI Mode Ready:")
        print("- Press 'q' to quit")
        print("- Press 's' to silence alerts for 5 seconds")
        print("- Video feed will display in separate window")
        print("-" * 40 + "\n")

    def _window_exists(self, window_name):
        """Check if an OpenCV window exists"""
        try:
            return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1
        except:
            return False

    def signal_handler(self, signum, frame):
        """Handle system signals gracefully"""
        print(f"Received signal {signum}, shutting down...")
        self.running = False
        self._stop_event.set()
        self.cleanup()
        sys.exit(0)

    def init_models(self):
        """Initialize dlib models with error handling"""
        try:
            print("Loading face detector...")
            self.detector = dlib.get_frontal_face_detector()
            
            print("Loading landmark predictor...")
            predictor_path = "shape_predictor_68_face_landmarks.dat"
            if not os.path.exists(predictor_path):
                print(f"ERROR: {predictor_path} not found!")
                print("Download it from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
                self.models_ready = False
                return
            
            self.predictor = dlib.shape_predictor(predictor_path)
            self.models_ready = True
            print("Models loaded successfully")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            self.models_ready = False

    def setup_gui(self):
        """Setup complete GUI interface with integrated video display"""
        try:
            self.root = tk.Tk()
            self.root.title("Drowsiness Detector")
            
            # Set window size based on platform
            if platform.machine() in ('arm', 'armv7l', 'aarch64'):
                # ARM (Orange Pi) optimized size
                self.root.geometry("800x480")
                self.video_width, self.video_height = 320, 240
            else:
                # x86_64 standard size
                self.root.geometry("1024x600")
                self.video_width, self.video_height = 480, 360
            
            self.root.configure(bg='#2c3e50')
            
            # Main container
            main_frame = ttk.Frame(self.root, padding="10")
            main_frame.pack(fill=tk.BOTH, expand=True)
            
            # Split into left (controls) and right (video) frames
            left_frame = ttk.Frame(main_frame, width=400)
            left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            
            right_frame = ttk.Frame(main_frame)
            right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=5)
            
            # Title
            title_label = ttk.Label(left_frame, text="Drowsiness Detector", 
                                  font=('Arial', 14, 'bold'))
            title_label.pack(pady=10)
            
            # System info frame
            info_frame = ttk.LabelFrame(left_frame, text="System Information", padding="10")
            info_frame.pack(fill=tk.X, pady=5)
            
            sys_info = self.get_system_info()
            info_text = f"Platform: {sys_info['platform']} | OpenCV: {sys_info['opencv_version']}"
            ttk.Label(info_frame, text=info_text, font=('Arial', 9)).pack()
            
            # Camera selection frame
            camera_frame = ttk.LabelFrame(left_frame, text="Camera Selection", padding="10")
            camera_frame.pack(fill=tk.X, pady=5)
            
            camera_control_frame = ttk.Frame(camera_frame)
            camera_control_frame.pack(fill=tk.X)
            
            ttk.Label(camera_control_frame, text="Camera:").pack(side=tk.LEFT)
            
            self.camera_var = tk.StringVar()
            self.camera_combo = ttk.Combobox(camera_control_frame, textvariable=self.camera_var, 
                                           width=15, state='readonly')
            self.camera_combo.pack(side=tk.LEFT, padx=5)
            
            self.scan_button = ttk.Button(camera_control_frame, text="🔍 Scan", 
                                         command=self.scan_cameras_gui)
            self.scan_button.pack(side=tk.LEFT, padx=5)
            
            self.connect_button = ttk.Button(camera_control_frame, text="📹 Connect", 
                                           command=self.connect_camera_gui)
            self.connect_button.pack(side=tk.LEFT, padx=5)
            
            # Status frame
            status_frame = ttk.LabelFrame(left_frame, text="Status", padding="10")
            status_frame.pack(fill=tk.X, pady=10)
            
            self.status_label = ttk.Label(status_frame, text="Ready to start...", 
                                         font=('Arial', 10))
            self.status_label.pack()
            
            self.fps_label = ttk.Label(status_frame, text="FPS: --")
            self.fps_label.pack()
            
            self.ear_label = ttk.Label(status_frame, text="EAR: --")
            self.ear_label.pack()
            
            self.frames_label = ttk.Label(status_frame, text="Closed Frames: 0/20")
            self.frames_label.pack()
            
            # Model status
            model_status = "✅ Models loaded" if self.models_ready else "❌ Models not loaded"
            self.model_label = ttk.Label(status_frame, text=model_status)
            self.model_label.pack()
            
            # Controls frame
            controls_frame = ttk.LabelFrame(left_frame, text="Controls", padding="10")
            controls_frame.pack(fill=tk.X, pady=10)
            
            # Control buttons
            button_frame = ttk.Frame(controls_frame)
            button_frame.pack()
            
            self.start_button = ttk.Button(button_frame, text="▶️ Start", 
                                          command=self.start_detection, width=10)
            self.start_button.pack(side=tk.LEFT, padx=5)
            
            self.stop_button = ttk.Button(button_frame, text="⏸️ Stop", 
                                         command=self.stop_detection, width=10, state='disabled')
            self.stop_button.pack(side=tk.LEFT, padx=5)
            
            self.sound_button = ttk.Button(button_frame, text="🔊 Sound ON", 
                                          command=self.toggle_sound, width=10)
            self.sound_button.pack(side=tk.LEFT, padx=5)
            
            self.silence_button = ttk.Button(button_frame, text="🔇 5s", 
                                            command=self.manual_silence_5s, width=10)
            self.silence_button.pack(side=tk.LEFT, padx=5)
            
            # Exit button
            self.exit_button = ttk.Button(controls_frame, text="❌ Exit", 
                                         command=self.close_application, width=10)
            self.exit_button.pack(pady=5)
            
            # Settings frame
            settings_frame = ttk.LabelFrame(left_frame, text="Settings", padding="10")
            settings_frame.pack(fill=tk.X, pady=10)
            
            # Sensitivity slider
            sensitivity_frame = ttk.Frame(settings_frame)
            sensitivity_frame.pack(fill=tk.X, pady=5)
            
            ttk.Label(sensitivity_frame, text="Sensitivity:").pack(side=tk.LEFT)
            self.sensitivity_var = tk.DoubleVar(value=self.CLOSED_THRESHOLD)
            self.sensitivity_scale = ttk.Scale(sensitivity_frame, from_=0.15, to=0.35, 
                                              variable=self.sensitivity_var, 
                                              command=self.update_sensitivity, length=150)
            self.sensitivity_scale.pack(side=tk.LEFT, padx=10)
            self.sensitivity_value_label = ttk.Label(sensitivity_frame, text=f"{self.CLOSED_THRESHOLD:.2f}")
            self.sensitivity_value_label.pack(side=tk.LEFT)
            
            # Video display frame
            video_frame = ttk.LabelFrame(right_frame, text="Camera Feed", padding="5")
            video_frame.pack(fill=tk.BOTH, expand=True)
            
            self.video_label = ttk.Label(video_frame)
            self.video_label.pack(fill=tk.BOTH, expand=True)
            
            # Bind close event
            self.root.protocol("WM_DELETE_WINDOW", self.close_application)
            
            # Schedule GUI updates
            self.root.after(100, self.update_gui_from_queue)
            self.root.after(100, self.update_video_display)
            
            # Auto-scan cameras on startup
            self.root.after(1000, self.scan_cameras_gui)
            
            self.gui_ready = True
            print("GUI initialized successfully")
            
        except Exception as e:
            print(f"Error setting up GUI: {e}")
            sys.exit(1)

    def safe_gui_update(self, func, *args, **kwargs):
        """Safely update GUI elements from threads"""
        if self._gui_running and hasattr(self, 'root'):
            try:
                if args or kwargs:
                    self.root.after(0, lambda: func(*args, **kwargs))
                else:
                    self.root.after(0, func)
            except RuntimeError:
                pass  # Tkinter has been destroyed

    def scan_cameras_gui(self):
        """Scan cameras and update GUI"""
        self.safe_gui_update(lambda: self.scan_button.config(state='disabled', text="🔍 Scanning..."))
        self.status_queue.put({'status': 'Scanning cameras...'})
        
        def scan_thread():
            cameras = self.scan_cameras()
            self.safe_gui_update(self.update_camera_list, cameras)
        
        threading.Thread(target=scan_thread, daemon=True).start()

    def scan_cameras(self):
        """Scan for available cameras"""
        self.available_cameras = []
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                self.available_cameras.append(i)
                cap.release()
            time.sleep(0.1)
        return self.available_cameras

    def update_camera_list(self, cameras):
        """Update camera list in GUI"""
        if not self._gui_running:
            return
            
        self.scan_button.config(state='normal', text="🔍 Scan")
        
        if cameras:
            camera_options = [f"Camera {i}" for i in cameras]
            self.camera_combo['values'] = camera_options
            self.camera_combo.current(0)
            self.status_queue.put({'status': f'Found {len(cameras)} camera(s)'})
        else:
            self.camera_combo['values'] = []
            self.status_queue.put({'status': 'No cameras found'})
            self.safe_gui_update(lambda: messagebox.showwarning("Camera Warning", "No cameras detected"))

    def connect_camera_gui(self):
        """Connect to selected camera"""
        if not self.camera_var.get():
            messagebox.showwarning("Selection Error", "Please select a camera first")
            return
        
        camera_index = int(self.camera_var.get().split()[-1])
        
        self.connect_button.config(state='disabled', text="📹 Connecting...")
        
        def connect_thread():
            success = self.init_camera(camera_index)
            self.safe_gui_update(self.update_connection_status, success)
        
        threading.Thread(target=connect_thread, daemon=True).start()

    def init_camera(self, camera_index=0):
        """Initialize camera with multiple backend support"""
        try:
            # Release existing camera if any
            if hasattr(self, 'cap') and self.cap is not None:
                self.cap.release()
                time.sleep(0.5)  # Allow time for release

            # Try different backends
            backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
            for backend in backends:
                try:
                    self.cap = cv2.VideoCapture(camera_index, backend)
                    if self.cap.isOpened():
                        # Set optimal parameters
                        if platform.machine() in ('arm', 'armv7l', 'aarch64'):
                            # Lower resolution for ARM
                            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
                            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
                            self.cap.set(cv2.CAP_PROP_FPS, 15)
                        else:
                            # Higher resolution for x86
                            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                            self.cap.set(cv2.CAP_PROP_FPS, 30)
                        
                        # Test frame capture
                        ret, _ = self.cap.read()
                        if ret:
                            print(f"Successfully initialized camera {camera_index} with backend {backend}")
                            self.selected_camera = camera_index
                            self.camera_ready = True
                            return True
                        
                    self.cap.release()
                except Exception as e:
                    print(f"Camera backend {backend} failed: {str(e)}")
                    continue

            print(f"Could not initialize camera {camera_index} with any backend")
            return False

        except Exception as e:
            print(f"Camera initialization error: {str(e)}")
            return False

    def update_connection_status(self, success):
        """Update connection status in GUI"""
        if not self._gui_running:
            return
            
        self.connect_button.config(state='normal', text="📹 Connect")
        
        if success:
            self.start_button.config(state='normal')
            messagebox.showinfo("Success", f"Camera {self.selected_camera} connected")
        else:
            messagebox.showerror("Error", "Failed to connect to camera")

    def start_detection(self):
        """Start or restart drowsiness detection"""
        if not self.camera_ready:
            messagebox.showwarning("Camera Error", "Please connect a camera first")
            return
        
        if not self.models_ready:
            messagebox.showerror("Model Error", "Face detection models not loaded")
            return
        
        # Stop any existing detection first
        self.stop_detection()
        
        # Reset stop event
        self._stop_event.clear()
        
        # Reset detection state
        self.detection_running = True
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')
        
        # Reset counters
        self.closed_eyes_frame = 0
        self.no_face_frame = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Start new video processing thread
        with self._video_thread_lock:
            if self.video_thread is None or not self.video_thread.is_alive():
                self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
                self.video_thread.start()

    def stop_detection(self):
        """Stop drowsiness detection while keeping camera alive"""
        if not self.detection_running:
            return

        self.detection_running = False
        self._stop_event.set()

        # Update GUI state safely
        self.safe_gui_update(lambda: self.start_button.config(state='normal'))
        self.safe_gui_update(lambda: self.stop_button.config(state='disabled'))

        # Don't join thread immediately — allow it to shut down gracefully
        if hasattr(self, 'root') and self._gui_running:
            self.root.after(500, self.check_video_thread_cleanup)

    def check_video_thread_cleanup(self):
        """Check if the video thread has stopped and clean up if needed"""
        with self._video_thread_lock:
            if self.video_thread and not self.video_thread.is_alive():
                print("Video thread stopped successfully.")
                self.video_thread = None
            else:
                print("Waiting for video thread to exit...")
                if hasattr(self, 'root') and self._gui_running:
                    self.root.after(500, self.check_video_thread_cleanup)

    def update_gui_from_queue(self):
        """Update GUI from queue (thread-safe)"""
        if not self._gui_running or not hasattr(self, 'root'):
            return
            
        try:
            while not self.status_queue.empty():
                status_data = self.status_queue.get_nowait()
                
                if 'status' in status_data:
                    self.status_label.config(text=status_data['status'])
                if 'fps' in status_data:
                    self.fps_label.config(text=f"FPS: {status_data['fps']:.1f}")
                if 'ear' in status_data:
                    self.ear_label.config(text=f"EAR: {status_data['ear']:.3f}")
                if 'closed_frames' in status_data:
                    self.frames_label.config(text=f"Closed Frames: {status_data['closed_frames']}/{self.CLOSED_CONSEC_FRAMES}")
                    
        except queue.Empty:
            pass
        
        if self._gui_running and hasattr(self, 'root'):
            self.root.after(100, self.update_gui_from_queue)

    def update_video_display(self):
        """Update the video display in the GUI"""
        if not self._gui_running or not hasattr(self, 'root'):
            return
            
        try:
            # Get the latest frame from the queue
            if not self.video_queue.empty():
                frame = self.video_queue.get_nowait()
                
                # Convert the image to RGB format
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize the frame to fit the display
                frame = cv2.resize(frame, (self.video_width, self.video_height))
                
                # Convert to PIL Image
                img = Image.fromarray(frame)
                
                # Convert to ImageTk format
                imgtk = ImageTk.PhotoImage(image=img)
                
                # Update the label
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
                
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Error updating video display: {e}")
            
        # Schedule the next update
        if self._gui_running and hasattr(self, 'root'):
            self.root.after(30, self.update_video_display)

    def toggle_sound(self):
        """Toggle sound on/off"""
        self.manual_silence = not self.manual_silence
        if self.manual_silence:
            self.sound_button.config(text="🔇 Sound OFF")
        else:
            self.sound_button.config(text="🔊 Sound ON")

    def manual_silence_5s(self):
        """Silence alert for 5 seconds"""
        self.manual_silence = True
        self.silence_button.config(text="🔇 Silenced...", state='disabled')
        
        def reset_silence():
            time.sleep(5)
            if self.running:
                self.manual_silence = False
                self.safe_gui_update(lambda: self.silence_button.config(text="🔇 5s", state='normal'))
                self.safe_gui_update(lambda: self.sound_button.config(text="🔊 Sound ON"))
        
        threading.Thread(target=reset_silence, daemon=True).start()

    def update_sensitivity(self, value):
        """Update sensitivity threshold"""
        self.CLOSED_THRESHOLD = float(value)
        self.sensitivity_value_label.config(text=f"{float(value):.2f}")

    def close_application(self):
        """Close the application safely"""
        print("Closing application...")
        self.running = False
        self._gui_running = False
        self._stop_event.set()
        
        # Stop detection first
        self.stop_detection()
        
        # Clean up resources
        self.cleanup()
        
        # Destroy the root window if it exists
        if hasattr(self, 'root'):
            try:
                self.root.quit()
                self.root.destroy()
            except:
                pass

    def video_loop(self):
        """Main video processing loop"""
        print("Starting video processing...")
        
        try:
            while self.running and not self._stop_event.is_set():
                if not self.detection_running:
                    break  # Exit if detection was stopped

                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame")
                    time.sleep(0.1)
                    continue
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Put frame in queue for GUI display
                if not self.cli_mode and processed_frame is not None:
                    try:
                        self.video_queue.put_nowait(processed_frame)
                    except queue.Full:
                        pass  # Skip frame if queue is full
                
                # Calculate FPS
                self.frame_count += 1
                elapsed = time.time() - self.start_time
                fps = self.frame_count / elapsed if elapsed > 0 else 0
                self.status_queue.put({'fps': fps})
                
                # Check stop event frequently
                if self._stop_event.is_set():
                    break
                    
        except Exception as e:
            print(f"Video processing error: {e}")
        finally:
            print("Exiting video loop")
            self._stop_event.clear()  # Reset for next detection

    def process_frame(self, frame):
        """Process frame for drowsiness detection"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Downscale for faster face detection on ARM
            if platform.machine() in ('arm', 'armv7l', 'aarch64'):
                small_gray = cv2.resize(gray, (0,0), fx=0.5, fy=0.5)
                faces = self.detector(small_gray, 0)
            else:
                faces = self.detector(gray)
            
            if len(faces) > 0:
                self.no_face_frame = 0
                
                for face in faces:
                    # Scale face coordinates back up if we downscaled
                    if platform.machine() in ('arm', 'armv7l', 'aarch64'):
                        face = dlib.rectangle(
                            left=face.left() * 2,
                            top=face.top() * 2,
                            right=face.right() * 2,
                            bottom=face.bottom() * 2
                        )
                    
                    shape = self.predictor(gray, face)
                    shape_np = np.array([[p.x, p.y] for p in shape.parts()])
                    
                    left_eye = shape_np[self.LEFT_EYE_IDX]
                    right_eye = shape_np[self.RIGHT_EYE_IDX]
                    left_ear = self.eye_aspect_ratio(left_eye)
                    right_ear = self.eye_aspect_ratio(right_eye)
                    avg_ear = (left_ear + right_ear) / 2.0
                    
                    # Update calibration
                    self.ear_history.append(avg_ear)
                    if len(self.ear_history) > 50:
                        self.ear_history.pop(0)
                    
                    # Auto-calibration
                    if len(self.ear_history) >= 20:
                        avg = np.mean(self.ear_history)
                        std = np.std(self.ear_history)
                        self.calibrated_threshold = max(0.18, avg - (std * 1.5))
                    
                    # Use calibrated threshold if available
                    current_threshold = self.calibrated_threshold if self.calibrated_threshold else self.CLOSED_THRESHOLD
                    
                    # Draw on frame
                    for eye in [left_eye, right_eye]:
                        hull = cv2.convexHull(eye)
                        cv2.drawContours(frame, [hull], -1, (0, 255, 0), 1)
                    
                    cv2.rectangle(frame, (face.left(), face.top()), 
                                (face.right(), face.bottom()), (255, 0, 0), 2)
                    
                    # Display EAR information
                    cv2.putText(frame, f"EAR: {avg_ear:.3f}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"Thresh: {current_threshold:.3f}", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Drowsiness detection
                    if avg_ear < current_threshold:
                        self.closed_eyes_frame += 1
                        cv2.putText(frame, f"Closed: {self.closed_eyes_frame}/{self.CLOSED_CONSEC_FRAMES}", (10, 90),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        
                        if self.closed_eyes_frame >= self.CLOSED_CONSEC_FRAMES:
                            cv2.putText(frame, "DROWSY!", (10, 120),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            if not self.manual_silence and time.time() > self.silence_until:
                                self.play_alert()
                                if self.cli_mode:
                                    print("\nALERT: Drowsiness detected! (Press 's' to silence)")
                    else:
                        self.closed_eyes_frame = 0
                    
                    # Update GUI status
                    if not self.cli_mode:
                        self.status_queue.put({
                            'status': 'Monitoring...',
                            'ear': avg_ear,
                            'closed_frames': self.closed_eyes_frame
                        })
            
            else:  # No face detected
                self.no_face_frame += 1
                cv2.putText(frame, "No face detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                if self.no_face_frame >= self.NO_FACE_FRAMES:
                    cv2.putText(frame, "WARNING: No face visible!", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if not self.manual_silence and time.time() > self.silence_until:
                        self.play_alert()
                        if self.cli_mode:
                            print("\nALERT: No face detected! (Press 's' to silence)")
                
                if not self.cli_mode:
                    self.status_queue.put({'status': 'No face detected'})
            
            return frame
            
        except Exception as e:
            error_msg = f"Frame processing error: {str(e)}"
            if self.cli_mode:
                print(f"\n{error_msg}")
            else:
                self.safe_gui_update(lambda: messagebox.showerror("Error", error_msg))
            return frame

    def eye_aspect_ratio(self, eye):
        """Calculate eye aspect ratio"""
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)

    def play_alert(self):
        """Play alert sound"""
        if not self.manual_silence and not self.sound_playing and time.time() > self.silence_until:
            self.sound_playing = True
            print("ALERT: Drowsiness detected!")
            try:
                if self.HAS_PYGAME and self.alert_sound:
                    self.alert_sound.play()
                else:
                    if platform.system() == 'Windows':
                        import winsound
                        winsound.Beep(1000, 500)
                    else:
                        os.system('echo -e "\a"')
            except Exception as e:
                print(f"Error playing alert: {e}")
            finally:
                self.sound_playing = False

    def get_system_info(self):
        """Get system information"""
        return {
            'platform': platform.system(),
            'platform_release': platform.release(),
            'opencv_version': cv2.__version__,
            'python_version': sys.version.split()[0]
        }

    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up...")
        self.running = False
        self._stop_event.set()
        self.detection_running = False
        
        # Wait for video thread to finish
        with self._video_thread_lock:
            if self.video_thread and self.video_thread.is_alive():
                self.video_thread.join(timeout=1.0)
                if self.video_thread.is_alive():
                    print("Force stopping video thread")
                self.video_thread = None
        
        # Release camera
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
            self.cap = None
        
        # Destroy window if it exists
        if self._window_exists(self.video_window_name):
            cv2.destroyWindow(self.video_window_name)
        
        if self.HAS_PYGAME:
            import pygame
            pygame.mixer.quit()

    def run(self):
        """Main run method"""
        if self.cli_mode:
            self.run_cli_mode()
        else:
            try:
                self.root.mainloop()
            except KeyboardInterrupt:
                print("Shutting down...")
            except Exception as e:
                print(f"GUI error: {e}")
            finally:
                self.cleanup()

    def run_cli_mode(self):
        """Run in command-line mode"""
        try:
            while self.running and self.camera_ready:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame")
                    time.sleep(0.1)
                    continue
                
                processed_frame = self.process_frame(frame)
                if processed_frame is not None:
                    cv2.imshow(self.video_window_name, processed_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):  # Quit
                        break
                    elif key == ord('s'):  # Silence
                        self.silence_until = time.time() + 5
                        print("\nAlerts silenced for 5 seconds")
                
                time.sleep(0.02)  # Reduce CPU usage
                
        except KeyboardInterrupt:
            print("\nDetection stopped by user")
        except Exception as e:
            print(f"\nError in CLI mode: {str(e)}")
        finally:
            if self._window_exists(self.video_window_name):
                cv2.destroyWindow(self.video_window_name)
            self.cleanup()
            print("\nCLI mode shutdown complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Driver Drowsiness Detection')
    parser.add_argument('--cli', action='store_true', help='Run in command-line mode')
    args = parser.parse_args()
    
    try:
        detector = DrowsinessDetector(cli_mode=args.cli)
        detector.run()
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
