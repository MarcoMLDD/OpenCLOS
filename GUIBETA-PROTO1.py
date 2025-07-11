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
import subprocess
import platform

# Remove problematic environment variables that might interfere
if 'QT_QPA_PLATFORM' in os.environ:
    del os.environ['QT_QPA_PLATFORM']

class DrowsinessDetector:
    def __init__(self):
        # Performance monitoring
        self.process = psutil.Process(os.getpid())
        print("CPU %:", self.process.cpu_percent())
        print("Memory MB:", self.process.memory_info().rss / 1024 / 1024)
        
        # Initialize flags first
        self.running = True
        self.gui_ready = False
        self.camera_ready = False
        
        # Sound management (simplified - no audio library for now)
        self.sound_playing = False
        self.manual_silence = False
        
        # Detection parameters
        self.CLOSED_THRESHOLD = 0.25
        self.CLOSED_CONSEC_FRAMES = 20
        self.NO_FACE_FRAMES = 5
        
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
        
        # Thread-safe communication
        self.status_queue = queue.Queue()
        self.command_queue = queue.Queue()
        
        # Camera selection
        self.available_cameras = []
        self.selected_camera = None
        
        # Initialize models safely
        self.init_models()
        
        # Setup GUI first
        self.setup_gui()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle system signals gracefully"""
        print(f"Received signal {signum}, shutting down...")
        self.running = False
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
    
    def get_system_info(self):
        """Get system information for debugging"""
        info = {
            'platform': platform.system(),
            'platform_release': platform.release(),
            'opencv_version': cv2.__version__,
            'python_version': sys.version
        }
        return info
    
    def scan_cameras_comprehensive(self):
        """Comprehensive camera scanning with multiple methods"""
        print("=== Comprehensive Camera Scan ===")
        sys_info = self.get_system_info()
        print(f"System: {sys_info['platform']} {sys_info['platform_release']}")
        print(f"OpenCV: {sys_info['opencv_version']}")
        
        self.available_cameras = []
        camera_info = {}
        
        # Method 1: Direct OpenCV scanning with multiple backends
        print("\n1. Testing OpenCV backends...")
        backends = [
            ('CAP_ANY', cv2.CAP_ANY),
            ('CAP_V4L2', cv2.CAP_V4L2),
            ('CAP_GSTREAMER', cv2.CAP_GSTREAMER),
            ('CAP_FFMPEG', cv2.CAP_FFMPEG),
        ]
        
        # Test more camera indices
        for i in range(10):
            camera_info[i] = {'available': False, 'backends': [], 'error': None}
            
            for backend_name, backend_id in backends:
                try:
                    cap = cv2.VideoCapture(i, backend_id)
                    if cap.isOpened():
                        # Test if we can actually read frames
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            camera_info[i]['available'] = True
                            camera_info[i]['backends'].append(backend_name)
                            if i not in self.available_cameras:
                                self.available_cameras.append(i)
                            print(f"  Camera {i}: ✓ Working with {backend_name}")
                        else:
                            print(f"  Camera {i}: ✗ Opens but can't read frames with {backend_name}")
                    cap.release()
                except Exception as e:
                    camera_info[i]['error'] = str(e)
                    print(f"  Camera {i}: ✗ Error with {backend_name}: {e}")
        
        # Method 2: System-specific detection
        print("\n2. System-specific detection...")
        if sys_info['platform'] == 'Linux':
            self.scan_linux_cameras()
        elif sys_info['platform'] == 'Windows':
            self.scan_windows_cameras()
        elif sys_info['platform'] == 'Darwin':  # macOS
            self.scan_macos_cameras()
        
        # Method 3: Try common camera paths/devices
        print("\n3. Testing common camera devices...")
        if sys_info['platform'] == 'Linux':
            self.test_linux_devices()
        
        print(f"\n=== Scan Results ===")
        print(f"Available cameras: {self.available_cameras}")
        
        return self.available_cameras
    
    def scan_linux_cameras(self):
        """Linux-specific camera detection"""
        try:
            # Method 1: v4l2-ctl
            result = subprocess.run(['v4l2-ctl', '--list-devices'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print("V4L2 devices found:")
                print(result.stdout)
                
                # Parse v4l2-ctl output to find device indices
                lines = result.stdout.split('\n')
                for line in lines:
                    if '/dev/video' in line:
                        try:
                            device_num = int(line.strip().split('video')[1])
                            if device_num not in self.available_cameras:
                                # Test this device
                                if self.test_camera_device(device_num):
                                    self.available_cameras.append(device_num)
                        except:
                            pass
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("v4l2-ctl not available")
        
        # Method 2: Check /dev/video* devices
        try:
            video_devices = []
            for i in range(10):
                device_path = f'/dev/video{i}'
                if os.path.exists(device_path):
                    video_devices.append(i)
                    print(f"Found device: {device_path}")
            
            for device in video_devices:
                if device not in self.available_cameras:
                    if self.test_camera_device(device):
                        self.available_cameras.append(device)
        except Exception as e:
            print(f"Error checking /dev/video devices: {e}")
    
    def scan_windows_cameras(self):
        """Windows-specific camera detection"""
        try:
            # Try DirectShow backend specifically
            for i in range(10):
                try:
                    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            if i not in self.available_cameras:
                                self.available_cameras.append(i)
                            print(f"DirectShow camera {i}: Available")
                    cap.release()
                except Exception as e:
                    print(f"DirectShow camera {i}: Error - {e}")
        except Exception as e:
            print(f"DirectShow scan error: {e}")
    
    def scan_macos_cameras(self):
        """macOS-specific camera detection"""
        try:
            # Try AVFoundation backend
            for i in range(10):
                try:
                    cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            if i not in self.available_cameras:
                                self.available_cameras.append(i)
                            print(f"AVFoundation camera {i}: Available")
                    cap.release()
                except Exception as e:
                    print(f"AVFoundation camera {i}: Error - {e}")
        except Exception as e:
            print(f"AVFoundation scan error: {e}")
    
    def test_linux_devices(self):
        """Test common Linux camera device paths"""
        common_devices = [
            '/dev/video0', '/dev/video1', '/dev/video2', '/dev/video3',
            '/dev/video4', '/dev/video5', '/dev/video6', '/dev/video7'
        ]
        
        for device_path in common_devices:
            if os.path.exists(device_path):
                device_num = int(device_path.split('video')[1])
                print(f"Testing {device_path}...")
                if self.test_camera_device(device_num):
                    if device_num not in self.available_cameras:
                        self.available_cameras.append(device_num)
                        print(f"  ✓ {device_path} working")
                else:
                    print(f"  ✗ {device_path} not working")
    
    def test_camera_device(self, device_index):
        """Test a specific camera device thoroughly"""
        try:
            # Test with different backends
            backends = [cv2.CAP_ANY, cv2.CAP_V4L2, cv2.CAP_GSTREAMER]
            
            for backend in backends:
                try:
                    cap = cv2.VideoCapture(device_index, backend)
                    if cap.isOpened():
                        # Set timeout for reading
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        
                        # Try to read a frame with timeout
                        start_time = time.time()
                        ret, frame = cap.read()
                        read_time = time.time() - start_time
                        
                        if ret and frame is not None and read_time < 2.0:
                            # Additional validation
                            if len(frame.shape) == 3 and frame.shape[2] == 3:
                                cap.release()
                                return True
                    cap.release()
                except Exception as e:
                    print(f"    Backend test error: {e}")
                    continue
            
            return False
            
        except Exception as e:
            print(f"Device test error for {device_index}: {e}")
            return False
    
    def scan_cameras(self):
        """Main camera scanning method"""
        return self.scan_cameras_comprehensive()
    
    def init_camera(self, camera_index=0):
        """Initialize camera with improved error handling"""
        try:
            print(f"Initializing camera {camera_index}...")
            
            # Release any existing camera
            if hasattr(self, 'cap') and self.cap is not None:
                self.cap.release()
                time.sleep(0.5)  # Give time for release
            
            # Determine best backend for the system
            system = platform.system()
            if system == 'Linux':
                backends = [cv2.CAP_V4L2, cv2.CAP_GSTREAMER, cv2.CAP_ANY]
            elif system == 'Windows':
                backends = [cv2.CAP_DSHOW, cv2.CAP_ANY]
            elif system == 'Darwin':
                backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
            else:
                backends = [cv2.CAP_ANY]
            
            success = False
            for backend in backends:
                try:
                    print(f"  Trying backend: {backend}")
                    self.cap = cv2.VideoCapture(camera_index, backend)
                    
                    if self.cap.isOpened():
                        # Set properties before testing
                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        self.cap.set(cv2.CAP_PROP_FPS, 30)
                        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        
                        # Test reading frames
                        for attempt in range(3):
                            ret, test_frame = self.cap.read()
                            if ret and test_frame is not None:
                                print(f"  ✓ Camera {camera_index} working with backend {backend}")
                                success = True
                                break
                            time.sleep(0.1)
                        
                        if success:
                            break
                    
                    self.cap.release()
                    
                except Exception as e:
                    print(f"  Backend {backend} failed: {e}")
                    if hasattr(self, 'cap'):
                        self.cap.release()
                    continue
            
            if not success:
                print(f"ERROR: Could not initialize camera {camera_index}")
                return False
            
            self.camera_ready = True
            self.selected_camera = camera_index
            print(f"Camera {camera_index} initialized successfully")
            
            # Update GUI
            self.status_queue.put({'status': f'Camera {camera_index} ready'})
            
            return True
            
        except Exception as e:
            print(f"Error initializing camera {camera_index}: {e}")
            return False
    
    def setup_gui(self):
        """Setup GUI with error handling"""
        try:
            self.root = tk.Tk()
            self.root.title("Driver Drowsiness Detection - Control Panel")
            self.root.geometry("600x500")
            self.root.configure(bg='#2c3e50')
            
            # Main frame
            main_frame = ttk.Frame(self.root, padding="20")
            main_frame.pack(fill=tk.BOTH, expand=True)
            
            # Title
            title_label = ttk.Label(main_frame, text="Dizon OpenCLOS Beta 1.0", 
                                   font=('Arial', 14, 'bold'))
            title_label.pack(pady=10)
            
            # System info frame
            info_frame = ttk.LabelFrame(main_frame, text="System Information", padding="10")
            info_frame.pack(fill=tk.X, pady=5)
            
            sys_info = self.get_system_info()
            info_text = f"Platform: {sys_info['platform']} | OpenCV: {sys_info['opencv_version']}"
            ttk.Label(info_frame, text=info_text, font=('Arial', 9)).pack()
            
            # Camera selection frame
            camera_frame = ttk.LabelFrame(main_frame, text="Camera Selection", padding="10")
            camera_frame.pack(fill=tk.X, pady=5)
            
            camera_control_frame = ttk.Frame(camera_frame)
            camera_control_frame.pack(fill=tk.X)
            
            ttk.Label(camera_control_frame, text="Camera:").pack(side=tk.LEFT)
            
            self.camera_var = tk.StringVar()
            self.camera_combo = ttk.Combobox(camera_control_frame, textvariable=self.camera_var, 
                                           width=15, state='readonly')
            self.camera_combo.pack(side=tk.LEFT, padx=5)
            
            self.scan_button = ttk.Button(camera_control_frame, text="🔍 Scan Cameras", 
                                         command=self.scan_cameras_gui)
            self.scan_button.pack(side=tk.LEFT, padx=5)
            
            self.connect_button = ttk.Button(camera_control_frame, text="📹 Connect", 
                                           command=self.connect_camera_gui)
            self.connect_button.pack(side=tk.LEFT, padx=5)
            
            # Test button for troubleshooting
            self.test_button = ttk.Button(camera_control_frame, text="🧪 Test Camera", 
                                        command=self.test_camera_gui)
            self.test_button.pack(side=tk.LEFT, padx=5)
            
            # Status frame
            status_frame = ttk.LabelFrame(main_frame, text="Status", padding="10")
            status_frame.pack(fill=tk.X, pady=10)
            
            self.status_label = ttk.Label(status_frame, text="Ready to start...", 
                                         font=('Arial', 10))
            self.status_label.pack()
            
            self.fps_label = ttk.Label(status_frame, text="FPS: --")
            self.fps_label.pack()
            
            self.ear_label = ttk.Label(status_frame, text="EAR: --")
            self.ear_label.pack()
            
            # Model status
            model_status = "✅ Models loaded" if getattr(self, 'models_ready', False) else "❌ Models not loaded"
            self.model_label = ttk.Label(status_frame, text=model_status)
            self.model_label.pack()
            
            # Controls frame
            controls_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
            controls_frame.pack(fill=tk.X, pady=10)
            
            # Control buttons
            button_frame = ttk.Frame(controls_frame)
            button_frame.pack()
            
            self.start_button = ttk.Button(button_frame, text="▶️ Start Detection", 
                                          command=self.start_detection, width=15)
            self.start_button.pack(side=tk.LEFT, padx=5)
            
            self.stop_button = ttk.Button(button_frame, text="⏸️ Stop Detection", 
                                         command=self.stop_detection, width=15, state='disabled')
            self.stop_button.pack(side=tk.LEFT, padx=5)
            
            self.sound_button = ttk.Button(button_frame, text="🔊 Sound ON", 
                                          command=self.toggle_sound, width=15)
            self.sound_button.pack(side=tk.LEFT, padx=5)
            
            self.silence_button = ttk.Button(button_frame, text="🔇 Silence 5s", 
                                            command=self.manual_silence_5s, width=15)
            self.silence_button.pack(side=tk.LEFT, padx=5)
            
            # Second row of buttons
            button_frame2 = ttk.Frame(controls_frame)
            button_frame2.pack(pady=5)
            
            self.exit_button = ttk.Button(button_frame2, text="❌ Exit", 
                                         command=self.close_application, width=15)
            self.exit_button.pack(side=tk.LEFT, padx=5)
            
            # Settings frame
            settings_frame = ttk.LabelFrame(main_frame, text="Settings", padding="10")
            settings_frame.pack(fill=tk.X, pady=10)
            
            # Sensitivity slider
            sensitivity_frame = ttk.Frame(settings_frame)
            sensitivity_frame.pack(fill=tk.X, pady=5)
            
            ttk.Label(sensitivity_frame, text="Sensitivity:").pack(side=tk.LEFT)
            self.sensitivity_var = tk.DoubleVar(value=0.25)
            self.sensitivity_scale = ttk.Scale(sensitivity_frame, from_=0.15, to=0.35, 
                                              variable=self.sensitivity_var, 
                                              command=self.update_sensitivity, length=200)
            self.sensitivity_scale.pack(side=tk.LEFT, padx=10)
            self.sensitivity_value_label = ttk.Label(sensitivity_frame, text="0.25")
            self.sensitivity_value_label.pack(side=tk.LEFT)
            
            # Bind close event
            self.root.protocol("WM_DELETE_WINDOW", self.close_application)
            
            # Schedule GUI updates
            self.root.after(100, self.update_gui_from_queue)
            
            # Auto-scan cameras on startup
            self.root.after(1000, self.scan_cameras_gui)
            
            self.gui_ready = True
            print("GUI initialized successfully")
            
        except Exception as e:
            print(f"Error setting up GUI: {e}")
            sys.exit(1)
    
    def scan_cameras_gui(self):
        """Scan cameras and update GUI"""
        self.scan_button.config(state='disabled', text="🔍 Scanning...")
        self.status_queue.put({'status': 'Scanning cameras...'})
        
        def scan_thread():
            cameras = self.scan_cameras()
            self.root.after(0, lambda: self.update_camera_list(cameras))
        
        threading.Thread(target=scan_thread, daemon=True).start()
    
    def test_camera_gui(self):
        """Test selected camera"""
        if not self.camera_var.get():
            messagebox.showwarning("Selection Error", "Please select a camera first")
            return
        
        camera_index = int(self.camera_var.get().split()[-1])
        
        def test_thread():
            try:
                print(f"Testing camera {camera_index}...")
                cap = cv2.VideoCapture(camera_index)
                
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        # Show test window for 3 seconds
                        cv2.imshow(f"Camera {camera_index} Test", frame)
                        cv2.waitKey(3000)
                        cv2.destroyAllWindows()
                        self.status_queue.put({'status': f'Camera {camera_index} test: SUCCESS'})
                    else:
                        self.status_queue.put({'status': f'Camera {camera_index} test: Can\'t read frames'})
                else:
                    self.status_queue.put({'status': f'Camera {camera_index} test: Can\'t open'})
                
                cap.release()
                
            except Exception as e:
                self.status_queue.put({'status': f'Camera {camera_index} test: ERROR - {e}'})
        
        threading.Thread(target=test_thread, daemon=True).start()
    
    def update_camera_list(self, cameras):
        """Update camera list in GUI"""
        self.scan_button.config(state='normal', text="🔍 Scan Cameras")
        
        if cameras:
            camera_options = [f"Camera {i}" for i in cameras]
            self.camera_combo['values'] = camera_options
            self.camera_combo.current(0)
            self.status_queue.put({'status': f'Found {len(cameras)} camera(s): {cameras}'})
        else:
            self.camera_combo['values'] = []
            self.status_queue.put({'status': 'No cameras found'})
            messagebox.showwarning("Camera Warning", 
                                 "No cameras detected. Please:\n"
                                 "1. Check camera connection\n"
                                 "2. Ensure camera permissions\n"
                                 "3. Close other applications using camera\n"
                                 "4. Try different USB ports\n"
                                 "5. Check if camera drivers are installed")
    
    def connect_camera_gui(self):
        """Connect to selected camera"""
        if not self.camera_var.get():
            messagebox.showwarning("Selection Error", "Please select a camera first")
            return
        
        camera_index = int(self.camera_var.get().split()[-1])
        
        self.connect_button.config(state='disabled', text="📹 Connecting...")
        
        def connect_thread():
            success = self.init_camera(camera_index)
            self.root.after(0, lambda: self.update_connection_status(success))
        
        threading.Thread(target=connect_thread, daemon=True).start()
    
    def update_connection_status(self, success):
        """Update connection status in GUI"""
        self.connect_button.config(state='normal', text="📹 Connect")
        
        if success:
            self.start_button.config(state='normal')
            messagebox.showinfo("Success", f"Camera {self.selected_camera} connected successfully!")
        else:
            messagebox.showerror("Error", f"Failed to connect to camera")
    
    def start_detection(self):
        """Start drowsiness detection"""
        if not self.camera_ready:
            messagebox.showwarning("Camera Error", "Please connect a camera first")
            return
        
        if not getattr(self, 'models_ready', False):
            messagebox.showerror("Model Error", 
                               "Face detection models not loaded. Please ensure "
                               "shape_predictor_68_face_landmarks.dat is in the current directory.")
            return
        
        self.detection_running = True
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')
        
        # Start video processing thread
        self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
        self.video_thread.start()
    
    def stop_detection(self):
        """Stop drowsiness detection"""
        self.detection_running = False
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        cv2.destroyAllWindows()
    
    def update_gui_from_queue(self):
        """Update GUI from queue (thread-safe)"""
        if not self.running:
            return
            
        try:
            # Process all queued status updates
            while not self.status_queue.empty():
                status_data = self.status_queue.get_nowait()
                
                if 'status' in status_data:
                    self.status_label.config(text=status_data['status'])
                if 'fps' in status_data:
                    self.fps_label.config(text=f"FPS: {status_data['fps']:.1f}")
                if 'ear' in status_data:
                    self.ear_label.config(text=f"EAR: {status_data['ear']:.3f}")
                    
        except queue.Empty:
            pass
        except Exception as e:
            print(f"GUI update error: {e}")
        
        # Schedule next update
        if self.running:
            self.root.after(100, self.update_gui_from_queue)
    
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
                self.silence_button.config(text="🔇 Silence 5s", state='normal')
                self.sound_button.config(text="🔊 Sound ON")
        
        threading.Thread(target=reset_silence, daemon=True).start()
    
    def update_sensitivity(self, value):
        """Update sensitivity threshold"""
        self.CLOSED_THRESHOLD = float(value)
        self.sensitivity_value_label.config(text=f"{float(value):.2f}")
    
    def close_application(self):
        """Close the application safely"""
        print("Closing application...")
        self.running = False
        self.detection_running = False
        self.cleanup()
        self.root.quit()
        self.root.destroy()
    
    def play_alert(self):
        """Play alert (simplified - just print for now)"""
        if not self.manual_silence:
            print("ALERT: DROWSINESS DETECTED!")
            # System beep - works on most systems
            try:
                if platform.system() == 'Windows':
                    import winsound
                    winsound.Beep(1000, 500)
                else:
                    os.system('echo -e "\\a"')
            except:
                pass
    
    def eye_aspect_ratio(self, eye):
        """Calculate eye aspect ratio"""
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)
    
    def calibrate_threshold(self, ear_value):
        """Auto-calibrate threshold"""
        self.ear_history.append(ear_value)
        
        if len(self.ear_history) > 100:
            self.ear_history.pop(0)
        
        if len(self.ear_history) >= 30:
            avg_ear = np.mean(self.ear_history)
            std_ear = np.std(self.ear_history)
            self.calibrated_threshold = max(0.15, avg_ear - 2 * std_ear)
    
    def process_frame(self, frame):
        """Process frame for drowsiness detection"""
        if frame is None:
            return None
            
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray)
            
            if len(faces) > 0:
                self.no_face_frame = 0
                
                for face in faces:
                    shape = self.predictor(gray, face)
                    shape_np = np.array([[p.x, p.y] for p in shape.parts()])
                    
                    left_eye = shape_np[self.LEFT_EYE_IDX]
                    right_eye = shape_np[self.RIGHT_EYE_IDX]
                    
                    left_ear = self.eye_aspect_ratio(left_eye)
                    right_ear = self.eye_aspect_ratio(right_eye)
                    avg_ear = (left_ear + right_ear) / 2.0
                    
                    # Calibrate threshold
                    self.calibrate_threshold(avg_ear)
                    
                    # Use calibrated threshold if available
                    current_threshold = self.calibrated_threshold if self.calibrated_threshold else self.CLOSED_THRESHOLD
                    
                    # Draw eye contours
                    for eye in [left_eye, right_eye]:
                        hull = cv2.convexHull(eye)
                        cv2.drawContours(frame, [hull], -1, (0, 255, 0), 1)
                    
                    # Draw face rectangle
                    cv2.rectangle(frame, (face.left(), face.top()), 
                                (face.right(), face.bottom()), (255, 0, 0), 2)
                    
                    # Display information
                    cv2.putText(frame, f"EAR: {avg_ear:.3f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"Threshold: {current_threshold:.3f}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Drowsiness detection
                    if avg_ear < current_threshold:
                        self.closed_eyes_frame += 1
                        cv2.putText(frame, f"Closed: {self.closed_eyes_frame}", (10, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        
                        if self.closed_eyes_frame >= self.CLOSED_CONSEC_FRAMES:
                            cv2.putText(frame, "WARNING: DROWSINESS!", (10, 120),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            self.play_alert()
                    else:
                        self.closed_eyes_frame = 0
                    
                    # Update GUI status
                    self.status_queue.put({
                        'status': 'Monitoring...',
                        'ear': avg_ear
                    })
                    
            else:
                self.no_face_frame += 1
                cv2.putText(frame, "No face detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                if self.no_face_frame >= self.NO_FACE_FRAMES:
                    cv2.putText(frame, "WARNING: Driver not visible!", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    self.play_alert()
                    
                self.status_queue.put({'status': 'No face detected'})
            
            return frame
            
        except Exception as e:
            print(f"Frame processing error: {e}")
            return frame
    
    def video_loop(self):
        """Main video processing loop"""
        print("Starting video loop...")
        
        while self.running and self.camera_ready and getattr(self, 'detection_running', False):
            try:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame")
                    break
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                if processed_frame is not None:
                    # Calculate FPS
                    self.frame_count += 1
                    elapsed = time.time() - self.start_time
                    fps = self.frame_count / elapsed if elapsed > 0 else 0
                    
                    # Update FPS in GUI
                    self.status_queue.put({'fps': fps})
                    
                    # Display frame
                    cv2.imshow("Drowsiness Detection", processed_frame)
                
                # Check for ESC key
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC key
                    break
                    
                # Small delay to prevent excessive CPU usage
                time.sleep(0.01)
                
            except Exception as e:
                print(f"Video loop error: {e}")
                break
        
        print("Video loop ended")
        cv2.destroyAllWindows()
    
    def run(self):
        """Main run method"""
        print("Driver Drowsiness Detection Started")
        print("=" * 50)
        
        try:
            # Start GUI main loop
            self.root.mainloop()
        except KeyboardInterrupt:
            print("Keyboard interrupt received")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up...")
        self.running = False
        
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
        
        cv2.destroyAllWindows()
        print("Cleanup completed")

# Camera diagnostic functions
def run_camera_diagnostics():
    """Run comprehensive camera diagnostics"""
    print("=" * 60)
    print("CAMERA DIAGNOSTICS")
    print("=" * 60)
    
    # System information
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    print(f"OpenCV: {cv2.__version__}")
    print(f"Current working directory: {os.getcwd()}")
    
    # Check permissions
    print("\n1. Checking permissions...")
    if platform.system() == 'Linux':
        # Check if user is in video group
        try:
            result = subprocess.run(['groups'], capture_output=True, text=True)
            if 'video' in result.stdout:
                print("✓ User is in video group")
            else:
                print("✗ User is NOT in video group (may cause issues)")
                print("  Fix: sudo usermod -a -G video $USER")
        except:
            print("? Could not check video group membership")
    
    # Check for video devices
    print("\n2. Checking for video devices...")
    if platform.system() == 'Linux':
        video_devices = []
        for i in range(10):
            device_path = f'/dev/video{i}'
            if os.path.exists(device_path):
                video_devices.append(device_path)
                # Check permissions
                try:
                    with open(device_path, 'rb') as f:
                        print(f"✓ {device_path} - accessible")
                except PermissionError:
                    print(f"✗ {device_path} - permission denied")
                except Exception as e:
                    print(f"? {device_path} - {e}")
        
        if not video_devices:
            print("✗ No /dev/video* devices found")
    
    # Test OpenCV backends
    print("\n3. Testing OpenCV backends...")
    backends = [
        ('CAP_ANY', cv2.CAP_ANY),
        ('CAP_V4L2', cv2.CAP_V4L2),
        ('CAP_GSTREAMER', cv2.CAP_GSTREAMER),
        ('CAP_DSHOW', cv2.CAP_DSHOW),
        ('CAP_AVFOUNDATION', cv2.CAP_AVFOUNDATION),
    ]
    
    for backend_name, backend_id in backends:
        try:
            cap = cv2.VideoCapture(0, backend_id)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    print(f"✓ {backend_name} - working")
                else:
                    print(f"✗ {backend_name} - opens but can't read")
            else:
                print(f"✗ {backend_name} - can't open")
            cap.release()
        except Exception as e:
            print(f"✗ {backend_name} - error: {e}")
    
    # Test camera indices
    print("\n4. Testing camera indices...")
    for i in range(5):
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    h, w = frame.shape[:2]
                    print(f"✓ Camera {i} - {w}x{h}")
                else:
                    print(f"✗ Camera {i} - can't read frames")
            else:
                print(f"✗ Camera {i} - can't open")
            cap.release()
        except Exception as e:
            print(f"✗ Camera {i} - error: {e}")
    
    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)

# Main execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Driver Drowsiness Detection')
    parser.add_argument('--diagnostics', action='store_true', 
                       help='Run camera diagnostics')
    parser.add_argument('--test-camera', type=int, metavar='INDEX',
                       help='Test specific camera index')
    
    args = parser.parse_args()
    
    if args.diagnostics:
        run_camera_diagnostics()
        sys.exit(0)
    
    if args.test_camera is not None:
        print(f"Testing camera {args.test_camera}...")
        try:
            cap = cv2.VideoCapture(args.test_camera)
            if cap.isOpened():
                print("Camera opened successfully")
                ret, frame = cap.read()
                if ret:
                    print(f"Frame captured: {frame.shape}")
                    cv2.imshow(f"Camera {args.test_camera} Test", frame)
                    print("Press any key to close...")
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                else:
                    print("Could not read frame")
            else:
                print("Could not open camera")
            cap.release()
        except Exception as e:
            print(f"Error: {e}")
        sys.exit(0)
    
    try:
        detector = DrowsinessDetector()
        detector.run()
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
