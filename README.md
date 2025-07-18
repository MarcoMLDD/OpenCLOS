# OpenCLOS: Driver Weariness Detection System

OpenCLOS is a project designed to detect driver weariness using a PERCLOS (Proportion of Eyelid Closure Over the Pupil Over Time) system. The files in this repository are still in development, mainly for ARM, and some support on x86_64 (Please read "Notes from Marco" below).

## Features (Beta)

* **PERCLOS-based Weariness Detection:** Utilizes the PERCLOS method to monitor eye closure and infer driver fatigue.
* **Simple Graphical User Interface (GUI):** Provides an intuitive interface for interacting with the system.
* **Real-time Camera Feed:** Displays live camera input for monitoring.

### Running the GUI

To launch the OpenCLOS GUI and begin detection:

1.  **Navigate to the OpenCLOS Directory:**
    ```bash
    cd /path/to/your/OpenCLOS/directory 
    # Or simply: cd OpenCLOS if you are in the parent directory after cloning
    ```

2.  **Execute the Python Script:**
    ```bash
    python3 OpenCLOSv_._.py #add --cli for the CLI version, available from GUIBETA-PROTO2 and further releases
    ```
    This command will open the OpenCLOS GUI window.

### Connecting Your Camera & Other Peripherals

Once the GUI is open:

1.  **Click the "Connect" Button:** Locate and tap the "Connect" button within the OpenCLOS GUI.
2.  **Camera Feedback Window:** A new window will appear, displaying the live feedback from your connected camera.
3.  **alert.wav:** Have a file named "alert.wav" in the same directory for the alarm to work. Sound support available from GUIBETA-PROTO2 and further releases.
4.  **shape_predictor_81_face_landmarks.dat:** This file needs to be in the same directory as the .py file. Check ## Acknowledgments for the GitHub repository.

### Notes from Marco
1. After release v1.1, updates/support for x86_64 and other operating systems such as Windows and macOS will end. This is so that the development would be focused on single-board ARM computers, as intentionally intended by the project team. Despite that, the code for x86_64 will remain in the source code (v1.0) and in v1.1.
2. On a single-board ARM computer, setting up the OpenCLOS system is easy. Just use the OpenCLOS-ARMINSTALL.sh (download it first lol). !!READ THE WARNINGS FIRST BEFORE PROCEEDING!!
```bash
cd /path/to/your/OpenCLOS-ARMINSTALL.sh/directory
chmod +x OpenCLOS-ARMINSTALL.sh
./OpenCLOS-ARMINSTALL.sh #add sudo if needed
```
3. This was programmed using a Fedora 42 desktop running Gnome. Do whatever you want with that information.

## Contributing

We welcome contributions to the OpenCLOS project! If you'd like to contribute, please fork the repository and submit a pull request.

## Acknowledgments
This project uses code and data from [codeniko/shape_predictor_81_face_landmarks](https://github.com/codeniko/shape_predictor_81_face_landmarks),
which is licensed under the BSD 3-Clause License.

To my research groupmates, this project wouldn't be possible without you!
* J. Rodriguez
* R. Punzalan
