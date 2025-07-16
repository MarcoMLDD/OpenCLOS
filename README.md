# OpenCLOS: Driver Weariness Detection System (Beta)

OpenCLOS is a project designed to detect driver weariness using a PERCLOS (Proportion of Eyelid Closure Over the Pupil Over Time) system. This repository contains the beta version of the OpenCLOS GUI application.

## Features (Beta)

* **PERCLOS-based Weariness Detection:** Utilizes the PERCLOS method to monitor eye closure and infer driver fatigue.
* **Simple Graphical User Interface (GUI):** Provides an intuitive interface for interacting with the system.
* **Real-time Camera Feed:** Displays live camera input for monitoring.

## Note: I'm really lazy and most of this README is AI-generated haha. So I'm REALLY REALLY sorry If I couldn't list the dependencies T_T Also, this was tested on Fedora 42. (do what you want with that info)

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
3.  **alert.wav:** Have a file named "alert.wav" on the same directory for the alarm to work. Sound support available from GUIBETA-PROTO2 and further releases.

## Contributing

We welcome contributions to the OpenCLOS project! If you'd like to contribute, please fork the repository and submit a pull request.

## Acknowledgments
This project uses code and data from [codeniko/shape_predictor_81_face_landmarks](https://github.com/codeniko/shape_predictor_81_face_landmarks),
which is licensed under the BSD 3-Clause License.

To my research groupmates, this project wouldn't be possible without you!
* J. Rodriguez
* R. Punzalan
