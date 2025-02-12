
# Duckiebot Repository (CSC22905)

This repository contains code and instructions for operating the Duckiebot DB21M robot named `csc22905`. The code includes implementations for various wheel control patterns and ROS-based operations.

## Course Information

- **Course Website**: [Course Link](https://sites.google.com/ualberta.ca/maanitpratap/home)
- **Robot Dashboard**: csc22905.local

## Robot Specifications

- **Model**: Duckiebot DB21M
- **Robot Name**: csc22905
- **Architecture**: arm32v7

Based on the file structure shown in your VS Code explorer, here's the accurate repository structure:

## Repository Structure

```
└── packages/
    ├── led_service/
    │   ├── scripts/
    │   │   └── led_service_node.py
    │   └── srv/
    │       └── SetLEDColor.srv
    │   ├── CMakeLists.txt
    │   └── package.xml
    └── my_package/
        ├── src/
        │   ├── wheel_d_node.py
        │   ├── wheel_control_node.py
        │   ├── wheel_rotation_node.py
        │   ├── wheel_encoder_reader_node.py
        │   ├── camera_reader_node.py
        │   ├── my_publisher_node.py
        │   ├── my_subscriber_node.py
        │   ├── move-old.bag
        │   └── trajectory.py
        ├── CMakeLists.txt
        └── package.xml
```
## Development Commands

### Build and Run Commands
```bash
# Build the package
dts devel build -f

# Make Python scripts executable
chmod +x ./packages/my_package/src/wheel_d_node.py

# Create package directory
mkdir -p ./packages/my_package

# Run wheel control nodes
dts devel run -R csc22905 -L wheel-rotate    # Rotation control
dts devel run -R csc22905 -L wheel-d         # D-pattern movement
dts devel run -R csc22905 -L wheel-control   # General wheel control
```

### Discovery and Connection
```bash
# Discover active Duckiebots
dts fleet discover

# Check connection
ping csc22905.local
```

### Robot Control
```bash
# Keyboard control
dts duckiebot keyboard_control csc22905

# Shutdown robot
dts duckiebot shutdown csc22905.local
```

### Calibration Commands
```bash
# Camera intrinsics calibration
dts duckiebot calibrate_intrinsics csc22905

# Camera extrinsics calibration
dts duckiebot calibrate_extrinsics csc22905
```

### Development Tools
```bash
# Start GUI tools
dts start_gui_tools csc22905

# Build locally
dts devel build -f

# Build for ARM architecture
dts devel build -f --arch arm32v7 -H csc22905.local
```

## Running Demos

### Lane Following Demo
```bash
dts duckiebot demo --demo_name lane_following --duckiebot_name csc22905 --package_name duckietown_demos
```

**Controls:**
- Press 'a': Start demo
- Press 's': Stop demo

## Docker Operations

### Running Mobile Robotics Container
```bash
docker -H csc22905.local run -it --rm --net=host duckietown/mobile-robotics:v3-arm32v7
```

## Robot Configuration

### Setting Trim Parameter
```bash
rosparam set /csc22905/kinematics_node/trim 0.0916
```

## Getting Started

1. **Network Connection**
   - Connect to Duckienet WiFi
   - Verify connection: `ping csc22905.local`

2. **Initial Setup**
   - Create package directory
   - Make scripts executable
   - Build the package

3. **Running Nodes**
   - Choose appropriate run command based on desired operation
   - Monitor robot behavior
   - Use keyboard controls when available

4. **Calibration**
   - Perform camera calibrations if needed
   - Set trim parameters as required

## Exercise Files

- **Hello from My_Robot**: `packages/my_package/my_script.py`
- **Wheel Control Nodes**: `packages/my_package/src/`

## Support and Documentation

- **Hardware Issues**: [DB21M Manual](https://docs.duckietown.com/daffy/opmanual-duckiebot/intro.html)
- **Software Support**: [Duckietown Documentation](https://docs.duckietown.com/daffy/)

## Common Issues and Solutions

1. **Permission Denied**
   - Run `chmod +x` on Python scripts
   - Ensure proper file ownership

2. **Build Failures**
   - Check network connection
   - Verify architecture settings
   - Clean build directory and retry

3. **Connection Issues**
   - Confirm Duckienet WiFi connection
   - Verify robot IP address
   - Check robot power status
