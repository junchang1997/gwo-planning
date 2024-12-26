## 基于灰狼算法的无人机全局路径规划

### install required python packages
```
cd gwo-path-planner
pip install -r requirements.txt
```
### install ffmpeg with MacOS
```
brew install ffmpeg
(need change some params for ffmpeg if you running in windows)
```

### usage
```
python main.py
```

## project structure 
```
├── src/
│   ├── core/                 # Core algorithm implementations
│   │   ├── gwo.py           # Grey Wolf Optimizer algorithm
│   │   ├── obj_fun.py       # Objective function for path evaluation
│   │   └── uav_setup.py     # UAV configuration and constraints
│   │
│   ├── utils/               # Utility functions
│   │   ├── encoders.py      # JSON encoders for NumPy arrays
│   │   └── export.py        # Data export functionality
│   │
│   └── visualization/       # Visualization tools
│       ├── animation.py     # 3D animation creation
│       ├── config.py        # Visualization settings
│       └── utils.py         # Visualization utilities
│
└── main.py                  # Main execution file
```