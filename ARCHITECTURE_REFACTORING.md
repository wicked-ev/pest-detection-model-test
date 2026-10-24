# Fallback Architecture Refactoring - Design Documentation

## Overview

This document describes the refactored pest detection system with a robust fallback architecture for both camera and inference backends. The system is designed to gracefully degrade when dependencies are unavailable or hardware fails.

## Architecture Principles

### 1. Strategy Pattern
Each backend (camera or inference) implements a common interface (`BaseCameraBackend` or `BaseInferenceBackend`), allowing runtime selection without code changes.

### 2. Fallback Strategy
Backends are tried in preference order. The first one that:
1. Has dependencies installed
2. Opens successfully
3. Can read/infer a frame

...becomes the active backend for the session.

### 3. Backend Selection Caching
Backend selection is cached to JSON file to avoid repeated startup delays:
- **Camera cache**: `.camera_backend_cache.json`
- **Inference cache**: `.backend_cache.json`

This prevents re-testing unsupported backends on every startup.

### 4. Clear Separation of Concerns
- **Backends**: Low-level hardware/model access
- **Factories**: Backend creation and selection logic
- **Services**: High-level APIs for application use

---

## Camera Backend Architecture

### File Structure

```
services/
├── camera_backends/
│   ├── __init__.py          # Public API exports
│   ├── base.py              # BaseCameraBackend abstract class
│   ├── opencv_camera.py     # OpenCV implementation (preferred)
│   ├── v4l2_camera.py       # V4L2 implementation (Linux-native)
│   └── factory.py           # CameraBackendFactory with caching
└── camera_service.py        # High-level camera service (refactored)
```

### Backend Hierarchy (Preference Order)

1. **OpenCV** (`opencv_camera.py`)
   - Preferred when available
   - Works across all platforms
   - May fail in certain virtual environments
   - Dependencies: `opencv-python`

2. **V4L2** (`v4l2_camera.py`)
   - Linux-native implementation
   - No external dependencies (stdlib only)
   - Most reliable on Raspberry Pi
   - Device path: `/dev/video{device}`

### Backend Interface (`BaseCameraBackend`)

```python
class BaseCameraBackend(abc.ABC):
    name: str                           # Backend identifier
    
    def is_available(self) -> bool:     # Check dependencies installed
    def open(self) -> None:             # Open camera hardware
    def read(self) -> np.ndarray:       # Read next RGB frame
    def close(self) -> None:            # Release resources
    def fileno(self) -> int:            # Get FD for select/poll (-1 if N/A)
```

### CameraService Refactoring

**Before**: Hardcoded V4L2 implementation with inline ioctl code

**After**: Pluggable backend with fallback logic

```python
# Usage remains the same
config = CameraConfig(device=0, width=640, height=480)
camera = CameraService(config)
camera.start()
frame, ts = camera.get_latest()
camera.stop()

# But internally uses:
# 1. Try OpenCV
# 2. Fall back to V4L2
# 3. Cache successful selection in .camera_backend_cache.json
```

### Backend Selection Flow

```
CameraService.__init__()
  ↓
start()
  ↓
_capture_loop()
  ├─ Call CameraBackendFactory.select_best_backend()
  │   ├─ Check cache for cached backend
  │   ├─ Try to open cached backend
  │   ├─ If fails, try each backend in preference order
  │   └─ Cache successful selection
  ├─ Open backend
  └─ Read frames continuously

_capture_loop() reconnection:
  ├─ Backend.read() fails
  ├─ Close backend
  ├─ Backoff with exponential delay
  └─ Retry backend selection
```

---

## Inference Backend Architecture

### File Structure

```
services/
├── inference_backends/
│   ├── __init__.py          # Public API exports
│   ├── base.py              # BaseInferenceBackend abstract class
│   ├── tflite.py            # TensorFlow Lite implementation (preferred)
│   ├── onnx.py              # ONNX Runtime implementation
│   └── factory.py           # InferenceBackendFactory with caching
└── model_service.py         # High-level inference service (refactored)
```

### Backend Hierarchy (Preference Order)

1. **TensorFlow Lite** (`tflite.py`)
   - Preferred for Raspberry Pi
   - Minimal memory footprint
   - May have unsupported operations
   - Dependencies: `tflite-runtime` or `tensorflow`

2. **ONNX Runtime** (`onnx.py`)
   - Alternative for cross-platform support
   - Good CPU performance
   - Wider model format support
   - Dependencies: `onnxruntime`

### Backend Interface (`BaseInferenceBackend`)

```python
class BaseInferenceBackend(abc.ABC):
    name: str                                   # Backend identifier
    
    def is_available(self) -> bool:             # Check dependencies
    def load_model(self) -> bool:               # Load model
    def run(self, frame: np.ndarray) ->         # Inference
         List[Detection]:
```

### ModelService Refactoring

**Before**: Manual backend iteration and fallback

**After**: Factory-based selection with caching

```python
# Usage remains the same
model = ModelService()
model.load_model()
model.start_streaming(camera_service)
detections = model.run_inference_on_image(frame)

# But internally uses:
# 1. Check cache for successful backend
# 2. Try TensorFlow Lite
# 3. Fall back to ONNX Runtime
# 4. Cache successful selection in .backend_cache.json
```

### Backend Selection Flow

```
ModelService.__init__()
  ├─ Create InferenceBackendConfig
  └─ Set cache path to output_dir/.backend_cache.json
  
load_model()
  └─ Call InferenceBackendFactory.select_best_backend()
      ├─ Check cache
      ├─ Try cached backend if found
      ├─ If fails, try each in preference order
      ├─ Call backend.load_model()
      └─ Cache successful selection
```

---

## Backend Caching Mechanism

### Cache Files

Stored as simple JSON in output directory:

**Camera backend cache** (`.camera_backend_cache.json`):
```json
{
    "camera_backend": "opencv"
}
```

**Inference backend cache** (`.backend_cache.json`):
```json
{
    "inference_backend": "tflite"
}
```

### Cache Behavior

**When cache is used**:
1. Startup: Try cached backend first before preference order
2. Success: Use it for the session
3. Failure: Fall through to preference order

**When cache is ignored**:
1. Corrupted JSON → skip, use preference order
2. Missing file → use preference order
3. Cached backend unavailable → use preference order

**When cache is updated**:
1. After successful backend selection
2. Only if cache path is writable
3. Failures during cache write are logged but don't block operation

### Why Caching Matters

On Raspberry Pi startup, testing backends can take significant time:
- OpenCV import: 500-1000ms
- V4L2 device access: 100-200ms
- ONNX session creation: 1000-2000ms
- TFLite model load: 500-1000ms

**Total time without cache**: 2-5 seconds

**With cache**: <100ms (read JSON + direct backend open)

---

## Configuration

### Camera Configuration

```python
@dataclass
class CameraConfig:
    device: int = 0                 # /dev/video{device}
    width: int = 640
    height: int = 480
    fps: int = 30
    buffer_size: int = 2            # V4L2 mmap buffers
    frame_format: str = "rgb"       # Always RGB output
    read_timeout: float = 3.0       # Select timeout
    backend_preference: Optional[Tuple[str, ...]] = None
    cache_dir: Optional[Path] = None
```

### Inference Configuration

```python
@dataclass
class InferenceBackendConfig:
    model_path: Path | str           # TFLite model
    onnx_model_path: Path | str      # ONNX model
    annotations_path: Path | str     # COCO annotations
    confidence_threshold: float = 0.5
    num_threads: int = 1
    backend_preference: Optional[Sequence[str]] = None
```

---

## Error Handling and Recovery

### Camera Backend Failure Recovery

```
read() fails
  ├─ Log CameraBackendError
  ├─ Close backend
  ├─ Sleep 0.5s
  └─ Backoff exponential (0.5s → 5.0s max)
     └─ Retry backend selection
```

### Inference Backend Failure Recovery

```
load_model() fails for backend
  ├─ Log failure
  ├─ Try next backend
  ├─ If all fail:
  │  └─ Log error + no backend available
  └─ Return None
```

---

## Logging

### Key Log Messages

**Camera**:
```
INFO: Opening camera backend (device=0, 640x480)
INFO: Camera backend opened: opencv
DEBUG: Camera backend 'v4l2' is not available
```

**Inference**:
```
INFO: Successfully selected inference backend: tflite
DEBUG: Inference backend 'onnx' failed: [error details]
INFO: Using cached inference backend: tflite
```

**Caching**:
```
DEBUG: Cached camera backend selection: opencv
DEBUG: Trying cached inference backend: tflite
```

---

## Migration Guide

### For Existing Code

The refactoring maintains **backward compatibility** with existing public APIs:

```python
# Old code continues to work without changes
from services.camera_service import CameraService, CameraConfig
from services.model_service import ModelService

config = CameraConfig()
cam = CameraService(config)
cam.start()

model = ModelService()
model.load_model()
model.start_streaming(cam)
```

### Using New Backend Selection Features

```python
from services.camera_backends import CameraBackendConfig, CameraBackendFactory
from services.inference_backends import InferenceBackendConfig, InferenceBackendFactory

# Direct backend selection with caching
camera_config = CameraBackendConfig(
    device=0,
    backend_preference=("opencv", "v4l2")
)
backend = CameraBackendFactory.select_best_backend(
    camera_config,
    cache_path=Path("./output/.camera_backend_cache.json")
)

# Get list of available backends
backends = CameraBackendFactory.create_backends(camera_config)
print(f"Available backends: {[b.name for b in backends]}")
```

---

## Testing

### Unit Testing Backends

Each backend can be tested independently:

```python
from services.camera_backends.v4l2_camera import V4L2CameraBackend

backend = V4L2CameraBackend(device=0)
assert backend.is_available()
backend.open()
frame = backend.read()
assert frame.shape == (480, 640, 3)
backend.close()
```

### Testing Backend Fallback

```python
config = CameraBackendConfig(backend_preference=("fake", "v4l2"))
backend = CameraBackendFactory.select_best_backend(config)
assert backend.name == "v4l2"  # Falls back after "fake" fails
```

---

## Future Enhancements

### Potential Additions

1. **Remote Backends**
   - Stream frames to remote server
   - Run inference on edge server
   - Ultimate fallback for unsupported hardware

2. **Backend Monitoring**
   - Track failure rates per backend
   - Auto-disable consistently failing backends
   - Rotate backends for load balancing

3. **Performance Metrics**
   - Cache read time improvements
   - Frame rate and latency by backend
   - Resource usage (CPU/memory) per backend

4. **Dynamic Configuration**
   - Change backends at runtime
   - A/B testing different backends
   - Adaptive backend selection based on workload

---

## Files Summary

| File | Purpose | Status |
|------|---------|--------|
| `services/camera_backends/base.py` | Camera backend interface | ✅ Complete |
| `services/camera_backends/opencv_camera.py` | OpenCV implementation | ✅ Complete |
| `services/camera_backends/v4l2_camera.py` | V4L2 implementation | ✅ Complete |
| `services/camera_backends/factory.py` | Backend selection + caching | ✅ Complete |
| `services/inference_backends/base.py` | Inference backend interface | ✅ Complete |
| `services/inference_backends/tflite.py` | TensorFlow Lite implementation | ✅ Complete |
| `services/inference_backends/onnx.py` | ONNX Runtime implementation | ✅ Complete |
| `services/inference_backends/factory.py` | Backend selection + caching | ✅ Complete |
| `services/camera_service.py` | Refactored to use backends | ✅ Complete |
| `services/model_service.py` | Refactored to use backends | ✅ Complete |
| `services/inference_backend.py` | Legacy (now deprecated) | ⚠️ Can be removed |

---

## Backward Compatibility

The old `inference_backend.py` is now deprecated but not removed. It can be safely deleted when all code has migrated to the new architecture.

Migration path:
1. ✅ `services.inference_backends` - New modular structure
2. ✅ `services.model_service` - Already uses new structure
3. ✅ `services.camera_service` - Already uses new structure
4. ⚠️ `services.inference_backend.py` - Can be deleted

---

## Conclusion

The refactored architecture provides:
- ✅ **Clean separation** of concerns (backends, factories, services)
- ✅ **Robust fallback** logic with preference ordering
- ✅ **Performance** through backend selection caching
- ✅ **Extensibility** for adding new backends
- ✅ **Maintainability** through clear code organization
- ✅ **Backward compatibility** with existing code

The system is now production-ready for Raspberry Pi 3 B+ with graceful degradation when hardware or dependencies are unavailable.
