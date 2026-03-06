# Video-Based Behavior Detection System — Research Documentation

## Research Overview

**Project:** Real-Time Surveillance Behavior Analysis System  
**Target Behaviors:** Fighting (*berkelahi*), Stealing (*mencuri*), Mass Brawls (*tawuran*)  
**Date:** March 2026  
**Hardware Target:** NVIDIA RTX 3060 (12 GB VRAM)

---

## 1. Problem Statement

Traditional CCTV surveillance relies on human operators monitoring multiple feeds — an approach that is error-prone, expensive, and unable to scale. This research evaluates state-of-the-art real-time object detection and pose estimation models for automated detection of three critical behaviors:

| Behavior | Indonesian Term | Detection Challenge | Typical Duration |
|----------|----------------|---------------------|------------------|
| Fighting | *Berkelahi* | Aggressive body movements between 2+ persons | 5–30 seconds |
| Stealing | *Mencuri* | Subtle hand–object interactions, concealment | 10–60 seconds |
| Mass Brawls | *Tawuran* | Dense crowds, many overlapping persons, weapons | 1–10 minutes |

### Requirements

- **Real-time processing:** ≥5 FPS per camera stream (acceptable for surveillance)
- **Deterministic latency:** Consistent inference time regardless of crowd density
- **Single-GPU deployment:** All models must fit within 12 GB VRAM (RTX 3060)
- **Multi-camera support:** Target 1–2 simultaneous camera streams

---

## 2. Models Evaluated

### 2.1 YOLOv26 (Ultralytics — September 2025)

YOLOv26 is a CNN-based, NMS-free, end-to-end real-time object detector developed by Ultralytics. It represents the latest evolution of the YOLO (You Only Look Once) family.

#### Key Architectural Innovations

| Innovation | Description | Impact |
|-----------|-------------|--------|
| **NMS-Free End-to-End** | Removes Non-Maximum Suppression post-processing entirely | Constant latency regardless of object count — critical for *tawuran* with dozens of people |
| **DFL Removal** | Eliminates Distribution Focal Loss (Softmax-heavy coordinate prediction) | 43% faster CPU inference; cleaner INT8/FP16 quantization for edge deployment |
| **MuSGD Optimizer** | Hybrid SGD + Muon optimizer (adapted from Moonshot AI's Kimi K2 LLM) | Faster training convergence, fewer epochs for fine-tuning |
| **STAL** | Small-Target-Aware Label Assignment with dynamic IoU thresholds | Better detection of small/distant persons in wide-angle CCTV |
| **RLE Pose** | Residual Log-Likelihood Estimation for keypoint localization | High-precision pose estimation under occlusion |

#### Model Variants

| Variant | Purpose | Key Capability |
|---------|---------|----------------|
| `yolo26{n,s,m,l,x}.pt` | Standard object detection | 80 COCO classes, NMS-free |
| `yolo26{s,m,l,x}-pose.pt` | Pose estimation | 17 keypoints per person (COCO format) |
| `yoloe-26{s,m,l,x}.pt` | Open-vocabulary detection | Zero-shot detection via text prompts |
| `yolo26{n,s,m,l,x}-seg.pt` | Instance segmentation | Pixel-level masks |

#### Usage

```python
from ultralytics import YOLO

# Standard detection
model = YOLO("yolo26s.pt")
results = model("frame.jpg")

# Pose estimation (for fighting/tawuran)
model = YOLO("yolo26s-pose.pt")
results = model("frame.jpg")

# Open-vocabulary (for stealing — zero-shot)
model = YOLO("yoloe-26s.pt")
model.set_classes(["person concealing object", "hand reaching into bag"])
results = model("frame.jpg")
```

### 2.2 RF-DETR (Roboflow — March 2025, ICLR 2026)

RF-DETR is a real-time transformer-based object detector developed by Roboflow. It is the first real-time model to achieve 60+ mAP on COCO and is built on a DINOv2 Vision Transformer backbone.

#### Key Architectural Features

| Feature | Description | Impact |
|---------|-------------|--------|
| **DINOv2 ViT Backbone** | Self-supervised Vision Transformer with global self-attention | Superior feature extraction; better understanding of spatial relationships |
| **Deformable Attention Decoder** | Attends to relevant image regions adaptively | Better handling of occluded and small objects |
| **NMS-Free** | End-to-end detection without post-processing | Deterministic latency (same benefit as YOLOv26) |
| **Neural Architecture Search** | Automated model design optimization | Optimized accuracy–latency trade-offs at every model size |
| **Fine-tuning Focused** | Designed for transfer learning on custom datasets | Proven generalization on RF100-VL (100 diverse domains) |

#### Model Variants

| Size | Class | COCO AP₅₀:₉₅ | Latency (ms) | Params (M) | Resolution | License |
|------|-------|--------------|--------------|------------|------------|---------|
| Nano | `RFDETRNano` | 48.4 | 2.3 | 30.5 | 384×384 | Apache 2.0 |
| Small | `RFDETRSmall` | 53.0 | 3.5 | 32.1 | 512×512 | Apache 2.0 |
| Medium | `RFDETRMedium` | 54.7 | 4.4 | 33.7 | 576×576 | Apache 2.0 |
| Large | `RFDETRLarge` | 56.5 | 6.8 | 33.9 | 704×704 | Apache 2.0 |
| XLarge | `RFDETRXLarge` | 58.6 | 11.5 | 126.4 | 700×700 | PML 1.0 |
| 2XLarge | `RFDETR2XLarge` | 60.1 | 17.2 | 126.9 | 880×880 | PML 1.0 |

#### Usage

```python
from rfdetr import RFDETRMedium
from rfdetr.util.coco_classes import COCO_CLASSES

model = RFDETRMedium()
detections = model.predict(image, threshold=0.5)
```

---

## 3. Performance Benchmarks

All benchmarks measured on **NVIDIA T4 GPU, TensorRT FP16, batch size 1**. Latency is "Total Latency" including all post-processing.

### 3.1 Object Detection — COCO val2017

| Model | COCO AP₅₀ | COCO AP₅₀:₉₅ | Latency (ms) | Params (M) | Resolution |
|-------|-----------|--------------|--------------|------------|------------|
| RF-DETR-N | **67.6** | **48.4** | 2.3 | 30.5 | 384×384 |
| RF-DETR-S | **72.1** | **53.0** | 3.5 | 32.1 | 512×512 |
| RF-DETR-M | **73.6** | **54.7** | 4.4 | 33.7 | 576×576 |
| RF-DETR-L | **75.1** | **56.5** | 6.8 | 33.9 | 704×704 |
| RF-DETR-XL | **77.4** | **58.6** | 11.5 | 126.4 | 700×700 |
| RF-DETR-2XL | **78.5** | **60.1** | 17.2 | 126.9 | 880×880 |
| YOLO26-N | 55.8 | 40.3 | **1.7** | **2.6** | 640×640 |
| YOLO26-S | 64.3 | 47.7 | **2.6** | **9.4** | 640×640 |
| YOLO26-M | 69.7 | 52.5 | **4.4** | **20.1** | 640×640 |
| YOLO26-L | 71.1 | 54.1 | **5.7** | **25.3** | 640×640 |
| YOLO26-X | 74.0 | 56.9 | **9.6** | **56.9** | 640×640 |
| YOLO11-N | 52.0 | 37.4 | 2.5 | 2.6 | 640×640 |
| YOLO11-S | 59.7 | 44.4 | 3.2 | 9.4 | 640×640 |
| YOLO11-M | 64.1 | 48.6 | 5.1 | 20.1 | 640×640 |
| YOLO11-L | 64.9 | 49.9 | 6.5 | 25.3 | 640×640 |
| YOLO11-X | 66.1 | 50.9 | 10.5 | 56.9 | 640×640 |
| D-FINE-S | 67.6 | 50.6 | 3.5 | 10.2 | 640×640 |
| D-FINE-M | 72.6 | 55.0 | 5.4 | 19.2 | 640×640 |
| D-FINE-L | 74.9 | 57.2 | 7.5 | 31.0 | 640×640 |
| LW-DETR-S | 66.8 | 48.0 | 2.6 | 14.6 | 640×640 |
| LW-DETR-M | 72.0 | 52.6 | 4.4 | 28.2 | 640×640 |
| LW-DETR-L | 74.6 | 56.1 | 6.9 | 46.8 | 640×640 |

### 3.2 Real-World Domain Adaptability — RF100-VL

| Model | RF100-VL AP₅₀ | RF100-VL AP₅₀:₉₅ |
|-------|---------------|-------------------|
| RF-DETR-N | **85.0** | **57.7** |
| RF-DETR-S | **86.7** | **60.2** |
| RF-DETR-M | **87.4** | **61.2** |
| RF-DETR-L | **88.2** | **62.2** |
| YOLO26-S | 82.7 | 57.0 |
| YOLO26-M | 84.4 | 58.7 |
| YOLO26-L | 85.0 | 59.3 |
| YOLO26-X | 85.6 | 60.0 |
| YOLO11-N | 81.4 | 55.3 |
| YOLO11-S | 82.3 | 56.2 |
| YOLO11-M | 82.5 | 56.5 |

### 3.3 Instance Segmentation — COCO val2017

| Model | COCO AP₅₀ | COCO AP₅₀:₉₅ | Latency (ms) | Params (M) |
|-------|-----------|--------------|--------------|------------|
| RF-DETR-Seg-N | **63.0** | **40.3** | 3.4 | 33.6 |
| RF-DETR-Seg-S | **66.2** | **43.1** | 4.4 | 33.7 |
| RF-DETR-Seg-M | **68.4** | **45.3** | 5.9 | 35.7 |
| YOLO26-N-Seg | 54.3 | 34.7 | **2.31** | **2.7** |
| YOLO26-S-Seg | 62.4 | 40.2 | **3.47** | **10.4** |
| YOLO26-M-Seg | 67.8 | 44.0 | **6.32** | **23.6** |
| YOLO11-N-Seg | 47.8 | 30.0 | 3.6 | 2.9 |
| YOLO11-S-Seg | 55.4 | 35.0 | 4.6 | 10.1 |
| YOLO11-M-Seg | 60.0 | 38.5 | 6.9 | 22.4 |

### 3.4 Analysis Summary

| Metric | Winner | Margin |
|--------|--------|--------|
| **COCO Accuracy (all sizes)** | RF-DETR | +2.2 to +8.1 mAP₅₀:₉₅ |
| **RF100-VL Domain Generalization** | RF-DETR | +1.2 to +3.2 mAP₅₀:₉₅ |
| **Inference Speed (all sizes)** | YOLOv26 | 16–26% faster |
| **Model Size / Memory** | YOLOv26 | 1.7x–12x fewer parameters |
| **Segmentation Accuracy** | RF-DETR | +0.1 to +5.6 mAP₅₀:₉₅ |
| **YOLOv26 vs YOLO11 (same arch)** | YOLOv26 | +2.9 to +6.0 mAP₅₀:₉₅, 26–43% faster |

---

## 4. Behavior Detection Strategy

### 4.1 Fighting (*Berkelahi*) — Pose-Based Detection ✅ Easiest

**Primary Model:** YOLOv26-Pose

Fighting detection relies on **skeletal keypoint analysis** across consecutive frames. Aggressive actions produce distinctive pose signatures.

**Detection Signals:**
- Rapid limb acceleration (punching, kicking)
- Close proximity between two or more persons (< 1 meter)
- Asymmetric body postures (one person lunging, other retreating)
- Keypoint velocity exceeding threshold over 5–10 frame window

**Keypoints Used (COCO 17-point format):**
- Wrists (IDs 9, 10) — punch/grab detection
- Ankles (IDs 15, 16) — kick detection
- Shoulders/Hips (IDs 5, 6, 11, 12) — body orientation and proximity

**Why YOLOv26-Pose:**
- Built-in RLE-based keypoint estimation — no separate model needed
- NMS-free ensures consistent detection even when fighters overlap
- 2.6 ms latency allows real-time analysis at 30+ FPS
- Pre-trained AVA dataset actions (push, hit, kick, punch) available via SlowFast for confirmation

### 4.2 Stealing (*Mencuri*) — Context-Aware Detection ⚠️ Hardest

**Primary Model:** YOLOE-26 (zero-shot first pass) + RF-DETR (fine-tuned confirmation)

Stealing is the most challenging behavior because it involves **subtle hand–object interactions** that are visually similar to normal activities.

**Detection Strategy (Multi-Stage):**

| Stage | Model | Role |
|-------|-------|------|
| 1. Object proximity | YOLOv26 | Detect persons near high-value objects (shelves, bags, displays) |
| 2. Zero-shot screening | YOLOE-26 | Text-prompted detection: `"person concealing object"`, `"hand reaching into bag"` |
| 3. Temporal analysis | SlowFast / X3D | Classify 2–4 second video clips for theft-specific motion patterns |
| 4. Fine-tuned confirmation | RF-DETR (custom) | Fine-tuned on labeled theft dataset for high-precision detection |

**Why Multi-Stage:**
- No single model reliably detects stealing out-of-the-box
- YOLOE-26 provides zero-shot capability for rapid prototyping without labeled data
- RF-DETR excels at fine-tuning on small custom datasets (proven on RF100-VL)
- SlowFast adds temporal context that frame-level detectors cannot capture

**Custom Dataset Requirements:**
- Minimum 500–1,000 labeled video clips showing theft behaviors
- Include diverse scenarios: shoplifting, pickpocketing, bag theft
- Negative samples: normal shopping, browsing, reaching for own items

### 4.3 Mass Brawls (*Tawuran*) — Crowd Density Analysis ✅ Moderate

**Primary Model:** YOLOv26 (detection + pose)

Tawuran detection combines **crowd density analysis** with **collective motion patterns**.

**Detection Signals:**
- Person count exceeding threshold in defined region (e.g., >15 persons in 50m²)
- Collective rapid movement in opposing directions (convergence pattern)
- Multiple aggressive pose signatures detected simultaneously
- Optional: weapon detection via YOLOE-26 text prompts (`"person with stick"`, `"person with weapon"`)

**Why NMS-Free Architecture is Critical:**
Traditional NMS-based models (YOLO11 and earlier) suppress overlapping bounding boxes — in a crowd of 30+ people, this causes:
- **Missed detections** (valid persons suppressed as duplicates)
- **Variable latency** (NMS processing time scales with object count)

Both YOLOv26 and RF-DETR are NMS-free, providing:
- **Constant inference time** regardless of crowd size
- **No suppression errors** — every person is detected independently

### 4.4 Comparison: Which Model for Which Behavior?

| Behavior | Best Model | Why | Accuracy | Speed |
|----------|-----------|-----|----------|-------|
| **Fighting** | YOLOv26-Pose | Built-in keypoint estimation; real-time pose analysis | High | Very Fast |
| **Stealing (prototype)** | YOLOE-26 | Zero-shot via text prompts; no labeled data needed | Moderate | Fast |
| **Stealing (production)** | RF-DETR (fine-tuned) | Best fine-tuning performance; highest detection accuracy | Very High | Fast |
| **Tawuran (detection)** | YOLOv26 | Lightweight; handles dense crowds; NMS-free | High | Very Fast |
| **Tawuran (confirmation)** | RF-DETR | Superior accuracy for counting persons in dense scenes | Very High | Fast |

---

## 5. System Architecture

### 5.1 Hybrid Pipeline Design

The system uses a **two-tier architecture** where a lightweight model runs continuously (Tier 1) and a heavier model is triggered on-demand (Tier 2).

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CCTV Camera Feed                            │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  TIER 1: Always-On (YOLOv26-Pose, ~2 ms/frame)                    │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │ Person       │  │ Pose         │  │ Behavior Trigger         │  │
│  │ Detection    │──│ Estimation   │──│ • Fighting → ALERT       │  │
│  │ (bbox)       │  │ (17 keypts)  │  │ • Crowd density → Tier 2 │  │
│  └──────────────┘  └──────────────┘  │ • Suspicious → Tier 2    │  │
│                                       └──────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ (triggered only when needed)
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  TIER 2: On-Demand (RF-DETR / SlowFast / YOLOE-26)                │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │ RF-DETR-M    │  │ SlowFast /   │  │ YOLOE-26               │  │
│  │ (fine-tuned) │  │ X3D-S        │  │ (open-vocabulary)       │  │
│  │ Stealing     │  │ Temporal     │  │ Text-prompted           │  │
│  │ confirmation │  │ analysis     │  │ zero-shot detection     │  │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  ALERT SYSTEM                                                       │
│  • Classification: fighting / stealing / tawuran                    │
│  • Confidence score                                                  │
│  • Bounding box / region of interest                                │
│  • Video clip extraction for review                                 │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 Processing Pipeline — Step by Step

| Step | Component | Model | Runs | Latency |
|------|-----------|-------|------|---------|
| 1 | Frame capture | — | Every frame | <1 ms |
| 2 | Person detection + pose | YOLOv26-Pose-S | Every 2nd–3rd frame | ~2.6 ms |
| 3 | Person tracking | ByteTrack | Every frame | <1 ms |
| 4 | Pose analysis (fighting) | Rule-based / ML classifier | Every frame with persons | <1 ms |
| 5 | Crowd density check (tawuran) | Person count per ROI | Every frame | <1 ms |
| 6 | Suspicious activity flag | Trigger logic | On threshold breach | <1 ms |
| 7a | Stealing confirmation | RF-DETR-M (fine-tuned) | On-demand | ~4.4 ms |
| 7b | Temporal behavior analysis | SlowFast / X3D-S | On-demand (2–4s clip) | ~50–100 ms |
| 7c | Zero-shot screening | YOLOE-26 | On-demand | ~2.6 ms |
| 8 | Alert dispatch | Alert system | On confirmed detection | <1 ms |

### 5.3 Implementation Example

```python
from ultralytics import YOLO
import numpy as np

# ── Tier 1: Always-on detection ──────────────────────────────────────
pose_model = YOLO("yolo26s-pose.pt")
tracker = ByteTrack()  # or DeepSORT

def process_frame(frame):
    # Step 1: Detect persons + keypoints
    results = pose_model(frame, conf=0.5, classes=[0])  # class 0 = person

    # Step 2: Track persons across frames
    detections = results[0].boxes
    keypoints = results[0].keypoints
    tracked = tracker.update(detections)

    # Step 3: Analyze behaviors
    alerts = []

    # Fighting detection: check for aggressive pose patterns
    for person_a, person_b in get_close_pairs(tracked, threshold_meters=1.5):
        if detect_fighting_pose(person_a.keypoints, person_b.keypoints):
            alerts.append({
                "type": "fighting",
                "confidence": calculate_confidence(person_a, person_b),
                "persons": [person_a.id, person_b.id]
            })

    # Tawuran detection: crowd density check
    person_count = len(tracked)
    if person_count > 15:  # threshold for mass gathering
        alerts.append({
            "type": "tawuran_warning",
            "person_count": person_count,
            "trigger": "crowd_density"
        })

    # Trigger Tier 2 for suspicious activities
    for alert in alerts:
        if alert["type"] in ["suspicious_proximity", "tawuran_warning"]:
            trigger_tier2(frame, alert)

    return alerts

# ── Tier 2: On-demand confirmation ──────────────────────────────────
from rfdetr import RFDETRMedium

rfdetr_model = RFDETRMedium()  # fine-tuned on stealing dataset
yoloe_model = YOLO("yoloe-26s.pt")

def trigger_tier2(frame, alert):
    if alert["type"] == "suspicious_proximity":
        # Zero-shot stealing detection
        yoloe_model.set_classes([
            "person concealing object",
            "hand reaching into bag",
            "person hiding item under clothing"
        ])
        results = yoloe_model(frame)

        # High-accuracy confirmation
        detections = rfdetr_model.predict(frame, threshold=0.6)

    elif alert["type"] == "tawuran_warning":
        # Accurate person count with RF-DETR
        detections = rfdetr_model.predict(frame, threshold=0.4)
        accurate_count = len(detections)
```

---

## 6. Hardware & Deployment Recommendations

### 6.1 VRAM Budget (RTX 3060 — 12 GB)

| Component | VRAM Usage (FP16) | Notes |
|-----------|-------------------|-------|
| YOLOv26-S-Pose | ~0.8 GB | Always loaded |
| ByteTrack | ~0.1 GB | CPU-based, minimal GPU |
| RF-DETR-M (on-demand) | ~2.5 GB | Loaded/unloaded as needed |
| YOLOE-26-S (on-demand) | ~1.0 GB | Shares backbone with YOLOv26 |
| SlowFast / X3D-S (on-demand) | ~1.5 GB | Loaded only for temporal analysis |
| **Total (worst case)** | **~5.9 GB** | Well within 12 GB budget |

### 6.2 Throughput Estimates (RTX 3060)

| Configuration | FPS | Cameras | Use Case |
|--------------|-----|---------|----------|
| YOLOv26-S-Pose only | ~120 FPS | 4–6 streams at 20 FPS | Fighting + tawuran only |
| YOLOv26-S-Pose + ByteTrack | ~80 FPS | 2–4 streams at 20 FPS | Full tracking pipeline |
| Hybrid (Tier 1 + Tier 2 on-demand) | ~30–60 FPS | 1–2 streams at 15 FPS | All 3 behaviors |
| RF-DETR-M continuous | ~45 FPS | 1–2 streams at 15 FPS | Maximum accuracy mode |

### 6.3 Optimization Techniques

| Technique | Impact | How |
|-----------|--------|-----|
| **TensorRT export** | 2–3x speedup | `model.export(format="engine", half=True)` |
| **FP16 inference** | ~40% less VRAM | Minimal accuracy loss (<0.1 mAP) |
| **Frame skipping** | 2–3x throughput | Process every 2nd or 3rd frame; sufficient for surveillance |
| **ROI cropping** | Reduced compute | Only process regions of interest, not full frame |
| **Dynamic model loading** | Lower peak VRAM | Load RF-DETR/SlowFast only when triggered |
| **INT8 quantization** | Further speedup | YOLOv26 DFL-free design makes INT8 cleaner |

### 6.4 Deployment Recommendations by Budget

| Budget | Hardware | Models | Cameras | Behaviors |
|--------|----------|--------|---------|-----------|
| **Low (~$300)** | 1× RTX 3060 | YOLOv26-Pose only | 1–2 | Fighting, tawuran |
| **Medium (~$500)** | 1× RTX 3080 | YOLOv26-Pose + RF-DETR-M | 1–2 | All 3 behaviors |
| **High (~$1,600)** | 1× RTX 4090 | Full hybrid pipeline | 3–4 | All 3, high accuracy |
| **Production** | 2× RTX 3060 or Cloud T4 | Dedicated per-tier GPUs | 4+ | All 3, redundancy |

### 6.5 Edge Deployment (Jetson / CPU)

| Aspect | YOLOv26 | RF-DETR |
|--------|---------|---------|
| **Jetson Orin Nano** | ✅ Excellent (2.6M params, DFL-free) | ⚠️ Heavy (30M+ params, ViT) |
| **CPU-only** | ✅ 43% faster than YOLO11 on CPU | ❌ Not recommended (ViT too slow) |
| **INT8 quantization** | ✅ Clean (no DFL Softmax) | ⚠️ ViT quantization more complex |
| **ONNX/TensorRT export** | ✅ Full support | ✅ Full support |
| **CoreML (Apple)** | ✅ Supported | ⚠️ In development |

**Edge recommendation:** Use YOLOv26-Pose exclusively on edge devices. Reserve RF-DETR for server/cloud deployment where GPU resources are available.

---

## 7. Conclusions & Recommendations

### 7.1 Model Selection Summary

| | YOLOv26 | RF-DETR |
|--|---------|---------|
| **Architecture** | CNN (lightweight, efficient) | Transformer (accurate, heavy) |
| **Strength** | Speed, model size, pose estimation, open-vocab | Accuracy, fine-tuning, domain generalization |
| **Weakness** | Lower accuracy than RF-DETR | No pose estimation, heavier model |
| **License** | AGPL-3.0 (restrictive for commercial) | Apache 2.0 (permissive) |
| **Best role** | Always-on primary detector (Tier 1) | On-demand high-accuracy confirmer (Tier 2) |

### 7.2 Final Recommendation

**Use a hybrid approach:**

1. **YOLOv26-Pose** as the always-on Tier 1 detector for fighting and tawuran — it provides built-in pose estimation, NMS-free deterministic latency, and the lightest resource footprint.

2. **YOLOE-26** for zero-shot stealing detection during prototyping — enables rapid iteration without labeled data.

3. **RF-DETR** (fine-tuned) as the Tier 2 high-accuracy confirmer for stealing and ambiguous cases — its superior accuracy (+2–8 mAP) and proven fine-tuning capability on custom datasets make it ideal for production-quality behavior classification.

4. **SlowFast / X3D** for temporal behavior analysis when frame-level detection is insufficient — particularly for stealing where the action unfolds over multiple seconds.

This hybrid approach maximizes both **speed** (YOLOv26 strengths) and **accuracy** (RF-DETR strengths) while staying within a single RTX 3060's 12 GB VRAM budget.

### 7.3 Recommended Development Roadmap

| Phase | Duration | Goal | Models |
|-------|----------|------|--------|
| **Phase 1: Prototype** | 2–4 weeks | Detect fighting with pre-trained models | YOLOv26-Pose |
| **Phase 2: Expand** | 2–4 weeks | Add tawuran detection; test YOLOE-26 for stealing | YOLOv26-Pose + YOLOE-26 |
| **Phase 3: Custom Data** | 4–8 weeks | Collect and label stealing dataset (500+ clips) | Data collection |
| **Phase 4: Fine-tune** | 2–4 weeks | Fine-tune RF-DETR on custom stealing dataset | RF-DETR-M |
| **Phase 5: Integration** | 2–4 weeks | Build full hybrid pipeline with alert system | All models |
| **Phase 6: Production** | 2–4 weeks | TensorRT optimization, monitoring, deployment | Optimized pipeline |

---

## References

1. **YOLOv26** — Ultralytics (September 2025). NMS-free, DFL-free real-time object detection. https://docs.ultralytics.com/models/yolo26/
2. **RF-DETR** — Roboflow (March 2025, ICLR 2026). Real-time detection transformer with DINOv2 backbone. https://github.com/roboflow/rf-detr
3. **SlowFast Networks** — Feichtenhofer et al. (ICCV 2019). Dual-pathway temporal action recognition. https://github.com/facebookresearch/SlowFast
4. **X3D** — Feichtenhofer (CVPR 2020). Efficient video recognition networks. Part of SlowFast repository.
5. **ByteTrack** — Zhang et al. (ECCV 2022). Multi-object tracking by associating every detection box. https://github.com/ifzhang/ByteTrack
6. **DINOv2** — Oquab et al. (2023). Self-supervised Vision Transformer features. https://github.com/facebookresearch/dinov2
7. **COCO Dataset** — Lin et al. (2014). Microsoft Common Objects in Context. https://cocodataset.org
8. **RF100-VL** — Roboflow. 100 diverse real-world detection datasets. https://github.com/roboflow/rf100-vl
9. **AVA Dataset** — Gu et al. (CVPR 2018). Atomic Visual Actions for action detection.

---
---

# Part II — Pipeline Architecture Analysis & Integration Guide

**Date:** March 2026
**Hardware:** 2× NVIDIA V100 32 GB
**Target:** Multi-camera CCTV surveillance with behavior detection

---

## 8. Current Pipeline Review

### 8.1 Current Architecture

```
Thread per CCTV stream (capture frame at 1 FPS)
  → Save frame in shared memory buffer
    → Pool/batch frames for YOLO detection pipeline
      → Get bounding boxes (object detection)
        → Crop bounding box regions and save as images
          → Pool/batch cropped images for async save to MongoDB
```

### 8.2 Identified Bottlenecks & Issues

| # | Issue | Severity | Impact |
|---|-------|----------|--------|
| 1 | **Thread-per-stream scaling** | 🔴 Critical | Python's GIL prevents true parallelism. At 200+ streams, thread overhead dominates. Context switching between hundreds of threads wastes CPU cycles. |
| 2 | **Synchronous RTSP decode** | 🔴 Critical | FFmpeg/OpenCV decode is CPU-bound. Each 1080p decode uses ~0.3–0.5 CPU core. 500 streams = 150–250 cores just for decode. |
| 3 | **Shared memory buffer — no backpressure** | 🟡 High | If YOLO batching falls behind capture rate, frames accumulate unbounded in memory. No drop policy = OOM risk. |
| 4 | **Single YOLO model bottleneck** | 🟡 High | All streams funnel into one YOLO instance. If batch queue stalls, all streams back up. No priority or fairness. |
| 5 | **Crop → save as image → MongoDB** | 🟡 High | JPEG encoding is CPU-bound (~2–5 ms per crop). Saving to disk then re-reading for MongoDB is redundant I/O. |
| 6 | **No tracking across frames** | 🟠 Medium | Without person tracking (ByteTrack), same person is re-detected every frame. No temporal identity = no behavior analysis possible. |
| 7 | **No behavior analysis layer** | 🟠 Medium | Pipeline ends at "save crops." No pose analysis, no temporal analysis, no alert system. |
| 8 | **No GPU decode (NVDEC)** | 🟠 Medium | CPU decode wastes cores that could process more streams. V100 NVDEC engines sit idle. |

### 8.3 Recommended Revised Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CAPTURE LAYER (async)                               │
│                                                                             │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐       ┌───────────┐          │
│  │ Camera 1  │  │ Camera 2  │  │ Camera 3  │  ...  │ Camera N  │          │
│  │ RTSP pull │  │ RTSP pull │  │ RTSP pull │       │ RTSP pull │          │
│  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘       └─────┬─────┘          │
│        │              │              │                    │               │
│        └──────────────┴──────┬───────┴────────────────────┘               │
│                              │                                             │
│                    ┌─────────▼──────────┐                                  │
│                    │  Ring Buffer Pool   │  ← Fixed-size, per-camera       │
│                    │  (drop-oldest)      │  ← Backpressure: drop frames    │
│                    └─────────┬──────────┘                                  │
└──────────────────────────────┼──────────────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────────────┐
│                    TIER 1: ALWAYS-ON (GPU 0)                                │
│                                                                             │
│  ┌──────────────────────────────────────────────────────┐                  │
│  │  Batch Assembler                                      │                  │
│  │  • Collect frames from ring buffers                   │                  │
│  │  • Dynamic batch size (8–32 based on queue depth)     │                  │
│  │  • Priority: cameras with recent alerts first         │                  │
│  └──────────────────────┬───────────────────────────────┘                  │
│                         │                                                   │
│  ┌──────────────────────▼───────────────────────────────┐                  │
│  │  YOLOv26-S-Pose (TensorRT FP16)                      │                  │
│  │  → Person bboxes + 17 keypoints per person            │                  │
│  │  → ~1.5 ms/image on V100                              │                  │
│  └──────────────────────┬───────────────────────────────┘                  │
│                         │                                                   │
│  ┌──────────────────────▼───────────────────────────────┐                  │
│  │  ByteTrack (CPU)                                      │                  │
│  │  → Track person IDs across frames                     │                  │
│  │  → Maintain per-person keypoint history (last 10 fr)  │                  │
│  └──────────────────────┬───────────────────────────────┘                  │
│                         │                                                   │
│  ┌──────────────────────▼───────────────────────────────┐                  │
│  │  Behavior Analyzer (CPU)                              │                  │
│  │  ├─ Fighting: keypoint velocity + proximity           │                  │
│  │  ├─ Tawuran: crowd density + collective motion        │                  │
│  │  └─ Suspicious: person-object proximity zones         │                  │
│  └──────────┬────────────────────────┬──────────────────┘                  │
│             │ direct alert           │ trigger Tier 2                       │
│             ▼                        ▼                                      │
│      ┌─────────────┐     ┌─────────────────────────┐                      │
│      │ Alert Queue  │     │ Tier 2 Request Queue    │                      │
│      └──────┬──────┘     └────────────┬────────────┘                      │
└─────────────┼─────────────────────────┼─────────────────────────────────────┘
              │                         │
              │    ┌────────────────────▼──────────────────────────────────┐
              │    │             TIER 2: ON-DEMAND (GPU 1)                 │
              │    │                                                       │
              │    │  ┌─────────────┐ ┌─────────────┐ ┌───────────────┐  │
              │    │  │ RF-DETR-M   │ │ SlowFast    │ │ YOLOE-26      │  │
              │    │  │ (fine-tuned)│ │ X3D-S       │ │ (open-vocab)  │  │
              │    │  │ Stealing    │ │ Temporal     │ │ Zero-shot     │  │
              │    │  │ ~2.7ms V100 │ │ ~50–80ms    │ │ ~1.5ms V100   │  │
              │    │  └──────┬──────┘ └──────┬──────┘ └───────┬───────┘  │
              │    │         └───────────┬────┴───────────────┘           │
              │    │                     ▼                                 │
              │    │           ┌──────────────────┐                       │
              │    │           │ Confirmation      │                       │
              │    │           │ Aggregator        │                       │
              │    │           └────────┬─────────┘                       │
              │    └───────────────────┼───────────────────────────────────┘
              │                        │
              ▼                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         OUTPUT LAYER                                        │
│                                                                             │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐                  │
│  │ MongoDB       │  │ Alert         │  │ Video Clip    │                  │
│  │ (crops+meta)  │  │ Dispatcher    │  │ Extractor     │                  │
│  │ Async bulk    │  │ WebSocket/API │  │ (evidence)    │                  │
│  └───────────────┘  └───────────────┘  └───────────────┘                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.4 Architectural Improvements — Detail

#### A. Replace Thread-per-Stream with Producer-Consumer + asyncio

**Problem:** Python threads hold the GIL; 500 threads = massive context-switch overhead.

**Solution:** Use a small **process pool** for CPU-bound decode, feeding into an **asyncio** event loop for I/O coordination.

```python
import asyncio
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from collections import deque
import cv2
import numpy as np

# ── Ring buffer per camera (fixed-size, drop-oldest) ──────────────────
class CameraRingBuffer:
    """Lock-free-ish ring buffer. Drops oldest frame when full."""
    def __init__(self, camera_id: str, max_size: int = 3):
        self.camera_id = camera_id
        self.buffer = deque(maxlen=max_size)  # auto-drops oldest
        self.frame_count = 0
        self.dropped = 0

    def put(self, frame: np.ndarray, timestamp: float):
        if len(self.buffer) == self.buffer.maxlen:
            self.dropped += 1
        self.buffer.append((frame, timestamp, self.frame_count))
        self.frame_count += 1

    def get_latest(self):
        return self.buffer[-1] if self.buffer else None

# ── RTSP decode in separate process (bypass GIL) ─────────────────────
def decode_worker(rtsp_url: str, shared_queue: mp.Queue, camera_id: str):
    """Runs in a subprocess. Decodes RTSP at 1 FPS."""
    cap = cv2.VideoCapture(rtsp_url)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            try:
                shared_queue.put_nowait((camera_id, frame, time.time()))
            except mp.queues.Full:
                pass  # backpressure: drop frame silently
        time.sleep(1.0)  # 1 FPS capture rate

# ── Main async coordinator ───────────────────────────────────────────
class PipelineCoordinator:
    def __init__(self, camera_urls: dict[str, str]):
        self.buffers: dict[str, CameraRingBuffer] = {}
        self.frame_queue = mp.Queue(maxsize=1000)
        self.decode_pool = ProcessPoolExecutor(max_workers=mp.cpu_count() // 2)

        for cam_id, url in camera_urls.items():
            self.buffers[cam_id] = CameraRingBuffer(cam_id, max_size=3)

    async def run(self):
        # Start decode workers as subprocesses
        for cam_id, url in camera_urls.items():
            self.decode_pool.submit(decode_worker, url, self.frame_queue, cam_id)

        # Main loop: drain queue → fill ring buffers → assemble batches
        while True:
            batch = self._assemble_batch(max_batch_size=32)
            if batch:
                results = await self._run_tier1(batch)
                await self._process_results(results)
            await asyncio.sleep(0.001)  # yield control
```

#### B. Backpressure Strategy

| Scenario | Action | Implementation |
|----------|--------|----------------|
| Frame queue full | **Drop newest frame** from that camera | `put_nowait` + catch `Full` |
| GPU batch queue full | **Drop oldest batch** (stale frames) | Ring buffer with `maxlen=3` per camera |
| Tier 2 queue full | **Drop lowest-confidence requests** | Priority queue sorted by confidence |
| MongoDB write slow | **Buffer in memory, bulk write** | Async bulk insert every 5 seconds |
| Network congestion | **Reduce capture FPS temporarily** | Adaptive sleep: `1.0 → 2.0s` |

#### C. Eliminate Redundant I/O (Crop → Disk → MongoDB)

**Current:** Crop → encode JPEG → save to disk → read from disk → upload to MongoDB
**Improved:** Crop → encode JPEG in-memory → bulk insert to MongoDB directly

```python
import motor.motor_asyncio  # async MongoDB driver
import cv2

async def save_crops_batch(crops: list[tuple[str, np.ndarray, dict]]):
    """Bulk save crops directly to MongoDB GridFS — no disk I/O."""
    client = motor.motor_asyncio.AsyncIOMotorClient("mongodb://localhost:27017")
    db = client.surveillance

    documents = []
    for camera_id, crop_img, metadata in crops:
        _, jpeg_bytes = cv2.imencode(".jpg", crop_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        documents.append({
            "camera_id": camera_id,
            "image": jpeg_bytes.tobytes(),  # binary, no disk round-trip
            "timestamp": metadata["timestamp"],
            "bbox": metadata["bbox"],
            "person_id": metadata.get("track_id"),
            "behavior_flags": metadata.get("flags", []),
        })

    if documents:
        await db.detections.insert_many(documents)  # single bulk operation
```

---

## 9. YOLOv26-Pose Integration (Fighting & Tawuran)

### 9.1 Integration Strategy: Replace, Don't Run Alongside

**Replace** the current YOLO detection model with YOLOv26-S-Pose. It provides:
- Everything the current YOLO does (bounding boxes, class detection)
- **Plus** 17 keypoints per person (pose estimation)
- **Plus** NMS-free deterministic latency
- **Faster** than YOLO11 equivalent (~26–43% faster)

There is no reason to run both — YOLOv26-Pose is a strict superset.

### 9.2 Where Pose Analysis Fits in the Data Flow

```
Frame from ring buffer
  │
  ▼
YOLOv26-S-Pose inference (GPU, ~1.5 ms on V100)
  │
  ├── outputs: bboxes + confidence + class + 17 keypoints per person
  │
  ▼
ByteTrack (CPU, <1 ms)
  │
  ├── outputs: tracked person IDs + keypoint history per track
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│ Behavior Analyzer (CPU, <1 ms total)                    │
│                                                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │ A. Fighting Detector                             │    │
│  │    • Input: keypoints of all close pairs         │    │
│  │    • Method: velocity + acceleration of wrists/  │    │
│  │      ankles over 5-frame sliding window          │    │
│  │    • Threshold: >X px/frame limb movement        │    │
│  │    • Output: ALERT if confirmed                  │    │
│  └─────────────────────────────────────────────────┘    │
│                                                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │ B. Tawuran Detector                              │    │
│  │    • Input: all person bboxes + keypoints        │    │
│  │    • Method 1: person_count > 15 in ROI          │    │
│  │    • Method 2: centroid convergence velocity     │    │
│  │    • Method 3: % of persons in fighting pose     │    │
│  │    • Output: ALERT if 2+ methods trigger         │    │
│  └─────────────────────────────────────────────────┘    │
│                                                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │ C. Suspicious Activity Flagger                   │    │
│  │    • Input: person tracks near defined zones     │    │
│  │    • Method: dwell time + hand movement pattern  │    │
│  │    • Output: TRIGGER TIER 2 for confirmation     │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

### 9.3 Fighting Detection — Keypoint Velocity Algorithm

```python
import numpy as np
from collections import defaultdict

class FightingDetector:
    """Detects fighting based on keypoint velocity analysis."""

    WRIST_IDS = [9, 10]      # left/right wrist
    ANKLE_IDS = [15, 16]     # left/right ankle
    SHOULDER_IDS = [5, 6]    # proximity reference
    HIP_IDS = [11, 12]       # proximity reference

    def __init__(
        self,
        velocity_threshold: float = 40.0,   # pixels/frame
        proximity_threshold: float = 150.0,  # pixels (approx 1.5m at typical CCTV)
        window_size: int = 5,                # frames to analyze
        min_aggressive_frames: int = 3,      # frames exceeding threshold
    ):
        self.velocity_threshold = velocity_threshold
        self.proximity_threshold = proximity_threshold
        self.window_size = window_size
        self.min_aggressive_frames = min_aggressive_frames
        self.keypoint_history: dict[int, list[np.ndarray]] = defaultdict(list)

    def update(self, track_id: int, keypoints: np.ndarray):
        """Store keypoints for a tracked person. keypoints shape: (17, 2)."""
        history = self.keypoint_history[track_id]
        history.append(keypoints.copy())
        if len(history) > self.window_size:
            history.pop(0)

    def _limb_velocity(self, track_id: int) -> float:
        """Compute max limb velocity over recent frames."""
        history = self.keypoint_history.get(track_id, [])
        if len(history) < 2:
            return 0.0

        max_vel = 0.0
        limb_ids = self.WRIST_IDS + self.ANKLE_IDS
        for i in range(1, len(history)):
            for lid in limb_ids:
                delta = np.linalg.norm(history[i][lid] - history[i-1][lid])
                max_vel = max(max_vel, delta)
        return max_vel

    def _torso_distance(self, kp_a: np.ndarray, kp_b: np.ndarray) -> float:
        """Distance between torso centers of two persons."""
        center_a = np.mean(kp_a[self.SHOULDER_IDS + self.HIP_IDS], axis=0)
        center_b = np.mean(kp_b[self.SHOULDER_IDS + self.HIP_IDS], axis=0)
        return np.linalg.norm(center_a - center_b)

    def check_pair(self, id_a: int, id_b: int) -> dict | None:
        """Check if two tracked persons are fighting."""
        hist_a = self.keypoint_history.get(id_a, [])
        hist_b = self.keypoint_history.get(id_b, [])
        if not hist_a or not hist_b:
            return None

        # Check proximity (must be close)
        dist = self._torso_distance(hist_a[-1], hist_b[-1])
        if dist > self.proximity_threshold:
            return None

        # Check velocity (both must have aggressive movement)
        vel_a = self._limb_velocity(id_a)
        vel_b = self._limb_velocity(id_b)

        if vel_a > self.velocity_threshold and vel_b > self.velocity_threshold:
            confidence = min(1.0, (vel_a + vel_b) / (4 * self.velocity_threshold))
            return {
                "type": "fighting",
                "person_ids": [id_a, id_b],
                "distance_px": dist,
                "velocity_a": vel_a,
                "velocity_b": vel_b,
                "confidence": round(confidence, 3),
            }
        return None
```

### 9.4 Tawuran Detection — Crowd Density + Collective Motion

```python
class TawuranDetector:
    """Detects mass brawls via crowd density and collective aggression."""

    def __init__(
        self,
        crowd_threshold: int = 15,          # persons in ROI
        convergence_threshold: float = 0.6,  # 60% moving toward center
        fighting_ratio_threshold: float = 0.3,  # 30% in aggressive poses
    ):
        self.crowd_threshold = crowd_threshold
        self.convergence_threshold = convergence_threshold
        self.fighting_ratio_threshold = fighting_ratio_threshold
        self.prev_centroids: dict[int, np.ndarray] = {}

    def analyze(
        self,
        tracked_persons: list[dict],  # [{id, bbox, keypoints}, ...]
        roi: tuple[int, int, int, int] | None = None,  # (x1, y1, x2, y2)
        fighting_detector: FightingDetector = None,
    ) -> dict | None:

        # Filter to ROI
        if roi:
            in_roi = [p for p in tracked_persons if self._in_roi(p["bbox"], roi)]
        else:
            in_roi = tracked_persons

        person_count = len(in_roi)
        if person_count < self.crowd_threshold:
            return None

        # Check 1: Crowd density exceeded
        signals = ["crowd_density"]

        # Check 2: Convergence — are people moving toward each other?
        centroids = {p["id"]: self._centroid(p["bbox"]) for p in in_roi}
        group_center = np.mean(list(centroids.values()), axis=0)

        converging = 0
        for pid, pos in centroids.items():
            prev = self.prev_centroids.get(pid)
            if prev is not None:
                prev_dist = np.linalg.norm(prev - group_center)
                curr_dist = np.linalg.norm(pos - group_center)
                if curr_dist < prev_dist:
                    converging += 1

        self.prev_centroids = centroids

        if person_count > 0 and converging / person_count > self.convergence_threshold:
            signals.append("convergence")

        # Check 3: Multiple fighting poses
        if fighting_detector:
            aggressive = sum(
                1 for p in in_roi
                if fighting_detector._limb_velocity(p["id"]) > fighting_detector.velocity_threshold
            )
            if person_count > 0 and aggressive / person_count > self.fighting_ratio_threshold:
                signals.append("collective_aggression")

        if len(signals) >= 2:  # at least 2 signals to confirm
            return {
                "type": "tawuran",
                "person_count": person_count,
                "signals": signals,
                "confidence": min(1.0, len(signals) / 3),
            }
        return None

    def _in_roi(self, bbox, roi) -> bool:
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        return roi[0] <= cx <= roi[2] and roi[1] <= cy <= roi[3]

    def _centroid(self, bbox) -> np.ndarray:
        return np.array([(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2])
```

---

## 10. SlowFast / X3D Integration (Temporal Behavior Analysis)

### 10.1 The Challenge: 1 FPS → Video Clips

SlowFast and X3D expect **multi-frame video clips** (typically 8–32 frames at ≥8 FPS). Your pipeline captures at **1 FPS**. This mismatch requires a buffering strategy.

| Approach | Clip Requirement | At 1 FPS | Clip Duration | Feasibility |
|----------|-----------------|----------|---------------|-------------|
| **SlowFast 8×8** | 8 frames, stride 8 | 8 frames = 8 seconds buffer | 8 seconds | ✅ Good — matches stealing/fighting duration |
| **SlowFast 4×16** | 4 frames, stride 16 | 4 frames = 4 seconds buffer | 4 seconds | ✅ Good — quick events |
| **X3D-S** | 13 frames, stride 6 | 13 frames = 13 seconds buffer | 13 seconds | ✅ Covers full tawuran build-up |
| **SlowFast 16×8** | 16 frames, stride 8 | 16 frames = 16 seconds | 16 seconds | ⚠️ Long buffer but better accuracy |

> **Key insight:** At 1 FPS, the temporal resolution is low, but the clip duration is inherently long. SlowFast can still extract useful **coarse temporal patterns** (person appearing → approaching → grabbing → leaving) even at 1 FPS. For fine-grained motion (punch trajectories), the keypoint velocity from YOLOv26-Pose (Tier 1) is already covering this at per-frame level.

### 10.2 When to Trigger SlowFast (On-Demand Only)

SlowFast should **never** run continuously. It runs only when Tier 1 flags suspicious activity:

| Trigger Source | Trigger Condition | SlowFast Task | Priority |
|---------------|-------------------|---------------|----------|
| Fighting detector | Fighting confidence > 0.5 but < 0.8 | Confirm/deny fight action | High |
| Suspicious activity | Person lingering near valuables > 10s | Classify stealing behavior | High |
| Tawuran detector | Crowd threshold met | Classify crowd action (running, fighting) | Medium |
| Periodic audit | Random 1% of cameras, every 60s | Background anomaly detection | Low |

### 10.3 Per-Person Clip Buffer

```python
from collections import deque
from dataclasses import dataclass, field
import numpy as np
import time

@dataclass
class PersonClipBuffer:
    """Maintains a rolling frame buffer per tracked person for SlowFast."""
    track_id: int
    max_frames: int = 16                    # 16 seconds at 1 FPS
    frames: deque = field(default_factory=lambda: deque(maxlen=16))
    crops: deque = field(default_factory=lambda: deque(maxlen=16))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=16))

    # Memory estimate: 16 frames × 224×224×3 (crop) ≈ 2.4 MB per person
    # 100 tracked persons ≈ 240 MB — manageable

    def add_frame(self, full_frame: np.ndarray, bbox: tuple, timestamp: float):
        """Add a frame crop for this tracked person."""
        x1, y1, x2, y2 = [int(c) for c in bbox]
        # Pad bbox by 20% for context
        h, w = full_frame.shape[:2]
        pad_x = int((x2 - x1) * 0.2)
        pad_y = int((y2 - y1) * 0.2)
        x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        x2, y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)

        crop = full_frame[y1:y2, x1:x2]
        # Resize to SlowFast input size
        crop_resized = cv2.resize(crop, (224, 224))

        self.frames.append(full_frame)  # keep full frame too (for RF-DETR)
        self.crops.append(crop_resized)
        self.timestamps.append(timestamp)

    def get_clip(self, num_frames: int = 8) -> np.ndarray | None:
        """Get a clip tensor for SlowFast. Returns (T, H, W, C) array."""
        if len(self.crops) < num_frames:
            return None
        recent = list(self.crops)[-num_frames:]
        return np.stack(recent, axis=0)  # (T, 224, 224, 3)

    @property
    def duration_seconds(self) -> float:
        if len(self.timestamps) < 2:
            return 0.0
        return self.timestamps[-1] - self.timestamps[0]

    @property
    def memory_mb(self) -> float:
        return len(self.crops) * 224 * 224 * 3 / (1024 * 1024)


class ClipBufferManager:
    """Manages clip buffers for all tracked persons. Evicts stale tracks."""

    def __init__(self, max_persons: int = 200, stale_timeout: float = 30.0):
        self.buffers: dict[int, PersonClipBuffer] = {}
        self.max_persons = max_persons
        self.stale_timeout = stale_timeout

    def update(self, track_id: int, frame: np.ndarray, bbox: tuple, ts: float):
        if track_id not in self.buffers:
            if len(self.buffers) >= self.max_persons:
                self._evict_oldest()
            self.buffers[track_id] = PersonClipBuffer(track_id=track_id)
        self.buffers[track_id].add_frame(frame, bbox, ts)

    def get_clip(self, track_id: int, num_frames: int = 8) -> np.ndarray | None:
        buf = self.buffers.get(track_id)
        return buf.get_clip(num_frames) if buf else None

    def _evict_oldest(self):
        """Remove the track with the oldest last-seen timestamp."""
        if not self.buffers:
            return
        oldest_id = min(self.buffers, key=lambda k: self.buffers[k].timestamps[-1])
        del self.buffers[oldest_id]

    def cleanup_stale(self, current_time: float):
        """Remove tracks not seen for stale_timeout seconds."""
        stale = [
            tid for tid, buf in self.buffers.items()
            if current_time - buf.timestamps[-1] > self.stale_timeout
        ]
        for tid in stale:
            del self.buffers[tid]

    @property
    def total_memory_mb(self) -> float:
        return sum(buf.memory_mb for buf in self.buffers.values())
```

### 10.4 SlowFast Inference Integration

```python
import torch
from pytorchvideo.models.hub import slowfast_r50_detection

class SlowFastAnalyzer:
    """Tier 2 temporal behavior analysis using SlowFast."""

    # AVA action labels relevant to our use case
    FIGHTING_ACTIONS = {"hit", "kick", "punch", "push", "grab", "fight"}
    STEALING_ACTIONS = {"take", "pick_up", "carry", "put_down", "grab"}

    def __init__(self, device: str = "cuda:1"):
        self.device = device
        self.model = slowfast_r50_detection()
        self.model.to(device).eval()

    @torch.no_grad()
    def analyze_clip(
        self,
        clip: np.ndarray,          # (T, 224, 224, 3)
        behavior_type: str,        # "fighting" or "stealing"
    ) -> dict:
        """Run SlowFast on a clip buffer."""
        # Preprocess: (T,H,W,C) → (C,T,H,W) normalized
        clip_tensor = torch.from_numpy(clip).float().permute(3, 0, 1, 2) / 255.0
        clip_tensor = clip_tensor.unsqueeze(0).to(self.device)  # (1,C,T,H,W)

        # SlowFast expects [slow_pathway, fast_pathway]
        # Slow: every 8th frame, Fast: all frames
        slow = clip_tensor[:, :, ::8, :, :]  # temporal stride 8
        fast = clip_tensor                     # full temporal resolution

        preds = self.model([slow, fast])

        # Map predictions to relevant actions
        target_actions = (
            self.FIGHTING_ACTIONS if behavior_type == "fighting"
            else self.STEALING_ACTIONS
        )

        action_scores = {}  # action_name → confidence
        for action, idx in ACTION_LABEL_MAP.items():
            if action in target_actions:
                action_scores[action] = float(preds[0, idx].cpu())

        top_action = max(action_scores, key=action_scores.get)
        return {
            "behavior": behavior_type,
            "top_action": top_action,
            "confidence": action_scores[top_action],
            "all_scores": action_scores,
        }
```

### 10.5 Where SlowFast Fits in the Hybrid Architecture

```
Tier 1 (GPU 0) flags suspicious activity
       │
       │  trigger_type: "fighting_uncertain" or "stealing_suspicious"
       │  includes: track_id, camera_id, confidence
       ▼
┌──────────────────────────────────────────────┐
│ ClipBufferManager.get_clip(track_id, 8)      │
│   → Returns (8, 224, 224, 3) ndarray         │
│   → If clip not ready (< 8 frames), WAIT     │
│     and re-trigger when buffer fills          │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│ SlowFastAnalyzer.analyze_clip(clip, type)    │
│   → Runs on GPU 1 (shared with RF-DETR)     │
│   → ~50–80 ms per clip                       │
│   → Returns action classification + conf     │
└──────────────────────┬───────────────────────┘
                       │
            ┌──────────┴──────────┐
            │                     │
    confidence > 0.7       confidence ≤ 0.7
            │                     │
     CONFIRM ALERT           DISMISS / LOG
```

---

## 11. RF-DETR Integration (Tier 2 High-Accuracy Confirmation)

### 11.1 GPU Assignment Strategy

**Recommended: Dedicated GPU per tier.**

| Component | GPU 0 (V100 #1) | GPU 1 (V100 #2) |
|-----------|-----------------|-----------------|
| YOLOv26-S-Pose | ✅ Always loaded | ❌ |
| ByteTrack | ✅ CPU-side | ✅ CPU-side |
| RF-DETR-M | ❌ | ✅ On-demand |
| SlowFast / X3D-S | ❌ | ✅ On-demand |
| YOLOE-26 | ❌ | ✅ On-demand |

**Why separate GPUs:**
- Tier 1 must have **guaranteed latency** — no competition for CUDA streams
- Tier 2 models share GPU 1 sequentially (they're triggered sporadically, not continuously)
- If Tier 2 queue backs up, Tier 1 is unaffected

### 11.2 Non-Blocking Routing to RF-DETR

The key requirement: triggering RF-DETR must **never block** the main YOLO pipeline.

```python
import asyncio
import torch
from rfdetr import RFDETRMedium
from queue import PriorityQueue
from dataclasses import dataclass, field
from typing import Any

@dataclass(order=True)
class Tier2Request:
    priority: int                          # lower = higher priority
    timestamp: float = field(compare=False)
    camera_id: str = field(compare=False)
    frame: Any = field(compare=False)      # np.ndarray
    trigger_type: str = field(compare=False)  # "stealing", "tawuran", "uncertain"
    track_ids: list = field(compare=False, default_factory=list)
    tier1_confidence: float = field(compare=False, default=0.0)

class Tier2Processor:
    """Runs on GPU 1. Processes RF-DETR / SlowFast requests from a priority queue."""

    PRIORITY_MAP = {"fighting_uncertain": 1, "stealing": 2, "tawuran": 3, "audit": 10}

    def __init__(self, device: str = "cuda:1", max_queue: int = 100):
        self.device = device
        self.queue = PriorityQueue(maxsize=max_queue)

        # Load models on GPU 1
        self.rfdetr = RFDETRMedium()  # fine-tuned weights
        self.slowfast = SlowFastAnalyzer(device=device)

        # Lazy-load YOLOE only when needed
        self._yoloe = None

    @property
    def yoloe(self):
        if self._yoloe is None:
            from ultralytics import YOLO
            self._yoloe = YOLO("yoloe-26s.pt").to(self.device)
        return self._yoloe

    def submit(self, request: Tier2Request) -> bool:
        """Non-blocking submit. Returns False if queue full (drops request)."""
        try:
            self.queue.put_nowait(request)
            return True
        except:
            return False  # backpressure: drop lowest-priority requests

    async def process_loop(self):
        """Main Tier 2 processing loop. Runs in background."""
        while True:
            if self.queue.empty():
                await asyncio.sleep(0.01)
                continue

            request = self.queue.get()
            result = await self._process_request(request)

            if result and result["confirmed"]:
                await self._dispatch_alert(request, result)

    async def _process_request(self, req: Tier2Request) -> dict:
        if req.trigger_type == "stealing":
            return await self._confirm_stealing(req)
        elif req.trigger_type == "fighting_uncertain":
            return await self._confirm_fighting(req)
        elif req.trigger_type == "tawuran":
            return await self._confirm_tawuran(req)

    async def _confirm_stealing(self, req: Tier2Request) -> dict:
        """Multi-model stealing confirmation."""
        # Step 1: RF-DETR high-accuracy detection
        rfdetr_detections = self.rfdetr.predict(req.frame, threshold=0.5)

        # Step 2: YOLOE zero-shot screening
        self.yoloe.set_classes([
            "person concealing object",
            "hand reaching into bag",
            "person hiding item under clothing",
            "shoplifting",
        ])
        yoloe_results = self.yoloe(req.frame)

        # Step 3: SlowFast temporal analysis (if clip available)
        clip = clip_buffer_manager.get_clip(req.track_ids[0], num_frames=8)
        slowfast_result = None
        if clip is not None:
            slowfast_result = self.slowfast.analyze_clip(clip, "stealing")

        # Aggregate: require at least 2/3 models to agree
        votes = 0
        if len(rfdetr_detections) > 0:
            votes += 1
        if len(yoloe_results[0].boxes) > 0:
            votes += 1
        if slowfast_result and slowfast_result["confidence"] > 0.6:
            votes += 1

        return {
            "confirmed": votes >= 2,
            "votes": votes,
            "rfdetr_count": len(rfdetr_detections),
            "yoloe_count": len(yoloe_results[0].boxes),
            "slowfast": slowfast_result,
        }

    async def _confirm_tawuran(self, req: Tier2Request) -> dict:
        """RF-DETR for accurate person count in dense crowd."""
        detections = self.rfdetr.predict(req.frame, threshold=0.3)
        person_dets = [d for d in detections if d.class_id == 0]

        return {
            "confirmed": len(person_dets) >= 15,
            "accurate_count": len(person_dets),
            "tier1_estimate": req.tier1_confidence,
        }

    async def _dispatch_alert(self, req: Tier2Request, result: dict):
        """Send confirmed alert to alert system."""
        alert = {
            "type": req.trigger_type,
            "camera_id": req.camera_id,
            "timestamp": req.timestamp,
            "result": result,
            "frame_evidence": req.frame,  # or save to GridFS
        }
        # → WebSocket push, API call, database insert, etc.
        await alert_dispatcher.send(alert)
```

### 11.3 RF-DETR Fine-Tuning for Stealing Detection

```python
from rfdetr import RFDETRMedium
from rfdetr.config import TrainConfig

def fine_tune_stealing_detector():
    """Fine-tune RF-DETR-M on custom stealing dataset."""

    model = RFDETRMedium()

    config = TrainConfig(
        dataset_dir="./data/stealing_dataset",    # COCO format
        num_classes=4,       # normal_interaction, concealing, grabbing, fleeing
        epochs=50,
        batch_size=8,
        lr=1e-4,             # lower LR for fine-tuning
        grad_accum_steps=4,  # effective batch 32
        resolution=576,      # RF-DETR-M native resolution
        augmentation="heavy",
    )

    # RF-DETR proven to generalize well on small datasets (RF100-VL benchmark)
    # Minimum recommended: 500 images per class
    # Optimal: 2,000+ images per class

    model.train(config)
    model.save("rfdetr_m_stealing_v1.pt")
```

### 11.4 Dataset Requirements for Stealing Fine-Tuning

| Class | Description | Min Images | Sources |
|-------|-------------|-----------|---------|
| `normal_interaction` | Person normally handling items | 1,000 | Existing CCTV footage |
| `concealing` | Person hiding item in clothing/bag | 500 | Staged + real footage |
| `grabbing` | Hand reaching for unattended item | 500 | Staged + retail CCTV |
| `fleeing` | Person rapidly leaving after taking | 300 | Staged scenarios |

---

## 12. Hardware Specifications

### 12.0 Design Principles

All specifications account for:
- **NVDEC zero-copy decode** — V100's built-in NVDEC engine decodes H.264/H.265 on dedicated fixed-function hardware, consuming 0% CUDA/Tensor cores and ~0.3 GB VRAM per GPU. Frames stay in GPU memory (no CPU copy).
- **NVDEC session limit** — Each V100 has 1 NVDEC engine supporting ~24–32 concurrent decode sessions. Cameras beyond this limit fall back to CPU software decode at ~0.05 core/stream (lightweight at 1 FPS).
- **1 FPS capture rate** — At 1 frame/second, CPU decode overhead is minimal even in software fallback.
- **Async bulk MongoDB writes** — Using `motor` (async driver) with `insert_many` batches of 100–500 docs, `ordered=False`.

### 12.1 GPU Internal Resource Allocation

NVDEC and CUDA/Tensor cores are **physically separate silicon** inside the V100. They run in parallel with zero interference:

```
┌─────────────────────────────────────────────────────────────┐
│                    NVIDIA V100 32 GB                         │
│                                                             │
│  ┌──────────────────────────────┐  ┌─────────────────────┐ │
│  │  CUDA + Tensor Cores         │  │  NVDEC Engine        │ │
│  │  (5,120 CUDA + 640 Tensor)   │  │  (fixed-function)    │ │
│  │                              │  │                     │ │
│  │  • YOLOv26-Pose inference    │  │  • H.264 decode     │ │
│  │  • RF-DETR inference         │  │  • H.265 decode     │ │
│  │  • SlowFast inference        │  │  • 24–32 sessions   │ │
│  │  • Preprocessing (resize)    │  │  • ~0.3 GB VRAM     │ │
│  │                              │  │                     │ │
│  │  100% available for AI       │  │  Runs in PARALLEL   │ │
│  │  workloads regardless of     │  │  with CUDA cores    │ │
│  │  NVDEC activity              │  │  at zero cost       │ │
│  └──────────────────────────────┘  └─────────────────────┘ │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  HBM2 VRAM (32 GB) — shared, only contention point   │  │
│  │  NVDEC uses ~0.3 GB → 31.7 GB free for inference      │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

**Impact:** NVDEC decode is effectively "free" — no separate decode GPU needed, no CPU core savings to calculate. The V100 handles both decode and inference simultaneously.

### 12.2 CPU Load Breakdown (per camera at 1 FPS)

| Task | Without NVDEC (CPU decode) | With NVDEC (GPU decode) | Notes |
|------|---------------------------|------------------------|-------|
| RTSP network I/O | 0.01 core | 0.01 core | Same — network stack is always CPU |
| H.264 frame decode | 0.30–0.50 core | **0.00 core** | Offloaded to NVDEC |
| Frame copy CPU→GPU | 0.02 core | **0.00 core** | Zero-copy: frame stays on GPU |
| ByteTrack tracking | 0.02 core | 0.02 core | CPU-side, lightweight |
| Behavior analysis | 0.01 core | 0.01 core | NumPy math on keypoint floats |
| Ring buffer management | 0.01 core | 0.01 core | deque operations |
| JPEG encode (crops) | 0.03 core | 0.03 core | CPU (or NVJPEG on GPU) |
| MongoDB async write | 0.01 core | 0.01 core | Amortized via bulk writes |
| **Total per camera** | **0.41–0.61 core** | **0.09 core** | **~78–85% reduction** |

| Camera Count | CPU Cores (no NVDEC) | CPU Cores (with NVDEC) | Savings |
|-------------|---------------------|----------------------|---------|
| 100 | 41–61 cores | **9 cores** | 78–85% |
| 300 | 123–183 cores | **27 cores** | 78–85% |
| 500 | 205–305 cores | **45 cores** | 78–85% |
| 800 | 328–488 cores | **72 cores** | 78–85% |
| 1,000 | 410–610 cores | **90 cores** | 78–85% |

> **Note:** Cameras 1–24 use GPU 0 NVDEC, cameras 25–48 use GPU 1 NVDEC (48 total hardware-decoded). Cameras 49+ fall back to CPU software decode at 1 FPS, which is still very light (~0.05 core/stream vs 0.3–0.5 for persistent decode). The 0.09 core/camera figure above is the blended average.

---

### 12.3 Spec A — With Behavior Detection (Fighting + Stealing + Tawuran)

#### Processing Pipeline

```
Camera RTSP → NVDEC (GPU decode) → GPU Memory → YOLOv26-Pose (Tensor Cores)
                                                        │
                                                   bbox + keypoints (tiny, few KB)
                                                        │
                                                   copy to CPU
                                                        │
                                        ┌───────────────┼───────────────┐
                                        ▼               ▼               ▼
                                   ByteTrack      Fighting Det.    Tawuran Det.
                                   (tracking)     (keypoint vel.)  (crowd density)
                                        │               │               │
                                        └───────┬───────┘               │
                                                ▼                       ▼
                                        Suspicious? ──────────→ Tier 2 (GPU 1)
                                                                RF-DETR / SlowFast
                                                                        │
                                                                   Confirmed?
                                                                        │
                                                                   ▼    ▼
                                                              Alert  MongoDB
```

#### GPU Allocation

| | GPU 0 (V100 #1) — Tier 1 | GPU 1 (V100 #2) — Tier 2 |
|--|--------------------------|--------------------------|
| **NVDEC** | Cameras 1–24 (H.264 decode) | Cameras 25–48 (H.264 decode) |
| **Inference** | YOLOv26-S-Pose (always-on) | RF-DETR-M + X3D-S + YOLOE-26 (on-demand) |
| **Role** | Primary detection + pose | Confirmation + temporal analysis |

#### VRAM Budget

| Component | GPU 0 | GPU 1 |
|-----------|-------|-------|
| NVDEC decode buffers (24 streams) | 0.3 GB | 0.3 GB |
| YOLOv26-S-Pose weights (TensorRT FP16) | 0.8 GB | — |
| RF-DETR-M weights | — | 2.5 GB |
| X3D-S weights | — | 0.8 GB |
| YOLOE-26-S weights | — | 1.0 GB |
| CUDA context + allocator | 0.8 GB | 0.8 GB |
| Inference batch buffers (batch=16) | 0.5 GB | 0.6 GB |
| Preprocessing buffers | 0.2 GB | 0.2 GB |
| **Total used** | **2.6 GB** | **6.2 GB** |
| **Free** | **29.4 GB** | **25.8 GB** |

#### Full Bill of Materials

| Component | Specification | Purpose | Est. Cost (USD) |
|-----------|--------------|---------|-----------------|
| **CPU** | AMD EPYC 9374F (32C/64T, 3.85 GHz) | RTSP I/O, ByteTrack, behavior analysis, overflow decode. 72 cores needed for 800 cams → 32C with HT (64T) covers it with headroom. | $2,500–3,000 |
| **Motherboard** | Supermicro H13SSL-N or equiv. (SP5 socket) | Single-socket EPYC, 2× PCIe Gen5 x16 for dual V100 | $500–700 |
| **RAM** | 128 GB DDR5-4800 ECC RDIMM (4×32 GB) | Ring buffers (~3 frames × 1080p × 800 cams ≈ 15 GB), clip buffers (~240 MB per 100 tracked persons), batch queues, OS overhead | $400–500 |
| **GPU 0** | NVIDIA V100 32 GB (already owned) | Tier 1: NVDEC decode (24 streams) + YOLOv26-S-Pose inference | — |
| **GPU 1** | NVIDIA V100 32 GB (already owned) | Tier 2: NVDEC decode (24 streams) + RF-DETR + X3D-S + YOLOE-26 | — |
| **NIC** | Mellanox ConnectX-5 25 GbE (dual-port) | 800 cameras × ~1 Mbps = 800 Mbps sustained. 25 GbE provides 3× headroom. Dual-port for redundancy. | $150–200 |
| **Network Switch** | 10/25 GbE managed switch (if cameras are IP-based) | Aggregate camera traffic. May need multiple switches for 800 cameras. | $500–2,000 (varies) |
| **Boot SSD** | 512 GB NVMe (PCIe Gen4) | OS (Ubuntu 22.04/24.04), CUDA toolkit, Python env, model weights (~5 GB total) | $50–70 |
| **Data SSD** | 2 TB NVMe (Samsung 990 Pro or Micron 7450) | MongoDB data directory (WiredTiger engine), alert clip buffer (10-sec clips per alert) | $150–200 |
| **Archive HDD** | 2× 8 TB HDD (RAID 1 mirror) | Long-term evidence storage. At 100 alerts/day × 10-sec clip × 5 MB = 500 MB/day → 8 TB lasts ~16,000 days. | $240–300 |
| **PSU** | 1,200W 80+ Platinum (redundant if rackmount) | 2× V100 (300W each) + EPYC 9374F (320W) + drives + fans = ~1,000W peak | $200–300 |
| **Chassis** | 4U rackmount server (e.g., Supermicro 4124GS-TNR) | Dual GPU cooling, sufficient airflow for V100 TDP | $300–500 |
| **UPS** | 1,500VA online UPS | Protect against power loss during inference + MongoDB writes | $300–400 |
| | | | |
| **Total (excl. V100s)** | | | **$5,300–8,200** |
| **Total (incl. V100s at ~$3,000 each used)** | | | **$11,300–14,200** |

#### MongoDB Sizing

| Cameras | Detections/sec (avg 5 persons/frame) | Crops/sec | Write Throughput | Daily Storage | MongoDB Config |
|---------|--------------------------------------|-----------|-----------------|---------------|---------------|
| 100 | 500 | 500 | ~5 MB/s | ~430 GB | Single node, WiredTiger, NVMe |
| 300 | 1,500 | 1,500 | ~15 MB/s | ~1.3 TB | Single node, NVMe, compression |
| 500 | 2,500 | 2,500 | ~25 MB/s | ~2.2 TB | Replica set (3 nodes) |
| 800 | 4,000 | 4,000 | ~40 MB/s | ~3.5 TB | Sharded cluster (2 shards) |

> **Storage note:** At 800 cameras, you generate ~3.5 TB/day of crop images. Implement a **TTL index** on MongoDB to auto-delete crops older than 7–30 days, or archive to cold storage (HDD/S3).

#### Power & Cooling

| Component | TDP / Draw | Notes |
|-----------|-----------|-------|
| 2× V100 32 GB | 600W (300W each) | Requires active cooling, 250W sustained typical |
| EPYC 9374F | 320W | High-clock SKU, sustained ~250W under load |
| NVMe SSDs (×2) | 15W | Negligible |
| HDDs (×2) | 20W | Negligible |
| NIC + fans + misc | 45W | — |
| **Total system** | **~1,000W sustained** | Peak ~1,100W |
| **Cooling** | ~3,400 BTU/hr | Standard 4U rackmount cooling sufficient |

---

### 12.4 Spec B — Without Behavior Detection (Detection + Crop + MongoDB Only)

This configuration runs YOLOv26 for object detection and bounding box extraction only. No pose estimation, no tracking, no fighting/stealing/tawuran analysis. No Tier 2 models.

#### Processing Pipeline (Simplified)

```
Camera RTSP → NVDEC (GPU decode) → GPU Memory → YOLOv26-S (Tensor Cores)
                                                        │
                                                   bbox + class + confidence
                                                        │
                                                   crop regions from frame
                                                        │
                                                   JPEG encode → MongoDB
```

#### GPU Allocation

| | GPU 0 (V100 #1) | GPU 1 (V100 #2) |
|--|-----------------|-----------------|
| **NVDEC** | Cameras 1–24 | Cameras 25–48 |
| **Inference** | YOLOv26-S (detection only, no pose) | YOLOv26-S (overflow / load balance) |
| **Role** | Primary detection | Overflow detection (or idle) |

> **Option:** Run both V100s for Tier 1 detection to double throughput, or leave GPU 1 idle to save power (~300W).

#### VRAM Budget (Single GPU Mode)

| Component | GPU 0 | GPU 1 (idle or overflow) |
|-----------|-------|--------------------------|
| NVDEC decode buffers | 0.3 GB | 0.3 GB (if used) |
| YOLOv26-S weights (TensorRT FP16) | 0.6 GB | 0.6 GB (if used) |
| CUDA context | 0.8 GB | 0.8 GB (if used) |
| Batch buffers (batch=16) | 0.5 GB | 0.5 GB (if used) |
| **Total used** | **2.2 GB** | **2.2 GB** |
| **Free** | **29.8 GB** | **29.8 GB** |

#### Full Bill of Materials

| Component | Specification | Purpose | Est. Cost (USD) |
|-----------|--------------|---------|-----------------|
| **CPU** | AMD EPYC 9274F (24C/48T, 4.05 GHz) or Intel Xeon w5-2465X (16C/32T) | RTSP I/O, overflow decode, JPEG encode, MongoDB writes. No tracking or behavior analysis. | $1,500–2,000 |
| **Motherboard** | Supermicro H13SSL-N or equiv. | Single-socket, 1–2× PCIe x16 | $500–700 |
| **RAM** | 64 GB DDR5-4800 ECC RDIMM (2×32 GB) | Ring buffers only (no clip buffers, no tracking history). 800 cams × 3 frames × 6 MB ≈ 14 GB. | $200–250 |
| **GPU** | 1× NVIDIA V100 32 GB (already owned) | NVDEC decode + YOLOv26-S inference. Second V100 optional for 2× throughput. | — |
| **NIC** | Intel X710-DA2 10 GbE (dual-port) | 500 cameras × ~1 Mbps = 500 Mbps. 10 GbE sufficient. | $80–120 |
| **Boot SSD** | 512 GB NVMe | OS + model weights | $50–70 |
| **Data SSD** | 2 TB NVMe | MongoDB data directory | $150–200 |
| **Archive HDD** | 1× 8 TB HDD | Long-term crop storage (optional) | $120–150 |
| **PSU** | 850W 80+ Gold | 1× V100 (300W) + EPYC 9274F (250W) + overhead | $120–150 |
| **Chassis** | 4U rackmount or tower workstation | Single GPU cooling | $200–400 |
| **UPS** | 1,000VA online UPS | Power protection | $200–250 |
| | | | |
| **Total (excl. V100)** | | | **$3,100–5,100** |
| **Total (incl. 1× V100 at ~$3,000 used)** | | | **$6,100–8,100** |

---

### 12.5 Comprehensive Comparison

| Specification | **With Behavior Detection** | **Without Behavior Detection** |
|--------------|---------------------------|-------------------------------|
| | *Fighting + Stealing + Tawuran* | *Detection + Crop + Save only* |
| | | |
| **CPU** | EPYC 9374F (32C/64T) | EPYC 9274F (24C/48T) |
| **CPU Cores needed** | ~72 (800 cams) | ~45 (800 cams) |
| **RAM** | 128 GB DDR5 ECC | 64 GB DDR5 ECC |
| **GPUs active** | 2× V100 32 GB | 1× V100 32 GB |
| **GPU 0 role** | Tier 1: NVDEC + YOLOv26-Pose | NVDEC + YOLOv26-S |
| **GPU 1 role** | Tier 2: NVDEC + RF-DETR + X3D-S + YOLOE | Idle (or overflow) |
| **GPU VRAM used** | 2.6 GB + 6.2 GB = 8.8 GB | 2.2 GB (single GPU) |
| **GPU VRAM free** | 29.4 + 25.8 = 55.2 GB | 29.8 GB |
| **NIC** | 25 GbE | 10 GbE |
| **Data SSD** | 2 TB NVMe | 2 TB NVMe |
| **PSU** | 1,200W Platinum | 850W Gold |
| **System power** | ~1,000W sustained | ~600W sustained |
| **Cooling** | ~3,400 BTU/hr | ~2,050 BTU/hr |
| | | |
| **Max cameras (1 FPS, batch=16)** | **800–1,120** | **1,120–2,240** |
| **Bottleneck** | GPU 1 (Tier 2 at high trigger rates) | GPU 0 (Tier 1 compute) |
| | | |
| **Models loaded** | YOLOv26-Pose + RF-DETR + X3D-S + YOLOE | YOLOv26-S only |
| **Tracking** | ✅ ByteTrack (person identity across frames) | ❌ No tracking |
| **Fighting detection** | ✅ Keypoint velocity + proximity | ❌ |
| **Stealing detection** | ✅ Multi-model confirmation | ❌ |
| **Tawuran detection** | ✅ Crowd density + convergence | ❌ |
| **Temporal analysis** | ✅ X3D-S / SlowFast on video clips | ❌ |
| **Zero-shot detection** | ✅ YOLOE-26 text prompts | ❌ |
| | | |
| **Cost (excl. V100s)** | **$5,300–8,200** | **$3,100–5,100** |
| **Cost (incl. V100s)** | **$11,300–14,200** | **$6,100–8,100** |
| **Cost difference** | — | **Saves $5,200–6,100** |

### 12.6 Scaling Guide — When to Upgrade

| Camera Count | Config Needed | CPU | RAM | GPUs | Notes |
|-------------|--------------|-----|-----|------|-------|
| **1–100** | Minimal | 16C (Xeon w5-2445) | 32 GB | 1× V100 | Single GPU handles everything |
| **100–300** | Standard | 24C (EPYC 9274F) | 64 GB | 1× V100 | Add behavior detection with same GPU |
| **300–600** | Recommended | 32C (EPYC 9374F) | 128 GB | 2× V100 | Full hybrid pipeline |
| **600–1,000** | Full | 32C (EPYC 9374F) | 128 GB | 2× V100 | Optimize batch size, reduce trigger rate |
| **1,000–2,000** | Scale-out | 64C (EPYC 9554) | 256 GB | 4× V100 or 2× A100 | Second server or larger GPU |
| **2,000+** | Distributed | Multi-server | 256+ GB each | GPU per server | Kubernetes / distributed inference |

### 12.7 Network Infrastructure

| Cameras | Bandwidth (1 FPS, H.264) | NIC Required | Switch Requirement |
|---------|-------------------------|-------------|-------------------|
| 100 | ~100 Mbps | 1 GbE (sufficient) | Standard managed switch |
| 300 | ~300 Mbps | 10 GbE | 10 GbE aggregation switch |
| 500 | ~500 Mbps | 10 GbE | 10 GbE with LACP bonding |
| 800 | ~800 Mbps | 25 GbE | 25 GbE aggregation |
| 1,000+ | ~1 Gbps+ | 25 GbE (dual-port) | 25 GbE with redundancy |

> **RTSP bandwidth note:** Each 1080p H.264 stream at 1 FPS uses ~0.5–1.5 Mbps depending on scene complexity and I-frame interval. The estimates above use ~1 Mbps average.

### 12.8 MongoDB Storage Planning

| Cameras | Crops/day (5 persons/frame avg) | Storage/day (50 KB avg crop) | 30-day retention | Recommended |
|---------|-------------------------------|-----------------------------|--------------------|-------------|
| 100 | 432,000 | 21 GB | 630 GB | 1 TB NVMe |
| 300 | 1,296,000 | 63 GB | 1.9 TB | 2 TB NVMe |
| 500 | 2,160,000 | 105 GB | 3.2 TB | 4 TB NVMe |
| 800 | 3,456,000 | 168 GB | 5.0 TB | 2× 4 TB NVMe (RAID 0) |

> **Recommendation:** Set a MongoDB **TTL index** to auto-expire documents after 7–30 days. Archive flagged alerts (with behavior detections) to cold storage before expiry.

## 13. V100 32 GB Deployment Estimate (2× GPUs, 1 FPS per Camera)

### 13.1 Baseline Latency on V100

V100 delivers ~1.5–2× the FP16 throughput of T4 (125 vs 65 TFLOPS). All estimates use **TensorRT FP16**.

| Model | T4 Benchmark | V100 Estimated | Per-inference VRAM |
|-------|-------------|----------------|-------------------|
| YOLOv26-S-Pose | 2.6 ms | **~1.5 ms** | ~0.8 GB (weights) |
| YOLOv26-M-Pose | 4.7 ms | **~2.8 ms** | ~1.5 GB |
| RF-DETR-M | 4.4 ms | **~2.7 ms** | ~2.5 GB |
| RF-DETR-L | 6.8 ms | **~4.2 ms** | ~3.2 GB |
| SlowFast R50 | 80–120 ms | **~50–80 ms** | ~1.5 GB |
| X3D-S | 40–60 ms | **~25–40 ms** | ~0.8 GB |
| YOLOE-26-S | 2.6 ms | **~1.5 ms** | ~1.0 GB |

### 13.2 Scenario (a): YOLOv26-Pose Only (Tier 1 Always-On)

**GPU 0 only.** GPU 1 idle. Detects: fighting, tawuran (no stealing confirmation).

| Model | Latency/frame | GPU 0 Budget: 1000 ms | GPU utilization |
|-------|--------------|----------------------|-----------------|
| YOLOv26-S-Pose | 1.5 ms | 1000 / 1.5 = **666 frames/sec** | — |

**VRAM Allocation (GPU 0):**

| Component | VRAM |
|-----------|------|
| YOLOv26-S-Pose weights | 0.8 GB |
| CUDA context | 0.8 GB |
| Batch buffer (batch=32, 640×640×3×FP16) | 0.8 GB |
| **Total** | **2.4 GB** |
| **Free** | **29.6 GB** |

| Batch Size | Throughput (frames/sec) | Practical (70% util) | **Max Cameras (1 FPS)** |
|-----------|------------------------|---------------------|------------------------|
| 1 | 666 | 466 | **466** |
| 8 | ~1,200 | 840 | **840** |
| 16 | ~1,600 | 1,120 | **1,120** |
| 32 | ~1,900 | 1,330 | **1,330** |

> **With 2× V100 both running YOLOv26-Pose:** double all numbers → **2,240–2,660 cameras** at batch=16–32.

### 13.3 Scenario (b): YOLOv26-Pose + RF-DETR Hybrid (10–20% Trigger)

**GPU 0:** YOLOv26-S-Pose (always-on)
**GPU 1:** RF-DETR-M (on-demand, triggered by 10–20% of frames)

**VRAM Allocation:**

| | GPU 0 | GPU 1 |
|--|-------|-------|
| YOLOv26-S-Pose | 0.8 GB | — |
| RF-DETR-M | — | 2.5 GB |
| YOLOE-26-S (lazy) | — | 1.0 GB |
| CUDA context | 0.8 GB | 0.8 GB |
| Batch buffers | 0.8 GB | 0.6 GB |
| **Total** | **2.4 GB** | **4.9 GB** |
| **Free** | **29.6 GB** | **27.1 GB** |

**Throughput Calculation:**

GPU 0 bottleneck (Tier 1 — determines max cameras):

| Batch | YOLOv26-S-Pose throughput | 70% util | Max cameras |
|-------|--------------------------|----------|-------------|
| 8 | 1,200 fps | 840 | **840** |
| 16 | 1,600 fps | 1,120 | **1,120** |
| 32 | 1,900 fps | 1,330 | **1,330** |

GPU 1 bottleneck (Tier 2 — must keep up with trigger rate):

| Cameras | Trigger Rate | Tier 2 frames/sec | RF-DETR-M capacity (370 fps) | Headroom |
|---------|-------------|-------------------|------------------------------|----------|
| 500 | 10% | 50 fps | 370 fps | ✅ 7.4× |
| 500 | 20% | 100 fps | 370 fps | ✅ 3.7× |
| 800 | 10% | 80 fps | 370 fps | ✅ 4.6× |
| 800 | 20% | 160 fps | 370 fps | ✅ 2.3× |
| 1,000 | 20% | 200 fps | 370 fps | ✅ 1.8× |
| 1,330 | 20% | 266 fps | 370 fps | ✅ 1.4× |

> **Tier 2 is never the bottleneck** at these camera counts and trigger rates. GPU 1 has massive headroom.

**Final capacity for Scenario (b):**

| Batch | **Max Cameras (1 FPS)** | Bottleneck |
|-------|------------------------|------------|
| 8 | **840** | GPU 0 (Tier 1) |
| 16 | **1,120** | GPU 0 (Tier 1) |
| 32 | **1,330** | GPU 0 (Tier 1) |

### 13.4 Scenario (c): Full Pipeline (YOLOv26-Pose + RF-DETR + SlowFast)

**GPU 0:** YOLOv26-S-Pose (always-on)
**GPU 1:** RF-DETR-M + SlowFast R50 + YOLOE-26-S (on-demand, time-shared)

**VRAM Allocation:**

| | GPU 0 | GPU 1 |
|--|-------|-------|
| YOLOv26-S-Pose | 0.8 GB | — |
| RF-DETR-M | — | 2.5 GB |
| SlowFast R50 | — | 1.5 GB |
| YOLOE-26-S | — | 1.0 GB |
| CUDA context | 0.8 GB | 0.8 GB |
| Batch buffers | 0.8 GB | 1.0 GB |
| **Total** | **2.4 GB** | **6.8 GB** |
| **Free** | **29.6 GB** | **25.2 GB** |

> VRAM is comfortable even with all 3 Tier 2 models loaded simultaneously.

**Throughput — GPU 1 becomes the constraint when SlowFast is active:**

SlowFast is expensive (~50–80 ms per clip). The question is how often it triggers.

| Trigger Source | Rate | SlowFast calls/sec | Time consumed |
|---------------|------|-------------------|---------------|
| Fighting uncertain (need temporal confirm) | 2% of cameras | At 800 cams: 16/sec | 16 × 60ms = **960 ms** ⚠️ |
| Stealing suspicious | 5% of cameras | At 800 cams: 40/sec | 40 × 60ms = **2,400 ms** ❌ |

> **SlowFast at 5% trigger rate on 800 cameras would exceed GPU 1's 1-second budget.** Solutions:

| Solution | Effect |
|----------|--------|
| **Reduce SlowFast trigger rate** to <2% | 800 × 2% = 16 calls/sec × 60ms = 960 ms ✅ fits |
| **Use X3D-S instead of SlowFast** | 25–40 ms vs 50–80 ms → 2× more capacity |
| **Batch SlowFast clips** | Batch=4 clips → ~120 ms total vs 4×60ms = 240 ms |
| **Time-share with RF-DETR** | RF-DETR handles most triggers; SlowFast only for uncertain cases |

**Practical capacity with mixed workload on GPU 1:**

Assume: RF-DETR handles 15% of triggers, SlowFast handles 2%, YOLOE handles 3%.

| Cameras | RF-DETR (15%) | SlowFast (2%) | YOLOE (3%) | Total GPU 1 time/sec | Feasible? |
|---------|-------------|--------------|------------|---------------------|-----------|
| 400 | 60 × 2.7ms = 162ms | 8 × 60ms = 480ms | 12 × 1.5ms = 18ms | **660 ms** | ✅ |
| 600 | 90 × 2.7ms = 243ms | 12 × 60ms = 720ms | 18 × 1.5ms = 27ms | **990 ms** | ✅ Tight |
| 800 | 120 × 2.7ms = 324ms | 16 × 60ms = 960ms | 24 × 1.5ms = 36ms | **1,320 ms** | ❌ Over budget |

**With X3D-S instead of SlowFast (25ms instead of 60ms):**

| Cameras | RF-DETR (15%) | X3D-S (2%) | YOLOE (3%) | Total GPU 1 time/sec | Feasible? |
|---------|-------------|-----------|------------|---------------------|-----------|
| 600 | 243ms | 12 × 25ms = 300ms | 27ms | **570 ms** | ✅ |
| 800 | 324ms | 16 × 25ms = 400ms | 36ms | **760 ms** | ✅ |
| 1,000 | 405ms | 20 × 25ms = 500ms | 45ms | **950 ms** | ✅ Tight |

**Final capacity for Scenario (c):**

| Config | Batch 16 | **Max Cameras** | Bottleneck |
|--------|---------|----------------|------------|
| SlowFast R50, mixed triggers | 16 | **~600** | GPU 1 (SlowFast) |
| X3D-S (recommended), mixed triggers | 16 | **~800–1,000** | GPU 1 (X3D-S) |
| X3D-S, conservative triggers (1%) | 16 | **~1,120** | GPU 0 (Tier 1) |

### 13.5 Summary — Camera Capacity (2× V100 32 GB, 1 FPS)

| Scenario | Batch=8 | Batch=16 | Batch=32 | Bottleneck |
|----------|---------|---------|---------|------------|
| **(a) YOLOv26-Pose only** | 840 | 1,120 | 1,330 | GPU 0 compute |
| **(b) + RF-DETR hybrid (20% trigger)** | 840 | 1,120 | 1,330 | GPU 0 compute |
| **(c) Full pipeline + SlowFast** | ~450 | ~600 | ~700 | GPU 1 (SlowFast) |
| **(c) Full pipeline + X3D-S** | ~650 | ~800–1,000 | ~1,100 | GPU 1 (X3D-S) |

> **Key takeaway:** At 1 FPS, the V100 GPUs are not the bottleneck for scenarios (a) and (b). The real constraints are:
> 1. **CPU decode** (need 64+ cores for 800+ RTSP streams)
> 2. **Network bandwidth** (800 cameras × 1 Mbps = 800 Mbps → need 10–25 GbE)
> 3. **SlowFast latency** in scenario (c) — mitigated by using X3D-S instead
> 4. **MongoDB write throughput** at 2,000+ docs/sec — use async bulk writes

---

## 14. Pipeline Scenario Analysis — YOLO vs RF-DETR with Behavior Detection

### 14.1 Candidate Pipeline Scenarios

Two base pipeline architectures are evaluated for integration with behavior detection. Both capture at **1 FPS per camera** and detect **multiple object classes** — not just persons.

#### Target Detection Classes

| Category | Objects | Notes |
|----------|---------|-------|
| **Person** | People (pedestrians, staff, intruders) | Primary target for behavior detection |
| **Vehicle** | Car, truck, bus, motorcycle, bicycle | Parking, traffic monitoring, suspicious vehicles |
| **Bag/Luggage** | Backpack, handbag, suitcase | Unattended bag detection, theft evidence |
| **Weapon** | Knife, gun (requires fine-tuning for many weapon types) | Critical security objects |
| **Animal** | Dog, cat, bird, horse, etc. | Stray animal detection, restricted area intrusion |
| **Other** | Umbrella, cell phone, laptop, etc. | Context-dependent objects |

> **COCO pretrained models** (both YOLO and RF-DETR) detect **80 object classes** out of the box. Custom classes (specific weapon types, uniform types) require fine-tuning.

#### Scenario 1 — YOLOv26-Pose Pipeline

```
Thread per CCTV stream (capture frame at 1 FPS)
  → Save frame in shared memory buffer
    → Pool/batch frames for YOLO detection pipeline
      → Get bounding boxes (object detection: person, vehicle, bag, weapon, animal, etc.)
        → Crop bounding box regions and save as images
          → Pool/batch cropped images for async save to MongoDB
```

**What YOLO Pose provides per frame:**
- Bounding boxes for **all 80 COCO classes** (person, car, truck, dog, backpack, knife, etc.)
- Class labels + confidence scores for every detected object
- **17 body keypoints** (nose, eyes, shoulders, elbows, wrists, hips, knees, ankles) **for person class only**
- Keypoint confidence scores per keypoint

> **Key point:** YOLOv26-Pose is NOT person-only. It detects all object classes like standard YOLO, but **additionally** outputs body keypoints for every detected person. The pose head adds negligible latency (~0.1 ms).

#### Scenario 2 — RF-DETR Pipeline

```
Thread per CCTV stream (capture frame at 1 FPS)
  → Save frame in shared memory buffer
    → Pool/batch frames for RF-DETR detection pipeline
      → Get bounding boxes (object detection: person, vehicle, bag, weapon, animal, etc.)
        → Crop bounding box regions and save as images
          → Pool/batch cropped images for async save to MongoDB
```

**What RF-DETR provides per frame:**
- Bounding boxes for **all 80 COCO classes** (same classes as YOLO)
- Class labels + confidence scores
- **Superior small object detection** (bags, weapons, distant animals are better detected)
- **No keypoints, no pose data** for any class

### 14.2 Key Difference — Output Capabilities

Both models detect the **same object classes**. The difference is what *extra* information they provide:

| Output | YOLOv26-Pose | RF-DETR |
|--------|-------------|---------|
| **Multi-class detection** (person, vehicle, bag, weapon, animal) | ✅ 80 COCO classes | ✅ 80 COCO classes |
| Bounding boxes + confidence | ✅ | ✅ |
| **17 body keypoints** (person class only) | ✅ | ❌ |
| Keypoint confidence scores | ✅ | ❌ |
| **Small object detection** (< 32×32 px: distant bags, weapons, animals) | Good | **Superior** (+7 AP) |
| Person detection (medium-large) | >90% AP | >93% AP |
| Vehicle detection | >85% AP | >88% AP |
| Bag/weapon detection (typically small) | Moderate | **Better** (deformable attention) |
| Inference latency (V100 TensorRT FP16) | **~1.5 ms** | ~2.7 ms |

#### Impact on Behavior Detection

| Behavior | What's Needed | YOLOv26-Pose | RF-DETR |
|----------|--------------|-------------|---------|
| **Fighting** | Person keypoints (arm velocity, proximity) | ✅ Built-in | ❌ Needs second model |
| **Tawuran** | Person keypoints + crowd count | ✅ Built-in | ⚠️ Count only (no pose) |
| **Stealing** | Hand-to-object proximity (keypoints + bag/object bbox) | ✅ Hand keypoints + bag bbox in same pass | ⚠️ Bag bbox only, no hand position |
| **Unattended bag** | Bag bbox without nearby person bbox | ✅ | ✅ (both detect bags) |
| **Suspicious vehicle** | Vehicle bbox + loitering time | ✅ | ✅ (both detect vehicles) |
| **Animal intrusion** | Animal bbox in restricted zone | ✅ | ✅ (both detect animals) |

**Critical implication:** For behaviors requiring **only bounding boxes** (unattended bag, vehicle loitering, animal intrusion), both models work equally well. For behaviors requiring **body keypoints** (fighting, tawuran, stealing), RF-DETR cannot do it alone — you must add YOLO Pose as a second model, doubling inference cost.

> **This is why YOLO Pose is recommended as Tier 1:** it handles ALL detection classes AND provides keypoints for person-specific behavior analysis in a single inference pass. RF-DETR's small object advantage is best leveraged on-demand (Tier 2) to confirm small objects like weapons, bags, or distant persons that YOLO may have missed.

---

### 14.3 SlowFast vs YOLO Pose for Behavior Detection at 1 FPS

#### How SlowFast Works

SlowFast is a **video-level temporal action recognition** model. It requires a **clip** (sequence of frames) as input:

| Pathway | Purpose | Typical Sampling | What Happens at 1 FPS |
|---------|---------|-----------------|----------------------|
| **Slow pathway** | Spatial semantics (what is happening) | 8 frames, stride 8 → 2.1s at 30 FPS | 8 frames = **8 seconds** of context |
| **Fast pathway** | Rapid motion capture (how it moves) | 32 frames, stride 2 → 2.1s at 30 FPS | 32 frames = **32 seconds** of context |

#### The Problem: Fast Pathway is Useless at 1 FPS

The fast pathway is designed to capture **rapid temporal changes** — a fist swing (~0.3s), a grab (~0.5s), a sudden lunge (~0.2s). At 1 FPS, these events happen **between** frames and are never captured. The fast pathway receives what is essentially a slideshow with no useful motion signal.

#### Comparison: SlowFast vs YOLO Pose at 1 FPS

| Factor | YOLO Pose (Recommended) | SlowFast |
|--------|------------------------|----------|
| **Works at 1 FPS?** | ✅ Yes — single-frame keypoints | ❌ Poorly — fast pathway is blind |
| **Latency per inference** | ~1.5 ms (single frame) | ~60 ms (8–32 frame clip) |
| **Memory overhead** | Zero buffering | Must buffer 8–32 frames per tracked person (~2.4 MB/person) |
| **Fighting detection** | Keypoint proximity + inter-frame velocity | Clip-level action class (degraded accuracy at 1 FPS) |
| **Tawuran detection** | Crowd density + centroid convergence | Could classify "crowd violence" but overkill |
| **Stealing detection** | Hand-object proximity heuristic | Needs additional model regardless |
| **Pipeline complexity** | Simple — inline with detection (one model) | Complex — clip buffer, separate inference, temporal alignment |
| **Accuracy at 1 FPS** | Good — poses are spatially informative | **Degraded** — temporal signal too sparse |
| **GPU cost (always-on)** | 1.5 ms/frame | 60 ms/clip + buffering VRAM |

#### When SlowFast Does Make Sense

SlowFast becomes valuable only with **temporarily increased frame rate** for suspicious cameras:

```
Normal operation:  Camera at 1 FPS → YOLO Pose → "suspicious pose detected"
Triggered mode:    Camera bumps to 15 FPS for 3 seconds → SlowFast confirms/denies
                   (45 frames captured → proper temporal signal for both pathways)
```

This is a valid **Tier 2 confirmation strategy** but not a primary detector at 1 FPS.

#### Verdict

> **At 1 FPS, YOLO Pose is strictly superior to SlowFast for behavior detection.** SlowFast's core architectural advantage (dual-pathway temporal modeling) is neutralized by the sparse frame rate. YOLO Pose provides spatially rich keypoint data on every single frame with no buffering overhead.
>
> **Recommendation:** Use SlowFast only as an optional Tier 2 confirmer when the camera can temporarily increase to 15+ FPS upon trigger. For the always-on pipeline at 1 FPS, rely on YOLO Pose keypoint analysis.

---

### 14.4 RF-DETR as On-Demand (Tier 2) — Rationale

#### RF-DETR's Small Object Advantage

RF-DETR uses **deformable attention** that focuses on arbitrary-scale features. This gives it a significant edge on small objects:

| Model | AP_small (< 32×32 px) | AP_medium | AP_large |
|-------|----------------------|----------|---------|
| RF-DETR-M | **~37** | ~58 | ~71 |
| YOLOv26-S | ~22 | ~52 | ~68 |
| YOLOv26-M | ~30 | ~56 | ~70 |

#### Why Small Object Advantage Doesn't Help for Always-On Behavior Detection

COCO "small" = **< 32×32 pixels**. In surveillance terms:

| Camera Setup | Person Size | COCO Category | Can Analyze Behavior? |
|-------------|------------|---------------|----------------------|
| 1080p, person at 5m | ~300×600 px | Large | ✅ Yes — full pose visible |
| 1080p, person at 15m | ~100×200 px | Medium | ✅ Yes — keypoints detectable |
| 1080p, person at 30m | ~50×100 px | Medium | ⚠️ Limited — coarse pose only |
| 1080p, person at 50m+ | ~25×50 px | **Small** | ❌ No — blurry blob, no useful pose |

> **Key insight:** If a person is small enough that YOLO misses it but RF-DETR detects it, that person is too small for behavior analysis anyway. There aren't enough pixels to determine fighting, stealing, or tawuran.

#### Where RF-DETR's Strength IS Valuable (On-Demand)

RF-DETR excels when triggered to confirm specific events:

| On-Demand Task | Why RF-DETR is Better |
|---------------|----------------------|
| **Stolen object detection** | Bags, phones, wallets are small objects — RF-DETR's sweet spot |
| **Weapon detection** | Knives, guns are small — RF-DETR detects more reliably |
| **Precise person-object interaction** | Higher AP means fewer false negatives on critical frames |
| **Distant crowd counting** | For tawuran, counting heads at distance where YOLO may miss some |

#### Cost Analysis: Always-On vs On-Demand

| RF-DETR Mode | Per-Frame Cost | At 800 Cameras | GPU Utilization |
|-------------|---------------|----------------|----------------|
| **Always-on (every frame)** | 2.7 ms/frame | 2,160 ms/sec (needs full GPU) | ~100% of GPU 1 |
| **On-demand (5% trigger rate)** | 2.7 ms × 5% = **0.14 ms/frame** | 108 ms/sec | **~5% of GPU 1** |
| **On-demand (20% trigger rate)** | 2.7 ms × 20% = **0.54 ms/frame** | 432 ms/sec | **~20% of GPU 1** |

Running RF-DETR on-demand at a 5% trigger rate uses **~95% less GPU** than always-on, while still providing high-accuracy confirmation exactly when it matters.

#### The On-Demand Trigger Flow

```
YOLO Pose (always-on, every frame, ~1.5 ms)
    │
    ├── 95% of frames: normal activity
    │   └── Crop → MongoDB (done, no Tier 2 needed)
    │
    └── 5% of frames: suspicious activity detected
        │  (e.g., aggressive pose, hand near another's bag,
        │   abnormal crowd convergence)
        │
        └── RF-DETR (on-demand, ~2.7 ms, this frame only)
                │
                ├── Confirm: stolen object in hand?
                ├── Confirm: weapon detected?
                ├── Confirm: precise person count in crowd?
                │
                └── If confirmed → Alert + Evidence saved
                    If denied → Suppress false alarm
```

> **Verdict:** RF-DETR should be **on-demand (Tier 2)**, not always-on. Its small object strength is most valuable for confirming specific detections (stolen objects, weapons), not for primary surveillance scanning. Running it always-on wastes GPU and provides no behavior detection capability without a second model.

---

### 14.5 Recommended Architecture — Final Pipeline

Based on the analysis in Sections 14.1–14.4, the recommended architecture is:

- **Tier 1 (always-on):** YOLOv26-S-Pose — detection + pose in a single 1.5 ms pass
- **Tier 2 (on-demand):** RF-DETR-M — high-accuracy confirmation for suspicious events
- **Behavior analysis:** CPU-side keypoint math (fighting, tawuran) + RF-DETR object confirmation (stealing)
- **SlowFast:** Optional Tier 2 only if camera can temporarily increase to 15+ FPS

#### Complete Pipeline Flowchart

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  CAPTURE LAYER                                                              │
│                                                                             │
│  Thread per CCTV stream (1 FPS capture)                                     │
│    → RTSP connection (TCP, persistent)                                      │
│    → NVDEC hardware decode (cameras 1–48 on GPU, rest on CPU)               │
│    → Frame saved to shared memory ring buffer (3 slots per camera)          │
│    → Backpressure: drop-oldest if buffer full                               │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  TIER 1 — ALWAYS-ON (GPU 0, every frame, every camera)                      │
│                                                                             │
│  Pool/batch frames (batch_size=16)                                          │
│    → YOLOv26-S-Pose inference (~1.5 ms/frame, TensorRT FP16)               │
│    → Output per frame:                                                      │
│         • Bounding boxes for ALL classes (person, vehicle, bag, animal...)  │
│         • Class labels + confidence scores for every detected object        │
│         • 17 body keypoints per PERSON (with confidence per keypoint)       │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  OBJECT-LEVEL PROCESSING (all detected objects)                     │    │
│  │                                                                     │    │
│  │  For ALL classes:                                                   │    │
│  │    • Crop bounding box → JPEG encode → batch for MongoDB            │    │
│  │    • Store metadata: class, confidence, bbox, camera_id, timestamp  │    │
│  │                                                                     │    │
│  │  For VEHICLES:                                                      │    │
│  │    • Track vehicle bbox across frames (loitering detection)         │    │
│  │    • Restricted zone violation (bbox inside forbidden region)        │    │
│  │                                                                     │    │
│  │  For BAGS/LUGGAGE:                                                  │    │
│  │    • Unattended bag: bag bbox with no person bbox within radius     │    │
│  │                                                                     │    │
│  │  For ANIMALS:                                                       │    │
│  │    • Restricted area intrusion: animal bbox in forbidden zone        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  PERSON-SPECIFIC BEHAVIOR ANALYSIS (CPU, ~0.01 ms/person)          │    │
│  │  (uses keypoints — only available for person class)                 │    │
│  │                                                                     │    │
│  │  FightingDetector:                                                  │    │
│  │    • Keypoint velocity between consecutive frames                   │    │
│  │    • Inter-person proximity (wrist-to-head distance < threshold)    │    │
│  │    • Aggressive pose classification (raised arms, lunging torso)    │    │
│  │                                                                     │    │
│  │  TawuranDetector:                                                   │    │
│  │    • Crowd density (persons per m² exceeds threshold)               │    │
│  │    • Centroid convergence (people moving toward each other)         │    │
│  │    • Collective aggression score (% of crowd with fighting pose)    │    │
│  │                                                                     │    │
│  │  StealingHeuristic (pre-screen):                                    │    │
│  │    • Hand keypoint proximity to detected bag/object bbox            │    │
│  │    • Unusual hand-to-bag/pocket trajectory                          │    │
│  │    • Person-to-person hand interaction near object                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└──────────────┬──────────────────────────────────┬──────────────────────────┘
               │                                  │
          Normal (95%)                     Suspicious (5%)
               │                                  │
               ▼                                  ▼
┌──────────────────────────────┐  ┌──────────────────────────────────────────────┐
│  STANDARD OUTPUT              │  │  TIER 2 — ON-DEMAND (GPU 1, triggered only)  │
│                               │  │                                              │
│  Crop ALL detected objects    │  │  RF-DETR-M inference (~2.7 ms/frame)         │
│    → JPEG encode              │  │    → High-accuracy re-detection of scene     │
│    → Batch (100–500 docs)     │  │    → Small object detection (bags, weapons,  │
│    → Async MongoDB write      │  │      phones, knives — RF-DETR excels here)   │
│      (motor driver)           │  │    → Detects objects YOLO may have missed    │
│                               │  │                                              │
│  Metadata saved per object:   │  │  Confirmation Logic:                          │
│    • camera_id                │  │    • Stealing: object in hand + proximity     │
│    • timestamp                │  │    • Weapon: object class = knife/gun/etc     │
│    • object_class (person,    │  │    • Unattended bag: re-confirm no owner      │
│      vehicle, bag, animal...) │  │    • False alarm: RF-DETR sees no anomaly     │
│    • bbox coordinates         │  │      → Suppress alert, save as normal frame   │
│    • confidence score         │  │                                              │
│    • person_count (per frame) │  │  If confirmed:                                │
│    • vehicle_count            │  │    → Alert (webhook / push notification)      │
│    • keypoints (person only)  │  │    → Evidence clip saved (10-sec buffer)      │
│                               │  │    → Behavior metadata → MongoDB              │
└──────────────────────────────┘  └──────────────────────────────────────────────┘
```

#### Processing Timeline (single frame, single camera)

```
Time (ms):  0        1.5      1.51         1.52              4.22
            │         │        │            │                  │
            ▼         ▼        ▼            ▼                  ▼
         YOLOv26   Result   Behavior     Crop+Save         RF-DETR
         -Pose     ready    Analysis     to MongoDB         (only if
         starts             (CPU,        (async,            suspicious)
                            ~0.01ms)     non-blocking)

Normal frame total:  ~1.52 ms (Tier 1 only)
Suspicious frame:    ~4.22 ms (Tier 1 + Tier 2)
Blended average:     ~1.52 + (0.05 × 2.7) = ~1.66 ms/frame
```

#### GPU Assignment

```
┌─────────────────────────────────────────────────────────────────┐
│  GPU 0 (V100 #1) — TIER 1: Always-On Detection + Pose          │
│                                                                 │
│  NVDEC: cameras 1–24 (H.264 → GPU memory, zero-copy)           │
│  Model: YOLOv26-S-Pose (TensorRT FP16, 0.8 GB)                 │
│  Batch: 16 frames → 16 × 1.5 ms = 24 ms → 667 cameras/sec     │
│  VRAM: 2.6 GB used / 29.4 GB free                              │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  GPU 1 (V100 #2) — TIER 2: On-Demand Confirmation              │
│                                                                 │
│  NVDEC: cameras 25–48 (H.264 → GPU memory, zero-copy)          │
│  Model: RF-DETR-M (TensorRT FP16, 2.5 GB)                      │
│  Load: 5% of frames → ~33–56 frames/sec (at 667–1120 cameras)  │
│  VRAM: 4.4 GB used / 27.6 GB free                              │
│                                                                 │
│  ⚡ ~95% idle — available for future models or scaling          │
└─────────────────────────────────────────────────────────────────┘
```

---

### 14.6 Hardware Requirements — Scenario Comparison

#### Scenario 1: YOLOv26-Pose (Tier 1) + RF-DETR On-Demand (Tier 2) — RECOMMENDED

This is the full behavior detection pipeline with NVDEC zero-copy decode.

| Component | Specification | Purpose | Est. Cost (USD) |
|-----------|--------------|---------|-----------------:|
| **CPU** | AMD EPYC 9374F (32C/64T, 3.85 GHz) | RTSP I/O, overflow decode (~0.09 core/cam with NVDEC), ByteTrack, behavior analysis | $2,500–3,000 |
| **Motherboard** | Supermicro H13SSL-N (SP5 socket) | Single-socket EPYC, 2× PCIe Gen5 x16 for dual V100 | $500–700 |
| **RAM** | 128 GB DDR5-4800 ECC RDIMM (4×32 GB) | Ring buffers, batch queues, tracking state, OS overhead | $400–500 |
| **GPU 0** | NVIDIA V100 32 GB *(already owned)* | Tier 1: NVDEC (24 streams) + YOLOv26-S-Pose (2.6 GB VRAM used) | — |
| **GPU 1** | NVIDIA V100 32 GB *(already owned)* | Tier 2: NVDEC (24 streams) + RF-DETR-M on-demand (4.4 GB VRAM used) | — |
| **NIC** | Mellanox ConnectX-5 25 GbE (dual-port) | 800 cameras × ~1 Mbps = 800 Mbps sustained | $150–200 |
| **Boot SSD** | 512 GB NVMe (PCIe Gen4) | OS, CUDA toolkit, Python env, model weights (~5 GB) | $50–70 |
| **Data SSD** | 2 TB NVMe (Samsung 990 Pro) | MongoDB data dir + alert clip buffer | $150–200 |
| **Archive HDD** | 2× 8 TB HDD (RAID 1) | Long-term evidence storage | $240–300 |
| **PSU** | 1,200W 80+ Platinum | 2× V100 (300W) + EPYC (320W) + overhead | $200–300 |
| **Chassis** | 4U rackmount server | Dual GPU cooling, sufficient airflow | $300–500 |
| **UPS** | 1,500VA online UPS | Power protection during writes | $300–400 |
| | | | |
| **Total (excl. V100s)** | | | **$4,800–6,200** |
| **Total (incl. V100s at ~$3K each)** | | | **$10,800–12,200** |

**Capacity:** 800–1,120 cameras at 1 FPS with behavior detection (fighting, stealing, tawuran).

#### Scenario 2: RF-DETR Always-On (No Behavior Detection)

Detection + crop + save only. No pose, no behavior analysis, no Tier 2.

| Component | Specification | Purpose | Est. Cost (USD) |
|-----------|--------------|---------|-----------------:|
| **CPU** | AMD EPYC 9274F (24C/48T, 4.05 GHz) | RTSP I/O, overflow decode, JPEG encode, MongoDB writes | $1,500–2,000 |
| **Motherboard** | Supermicro H13SSL-N (SP5 socket) | Single-socket EPYC | $500–700 |
| **RAM** | 64 GB DDR5-4800 ECC RDIMM (2×32 GB) | Ring buffers only (no clip buffers, no tracking state) | $200–250 |
| **GPU** | 1× NVIDIA V100 32 GB *(already owned)* | NVDEC (24 streams) + RF-DETR-M always-on (4.4 GB VRAM used) | — |
| **NIC** | Intel X710-DA2 10 GbE (dual-port) | 500 cameras × ~1 Mbps = sufficient | $80–120 |
| **Boot SSD** | 512 GB NVMe | OS + model weights | $50–70 |
| **Data SSD** | 2 TB NVMe | MongoDB data directory | $150–200 |
| **Archive HDD** | 1× 8 TB HDD | Crop storage (optional) | $120–150 |
| **PSU** | 850W 80+ Gold | 1× V100 (300W) + EPYC (250W) + overhead | $120–150 |
| **Chassis** | 4U rackmount or tower workstation | Single GPU cooling | $200–400 |
| **UPS** | 1,000VA online UPS | Power protection | $200–250 |
| | | | |
| **Total (excl. V100)** | | | **$3,100–4,300** |
| **Total (incl. 1× V100 at ~$3K)** | | | **$6,100–7,300** |

**Capacity:** ~370 cameras at 1 FPS (RF-DETR-M at 2.7 ms/frame, batch=16). Detection + crop only, no behavior detection.

> **Note on Scenario 2 throughput:** RF-DETR is 1.8× slower than YOLOv26-S per frame (2.7 ms vs 1.5 ms). If you use a single V100 with RF-DETR always-on, maximum cameras drops from ~667 (YOLO) to **~370** per GPU. Using both V100s for RF-DETR would recover to ~740, but then you have no GPU for Tier 2 confirmation.

---

### 14.7 Side-by-Side Comparison

| Specification | **Scenario 1 (Recommended)** | **Scenario 2** |
|--------------|------------------------------|----------------|
| | *YOLO Pose + RF-DETR on-demand* | *RF-DETR always-on, no behavior* |
| | | |
| **Primary model** | YOLOv26-S-Pose (1.5 ms) | RF-DETR-M (2.7 ms) |
| **Secondary model** | RF-DETR-M (on-demand, 5%) | None |
| **Detection classes** | 80 COCO (person, vehicle, bag, animal, etc.) | 80 COCO (same classes) |
| **Keypoints (person)** | ✅ 17 body keypoints per person | ❌ None |
| **Behavior analysis** | CPU keypoint math | ❌ Not available |
| | | |
| **CPU** | EPYC 9374F (32C/64T) | EPYC 9274F (24C/48T) |
| **RAM** | 128 GB DDR5 ECC | 64 GB DDR5 ECC |
| **GPUs active** | 2× V100 32 GB | 1× V100 32 GB |
| **NIC** | 25 GbE | 10 GbE |
| **PSU** | 1,200W Platinum | 850W Gold |
| **System power** | ~1,000W sustained | ~600W sustained |
| | | |
| **Max cameras (1 FPS)** | **800–1,120** | **~370** (1 GPU) / ~740 (2 GPU) |
| **Small object detection** (bags, weapons, animals at distance) | On-demand via RF-DETR (Tier 2) | ✅ Always-on |
| **Person detection AP** | ~90%+ (YOLO, sufficient) | ~93%+ (RF-DETR, higher) |
| **Vehicle detection** | ✅ (large objects, both models excel) | ✅ |
| **Bag/weapon detection** | ✅ YOLO detects + RF-DETR confirms small ones | ✅ Better for small/distant objects |
| **Animal detection** | ✅ | ✅ |
| | | |
| **Fighting detection** | ✅ Keypoint velocity + proximity | ❌ No keypoints |
| **Tawuran detection** | ✅ Crowd density + convergence | ❌ No keypoints |
| **Stealing detection** | ✅ Hand-to-bag keypoint + RF-DETR confirm | ❌ No hand position data |
| **Unattended bag** | ✅ Bag bbox + no nearby person bbox | ✅ Same logic, slightly better bag AP |
| **Vehicle loitering** | ✅ Track vehicle bbox across frames | ✅ Same capability |
| **Animal intrusion** | ✅ Animal bbox in restricted zone | ✅ Same capability |
| **Weapon detection** | ⚠️ YOLO detects + RF-DETR Tier 2 confirms | ✅ Better small weapon AP always-on |
| **False alarm suppression** | ✅ RF-DETR Tier 2 confirmation | ❌ No confirmation layer |
| | | |
| **Cost (excl. V100s)** | **$4,800–6,200** | **$3,100–4,300** |
| **Cost (incl. V100s)** | **$10,800–12,200** | **$6,100–7,300** |
| **Cost difference** | — | Saves $4,700–4,900 |
| | | |
| **Best for** | Full surveillance: multi-class detection + behavior analysis | Simple multi-class detection + crop + save |

### 14.8 Final Recommendation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   ✅  RECOMMENDED: Scenario 1 — YOLOv26-Pose (Tier 1) + RF-DETR (Tier 2)  │
│                                                                             │
│   • Detects ALL object classes (person, vehicle, bag, weapon, animal)       │
│   • ADDITIONALLY provides 17 body keypoints for every detected person      │
│   • One model handles multi-class detection + behavior analysis in ~1.5 ms │
│   • RF-DETR confirms suspicious events on-demand (~5% of frames)           │
│     → excels at confirming small objects: bags, weapons, distant animals    │
│   • 800–1,120 cameras on 2× V100 at 1 FPS                                  │
│   • Full fighting, stealing, and tawuran detection                          │
│   • Bbox-only behaviors (unattended bag, vehicle loitering, animal          │
│     intrusion) work with both models — no keypoints needed                  │
│   • GPU 1 is ~95% idle → room for future expansion                         │
│                                                                             │
│   ❌  NOT RECOMMENDED: SlowFast at 1 FPS                                   │
│   • Fast pathway receives no useful temporal signal at 1 FPS                │
│   • Use only as optional Tier 2 with temporary FPS increase (15+ FPS)      │
│                                                                             │
│   ❌  NOT RECOMMENDED: RF-DETR as always-on primary                        │
│   • Detects all classes well, but NO keypoints for any class               │
│   • Cannot detect fighting/tawuran/stealing without adding YOLO Pose       │
│   • 1.8× slower → fewer cameras per GPU                                    │
│   • Small object advantage is valuable but better leveraged on-demand      │
│   • Best role: Tier 2 confirmer for stolen objects / weapons / small items  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
