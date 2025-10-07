<div align="center">

# 📝 FormLens
**From Ink to Insight with Adapting Vision-Language Models for Handwritten Form Digitization**

The majority of the code is sourced from the MS-SWIFT GOT OCR 2.0 repository--->https://github.com/Ucas-HaoranWei/GOT-OCR2.0. The data preprocessing codes are available here.

*Adapting Vision-Language Models for End-to-End Handwritten Form Digitization*

</div>

---

## 🎯 **Overview**

**FormLens** is an innovative adaptation of the vision-language model GOT 2.0, specifically tailored for end-to-end handwritten form digitization. Our model directly transforms full input images into structured key-value pairs, eliminating the need for region detection, OCR, or rule-based processing.

### ✨ **Key Features**
- 🔄 **End-to-End Processing**: Direct image-to-structured data transformation
- 📱 **Real-World Robustness**: Handles diverse layouts, backgrounds, and orientations
- 🎨 **Handwritten Text Support**: Optimized for handwritten form fields
- 🌐 **Multilingual Capabilities**: Supports multiple languages including Hindi
- 📊 **High Accuracy**: Outperforms commercial and open-source alternatives

---

## 🚀 **Quick Start**

### **Prerequisites**
```bash
# Install required dependencies
pip install torch torchvision
pip install opencv-python numpy
pip install transformers datasets
```

### **Model Fine-tuning**
```bash
# Fine-tune FormLens model
CUDA_VISIBLE_DEVICES=1 \
swift sft \
    --model stepfun-ai/GOT-OCR2_0 \
    --train_type lora \
    --resume_from_checkpoint /ssd_scratch/shaon/got_output_form/v0-20250122-114933/checkpoint-965000 \
    --dataset /ssd_scratch/shaon/Hindi_original_data_format1/splitted_word/train/filtered_word.jsonl \
    --val_dataset /ssd_scratch/shaon/Hindi_original_data_format1/splitted_word/val/filtered_word.jsonl \
    --max_steps 1000000 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --output_dir /ssd_scratch/shaon/got_output_printed_word \
    --eval_steps 5000 \
    --save_steps 5000
```

---

## 🛠️ **Tools & Utilities**

### **Data Preprocessing**
Our comprehensive data preprocessing toolkit includes:

- **`data_preprocess.py`** - Image augmentation and preprocessing
- **`create_jsonl.py`** - Training data format conversion
- **Augmentation Features**:
  - 🔄 Rotation augmentation
  - 📸 Noise injection (Gaussian, Salt-Pepper)
  - 🎭 Perspective transformation
  - 🌫️ Blur effects (Gaussian, Motion)
  - 🌟 Brightness/Contrast adjustment

### **Usage Examples**

#### **Image Augmentation**
```bash
# Apply various augmentations to training images
python data_preprocess.py \
    --input your_form_image.jpg \
    --output augmented_results \
    --rotation_range -15 15 \
    --noise_type gaussian \
    --blur_type motion
```

#### **JSONL Data Creation**
```bash
# Create training data from CSV
python create_jsonl.py \
    --csv your_data.csv \
    --output training_data.jsonl \
    --image_path_column image_path \
    --response_column response
```

---

## 📊 **Performance**

FormLens demonstrates superior performance compared to existing solutions:

| Method | WRR (%) | CRR (%) | F1 (%) |
|--------|---------|---------|--------|
| Google Form Parser | 92.14 | 96.38 | 89.67 |
| Azure Form Recognizer | 93.29 | 97.25 | 91.85 |
| **FormLens (Ours)** | **95.44** | **98.33** | **94.71** |

---

## 📁 **Project Structure**

```
FormLens/
├── 📄 README.md                    # This file
├── 🐍 data_preprocess.py          # Image augmentation toolkit
├── 🐍 create_jsonl.py             # Training data formatter
├── 📋 README_preprocessing.md      # Detailed preprocessing docs
└── 📊 datasets/                   # Training datasets
    ├── Form6000/                  # Our benchmark dataset
    └── processed/                 # Preprocessed training data
```

---

## 🎮 **Live Demo**

Try our FormLens model live:
- **🔗 Demo URL**: [http://10.10.16.13:5000/form_ocr](http://10.10.16.13:5000/form_ocr)
- **📱 Features**: Upload handwritten forms, get structured output
- **🌐 Access**: Available 24/7 for testing and evaluation

---

## 📚 **Dataset: Form6000**

We release **Form6000**, a comprehensive benchmark dataset:

| Characteristic | Count |
|----------------|-------|
| Unique form templates | 50 |
| Participants (writers) | 650 |
| Total handwritten forms | 650 |
| Captured mobile images | 5,350 |
| **Total dataset size** | **6,000** |

### **Dataset Features**
- 📝 Real-world handwritten forms
- 📱 Mobile-captured images (7-10 per form)
- 🎨 Diverse handwriting styles
- 🌍 Multiple form layouts and backgrounds

---

## 🤝 **Contributing**

We welcome contributions! Please see our contributing guidelines:

1. **🐛 Bug Reports**: Use GitHub Issues
2. **💡 Feature Requests**: Submit enhancement proposals
3. **📝 Code Contributions**: Fork, branch, and submit PRs
4. **📊 Dataset Contributions**: Help expand Form6000

---

## 📄 **Citation**

If you use FormLens in your research, please cite our paper:

```bibtex
<!-- @article{bhattacharya2024formlens,
  title={FormLens: From Ink to Insight with Adapting Vision-Language Models for Handwritten Form Digitization},
  author={Bhattacharya, Shaon and Mondal, Ajoy and Jawahar, C V},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
} -->
```

---

## 📞 **Contact & Support**

- **👨‍💻 Authors**: Shaon Bhattacharya, Ajoy Mondal, C V Jawahar
- **🏛️ Institution**: CVIT, International Institute of Information Technology
- **📧 Email**: [Contact Information]
- **🌐 Project Page**: [https://formlens.github.io](https://formlens.github.io)

---

## 📜 **License**

<!-- This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. -->

---

## 🙏 **Acknowledgments**

- **GOT 2.0**: Based on the excellent work from [MS-SWIFT GOT OCR 2.0](https://github.com/Ucas-HaoranWei/GOT-OCR2.0)
- **Community**: Thanks to all contributors and the open-source community
- **Institution**: CVIT, IIIT Hyderabad for research support

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

Made with ❤️ by the FormLens Team

</div>