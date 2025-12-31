
# ADAgent
This is the official PyTorch implementation of ADAgent from the paper "ADAgent: LLM Agent for Alzheimer’s Disease Analysis with Collaborative Coordinator" accepted by MICCAI 2025 Workshop (Agentic AI for Medicine).

## Demo

https://github.com/user-attachments/assets/3c51b397-9f29-4090-a3f6-870adcf36dcc

## Abstract
Alzheimer’s disease (AD) is a progressive and irreversible neurodegenerative disease. Early and precise diagnosis of AD is crucial for timely intervention and treatment planning to alleviate the progressive neurodegeneration. However, most existing methods rely on single-modality data, which contrasts with the multifaceted approach used by medical experts. While some deep learning approaches process multi-modal data, they are limited to specific tasks with a small set of input modalities and cannot handle arbitrary combinations. This highlights the need for a system that can address diverse AD-related tasks, process multi-modal or missing input, and integrate multiple advanced methods for improved performance. In this paper, we propose ADAgent, the first specialized AI agent for AD analysis, built on a large language model (LLM) to address user queries and support decision-making. ADAgent integrates a reasoning engine, specialized medical tools, and a collaborative outcome coordinator to facilitate multi-modal diagnosis and prognosis tasks in AD. Extensive experiments demonstrate that ADAgent outperforms SOTA methods, achieving significant improvements in accuracy, including a 2.7% increase in multi-modal diagnosis, a 0.7% improvement in multi-modal prognosis, and enhancements in MRI and PET diagnosis tasks.

## Workflow
![image](https://github.com/willenhou/ADAgent/blob/main/assets/w6.png)

## 🔧 Tool

ADAgent provides the following diagnostic tools:
- **MRI Diagnosis Tool** (`MriDiagnosisTool`): Diagnose Alzheimer's disease based on MRI images
- **PET Diagnosis Tool** (`PetDiagnosisTool`): Diagnose Alzheimer's disease based on PET images
- **MRI+PET Combined Diagnosis Tool** (`MriPetDiagnosisTool`): Comprehensive diagnosis combining MRI and PET images

All diagnostic tools support NIFTI format (`.nii`) files. The diagnosis results provide probabilities for three stages:
- **0**: CN (Cognitive Normal) - Cognitively normal
- **1**: MCI (Mild Cognitive Impairment) - Mild cognitive impairment
- **2**: AD (Alzheimer's Disease) - Alzheimer's disease

## 📦 Installation Guide

### 1. Clone the Repository

```bash
git clone https://github.com/wlhou/ADAgent.git
cd ADAgent
```

### 2. Create Main Virtual Environment (Recommended)

```bash
conda create -n adagent python=3.12 -y
conda activate adagent
pip install -e .
```

### 3. Create Sub Virtual Environment for CmVim and nnMamba

This is used to run CMViM and nnMamba. Please name the sub virtual enviroment as vimMamba. To install vimMamba, please refer to [Vim](https://github.com/hustvl/Vim), which must be installed with cuda 11.8. And you also need add the path of vimMamba in the file ADAgent/adagent/tools/mri_pet_diagnosis.py. For example, in my own project:
```bash
new_env['PATH'] = '/home/wlhou/anaconda3/envs/vimMamba/bin:' + new_env['PATH']  # 修改 PATH 环境变量
```

### 4. Download Model Weights

Please download the [model weights](https://huggingface.co/Willenhou/ADAgent/tree/main) in the file ADAgent/model-weights/.

## 🚀 Usage

### Launch the Application

Run the main program:

```bash
python main.py
```

After the program starts, you will see output similar to:

```
Starting server...
Agent initialized
Running on local URL:  http://127.0.0.1:7860
Running on public URL: https://xxxxx.gradio.live
```

### Using the Web Interface

1. Open your browser and visit the displayed local URL (usually `http://127.0.0.1:7860`)
2. In the interface:
   - **Upload MRI file**: Click the "📎 Upload MRI" button and select an MRI file in NIFTI format
   - **Upload PET file**: Click the "📄 Upload PET" button and select a PET file in NIFTI format
   - **Enter question**: Type your question or instruction in the text box
   - **View results**: Diagnosis results will be displayed in the chat interface

### Example Questions

- "Please analyze this MRI image and diagnose whether the patient has Alzheimer's disease"
- "Based on the uploaded PET image, provide a diagnosis result"
- "Comprehensively analyze the MRI and PET images to assess the patient's cognitive status"

### Command Line Parameters (Optional)

If you need to customize the configuration, you can modify the parameters in `main.py`:

```python
# Modify in main.py
selected_tools = [
    "MriDiagnosisTool",
    "PetDiagnosisTool",
    "MriPetDiagnosisTool",
]

# Modify server configuration
demo.launch(
    server_name="0.0.0.0",  # Listen on all network interfaces
    server_port=8585,       # Specify port
    share=False             # Whether to create public link
)
```

## 🔍 Troubleshooting
### Issue 1: API Key Error

**Error**: `Invalid API key` or `Authentication failed`

**Solution**:
- Check if `OPENAI_API_KEY` in the `.env` file is correct
- Ensure the `.env` file is in the project root directory
- Verify the API key is valid and has sufficient credits

### Issue 2: Missing Model Files

**Error**: `FileNotFoundError: model-weights/xxx.pt`

**Solution**:
- Ensure the `model-weights/` directory exists and contains the required `.pt` files
- Contact the project maintainer to obtain model weight files

### Issue 3: Port Already in Use

**Error**: `Address already in use`

**Solution**:
- Modify the port number in `main.py`: `demo.launch(server_port=8586)`
- Or close the program that's using the port


## 📝 Notes

1. **Data Privacy**: This project processes medical imaging data. Please ensure compliance with relevant privacy regulations
2. **Model Weights**: Model weight files are large and may need to be downloaded separately
3. **API Costs**: Using OpenAI API will incur costs. Please monitor your usage
4. **GPU Requirements**: While it can run on CPU, GPU will significantly improve performance

## 🙏 Acknowledgments

Thanks to all developers and researchers who contributed to this project.

## Citation

If you find this work useful, please cite our paper:
```
@inproceedings{hou2025adagent,
  title={{ADAgent}: {LLM} agent for {Alzheimer’s} disease analysis with collaborative coordinator},
  author={Hou, Wenlong and Yang, Guangqian and Du, Ye and Lau, Yeung and Liu, Lihao and He, Junjun and Long, Ling and Wang, Shujun},
  booktitle={International Workshop on Agentic AI for Medicine},
  pages={23--32},
  year={2025},
  organization={Springer}
}
```

---

**Disclaimer**: This tool is for research and educational purposes only and cannot replace professional medical diagnosis. Any medical decisions should be made in consultation with professional doctors.
