import os
import subprocess
import torch
import torch.nn as nn
import json
from pathlib import Path

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from typing import Optional, Type

# Get project root (go up 2 levels from this file: tools -> adagent -> root)
_PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
_SUBTOOLS_DIR = _PROJECT_ROOT / "adagent" / "tools" / "subtools"
_TEMP_DIR = _PROJECT_ROOT / "temp"

class MriPetDiagnosisInput(BaseModel):
    """Input schema for the AD diagnosis Tool whose inputs are MRI and PET. Only supports NIFTI files."""

    mri_path: str = Field(
        ...,
        description="Path to MRI file, only supports NIFTI files",
    )
    pet_path: str = Field(
        ...,
        description="Path to PET file, only supports NIFTI files",
    )

class MriPetDiagnosisTool(BaseTool):
    """
    """

    name: str = "mri_pet_diagnosis"
    description: str = (
        "To diagnose whether a patient has Alzheimer's Disease(AD) or not based on MRI file and PET file?"
        "The inputs are MRI file and PET file. To input the MRI file path first, then enter the PET file path. Example input: {'mri_path': '/path/to/mri.nii', 'pet_path': '/path/to/mri.nii'}"
        "If the input is a MRI file, please use other tool. If the input is a PET file, please use other tool."
        "The tool will give you some results predicted by multiple models. Each model will give you an array consisting three elements, which represent the probabilities of a prediction of 0, 1, and 2 in sequence."
        "O represents the patient is in CN stage. 1 represents the patient is in MCI stage. 2 represents the patient is in AD stage."
    )
    args_schema: Type[BaseModel] = MriPetDiagnosisInput
    device: str = "cuda"

    def __init__(self, device: Optional[str] = "cuda"):
        super().__init__()
        self.device = torch.device(device) if device else "cuda"

    def _run(
        self,
        mri_path: str,
        pet_path: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ):
        print("调用成功")
        try:
            print("准备进入子程序")
            new_env = os.environ.copy()

            # 修改或添加新的环境变量
            new_env['MY_VARIABLE'] = 'my_value'
            # Note: Remove hardcoded conda path - users should configure their own environment
            new_env['PATH'] = '/home/wlhou/anaconda3/envs/vimMamba/bin:' + new_env['PATH']  # 修改 PATH 环境变量
            #new_env['PATH'] = '/home/gqyang/miniconda3/envs/mamba/bin:' + new_env['PATH']
            #new_env['PYTHONPATH'] = '/data/Data2/wlhou/anaconda3/envs/ResNet/lib/python3.12/site-packages'  # 设置 PYTHONPATH
            new_env['DEBUG'] = '1'  # 设置一个自定义的调试标志
            proc = subprocess.Popen(
                ["python", str(_SUBTOOLS_DIR / "mri_pet_diagnosis.py")],
                stdin=subprocess.PIPE,           # 向子进程发送数据
                stdout=subprocess.PIPE,          # 接收子进程输出
                stderr=subprocess.PIPE,          # 接收子进程错误
                universal_newlines=True,        # 文本模式
                env=new_env                    # 自定义子进程环境变量
            )
            print("Popen建立成功")
            # 发送输入数据到子进程（可传复杂对象）
            input_data = {
            "mri_path": mri_path,
            "pet_path": pet_path
            }
            print("通过Popen.communicate发送input")
            # 通过 JSON 序列化通信数据
            stdout, stderr = proc.communicate(json.dumps(input_data))
            print("接收子进程返回值成功")
            outputs = None
            _TEMP_DIR.mkdir(exist_ok=True)
            with open(str(_TEMP_DIR / "child_output.json"), "r") as f:
                outputs = json.load(f)

            #os.remove(str(_TEMP_DIR / "child_output.json"))
            if proc.returncode == 0:
                print("解析子进程返回结果成功")
            else:
                print(f"子进程错误 (Code {proc.returncode}):")
                print("错误信息:", stderr)

            metadata = {
                "mri_path": mri_path,
                "pet_path": pet_path,
                "analysis_status": "completed",
                "note": "The tool will give you some results predicted by multiple models. Each model will give you an array consisting three elements, which represent the probabilities of a prediction of 0, 1, and 2 in sequence. O represents the patient is in CN stage. 1 represents the patient is in MCI stage. 2 represents the patient is in AD stage.",
            }
            return outputs, metadata
        
        except Exception as e:
            print("进入子程序失败")
            return {"error": str(e)}, {
                "image_path": mri_path,
                "analysis_status": "failed",
            }

    async def _arun(
        self,
        mri_path: str,
        pet_path: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ):
        """Asynchronous version of _run."""
        return self._run(mri_path, pet_path, run_manager)