import re
import gradio as gr
from pathlib import Path
import time
import shutil
from typing import AsyncGenerator, List, Optional, Tuple
from gradio import ChatMessage
import nibabel as nib
import os
import cv2 as cv

class ChatInterface:
    """
    A chat interface for interacting with a medical AI agent through Gradio.

    Handles file uploads, message processing, and chat history management.
    Supports both regular image files and DICOM medical imaging files.
    """

    def __init__(self, agent, tools_dict):
        """
        Initialize the chat interface.

        Args:
            agent: The medical AI agent to handle requests
            tools_dict (dict): Dictionary of available tools for image processing
        """
        self.agent = agent
        self.tools_dict = tools_dict
        self.upload_dir = Path("temp") #Q：Path函数是什么？ Q：upload_dir是什么变量
        self.upload_dir.mkdir(exist_ok=True) #？
        self.current_thread_id = None
        # Separate storage for original and display paths
        self.original_file_path = None  # For LLM (.dcm or other)
        self.original_mri_path = None
        self.original_pet_path = None
        self.display_mri_path = None
        self.display_pet_path = None
        self.display_file_path = None   # For UI (always viewable format)
    
    def load_nifti_image(self, file_path):
        # Load the NIfTI file
        img = nib.load(file_path)
        #print(img.shape)
        data = img.get_fdata()
        return data
    
    def plot_slice(self, data, slice_index, axis):
        """
        Plot a slice of the 3D MRI data along the specified axis.
        :param data: 3D numpy array of MRI data
        :param slice_index: Index of the slice to plot
        :param axis: Axis along which to take the slice (0, 1, or 2)
        """
        if axis == 0:
            slice_data = data[slice_index, :, :]
        elif axis == 1:
            slice_data = data[:, slice_index, :]
        elif axis == 2:
            slice_data = data[:, :, slice_index]
        else:
            raise ValueError("Axis must be 0, 1, or 2")
        return slice_data

    def handle_upload(self, file_path: str) -> str:
        """
        Handle new file upload and set appropriate paths.

        Args:
            file_path (str): Path to the uploaded file

        Returns:
            str: Display path for UI, or None if no file uploaded
        """
        if not file_path:
            return None

        source = Path(file_path)
        timestamp = int(time.time())

        # Save original file with proper suffix
        suffix = source.suffix.lower()
        saved_path = self.upload_dir / f"upload_{timestamp}{suffix}"
        shutil.copy2(file_path, saved_path)  # Use file_path directly instead of source
        self.original_file_path = str(saved_path)

        # Handle DICOM conversion for display only
        if suffix == ".dcm":
            output, _ = self.tools_dict["DicomProcessorTool"]._run(str(saved_path))
            self.display_file_path = output["image_path"]
        else:
            self.display_file_path = str(saved_path)

        return self.display_file_path
    
    def handle_mri_upload(self, file_path: str) -> str:
        """
        Handle new mri upload and set appropriate paths.

        Args:
            file_path (str): Path to the uploaded file

        Returns:
            str: Display path for UI, or None if no file uploaded
        """
        if not file_path:
            return None

        source = Path(file_path)
        timestamp = int(time.time())

        # Save original file with proper suffix
        suffix = source.suffix.lower()
        saved_path = self.upload_dir / f"upload_{timestamp}{suffix}"
        shutil.copy2(file_path, saved_path)  # Use file_path directly instead of source
        self.original_mri_path = str(saved_path.resolve())

        nii_data = self.load_nifti_image(str(saved_path))
        sliced_image = self.plot_slice(nii_data, slice_index=90, axis=2)
        sliced_image = ((sliced_image - sliced_image.min()) / (sliced_image.max() - sliced_image.min()))*255
        sliced_image = sliced_image.astype("uint8")
        saved_slice_path = os.path.join(saved_path.parent, "mri_sliced.png")
        cv.imwrite(saved_slice_path, sliced_image)

        self.display_mri_path = str(saved_slice_path)
        return self.display_mri_path

    def handle_pet_upload(self, file_path: str) -> str:
        """
        Handle new pet upload and set appropriate paths.

        Args:
            file_path (str): Path to the uploaded file

        Returns:
            str: Display path for UI, or None if no file uploaded
        """
        if not file_path:
            return None

        source = Path(file_path)
        timestamp = int(time.time())

        # Save original file with proper suffix
        suffix = source.suffix.lower()
        saved_path = self.upload_dir / f"upload_{timestamp}{suffix}"
        shutil.copy2(file_path, saved_path)  # Use file_path directly instead of source
        self.original_pet_path = str(saved_path.resolve())

        nii_data = self.load_nifti_image(str(saved_path))
        sliced_image = self.plot_slice(nii_data, slice_index=90, axis=2)
        sliced_image = ((sliced_image - sliced_image.min()) / (sliced_image.max() - sliced_image.min()))*255
        sliced_image = sliced_image.astype("uint8")
        saved_slice_path = os.path.join(saved_path.parent, "pet_sliced.png")
        cv.imwrite(saved_slice_path, sliced_image)

        self.display_pet_path = str(saved_slice_path)
        return self.display_pet_path

    def add_message(
        self, message: str, display_mri: str, display_pet: str ,history: List[dict]
    ) -> Tuple[List[dict], gr.Textbox]:
        """
        Add a new message to the chat history.

        Args:
            message (str): Text message to add
            display_mri (str): Path to MRI being displayed
            display_pet (str): Path to PET being displayed
            history (List[dict]): Current chat history

        Returns:
            Tuple[List[dict], gr.Textbox]: Updated history and textbox component
        """
        mri_path = display_mri
        pet_path = display_pet
        # if mri_path is not None:
        #     history.append({"role": "user", "content": {"mri_path": mri_path}})
        # if pet_path is not None:
        #     history.append({"role": "user", "content": {"pet_path": pet_path}})
        if message is not None:
            history.append({"role": "user", "content": message})
        return history, gr.Textbox(value=message, interactive=False)

    async def process_message(
        self, message: str, display_mri: Optional[str], display_pet: Optional[str], chat_history: List[ChatMessage]
    ) -> AsyncGenerator[Tuple[List[ChatMessage], Optional[str], str], None]:
        """
        Process a message and generate responses.

        Args:
            message (str): User message to process
            display_image (Optional[str]): Path to currently displayed image
            chat_history (List[ChatMessage]): Current chat history

        Yields:
            Tuple[List[ChatMessage], Optional[str], str]: Updated chat history, display path, and empty string
        """
        chat_history = chat_history or []

        # Initialize thread if needed
        if not self.current_thread_id:
            self.current_thread_id = str(time.time())

        messages = []
        mri_path = self.original_mri_path or display_mri
        pet_path = self.original_pet_path or display_pet
        if mri_path is not None:
            messages.append({"role": "user", "content": f"mri_path: {mri_path}"}) #image是以地址的形式进行传输
        if pet_path is not None:
            messages.append({"role": "user", "content": f"pet_path: {pet_path}"})
        if message is not None:
            messages.append({"role": "user", "content": message})

        try:
            for event in self.agent.workflow.stream(
                {"messages": messages}, {"configurable": {"thread_id": self.current_thread_id}}
            ):
                if isinstance(event, dict):
                    if "process" in event:
                        content = event["process"]["messages"][-1].content   #提取最新一条消息内容
                        if content:
                            content = re.sub(r"temp/[^\s]*", "", content) #使用正则表达式 re.sub 移除内容中的临时路径（如 temp/xxx）
                            chat_history.append(ChatMessage(role="assistant", content=content))
                            yield chat_history, self.display_mri_path, self.display_pet_path, ""

                    elif "execute" in event:
                        for message in event["execute"]["messages"]:
                            tool_name = message.name
                            tool_result = eval(message.content)[0]

                            if tool_result:
                                metadata = {"title": f"🖼️ Image from tool: {tool_name}"}
                                formatted_result = " ".join(
                                    line.strip() for line in str(tool_result).splitlines()
                                ).strip()
                                metadata["description"] = formatted_result
                                chat_history.append(
                                    ChatMessage(
                                        role="assistant",
                                        content=formatted_result,
                                        metadata=metadata,
                                    )
                                )

                            yield chat_history, self.display_mri_path, self.display_pet_path, ""

        except Exception as e:
            chat_history.append(
                ChatMessage(
                    role="assistant", content=f"❌ Error: {str(e)}", metadata={"title": "Error"}
                )
            )
            yield chat_history, self.display_mri_path, self.display_pet_path, ""


def create_demo(agent, tools_dict):
    """
    Create a Gradio demo interface for the medical AI agent.

    Args:
        agent: The medical AI agent to handle requests
        tools_dict (dict): Dictionary of available tools for image processing

    Returns:
        gr.Blocks: Gradio Blocks interface
    """
    interface = ChatInterface(agent, tools_dict)

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        with gr.Column():
            gr.Markdown(
                """
            # 🏥 ADAgent
            ADAgent for diagnosis of Alzheimer's disease with multi-modal input.
            """
            )

            with gr.Row(): #它将所有子组件水平排列
                with gr.Column(scale=3): #它将所有子组件垂直排列
                    chatbot = gr.Chatbot(
                        [],
                        height=800,
                        container=True,
                        show_label=True,
                        elem_classes="chat-box",
                        type="messages",
                        label="Agent",
                        avatar_images=( #用户消息无头像，AI消息使用 adagent_logo.jpg
                            None,
                            "assets/adagent_logo.jpg",
                        ),
                    )
                    with gr.Row():
                        with gr.Column(scale=3):
                            txt = gr.Textbox( #用户输入文本框，占位符提示输入问题
                                show_label=False,
                                placeholder="Please input your command...",
                                container=False,
                            )

                with gr.Column(scale=3):
                    with gr.Row():
                        mri_display = gr.Image( #显示上传的X光片或者DICOM图像
                            label="MRI", type="filepath", height=700, container=True
                        )
                        pet_display = gr.Image( #显示上传的X光片或者DICOM图像
                            label="PET", type="filepath", height=700, container=True
                        )
                    with gr.Row():
                        upload_mri_button = gr.UploadButton(
                            "📎 Upload MRI",
                            file_types=["file"],
                        )
                        upload_pet_button = gr.UploadButton(
                            "📄 Upload PET",
                            file_types=["file"],
                        )
                    with gr.Row():
                        clear_btn = gr.Button("Clear Chat")
                        new_thread_btn = gr.Button("New Thread") #创建新会话线程

        # Event handlers
        def clear_chat():
            interface.original_mri_path = None
            interface.original_pet_path = None
            interface.display_mri_path = None
            interface.display_pet_path = None
            return [], None, None

        def new_thread():
            interface.current_thread_id = str(time.time())
            return [], interface.display_mri_path, interface.display_pet_path

        def handle_mri_upload(file):
            print(file)
            print(file.name)
            return interface.handle_mri_upload(file.name)
        
        def handle_pet_upload(file):
            return interface.handle_pet_upload(file.name)

        #事件绑定
        chat_msg = txt.submit(
            interface.add_message, inputs=[txt, mri_display, pet_display, chatbot], outputs=[chatbot, txt]
        )
        #用户按下回车时触发 interface.add_message，更新聊天记录。

        bot_msg = chat_msg.then(
            interface.process_message,
            inputs=[txt, mri_display, pet_display, chatbot],
            outputs=[chatbot, mri_display, pet_display, txt],
        )
        bot_msg.then(lambda: gr.Textbox(interactive=True), None, [txt])

        upload_mri_button.upload(handle_mri_upload, inputs=upload_mri_button, outputs=mri_display)

        upload_pet_button.upload(handle_pet_upload, inputs=upload_pet_button, outputs=pet_display)

        clear_btn.click(clear_chat, outputs=[chatbot, mri_display, pet_display])
        new_thread_btn.click(new_thread, outputs=[chatbot, mri_display, pet_display])

    return demo
