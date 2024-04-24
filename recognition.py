import argparse
import functools
import tkinter as tk
from tkinter import simpledialog
import numpy as np
from mvector.predict import MVectorPredictor
from mvector.utils.record import RecordAudio
from mvector.utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',          str,    'configs/ecapa_tdnn.yml',   '配置文件')
add_arg('use_gpu',          bool,   False,                       '是否使用GPU预测')
add_arg('audio_db_path',    str,    'audio_db/',                '音频库的路径')
add_arg('model_path',       str,    'models/EcapaTdnn_Fbank/best_model/', '导出的预测模型文件路径')
args = parser.parse_args()
print_arguments(args=args)


class VoiceRecognitionGUI:
    def __init__(self, master):
        master.title("声纹识别")
        master.geometry('380x200')
        # 识别使用时间，单位秒
        self.infer_time = 2
        # 录音采样率
        self.samplerate = 16000
        # 录音块大小
        self.numframes = 1024
        # 模型输入长度
        self.infer_len = int(self.samplerate * self.infer_time / self.numframes)
        self.recognizing = False
        self.record_data = []
        self.record_audio = RecordAudio()
        # 录音长度标签和输入框
        self.record_seconds_label = tk.Label(master, text="录音长度(s):")
        self.record_seconds_label.place(x=3, y=3)
        self.record_seconds = tk.StringVar(value='4')
        self.record_seconds_entry = tk.Entry(master, width=30, textvariable=self.record_seconds)
        self.record_seconds_entry.place(x=90, y=3)
        # 判断是否为同一个人的阈值标签和输入框
        self.threshold_label = tk.Label(master, text="判断阈值:")
        self.threshold_label.place(x=4, y=40)
        self.threshold = tk.StringVar(value='0.6')
        self.threshold_entry = tk.Entry(master, width=30, textvariable=self.threshold)
        self.threshold_entry.place(x=90, y=40)

        # 选择功能标签和按钮
        self.label = tk.Label(master, text="请选择功能：")
        self.label.place(x=12, y=90)
        self.register_button = tk.Button(master, text="注册音频", command=self.register)
        self.register_button.place(x=90, y=90)
        self.recognize_button = tk.Button(master, text="执行声纹识别", command=self.recognize)
        self.recognize_button.place(x=150, y=90)
        self.remove_user_button = tk.Button(master, text="删除用户", command=self.remove_user)
        self.remove_user_button.place(x=240, y=90)
        self.result_label = tk.Label(master, text="结果显示", font=('Arial', 16))
        self.result_label.place(relx=0.5, y=160, anchor=tk.CENTER)
        # 识别器
        self.predictor = MVectorPredictor(configs=args.configs,
                                          threshold=float(self.threshold.get()),
                                          audio_db_path=args.audio_db_path,
                                          model_path=args.model_path,
                                          use_gpu=args.use_gpu)

    # 注册
    def register(self):
        record_seconds = int(self.record_seconds.get())
        # 开始录音
        self.result_label.config(text="正在录音...")
        audio_data = self.record_audio.record(record_seconds=record_seconds)
        self.result_label.config(text="录音结束")
        name = simpledialog.askstring(title="注册", prompt="请输入注册名")
        if name is not None and name != '':
            self.predictor.register(user_name=name, audio_data=audio_data, sample_rate=self.record_audio.sample_rate)
            self.result_label.config(text="注册成功")

    # 声纹识别
    def recognize(self):
        threshold = float(self.threshold.get())
        record_seconds = int(self.record_seconds.get())
        # 开始录音
        self.result_label.config(text="开始录音...")
        audio_data = self.record_audio.record(record_seconds=record_seconds)
        self.result_label.config(text="录音结束")
        name, score = self.predictor.recognition(audio_data, threshold, sample_rate=self.record_audio.sample_rate)
        if name:
            self.result_label.config(text=f"说话人为：{name}，得分：{score}")
        else:
            self.result_label.config(text="没有识别到说话人")
# 删除用户
    def remove_user(self):
        name = simpledialog.askstring(title="删除用户", prompt="请输入删除用户名称")
        if name is not None and name != '':
            result = self.predictor.remove_user(user_name=name)
            if result:
                self.result_label.config(text="删除成功")
            else:
                self.result_label.config(text="删除失败")


    # 识别
    def recognize_real(self):
        threshold = float(self.threshold.get())  # 获取阈值
        while self.recognizing:     # 当正在识别时执行循环
            if len(self.record_data) < self.infer_len: continue  # 如果录音数据长度小于指定长度，则跳过当前循环
            # 截取最新的音频数据
            seg_data = self.record_data[-self.infer_len:]
            audio_data = np.concatenate(seg_data)  # 将截取的音频数据拼接成一个数组
            # 删除旧的音频数据
            del self.record_data[:len(self.record_data) - self.infer_len]
            name, score = self.predictor.recognition(audio_data, threshold,
                                                     sample_rate=self.record_audio.sample_rate)  # 调用预测器进行识别

            if name:
                self.result_label.config(text=f"【{name}】正在说话")  # 如果识别到人名，则更新结果标签显示正在说话的人名
            else:
                self.result_label.config(text="")  # 如果没有识别到人名，则清空结果标签


if __name__ == '__main__':
    root = tk.Tk()
    gui = VoiceRecognitionGUI(root)
    root.mainloop()
