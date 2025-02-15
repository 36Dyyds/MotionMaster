import datetime
import os

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageFont, ImageDraw
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from keras.models import Sequential
from moviepy.editor import VideoFileClip


def paint_chinese_opencv(im, chinese, pos, color):
    img_PIL = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    font = ImageFont.truetype('font/simsun.ttc', 25, encoding="utf-8")
    fillColor = color
    position = pos
    draw = ImageDraw.Draw(img_PIL)
    draw.text(position, chinese, fillColor, font)

    img = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
    return img


def get_angle(v1, v2):
    angle = np.dot(v1, v2) / (np.sqrt(np.sum(v1 * v1)) * np.sqrt(np.sum(v2 * v2)))
    angle = np.arccos(angle) / 3.14 * 180

    cross = v2[0] * v1[1] - v2[1] * v1[0]
    if cross < 0:
        angle = -angle
    return angle


def get_pos(keypoints):
    # 计算右臂与水平方向的夹角
    keypoints = np.array(keypoints)
    v1 = keypoints[5] - keypoints[6]
    v2 = keypoints[8] - keypoints[6]
    angle_right_arm = get_angle(v1, v2)

    # 计算左臂与水平方向的夹角
    v1 = keypoints[7] - keypoints[5]
    v2 = keypoints[6] - keypoints[5]
    angle_left_arm = get_angle(v1, v2)

    # 计算左肘的夹角
    v1 = keypoints[6] - keypoints[8]
    v2 = keypoints[10] - keypoints[8]
    angle_right_elbow = get_angle(v1, v2)

    # 计算右肘的夹角
    v1 = keypoints[5] - keypoints[7]
    v2 = keypoints[9] - keypoints[7]
    angle_left_elbow = get_angle(v1, v2)

    str_pos = "正常"
    # 设计动作识别规则
    if angle_right_arm < 0 and angle_left_arm < 0:
        str_pos = "正常"
        if abs(angle_left_elbow) < 120 and abs(angle_right_elbow) < 120:
            str_pos = "叉腰"
    elif angle_right_arm < 0 and angle_left_arm > 0:
        str_pos = "抬左手"
    elif angle_right_arm > 0 and angle_left_arm < 0:
        str_pos = "抬右手"
    elif angle_right_arm > 0 and angle_left_arm > 0:
        str_pos = "抬双手"
        if abs(angle_left_elbow) < 120 and abs(angle_right_elbow) < 120:
            str_pos = "三角形"
    return str_pos


# 启动入口
if __name__ == '__main__':

    # 视频文件名称列表
    video_paths = [
        'test',
        '3e31e104fc3b17758ea5ca4b311bf1d5',
        '624cf417851ea7b03518eb9df38d68cb'
    ]

    video_path = video_paths[0]

    # 指定要创建的文件夹路径
    folder_path = 'output/' + video_path

    # 使用os.makedirs()创建文件夹，如果路径中的父文件夹不存在，也会创建它们
    os.makedirs(folder_path, exist_ok=True)

    print(f'文件夹 "{folder_path}" 创建成功')

    cap = cv2.VideoCapture('video/' + video_path + '.mp4', cv2.CAP_FFMPEG)

    # 检查视频文件是否成功打开
    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()

    # 获取视频的宽度和高度
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 自动获取视频的帧率
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 MP4V 编解码器
    # out = cv2.VideoWriter('output/' + video_path + '/' + video_path + '.mp4', fourcc, fps, (width, height))

    # 无音频视频
    output_video_path_no_audio = 'output/' + video_path + '/' + video_path + '_no_audio.mp4'
    out = cv2.VideoWriter(output_video_path_no_audio, fourcc, fps, (width, height))

    # 声明空的 DataFrame，以及相关的列名
    columns = ['Timestamp', 'FrameNumber', 'Expression',
               'LeftShoulder_X', 'LeftShoulder_Y', 'RightShoulder_X', 'RightShoulder_Y',
               'LeftElbow_X', 'LeftElbow_Y', 'RightElbow_X', 'RightElbow_Y',
               'LeftHand_X', 'LeftHand_Y', 'RightHand_X', 'RightHand_Y',
               'LeftLeg_X', 'LeftLeg_Y', 'RightLeg_X', 'RightLeg_Y']
    data_list = []
    data_frame = pd.DataFrame(columns=columns)

    file_model = "model/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite"
    interpreter = tf.lite.Interpreter(model_path=file_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    print('input_details\n', input_details)
    output_details = interpreter.get_output_details()
    print('output_details', output_details)

    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    frame_rate_calc = 1
    freq = cv2.getTickFrequency()
    frame_number = 0

    # ------------------------------------
    class_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    img_width, img_height = 48, 48

    # 构建卷积神经网络模型
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(img_width, img_height, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7))
    model.add(Activation('softmax'))

    # 载入预训练的权重
    model.load_weights('model/first_try.h5')

    # 加载人脸级联分类器
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # 打开视频文件
    # cap = cv2.VideoCapture('test.mp4')
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    expression_duration = {}
    # ------------------------------------

    while True:
        t1 = cv2.getTickCount()

        success, img = cap.read()
        if not success:
            break

        # ---------------------------------------------------
        # 获取当前帧时间
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # 毫秒转秒
        # 显示帧率和当前帧参数
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        # 对帧进行预处理
        frame_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 在视频帧上检测人脸
        faces = face_cascade.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        expression = 'neutral'

        # 遍历检测到的人脸
        for (x, y, w, h) in faces:
            # 在人脸周围画一个蓝色方框
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # 提取人脸区域并进行预测
            face_roi = frame_gray[y:y + h, x:x + w]
            face_roi = cv2.resize(face_roi, (img_width, img_height))
            face_roi = np.expand_dims(face_roi, axis=0)
            face_roi = np.expand_dims(face_roi, axis=-1)
            face_roi = face_roi.astype('float32') / 255.0  # 归一化

            # 使用模型进行表情预测
            prediction = model.predict(face_roi)
            predicted_class = np.argmax(prediction, axis=1)[0]

            # 输出预测表情
            expression = class_labels[predicted_class]

            # 在方框旁边显示预测的表情
            cv2.putText(img, f"Expression: {expression}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 1, cv2.LINE_AA)

        # 更新表情持续时间
        expression_duration[expression] = expression_duration.get(expression, 0) + 1

        # ---------------------------------------------------

        frame_number += 1

        imH, imW, _ = np.shape(img)

        # img = cv2.resize(img, (int(imW * 0.5), int(imH * 0.5)))
        # imH, imW, _ = np.shape(img)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (width, height))
        input_data = np.expand_dims(img_resized, axis=0)
        input_data = (np.float32(input_data) - 128.0) / 128.0
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        hotmaps = interpreter.get_tensor(output_details[0]['index'])[0]
        offsets = interpreter.get_tensor(output_details[1]['index'])[0]
        h_output, w_output, n_KeyPoints = np.shape(hotmaps)
        keypoints = []
        score = 0

        str_pos = '正常'

        for i in range(n_KeyPoints):
            # 遍历每一张hotmap
            hotmap = hotmaps[:, :, i]

            # 获取最大值 和最大值的位置
            max_index = np.where(hotmap == np.max(hotmap))
            max_val = np.max(hotmap)

            # 获取y，x偏移量 前n_KeyPoints张图是y的偏移 后n_KeyPoints张图是x的偏移
            offset_y = offsets[max_index[0], max_index[1], i]
            offset_x = offsets[max_index[0], max_index[1], i + n_KeyPoints]

            # 计算在posnet输入图像中具体的坐标
            pos_y = max_index[0] / (h_output - 1) * height + offset_y
            pos_x = max_index[1] / (w_output - 1) * width + offset_x

            # 计算在源图像中的坐标
            pos_y = pos_y / (height - 1) * imH
            pos_x = pos_x / (width - 1) * imW

            # 取整获得keypoints的位置
            keypoints.append([int(round(pos_x[0])), int(round(pos_y[0]))])

            # 利用sigmoid函数计算置每一个点的置信度
            score = score + 1.0 / (1.0 + np.exp(-max_val))

        # 取平均得到最终的置信度
        score = score / n_KeyPoints

        if score > 0.5:
            # 标记关键点
            for point in keypoints:
                cv2.circle(img, (point[0], point[1]), 5, (255, 255, 0), 5)

            # 画关节连接线
            # 左臂
            cv2.polylines(img, [np.array([keypoints[5], keypoints[7], keypoints[9]])], False, (0, 255, 0), 3)
            # # 右臂
            cv2.polylines(img, [np.array([keypoints[6], keypoints[8], keypoints[10]])], False, (0, 0, 255), 3)
            # # 左腿
            cv2.polylines(img, [np.array([keypoints[11], keypoints[13], keypoints[15]])], False, (0, 255, 0), 3)
            # # 右腿
            cv2.polylines(img, [np.array([keypoints[12], keypoints[14], keypoints[16]])], False, (0, 255, 255), 3)
            # 身体部分
            cv2.polylines(img, [np.array([keypoints[5], keypoints[6], keypoints[12], keypoints[11], keypoints[5]])], False,
                          (255, 255, 0), 3)

            # 计算位置角
            str_pos = get_pos(keypoints)

        img = paint_chinese_opencv(img, str_pos, (0, 5), (255, 0, 0))

        # 在视频右上角显示信息
        time_text = f"Time: {str(datetime.timedelta(seconds=current_time))}"
        frame_text = f"Frame: {current_frame}/{int(total_frames)}"
        print(frame_text)
        expression_text = f"Expression: {expression}"
        left_shoulder = f"Left Shoulder: {keypoints[5]}"
        right_shoulder = f"Right Shoulder: {keypoints[6]}"
        left_elbow = f"Left Elbow: {keypoints[7]}"
        right_elbow = f"Right Elbow: {keypoints[8]}"
        left_hand = f"Left Hand: {keypoints[9]}"
        right_hand = f"Right Hand: {keypoints[10]}"
        left_leg = f"Left Leg: {keypoints[11]}"
        right_leg = f"Right Leg: {keypoints[12]}"

        # 生成黑色背景
        text_bg = np.zeros((230, 280, 3), dtype=np.uint8)

        # 将文本放置在黑色背景上
        cv2.putText(text_bg, time_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(text_bg, frame_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(text_bg, expression_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(text_bg, left_shoulder, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(text_bg, right_shoulder, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(text_bg, left_elbow, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(text_bg, right_elbow, (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(text_bg, left_hand, (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(text_bg, right_hand, (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(text_bg, left_leg, (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(text_bg, right_leg, (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # 将黑色背景叠加到视频帧上
        img[:230, -280:] = text_bg

        # 将帧写入视频文件
        out.write(img)

        # 显示当前帧
        # cv2.imshow('Frame', img)

        t2 = cv2.getTickCount()
        time1 = (t2 - t1) / freq
        frame_rate_calc = 1 / time1

        # 将相关数据添加到列表中
        data_list.append({
            'Timestamp': current_time,
            'FrameNumber': frame_number,
            'Expression': expression,  # 表情识别的结果，将其替换为实际的表情数据
            # Left肩
            'LeftShoulder_X': keypoints[5][0],
            'LeftShoulder_Y': keypoints[5][1],
            # Left肘
            'LeftElbow_X': keypoints[7][0],
            'LeftElbow_Y': keypoints[7][1],
            # Left手
            'LeftHand_X': keypoints[9][0],
            'LeftHand_Y': keypoints[9][1],
            # Left leg
            'LeftLeg_X': keypoints[11][0],
            'LeftLeg_Y': keypoints[11][1],

            # Right肩
            'RightShoulder_X': keypoints[6][0],
            'RightShoulder_Y': keypoints[6][1],
            # Right肘
            'RightElbow_X': keypoints[8][0],
            'RightElbow_Y': keypoints[8][1],
            # Right手
            'RightHand_X': keypoints[10][0],
            'RightHand_Y': keypoints[10][1],
            # Right leg
            'RightLeg_X': keypoints[12][0],
            'RightLeg_Y': keypoints[12][1],
        })

        # 按下 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 将列表转换为 DataFrame
    data_frame = pd.DataFrame(data_list, columns=columns)
    # 将 DataFrame 写入 Excel 文件
    excel_output_file = 'output/' + video_path + '/' + video_path + '.xlsx'
    data_frame.to_excel(excel_output_file, index=False)
    print(f"Data has been saved to {excel_output_file}")

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # 使用 moviepy 提取原视频的音频
    input_video_path = 'video/' + video_path + '.mp4'
    input_clip = VideoFileClip(input_video_path)
    audio = input_clip.audio

    # 加载处理后的无音频视频
    output_clip = VideoFileClip(output_video_path_no_audio)

    # 将音频添加到无音频视频中
    final_clip = output_clip.set_audio(audio)

    # 保存最终带有音频的视频
    final_output_video_path = 'output/' + video_path + '/' + video_path + '.mp4'
    final_clip.write_videofile(final_output_video_path, codec='libx264', audio_codec='aac')

    # 删除临时无音频视频文件
    os.remove(output_video_path_no_audio)
