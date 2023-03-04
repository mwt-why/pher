import json
import os
import base64
import cv2 as cv
import numpy as np
from flask import Flask
from flask import request
from flask import jsonify
from paddleocr import PaddleOCR
from PIL import Image
from io import BytesIO

app = Flask(__name__)
paddle_ocr = PaddleOCR(use_angle_cls=True, lang="ch")  # need to run only once to download and load model into memory


@app.route('/home', methods=['GET', 'POST'])
def home():
    return '<h1>this is a photo handle server</h1>'


@app.route("/ocr", methods=["POST"])
def ocr():
    data = json.loads(request.data)
    base64_img = data['data']
    binary_img = base64.b64decode(base64_img)

    ocr_results = paddle_ocr.ocr(binary_img, cls=True)
    response = []
    for orc_result in ocr_results:
        label = orc_result[1][0]
        box_x1 = orc_result[0][0][0]
        box_y1 = orc_result[0][0][1]
        box_x2 = orc_result[0][2][0]
        box_y2 = orc_result[0][2][1]
        response.append({'label': label, 'box': [box_x1, box_y1, box_x2, box_y2]})
    return jsonify(response)


@app.route("/stitching", methods=["POST"])
def stitching():
    data = json.loads(request.data)
    base64_img = data['data']
    picture_set = data['set']
    # 打开准备添加图片
    binary_img = base64.b64decode(base64_img)
    img = Image.open(BytesIO(binary_img))
    img_width, img_height = img.size

    # 打开之前的图片
    filename = picture_set + ".png"
    filename = os.path.join("dataset", filename)
    response = {}
    if os.path.exists(filename):
        before_img = Image.open(filename)
        before_img_width, before_img_height = before_img.size
        target = Image.new(before_img.mode, (before_img_width, img_height + before_img_height))
        target.paste(before_img, box=(0, 0))
        target.paste(img, box=(0, before_img_height))
        target.save(filename)
        response['top'] = before_img_height
        response['bottom'] = before_img_height + img_height
    else:
        target = Image.new(img.mode, (img_width, img_height))
        target.paste(img)
        target.save(filename)
        response['top'] = 0
        response['bottom'] = img_height

    return jsonify(response)


@app.route("/detect", methods=["POST"])
def detect():
    data = json.loads(request.data)
    picture_set = data['set']
    base64_img = data['data']
    # 转byte
    binary_img = base64.b64decode(base64_img)
    img = np.frombuffer(binary_img, np.uint8)
    template = cv.imdecode(img, 0)
    w, h = template.shape[::-1]

    filename = picture_set + ".png"
    filename = os.path.join("dataset", filename)

    img2 = cv.imread(filename, 0)
    methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
               'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
    for meth in methods:
        img = img2.copy()
        method = eval(meth)
        # Apply template Matching
        res = cv.matchTemplate(img, template, method)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        print(top_left[1], bottom_right[1])
        return jsonify({"top": top_left[1], "bottom": bottom_right[1]})


def img_pixel():
    data = json.loads(request.data)
    base64_img = data['data']
    binary_img = base64.b64decode(base64_img)
    memory_stream = BytesIO(binary_img)
    img = Image.open(memory_stream)
    img = img.convert("RGB")

    position = data['position']
    r, g, b = img.getpixel((position['x'], position['y']))
    print(f"Color at ({position['x']}, {position['y']}): R={r}, G={g}, B={b}")
    return jsonify({"R": r, "G": g, "B": b})


def rgb_detect():
    data = json.loads(request.data)
    base64_img = data['data']
    binary_img = base64.b64decode(base64_img)
    memory_stream = BytesIO(binary_img)
    img = Image.open(memory_stream)
    img = img.convert("RGB")

    rgb = data['RGB']
    position_list = []
    for x in range(img.width):
        for y in range(img.height):
            r, g, b = img.getpixel((x, y))
            if r == rgb['R'] and g == rgb['G'] and b == rgb['B']:
                print(f"Found at ({x}, {y})")
                position_list.append({"x": x, "y": y})
    return jsonify(position_list)


if __name__ == '__main__':
    app.run(host="192.168.10.3", port=9090)
