import json

from flask import Flask
from flask import request
from flask import jsonify
from paddleocr import PaddleOCR
import base64

app = Flask(__name__)
paddle_ocr = PaddleOCR(use_angle_cls=True, lang="ch")  # need to run only once to download and load model into memory


@app.route('/home', methods=['GET', 'POST'])
def home():
    return '<h1>this is a photo handle server</h1>'


@app.route("/ocr", methods=['POST'])
def ocr():
    data = json.loads(request.data)
    content = data['data']
    img = base64.b64decode(content)
    ocr_results = paddle_ocr.ocr(img, cls=True)
    response = []
    for orc_result in ocr_results:
        label = orc_result[1][0]
        box_x1 = orc_result[0][0][0]
        box_y1 = orc_result[0][0][1]
        box_x2 = orc_result[0][2][0]
        box_y2 = orc_result[0][2][1]
        response.append({'label': label, 'box': [box_x1, box_y1, box_x2, box_y2]})
    return jsonify(response)


if __name__ == '__main__':
    app.run(host="192.168.10.3", port=9090)
