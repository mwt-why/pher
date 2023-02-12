from flask import Flask
from flask import request
from paddleocr import PaddleOCR
import base64

app = Flask(__name__)
ocr = PaddleOCR(use_angle_cls=True, lang="ch")  # need to run only once to download and load model into memory


@app.route('/', methods=['GET', 'POST'])
def home():
    return '<h1>this is a photo handle server</h1>'


@app.route('/ocr')
def ocr():
    content = request.form['content']
    img = base64.b64decode(content)
    ocr_results = ocr.ocr(img, cls=True)
    response = []
    for orc_result in ocr_results:
        response.append(
            {orc_result[1][0], [orc_result[0][0][0], orc_result[0][0][1], orc_result[0][2][0], orc_result[0][2][1]]})
    return response


if __name__ == '__main__':
    app.run(host="192.168.10.3", port=9090)
