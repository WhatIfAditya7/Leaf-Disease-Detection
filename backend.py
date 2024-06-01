import predict
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from PIL import Image
import io
import os
import base64
import logging

path = "project dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid/Blueberry___healthy/0a0b8f78-df2d-4cfc-becf-cde10fa2766b___RS_HL 5487.JPG"

# creating a Flask app
app = Flask(__name__)
CORS(app)

@app.route('/getDiseasesImage', methods=['POST'])
@cross_origin()
def getDiseasesImage():
    try:
        req = request.json
        imageString = req['imageData']
        print("imageString", imageString)
        image = base64.b64decode(imageString)

        img = Image.open(io.BytesIO(image))
        imageName = "diseasesImage.jpg"
        img.save(imageName)
        print("image saved")
        diseases = predict.prediction(imageName)
        # os.remove(imageName)

        response = {
            "success": 1,
            "message": "Successfully find diseases",
            "diseases": diseases,
        }

        return response

    except Exception as e:
        print(e)
        response = {
            "success": 0,
            "message": "Failed to find diseases",
        }

        return response


# @app.route('/captchaSolver', methods=['post'])
# @cross_origin()
# def captchaSolver():
#     try:
#         req = request.json
#
#         imageString = req['imageData']
#         image = base64.b64decode(imageString)
#
#         img = Image.open(io.BytesIO(image))
#         img.save("fetchedImage.png", "png")
#         imgNumpy = np.asarray(img)
#         # print("image",imgNumpy.shape)
#
#         predictedText = predictor_.predict_image("fetchedImage.png")
#         # print(predictedText)
#
#         response = jsonify(
#             {
#                 "success": 1,
#                 "message": "Successfully solved Captcha Image",
#                 "predictedText": predictedText
#             }
#         )
#         return response
#
#     except:
#
#         response = jsonify(
#             {
#                 "success": 0,
#                 "message": "Failed to solved Captcha Image",
#                 "predictedText": ""
#             }
#         )
#         return response


# driver function
if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True)
