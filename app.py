from flask import Flask,request, send_from_directory, render_template
import os
from flask_restful import Resource, Api
from random import randint
import random
import shutil
from shutil import copyfile
from pathlib2 import Path
import mmh3
import requests
import json

app = Flask(__name__)
api = Api(app)

#@app.route("/")
class route(Resource):
    def get(self):
        return render_template("home.html")


#@app.route("/upload", methods=["POST"])
class upload(Resource):
    def post(self):
        number = randint(0,9999)
        file = request.files["file"]
        file.save("static/%s.json"%number)
        with open("static/%s.json"%number,"r") as json_file:
            data = json.load(json_file)
            for p in data["info"]:
                name = p["name"]
                image = p["image"]
        hashed = mmh3.hash("%s"%name)
        rand_int_32 = hashed & 0xffffffff
        id = str(hashed)  #random number from your name
        fh = open("%s.jpg"%id,"wb")
        fh.write(str(image).decode("base64"))
        fh.close()
        try:
            if not os.path.exists("%s"%id):
                os.makedirs("\src\images\%s"%(id), 0755)
        except OSError:
            print("ERROR: Creating Directory %s" %id)

        my_file = Path("\src\images\%s"%id)
        fileno = randint(0,9999)
        if my_file.is_dir()==True:
            fileno = randint(0,9999)
            copyfile("\%s.jpg"%id, "\src\images\%s\%s.jpg"%(id,fileno))
        if my_file.is_dir()==False:
            copyfile("\%s.jpg"%id, "\src\images\%s\%s.jpg"%(id,fileno))
        os.remove("static/%s.json"%number)
        return("Your ID is: %s\nDone! .json successfully extracted\n now waiting training..."%id) # to train do CTRL+C on terminal

api.add_resource(route, "/")
api.add_resource(upload, "/upload")
if __name__ == '__main__':
    app.run(host="0.0.0.0")
import train
    