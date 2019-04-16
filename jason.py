import base64
import json


#with open("teresa.jpg", "rb") as imageFile:
#    stri = base64.b64encode(imageFile.read())
#    jason_data = json.dumps(stri)
#file = open("jspy.json","w")
#file.write(str(stri))
#file.close()

#fh = open("imageToSave.jpg", "wb")
#fh.write(stri.decode('base64'))
#fh.close()


data = {}
with open("teresa.jpg", "rb") as imageFile:
    stri = base64.b64encode(imageFile.read())
    yourname=raw_input("What's your name? ")
    data["info"] = []
    data["info"].append({
        "name": ("%s"%yourname),
        "image": ("%s"%stri)
    })
    #data["name"] = ["%s"%yourname]
    #data["image"]=["%s" %stri]
import mmh3
hashed = mmh3.hash("%s"%yourname)
rand_int_32 = hashed & 0xffffffff
id = str(hashed)
with open('%s.json'%id, 'w') as outfile:
    json.dump(data, outfile)
   