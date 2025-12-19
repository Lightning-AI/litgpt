import torch
from litgpt.tokenizer import Tokenizer

tokenizer = Tokenizer("checkpoints/soketlabs/pragna-1b")

def encode_text(text):
  text = """
  Visualator in practice

  Visualator, our iPhone and iPad app, is available to download now and is a great little tool for creating abstract compositions. With two design platforms, Triangulate and Gradulate, Visualator enables you to design on the fly and then save your art to your device's internal memory.

  Although Visualator is fun to use, it needn't be just a fancy play-thing. In this tutorial I'll demonstrate how I took my Visualator compositions and combined them into my own original piece of graphic art using Illustrator and Photoshop.

  The possibilities are endless, and this tutorial should act as a starting point for your own experiments. We'd love to see your Visualator creations, whether you've created them in-app or exported the imagery. You can upload your creations to the Computer Arts Visualator Flickr group.

  01 To begin with you'll need an iPhone, iPod Touch or iPad with a Wi-Fi or 3G connection. Head to the App Store and download Visualator. The app is free to install, and should take no longer than a few minutes to download.

  02 Once installation is complete, launch Visualator and choose your drawing mode. For the purposes of this tutorial, I'm using Triangulate, as I want to create an angular, geometric piece. Choose a background colour and a range of colours to draw with, and experiment with toggling the Pattern Modes on and off.

  03 Create a number of different, random compositions, connect your device to a desktop or laptop, and import the images. Open them in Photoshop, convert to Grayscale, and up the levels and curves so that the contrast is quite severe. Save all of your black and while images in a folder for use later.

  04 Switch over to Illustrator and set up a new document. Using the Pen tool and the Basic Shapes tools, begin to flesh out a rough composition that we'll later drop our Visualator comps into.

  05 To create the circular parallel bars, simply draw a circle and fill it with a Linear Gradient. Now go to Object> Expand and specify 10 objects. Hold Shift+R to rotate the shape, and then select Divide from the Pathfinder palette and hit Ctrl/Cmd+Shift+G to Ungroup the objects.

  06 Ensure that all your elements are on separate layers, then export your document as a PSD with Write Layers selected. Open it up in Photoshop and turn all the layers off, apart from your parallel lines.

  07 Open up one of your black and white Visualator comps and copy it. Back in your main Photoshop document, use the Magic Wand to select one of the parallel lines, and go to Edit> Paste Into. Repeat this process using different Visualator comps for all the separate bars until you have something that looks roughly like the image above.

  08 Switch your other layers back on, and source some imagery to drop in. I want my piece to have a surreal look so I've found a desert scene from my own iPhoto library. Hit Ctrl/Cmd+I to invert the image, and ramp the contrast right up. Now, using the same technique as before, I paste the desert image into the central circle shape.

  09 Now to add some colour. Select all of your Visualator layers, and click Merge Layers in the Layers palette drop-down menu. In the Layer Styles palette choose Colour Overlay, select Multiply as a Blending Mode, and choose your Overlay colour.

  10 To finish off this simple graphic piece, I've pasted a number of inverted Visualator comps into the background, with Lighten selected as the Blending Mode. Experiment with the size and scale of the comps, and try to create some interplay between the angular elements. I've also drawn some diagonal lines and a diamond shape in Illustrator, and pasted them over, knocking the Opacity of the large triangle back to 75%. There you have it: a very quick and easy way to put Visualator to use in your own work.

  Luke O'Neill
  Deputy art editor of Computer Arts, Luke is a graphic designer and illustrator able to turn his hand to anything from complex layouts to branding projects. He is currently broadening his skill set by working on the design of Computer Arts' next must-have iPad applicatio
  """

  enc = tokenizer.encode(text)

  print(enc.shape)

def decode_text():
  enc = torch.tensor([    1,   529, 29989,  5205, 29989, 29958, 36621,  3492,   526,   263,
         1424,   347,   299,   368, 13563,  7451,  1058,  2337,  4076,  8444,
        29892,   316,   941,  2356, 29892,   322,   772,   492,   371,  6089,
        21106, 29879, 29958, 36621, 29966, 29989,  1792, 29989, 29958, 36621,
        36904, 38788, 39860, 32514, 34206, 36702, 29973,   829, 29879, 29958,
        36621, 29966, 29989,   465, 22137, 29989, 29958, 36621, 34648, 38788,
        39623, 36350, 40013, 29871, 35548, 34860, 38423, 32644, 39352, 38306,
        36702, 37626, 38911, 34608, 36410, 37023, 33441, 34330, 30580, 29892,
        33590, 34919, 39655, 40025, 35638, 34835, 33707, 39964, 37147, 33983,
        32328, 37666, 33564, 38851, 36874, 33333, 39170, 39972, 30444, 39101,
        39621, 37540, 36241, 33679, 39188, 39762, 36893, 29871, 39352, 39140,
        39697, 31329, 36272, 33719, 39619, 31377, 30269, 36943, 38313, 31732,
        37306, 31330, 38869, 37554, 37005, 37603, 35293, 38951, 36389, 39211,
        38207, 29892, 34243, 36762, 32782, 38241, 31009, 30444, 34542, 35109,
        38160, 32715, 32890, 39311, 32715, 37076, 35894, 37929, 32856, 30603,
        31380, 37104, 38207, 31776, 34648, 35183, 32197, 32618, 37501, 35525,
        36733, 34542, 35109, 34627, 33053, 34847, 38754, 35172, 37626, 34608,
        37920, 35790, 36241, 36874, 31732, 35636, 37389, 39972, 30444, 34498,
        31132, 33286, 37782, 39752, 34218, 35096, 32852, 39884, 38831, 35760,
        38990, 35612, 39352, 30424, 32105, 38911, 34608, 34918, 35155, 34674,
        29892, 37432, 39450, 36096, 33003, 37023, 33441, 34330, 39374, 33733,
        37313, 34919, 34926, 32720, 34008, 36874, 37583, 35638, 39402, 34918,
        34691, 34008, 34259, 37147, 34542, 35109, 38160, 32715, 32890, 39311,
        32715, 37076, 37513, 35750, 33908, 31776, 34648, 38965, 38549, 31413,
        34528, 39090, 35144, 32569, 34850, 38532, 35109, 33151, 31927, 30444,
        36868, 36716, 32890, 39619, 31377, 30269, 32987, 31667, 39656, 33055,
        32575, 37531, 36173, 39188, 38390, 36348, 38086, 35612, 39352, 39140,
        33604, 33205, 38911, 36110, 38592, 33566, 33405, 37147, 33983, 39867,
        33863, 38838, 35567, 32576, 37072, 37041, 37072, 37334, 36874, 36716,
        32890, 39619, 31377, 30269, 32987, 37837, 36096, 34218, 36200, 37920,
        33564, 37072, 29899, 39252, 38390, 36997, 30621, 36313, 30580, 34494,
        34918, 39735, 35810, 31776, 32595, 33499, 36313, 34330, 30799, 31377,
        36620, 36904, 38788, 33286, 38925, 30489, 38593, 36874, 30489, 33310,
        33596, 35640, 37540, 33181, 35109, 36117, 36350, 40013, 38736, 32634,
        37074, 35862, 33161, 33096, 39698, 39182, 35714, 36893, 37048, 37013,
        32357, 37644, 29892, 32735, 34294, 34933, 37396, 31667, 37675, 29892,
        39637, 34608, 32757, 38160, 39656, 36241, 36874, 36468, 34556, 39129,
        37564, 33031, 35714, 33978, 36430, 33773, 33096, 36618, 35177, 33064,
        30702, 30444, 33718, 35648, 36389, 36904, 38788, 39623, 39624, 36173,
        35990, 35739, 33397, 36893,     2])
  dec = tokenizer.decode(enc)

  print(dec)  

if __name__ == "__main__":
  # encode_text()
  decode_text()