import os
from moondream_caption import describe_scene_with_moondream

files = os.listdir("frames")
files.sort()

scene_frames = [os.path.join("frames", file_name) for file_name in files[:3]]

print("Frames utilisées :", scene_frames)

result = describe_scene_with_moondream(scene_frames)

print("Type du résultat :", type(result))
print("Valeur brute :", repr(result))
print("Description de la scène :")
print(result)