from noise import snoise2, pnoise2
import numpy as np
from PIL import Image

import random
import math


# Settings that control the noise
shape = (1024,1024)
scale = 100.0
octaves = 6
persistence = 0.5
lacunarity = 2.0

print("Calculating 2D Noise...")

world_squished = np.zeros(shape)
for i in range(shape[0]):
    for j in range(shape[1]):
        world_squished[i][j] = pnoise2(i / scale,
                                j / scale,
                                octaves,
                                persistence=persistence,
                                lacunarity=lacunarity,
                                base=100)


print("Creating Spherical Mask")

center_x, center_y = shape[1] // 2, shape[0] // 2
circle_grad = np.zeros_like(world_squished)

for y in range(world_squished.shape[0]):
    for x in range(world_squished.shape[1]):
        distx = abs(x - center_x)
        disty = abs(y - center_y)
        dist = math.sqrt(distx*distx + disty*disty)
        circle_grad[y][x] = dist

# get it between -1 and 1
max_grad = np.max(circle_grad)
circle_grad = circle_grad / max_grad
circle_grad -= 0.5
circle_grad *= 2.0
circle_grad = -circle_grad

# shrink gradient
for y in range(world_squished.shape[0]):
    for x in range(world_squished.shape[1]):
        if circle_grad[y][x] > 0:
            circle_grad[y][x] *= 20

# get it between 0 and 1
circle_grad = (circle_grad - np.min(circle_grad))/np.ptp(circle_grad)


# Adding in world noise
print("Creating World Noise")
world_noise = np.zeros_like(world_squished)

for i in range(shape[0]):
    for j in range(shape[1]):
        world_noise[i][j] = (world_squished[i][j] * circle_grad[i][j])
        if world_noise[i][j] > 0:
            world_noise[i][j] *= 20

# get it between 0 and 1
world_noise = (world_noise - np.min(world_noise))/np.ptp(world_noise)


lightblue = [0,191,255]
blue = [65,105,225]
green = [34,139,34]
darkgreen = [0,100,0]
sandy = [210,180,140]
beach = [238, 214, 175]
snow = [255, 250, 250]
mountain = [139, 137, 137]

threshold = 0.2


def add_color2(world):
    print("Adding Colour")

    color_world = np.zeros(world.shape+(3,))
    for i in range(shape[0]):
        for j in range(shape[1]):
            if world[i][j] < threshold + 0.05:
                color_world[i][j] = blue
            elif world[i][j] < threshold + 0.055:
                color_world[i][j] = sandy
            elif world[i][j] < threshold + 0.1:
                color_world[i][j] = beach
            elif world[i][j] < threshold + 0.25:
                color_world[i][j] = green
            elif world[i][j] < threshold + 0.6:
                color_world[i][j] = darkgreen
            elif world[i][j] < threshold + 0.7:
                color_world[i][j] = mountain
            elif world[i][j] < threshold + 1.0:
                color_world[i][j] = snow

    return color_world

island_world_grad = add_color2(world_noise)

img = Image.fromarray((island_world_grad).astype(np.uint8))
img.save("map.png", "PNG")
img.show()
