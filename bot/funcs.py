import torch
import torch.nn as nn
from torchvision.models import resnet34, resnet152, ResNet34_Weights, ResNet152_Weights
from torchvision.io import read_image
import torchvision.transforms as transforms

import os
import cv2
import joblib
import asyncio
import PIL.Image as Image
import itertools

from scipy.spatial import distance
import numpy as np
from sklearn.cluster import KMeans

from sklearn.metrics.pairwise import cosine_distances, euclidean_distances, manhattan_distances

def load_model():
    model_master = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)

    for params in model_master.parameters():
        params.requires_grad = False

    masterCategory_head = nn.Sequential(
                                        nn.Linear(512, 3)
                                        )

    model_master.fc = masterCategory_head
    model_master.fc[0].weight.requires_grad = False
    model_master.fc[0].bias.requires_grad = False

    model_master.load_state_dict(torch.load('weigths/baseline_weights.pt', map_location=torch.device('mps')))
    model_master.eval()

    # subType
    model_subtype = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)

    for params in model_subtype.parameters():
        params.requires_grad = False

    masterCategory_head = nn.Linear(512, 1)
                                        
    model_subtype.fc = masterCategory_head
    model_subtype.fc.weight.requires_grad = False
    model_subtype.fc.bias.requires_grad = False

    model_subtype.load_state_dict(torch.load('weigths/subtype_weights_new.pt', map_location=torch.device('mps')))
    model_subtype.eval()

    # articaType
    weights = ResNet152_Weights.IMAGENET1K_V1
    model_type = resnet152(weights=weights)
    model_type.eval()


    # Shoes
    model_shoes = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)

    for params in model_shoes.parameters():
        params.requires_grad = False

    masterCategory_head = nn.Sequential(
                                        nn.Linear(512, 4)
                                        )

    model_shoes.fc = masterCategory_head
    model_shoes.fc[0].weight.requires_grad = False
    model_shoes.fc[0].bias.requires_grad = False

    model_shoes.load_state_dict(torch.load('weigths/shoes_weights_1.pt', map_location=torch.device('mps')))
    model_shoes.eval()

    return model_master, model_subtype, model_type, model_shoes


def prediction(model1, model2, model3, model4, path_for_pic) -> dict:

    result_cloth = {}
    topwear = ['Tshirts', 'Shirts', 'Tops', 'Sweaters']
    bottomwear = ['Jeans', 'Shorts', 'Trousers', 'Track Pants', 'Leggings', 'Capris', 'Skirts']
    shoes = ['Casual Shoes', 'Sports Shoes', 'Heels', 'Formal Shoes']
    seasons = ['offseason', 'summer', 'winter']
    weights = ResNet152_Weights.IMAGENET1K_V1

    pic_for_pred = read_image(path_for_pic) / 255
    pred = model1(pic_for_pred.unsqueeze(0)).argmax().item()

    if pred == 0:
        result_cloth.setdefault('masterCategory', 'apparel')
    elif pred == 1:
        result_cloth.setdefault('masterCategory', 'accs')
    else:
        result_cloth.setdefault('masterCategory', 'shoes')
    
    try:
        if result_cloth['masterCategory'] == 'apparel':
            pred_subcat = model2(pic_for_pred.unsqueeze(0))
            sig = nn.Sigmoid()
            pred_subcat = torch.round(sig(pred_subcat))

            if pred_subcat == 0:
                result_cloth.setdefault('subCat', 'topwear')
            else:
                result_cloth.setdefault('subCat', 'bottomwear')

            with torch.no_grad():
                prediction = model3(pic_for_pred.unsqueeze(0))
            # Get the  predicted class label
            class_id = prediction.argmax().item()
            category_name = weights.meta["categories"][class_id]
            result_cloth.setdefault('articalType', category_name)

        elif result_cloth['masterCategory'] == 'shoes':
            pred_shoes = model4(pic_for_pred.unsqueeze(0)).argmax().item()
            result_cloth.setdefault('articalType', shoes[pred_shoes])

        return result_cloth
    
    except:
        return result_cloth
    

def find_dominant_color(image_path): 
    image = Image.open(image_path)
    image_array = np.array(image)

    pixels = image_array.reshape(-1, 3)

    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(pixels)

    dominant_colors = kmeans.cluster_centers_.round().astype(int)
    dominant_colors_hex = ['#%02x%02x%02x' % (r, g, b) for r, g, b in dominant_colors]

    dists = []
    for row in dominant_colors:
        dists.append(euclidean_distances(np.array([255, 255, 255])[None, :], np.array([row])))

    index = np.array(dists).argmax()
    dominant_color = tuple(dominant_colors[index])
    
    return dominant_color

def find_hexagon_colors(rgb):
    r, g, b = rgb
    hexagon_colors = [(r, (g + 60) % 256, (b + 60) % 256), 
                      ((r + 60) % 256, (g + 120) % 256, b), 
                      ((r + 120) % 256, g, (b + 60) % 256),
                      ((r + 60) % 256, (g + 120) % 256, (b + 60) % 256), 
                      (r, (g + 60) % 256, (b + 120) % 256), 
                      ((r + 60) % 256, g, (b + 120) % 256)]
    return hexagon_colors

def find_triadic_colors(rgb):
    r, g, b = rgb
    triadic_colors = [(r, (g + 120) % 256, (b + 120) % 256), 
                      ((r + 120) % 256, (g + 120) % 256, b)]
    return triadic_colors

def find_analogous_colors(rgb):
    r, g, b = rgb
    analogous_colors = [(r, (g + 30) % 256, (b + 30) % 256), 
                        ((r + 30) % 256, g, (b + 30) % 256), 
                        ((r + 30) % 256, (g + 30) % 256, b), 
                        (r, (g + 30) % 256, b), 
                        (r, g, (b + 30) % 256), 
                        ((r + 30) % 256, g, b)]
    return analogous_colors

def find_matching_clothes(folder, user_id, user_input, color_func=find_hexagon_colors):

    names = os.listdir(folder)
    names.remove('alltypes.json')
    cloth_dict = {}
    for item in names:
        cloth_dict[item.split('.')[0] + '.jpg'] = joblib.load(f'{folder}{item}')
        photo_files = list(cloth_dict.keys())


    main_cloths = {}
    second_cloths = {}
    for key in photo_files:
        if user_input in cloth_dict[key].values():
            if 'subCat' in cloth_dict[key].keys():
                main_cloths[key] = cloth_dict[key]['subCat']
            else:
                main_cloths[key] = cloth_dict[key]['masterCategory']

    for key in photo_files:
        if list(main_cloths.values())[0] not in cloth_dict[key].values():
            if 'subCat' in cloth_dict[key].keys():
                second_cloths[key] = cloth_dict[key]['subCat']
            else:
                second_cloths[key] = cloth_dict[key]['masterCategory']


    unique_second = list(set(second_cloths.values()))

    cat_second_cloths = []
    for item in unique_second:
        a = []
        for key in second_cloths:
            if item == second_cloths[key]:
                a.append(key)
        cat_second_cloths.append(a)

    cat_second_cloths = dict(zip(unique_second, cat_second_cloths))


    main_rgb_color = [find_dominant_color(f'photos/user_{user_id}/{path}') for path in photo_files]
    items_colors = dict(zip(photo_files, main_rgb_color))

    main_cloths_color = {key: items_colors[key] for key in main_cloths.keys()}

    matching_colors = {color: color_func(color) for color in list(main_cloths_color.values())} # {(осн. цвет): [список кортежей с компл-ми цветами]} - можно использовать find_triadic_colors / find_analogous_colors
    matching_colors_list = list(itertools.chain.from_iterable(matching_colors.values())) # все значения из matching_colors переводим в список для дальнейшей итерации по ним


    result_nearest = []
    for key in cat_second_cloths:
        second_cloths_color = {key: items_colors[key] for key in cat_second_cloths[key]}

        distances = {color: {filename: distance.cosine(color, rgb) for filename, rgb in second_cloths_color.items()} for color in matching_colors_list} # вычисляем близость компл-х цветов и всех имеющихся предметов одежды
        nearest_dict = {key: dict(sorted(values.items(), key=lambda x: x[1])[:1]) for key, values in distances.items()} # выбираем наиболее близкую по цвету вещь (кол-во можем менять)
        result_nearest.append(nearest_dict)


    result_cat_dict = dict(zip(unique_second, result_nearest))
   
    num_clothes = len(main_cloths)
    x = 0 # стартовые точки с разницей в window
    y = int(len(nearest_dict) / len(main_cloths))
    window = y
    nearest_dict_spliced = []
    for i in range(num_clothes):
        for key in result_cat_dict:
            m = result_cat_dict[key]
            l = list({k: m[k] for k in list(m)[x:y]}.values())
            keys_list = []
            values_list = []
            for j in l:
                keys_list.append((j.keys()))
                values_list.append(list(j.values()))
            t = values_list.index(min(values_list))
            nearest_dict_spliced.append(list(keys_list[t])[0])
        x += window
        y += window

    result = {}
    w = 0
    z = len(unique_second)
    window = z
    for item in main_cloths:
        result[item] = nearest_dict_spliced[w:z]
        w += window
        z += window

    return result
