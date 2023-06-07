import torch
import torch.nn as nn
from torchvision.models import resnet34, resnet50, ResNet34_Weights, ResNet50_Weights
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
    weights = ResNet50_Weights.IMAGENET1K_V1
    model_type = resnet50(weights=weights)
    model_type.eval()


    # # Topwear
    # model_topwear = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)

    # for params in model_topwear.parameters():
    #     params.requires_grad = False

    # masterCategory_head = nn.Sequential(
    #                                     nn.Linear(512, 4)
    #                                     )

    # model_topwear.fc = masterCategory_head
    # model_topwear.fc[0].weight.requires_grad = False
    # model_topwear.fc[0].bias.requires_grad = False

    # model_topwear.load_state_dict(torch.load('weigths/topwear_weights_1.pt', map_location=torch.device('mps')))
    # model_topwear.eval()

    # # Bottomwear
    # model_bottomwear = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)

    # for params in model_bottomwear.parameters():
    #     params.requires_grad = False

    # masterCategory_head = nn.Sequential(
    #                                     nn.Linear(512, 7)
    #                                     )

    # model_bottomwear.fc = masterCategory_head
    # model_bottomwear.fc[0].weight.requires_grad = False
    # model_bottomwear.fc[0].bias.requires_grad = False

    # model_bottomwear.load_state_dict(torch.load('weigths/bottomwear_weights_1.pt', map_location=torch.device('mps')))
    # model_bottomwear.eval()

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

    # # Season_shoes
    # model_season_shoes = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)

    # for params in model_season_shoes.parameters():
    #     params.requires_grad = False

    # masterCategory_head = nn.Sequential(
    #                                     nn.Linear(512, 3)
    #                                     )

    # model_season_shoes.fc = masterCategory_head
    # model_season_shoes.fc[0].weight.requires_grad = False
    # model_season_shoes.fc[0].bias.requires_grad = False

    # model_season_shoes.load_state_dict(torch.load('weigths/season_footwear_weights_1_5.pt', map_location=torch.device('mps')))
    # model_season_shoes.eval()

    # # Season_wear
    # model_season_wear = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)

    # for params in model_season_wear.parameters():
    #     params.requires_grad = False

    # masterCategory_head = nn.Sequential(
    #                                     nn.Linear(512, 3)
    #                                     )

    # model_season_wear.fc = masterCategory_head
    # model_season_wear.fc[0].weight.requires_grad = False
    # model_season_wear.fc[0].bias.requires_grad = False

    # model_season_wear.load_state_dict(torch.load('weigths/season_apparel_weights_1_5.pt', map_location=torch.device('mps')))
    # model_season_wear.eval()

    return model_master, model_subtype, model_type, model_shoes


def prediction(model1, model2, model3, model4, path_for_pic) -> dict:

    result_cloth = {}
    topwear = ['Tshirts', 'Shirts', 'Tops', 'Sweaters']
    bottomwear = ['Jeans', 'Shorts', 'Trousers', 'Track Pants', 'Leggings', 'Capris', 'Skirts']
    shoes = ['Casual Shoes', 'Sports Shoes', 'Heels', 'Formal Shoes']
    seasons = ['offseason', 'summer', 'winter']
    weights = ResNet50_Weights.IMAGENET1K_V1

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

def find_matching_clothes(folder, cat, user_id, color_func = find_hexagon_colors):
    names = os.listdir(folder)
    my_dict = {}
    for name in names:
        if name.endswith('.json'):
            file_paths = [(name.split('.')[0]+'.jpg') for name in names]
            sub_dict = joblib.load(f'jsons/user_{user_id}/{name}')
            my_dict[f'{name.split(".")[0]}.jpg'] = sub_dict

    # file_paths = [os.path.join(folder, name) for name in names if name.endswith('.jpeg') or name.endswith('.jpg')] # получаем пути к файлам
    main_rgb_color = [find_dominant_color(f'photos/user_{user_id}/{path}') for path in file_paths] # получаем основной цвет для каждого файла
    items_colors = dict(zip(file_paths, main_rgb_color)) # создаем словарь {"путь к файлу": цвет}
    matching_colors = {color: color_func(color) for color in list(items_colors.values())} # {(осн. цвет): [список кортежей с компл-ми цветами]} - можно использовать find_triadic_colors / find_analogous_colors
    matching_colors_list = list(itertools.chain.from_iterable(matching_colors.values())) # все значения из matching_colors переводим в список для дальнейшей итерации по ним
    distances = {color: {filename: distance.cosine(color, rgb) for filename, rgb in items_colors.items()} for color in matching_colors_list} # вычисляем близость компл-х цветов и всех имеющихся предметов одежды
    nearest_dict = {key: dict(sorted(values.items(), key=lambda x: x[1])[:2]) for key, values in distances.items()} # выбираем наиболее близкую по цвету вещь (кол-во можем менять)
    result_dict = {}
    for key1, value1 in items_colors.items(): # создаем словарь {"файл с предметом одежды": [список ссылок на подходящие по цвету вещи]}
        inner_list = []
        inner_value = matching_colors.get(value1)
        if inner_value:
            for inner_item in inner_value:
                inner_dict = nearest_dict.get(inner_item, {})
                inner_list.extend(list(inner_dict.keys()))
        result_dict[key1] = list(set(inner_list))

    # for key, value in result_dict.items(): # эту часть нужно доработать, чтобы вещь не рекомендовалась сама себе. Пока просто удаляем ссылку на вещь из значения, если она совпадает с ключом
    #     if key in value:
    #         value.remove(key)

    for key in file_paths:
        try:
            if my_dict[key]['subCat'] != cat:
                del result_dict[key]
        except:
            continue
    
    return result_dict
