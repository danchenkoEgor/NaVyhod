import os
import cv2
import joblib
import asyncio
import PIL.Image as Image

from scipy.spatial import distance
import numpy as np
from sklearn.cluster import KMeans

from sklearn.metrics.pairwise import cosine_distances, euclidean_distances, manhattan_distances

from aiogram import Bot, Dispatcher, executor, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import StatesGroup, State
from aiogram.dispatcher.handler import CancelHandler
from aiogram.dispatcher.middlewares import BaseMiddleware
from aiogram.types import ReplyKeyboardRemove, ReplyKeyboardMarkup, KeyboardButton

import warnings
warnings.filterwarnings("ignore")

from funcs import load_model, prediction, find_matching_clothes, find_hexagon_colors, find_analogous_colors, find_triadic_colors

TOKEN = ''

bot = Bot(token = TOKEN)
dp = Dispatcher(bot=bot)

# Загружаем модели

model1, model2, model3, model4 = load_model()



# Бот начинает работать
@dp.message_handler(commands=['start'])
async def start_answer(message: types.Message):
    await message.answer('Привет!')

class AlbumMiddleware(BaseMiddleware):
    """This middleware is for capturing media groups."""

    album_data: dict = {}

    def __init__(self, latency: int | float = 0.01):
        """
        You can provide custom latency to make sure
        albums are handled properly in highload.
        """
        self.latency = latency
        super().__init__()

    async def on_process_message(self, message: types.Message, data: dict):
        if not message.media_group_id:
            return

        try:
            self.album_data[message.media_group_id].append(message)
            raise CancelHandler()  # Tell aiogram to cancel handler for this group element
        except KeyError:
            self.album_data[message.media_group_id] = [message]
            await asyncio.sleep(self.latency)

            message.conf["is_last"] = True
            data["album"] = self.album_data[message.media_group_id]

    async def on_post_process_message(self, message: types.Message, result: dict, data: dict):
        """Clean up after handling our album."""
        if message.media_group_id and message.conf.get("is_last"):
            del self.album_data[message.media_group_id]

@dp.message_handler(content_types=['photo'])
async def get_message(message: types.Message, album: list[types.Message] = None):
    if not album:
        album = [message]

    media_group = types.MediaGroup()
    
    all_types = []
    
    for obj in album:
        if obj.photo:

            user_id = obj.from_user.id
            photo = obj.photo[-1]

            try:
                os.mkdir(f'jsons/user_{user_id}')
                os.mkdir(f'photos/user_{user_id}')
                ind = len(os.listdir(f'photos/user_{user_id}'))
                photo_id = photo.file_id
                photo_file = await bot.get_file(photo_id)

                # Сохраняем фотографию
                file_name = f'photos/user_{user_id}/photo_{ind}.jpg'  # Генерируем уникальное имя файла
                await bot.download_file(photo_file.file_path, file_name)
            
            except:
                ind = len(os.listdir(f'photos/user_{user_id}'))
                photo_id = photo.file_id
                photo_file = await bot.get_file(photo_id)

                # Сохраняем фотографию
                file_name = f'photos/user_{user_id}/photo_{ind}.jpg'  # Генерируем уникальное имя файла
                await bot.download_file(photo_file.file_path, file_name)

            image = cv2.imread(file_name)
            image_resize = cv2.resize(image, (224, 224))
            try:
                os.mkdir(f'photos/user_{user_id}/resize_photos')
                cv2.imwrite(f'photos/user_{user_id}/resize_photos/photo_{ind}.jpg', image_resize)
            except:
                cv2.imwrite(f'photos/user_{user_id}/resize_photos/photo_{ind}.jpg', image_resize)


            cloth_dict = prediction(model1, model2, model3, model4,
                                    path_for_pic=f'photos/user_{user_id}/resize_photos/photo_{ind}.jpg')
            
            joblib.dump(cloth_dict, f'jsons/user_{user_id}/photo_{ind}.json')

            r = joblib.load(f'jsons/user_{user_id}/photo_{ind}.json')
            print(r)

    
    all_files = os.listdir(f'jsons/user_{user_id}')
    try:
        all_files.remove('alltypes.json')
        for item in all_files:
            photo_info = joblib.load(f'jsons/user_{user_id}/{item}')
            if 'articalType' in photo_info:
                all_types.append(photo_info['articalType'])
    except:
        for item in all_files:
            photo_info = joblib.load(f'jsons/user_{user_id}/{item}')
            if 'articalType' in photo_info:
                all_types.append(photo_info['articalType'])
       
    print(set(all_types))
    joblib.dump(set(all_types), f'jsons/user_{user_id}/alltypes.json')

    await message.answer('Все фотографии загружены!')



@dp.message_handler(commands=['outfits_topwear'])
async def outfit_recomendation(messege: types.Message):
    user_id = messege.from_user.id
    try:
        dict = find_matching_clothes(f'jsons/user_{user_id}', cat='topwear', user_id=user_id, color_func=find_triadic_colors)
        print(dict)
        keys = list(dict)
        media = types.MediaGroup()
        photos = dict[keys[0]]
        for item in photos:
            media.attach_photo(types.InputFile(f'photos/user_{user_id}/{item}'))

        await messege.answer_media_group(media=media)
        await messege.answer('Твой первый лук!')

        media = types.MediaGroup()
        photos = dict[keys[1]]
        for item in photos:
            media.attach_photo(types.InputFile(f'photos/user_{user_id}/{item}'))

        await messege.answer_media_group(media=media)
        await messege.answer('Твой второй лук!')

    except:
        await messege.answer('Загрузите фотографии верхней одежды')

@dp.message_handler(commands=['outfits_bottomwear'])
async def outfit_recomendation(messege: types.Message):
    user_id = messege.from_user.id
    try:
        dict = find_matching_clothes(f'jsons/user_{user_id}', cat='bottomwear', user_id=user_id, color_func=find_triadic_colors)
        print(dict)
        keys = list(dict)
        media = types.MediaGroup()
        photos = dict[keys[0]]
        for item in photos:
            media.attach_photo(types.InputFile(f'photos/user_{user_id}/{item}'))

        await messege.answer_media_group(media=media)
        await messege.answer('Твой первый лук!')

        media = types.MediaGroup()
        photos = dict[keys[1]]
        for item in photos:
            media.attach_photo(types.InputFile(f'photos/user_{user_id}/{item}'))

        await messege.answer_media_group(media=media)
        await messege.answer('Твой второй лук!')

    except:
        await messege.answer('Загрузите фотографии нижней одежды')

@dp.message_handler(commands='delete')
async def delete_photos(messege: types.Message):

    user_id = messege.from_user.id
    photo_files = os.listdir(f'photos/user_{user_id}')
    json_files = os.listdir(f'jsons/user_{user_id}')

    try:
        for item in photo_files:
            os.remove(f'photos/user_{user_id}/{item}')
    except:
        for file in json_files:
            os.remove(f'jsons/user_{user_id}/{file}')

    await messege.answer('Фотографии удалены')


if __name__ == '__main__':
    dp.middleware.setup(AlbumMiddleware())
    executor.start_polling(dp, skip_updates=True)