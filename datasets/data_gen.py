import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm 
import cv2
import os

SIZE = 400
SAVE_DIR = "Full_Dataset"
SAVE_THRESH = [0.1, .9]
IMG_THRESH = 0.20
PRINT_FAILS = False
px_pr = []

def get_image(path, divs = 1):

    img = Image.open(path).resize((SIZE*divs, SIZE*divs))
    np_img = np.asarray(img)
    return np_img

def px_percent(mask):

    return np.mean(mask > 0)

def img_white_percent(img):

    return np.mean(img == [255, 255, 255])

def get_road(mask, idx = 0):

    if (len(mask.shape) == 3):
        road = ~mask[:,:,idx]
    else:
        road = mask

    road = road/(max(road.max(), 1))
    road = road*255
    return road

def save_img(image, out_path):
    # plt.imsave(out_path, image, cmap = "grey")
    
    image = image.astype(np.uint8)
    img = Image.fromarray(image)
    img.save(out_path)

def split_image(image, divs):
    images = []
    for i in range(divs):
        for j in range(divs):
            images.append(image[SIZE*i:SIZE*(i+1), SIZE*j:SIZE*(j+1)])

    return images

        

def save_dirs(out_path):
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    if not os.path.exists(os.path.join(out_path, "images")):
        os.mkdir(os.path.join(out_path, "images"))

    if not os.path.exists(os.path.join(out_path, "groundtruth")):
        os.mkdir(os.path.join(out_path, "groundtruth"))

def aerial_seg(dir = "aerial"):
    city_list = [city for city in os.listdir(f"{dir}") if os.path.isdir(os.path.join(dir, city))]
    image_list = []
    mask_list = []

    for city in city_list:
        files = [file for file in os.listdir(os.path.join(dir, city)) if file[-5] == 'e']
        files = sorted(files)
        for file in tqdm(files,f"Aerial {city} scenes"):
            image = get_image(os.path.join(dir, city, file))
            mask = get_image(os.path.join(dir, city, file.split('_')[0] + "_labels.png"))
            mask = get_road(mask, idx = 0)
            mask_prx = px_percent(mask)
            
            if mask_prx > SAVE_THRESH[0] and mask_prx < SAVE_THRESH[1] and img_white_percent(image) < IMG_THRESH:

                px_pr.append(mask_prx)

                image_list.append("images/" + file)
                mask_list.append("groundtruth/" + file.split('_')[0] + "_labels.png")

                save_img(image, f"{SAVE_DIR}/images/"  + file)
                save_img(mask, f"{SAVE_DIR}/groundtruth/"  + file.split('_')[0] + "_labels.png")
            else:
                if PRINT_FAILS:
                    print(f'File {file} was skipped')
    
        with open(f'{SAVE_DIR}/images_aerial.txt', 'w') as fp:
            fp.write('\n'.join(image_list))

        with open(f'{SAVE_DIR}/masks_aerial.txt', 'w') as fp:
            fp.write('\n'.join(mask_list))

    print(f'{len(image_list)} images were converted')

    return image_list, mask_list

def mass_seg(DIVS = 5, dir = "mass/tiff"):
    splits = [split for split in os.listdir(f"{dir}") if os.path.isdir(f"{dir}/" + split) and split[-1] != 's']
    image_list = []
    mask_list = []

    for split in splits:
        images = [file for file in os.listdir(os.path.join(f"{dir}", split))]
        masks = [file for file in os.listdir(os.path.join(f"{dir}", split + "_labels"))]
        images = sorted(images)
        masks = sorted(masks)

        for image_path, mask_path in tqdm(zip(images, masks),f" Mass {split} scenes", total = len(images)):

            image = get_image(os.path.join(f"{dir}", split, image_path), DIVS)
            mask = get_image(os.path.join(f"{dir}", split + "_labels", mask_path), DIVS)
            mask = get_road(mask)

            image_divs = split_image(image, DIVS)
            mask_divs = split_image(mask, DIVS)
            for idx in range(len(image_divs)):
                mask_prx = px_percent(mask_divs[idx])
                if mask_prx > SAVE_THRESH[0] and mask_prx < SAVE_THRESH[1] and img_white_percent(image_divs[idx]) < IMG_THRESH:
                    
                    px_pr.append(mask_prx)
                    image_list.append("images/" + image_path.split('.')[0]+f'_{idx}.png')
                    mask_list.append("groundtruth/" + mask_path.split('.')[0]+f'_{idx}_labels.png')

                    save_img(image_divs[idx], f"{SAVE_DIR}/images/"  + image_path.split('.')[0]+f'_{idx}.png')
                    save_img(mask_divs[idx], f"{SAVE_DIR}/groundtruth/"  + mask_path.split('.')[0]+f'_{idx}_labels.png')
                else:
                    if PRINT_FAILS:
                        print(f'File {image_path} div ({idx%DIVS}, {idx//DIVS}) was skipped')
    
        with open(f'{SAVE_DIR}/images_mass.txt', 'w') as fp:
            fp.write('\n'.join(image_list))

        with open(f'{SAVE_DIR}/masks_mass.txt', 'w') as fp:
            fp.write('\n'.join(mask_list))

    print(f'{len(image_list)} images were converted')

    return image_list, mask_list

def city_scale_seg(DIVS = 8, dir = "cityScale/data"):
    splits = [split for split in os.listdir(f"{dir}") if os.path.isdir(f"{dir}/" + split)]
    image_list = []
    mask_list = []

    for split in splits:
        images = [file for file in os.listdir(os.path.join(f"{dir}", split)) if file.endswith("sat.png")]
        masks = [file for file in os.listdir(os.path.join(f"{dir}", split)) if file.endswith("gt.png")]
        images = sorted(images)
        masks = sorted(masks)

        for image_path, mask_path in tqdm(zip(images, masks),f" City Scales {split} scenes", total = len(images)):

            image = get_image(os.path.join(f"{dir}", split, image_path), DIVS)
            mask = get_image(os.path.join(f"{dir}", split, mask_path), DIVS)
            mask = get_road(mask)

            image_divs = split_image(image, DIVS)
            mask_divs = split_image(mask, DIVS)
            for idx in range(len(image_divs)):
                mask_prx = px_percent(mask_divs[idx])
                
                if mask_prx > SAVE_THRESH[0] and mask_prx < SAVE_THRESH[1] and img_white_percent(image_divs[idx]) < IMG_THRESH:

                    px_pr.append(mask_prx)
                    image_list.append("images/" + image_path.split('.')[0]+f'_{idx}.png')
                    mask_list.append("groundtruth/" + mask_path.split('.')[0]+f'_{idx}_labels.png')

                    save_img(image_divs[idx], f"{SAVE_DIR}/images/"  + image_path.split('.')[0]+f'_{idx}.png')
                    save_img(mask_divs[idx], f"{SAVE_DIR}/groundtruth/"  + mask_path.split('.')[0]+f'_{idx}_labels.png')
                else:
                    if PRINT_FAILS:
                        print(f'File {image_path} div ({idx%DIVS}, {idx//DIVS}) was skipped')
    
        with open(f'{SAVE_DIR}/images_city_scales.txt', 'w') as fp:
            fp.write('\n'.join(image_list))

        with open(f'{SAVE_DIR}/masks_city_scales.txt', 'w') as fp:
            fp.write('\n'.join(mask_list))

    print(f'{len(image_list)} images were converted')

    return image_list, mask_list

if __name__ == "__main__":
    save_dirs(SAVE_DIR)
    image_list = []
    mask_list = []

    images, masks = city_scale_seg()
    image_list.extend(images)
    mask_list.extend(masks)

    print(np.mean(px_pr))

    images, masks = aerial_seg()
    image_list.extend(images)
    mask_list.extend(masks)

    print(np.mean(px_pr))

    images, masks = mass_seg()
    image_list.extend(images)
    mask_list.extend(masks)

    print(np.mean(px_pr))

    with open(f'{SAVE_DIR}/images.txt', 'w') as fp:
        fp.write('\n'.join(image_list))

    with open(f'{SAVE_DIR}/masks.txt', 'w') as fp:
        fp.write('\n'.join(mask_list))