import json
#读取image1_dict.json文件
image1_file = open('D:/FILM_Network/film/utils/image2_join_dict_film3.json', 'r')
str_image1 = image1_file.read()
image1_data = json.loads(str_image1)
#读取gid2_konwn.json文件
gid1_konwn_file = open('D:/FILM_Network/film/results/json/gid2_konwn_dict.json', 'r')
str_gid1_konwn = gid1_konwn_file.read()
gid1_konwn_data = json.loads(str_gid1_konwn)
#计算farmland的percision和recall
farmland_TP = 0
farmland_FP = 0
farmland_FN = 0
for i in image1_data:
    if image1_data[i] == gid1_konwn_data[i] and image1_data[i] == "farmland" :
        farmland_TP = farmland_TP + 1
    if image1_data[i] != "farmland" and gid1_konwn_data[i] == "farmland" :
        farmland_FP = farmland_FP + 1
    if image1_data[i] == "farmland" and gid1_konwn_data[i] != "farmland" :
        farmland_FN = farmland_FN + 1
farmland_percison = farmland_TP / (farmland_TP + farmland_FP)
farmland_recall = farmland_TP / (farmland_TP + farmland_FN)
farmland_F1=(2*farmland_percison*farmland_recall)/(farmland_percison+farmland_recall)
print('Farmland')

print(farmland_F1)
#计算build_up的percision和recall
bulit_up_TP = 0
bulit_up_FP = 0
bulit_up_FN = 0
for i in image1_data:
    if image1_data[i] == gid1_konwn_data[i] and image1_data[i] == "bulit_up" :
        bulit_up_TP = bulit_up_TP + 1
    if image1_data[i] != "bulit_up" and gid1_konwn_data[i] == "bulit_up" :
        bulit_up_FP = bulit_up_FP + 1
    if image1_data[i] == "bulit_up" and gid1_konwn_data[i] != "bulit_up" :
        bulit_up_FN = bulit_up_FN + 1
bulit_up_percison = bulit_up_TP / (bulit_up_TP + bulit_up_FP)
bulit_up_recall = bulit_up_TP / (bulit_up_TP + bulit_up_FN)
bulit_up_F1=(2*bulit_up_percison*bulit_up_recall)/(bulit_up_percison+bulit_up_recall)
print('bulitup')

print(bulit_up_F1)
#计算meadow的percision和recall
meadow_TP = 0
meadow_FP = 0
meadow_FN = 0
for i in image1_data:
    if image1_data[i] == gid1_konwn_data[i] and image1_data[i] == "meadow" :
        meadow_TP = meadow_TP + 1
    if image1_data[i] != "meadow" and gid1_konwn_data[i] == "meadow" :
        meadow_FP = meadow_FP + 1
    if image1_data[i] == "meadow" and gid1_konwn_data[i] != "meadow" :
        meadow_FN = meadow_FN + 1
meadow_F1=0
if meadow_TP != 0:
    meadow_percison = meadow_TP / (meadow_TP + meadow_FP)
    meadow_recall = meadow_TP / (meadow_TP + meadow_FN)
    meadow_F1 = (2 * meadow_percison * meadow_recall) / (meadow_percison + meadow_recall)
    print('meadow')

    print(meadow_F1)
else:
    print(0)
    print(0)
#计算forest的percision和recall
forest_TP = 0
forest_FP = 0
forest_FN = 0
forest_F1=0
for i in image1_data:
    if image1_data[i] == gid1_konwn_data[i] and image1_data[i] == "forest" :
        forest_TP = forest_TP + 1
    if image1_data[i] != "forest" and gid1_konwn_data[i] == "forest" :
        forest_FP = forest_FP + 1
    if image1_data[i] == "forest" and gid1_konwn_data[i] != "forest" :
        forest_FN = forest_FN + 1
forest_percison = forest_TP / (forest_TP + forest_FP)
forest_recall = forest_TP / (forest_TP + forest_FN)
forest_F1 = (2 * forest_percison * forest_recall) / (forest_percison + forest_recall)
print('forest')
print(forest_F1)
#计算water的percision和recall
water_TP = 0
water_FP = 0
water_FN = 0
water_F1=0
for i in image1_data:
    if image1_data[i] == gid1_konwn_data[i] and image1_data[i] == "water" :
        water_TP = water_TP + 1
    if image1_data[i] != "water" and gid1_konwn_data[i] == "water" :
        water_FP = water_FP + 1
    if image1_data[i] == "water" and gid1_konwn_data[i] != "water" :
        water_FN = water_FN + 1
if water_TP != 0:
    water_percison = water_TP / (water_TP + water_FP)
    water_recall = water_TP / (water_TP + water_FN)
    water_F1 = (2 * water_percison * water_recall) / (water_percison + water_recall)
    print('water')

    print(water_F1)
else:
    print(0)
    print(0)

print(farmland_F1,bulit_up_F1,meadow_F1,forest_F1,water_F1)