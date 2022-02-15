import json
o_dict_path=r'D:\FILM_Network\film\results\json\gid2_konwn_dict.json'
t_dict_path=r'D:\FILM_Network\film\image2_dict_film3.json'
o_dict = json.load(open(o_dict_path))
t_dict = json.load(open(t_dict_path))

for i in (o_dict):
    if o_dict[i]=='undefined':
        t_dict[i]='undefined'

jsObj = json.dumps(t_dict, indent=4)  # indent参数是换行和缩进

fileObject = open('image2_join_dict_film3.json', 'w')
fileObject.write(jsObj)
fileObject.close()