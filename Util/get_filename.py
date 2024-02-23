import os
import re


if __name__ == '__main__':
    name_list = []
    folder_dir = "/Users/sunmeijie/Downloads/毕业设计/YOLO/data/MAFA/val/"
    for root, dirs, files in os.walk(folder_dir):
        for file in files:
            file = file.strip()
            if file != '.DS_Store' and file != 'filename.txt':
                file = re.sub(r'.(xml|jpg|png)$', "", file)
                name_list.append(file + '\n')
    print(len(name_list), sep='\n')
    name_list = list(set(name_list))
    all_len = len(name_list)
    print(all_len, sep='\n')

    # train = name_list[:int(all_len * 0.85)]
    # val = name_list[int(all_len * 0.85):]
    # print(len(train))
    # print(len(val))

    # with open(folder_dir + '/train.txt', 'w') as f:
    #     f.writelines(train)
    # with open(folder_dir + '/val.txt', 'w') as f:
    #     f.writelines(val)
    with open(folder_dir + '/filename.txt', 'w') as f:
        f.writelines(name_list)


