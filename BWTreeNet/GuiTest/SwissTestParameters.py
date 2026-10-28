img_rows = 1000
img_cols = 1000
window_shape = 1000
source_img_band = 1
classnum = 2
result_img_band = 1

gpu = 0
test_big_1983 = '../data/ori/1983_big/'
test_big_1985 = '../data/ori/1985_big/'
test_big_1980 = '../data/ori/1980_big/'
test_big_1981 = '../data/ori/1981_big/'
test_big_1982 = '../data/ori/1982_big/'
test_big_1984 = '../data/ori/1984_big/'
result_path = '../code/GuiTest/result/'

#strategy
normalization_strategy = True

overlap_strategy = True
stride =600

abandon_strategy = True
delete_edge_strategy = True
delete_edge_value = 200

flip_strategy = True

ratio_output_strategy = False

test_path = [test_big_1980,test_big_1981,test_big_1982,test_big_1983,test_big_1984,test_big_1985]

model_name = 'BWTreeNet'
model_path = '../code/GuiTest/BWTreeNet_xxxx.pth'
project_name = 'Test1980'
